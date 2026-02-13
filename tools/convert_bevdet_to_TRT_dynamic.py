import argparse
import os
from typing import Dict, Optional, Sequence, Union

import h5py
import mmcv
import numpy as np
import onnx
import pycuda.driver as cuda
import tensorrt as trt
import torch
import tqdm
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdeploy.apis.core import no_mp
from mmdeploy.backend.tensorrt.calib_utils import HDF5Calibrator
from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin
from mmdeploy.backend.tensorrt.utils import save, search_cuda_version
from mmdeploy.utils import load_config
from packaging import version
from torch.utils.data import DataLoader

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module

'''
Example command 1 to run:
python tools/convert_bevdet_to_TRT_dynamic.py \
  configs/bevdet/bevdet-r50.py \
  /home/byounghun/workspace/BEVDet/ckpts/bevdet-r50.pth \
  /home/byounghun/workspace/BEVDet/tensorrt \
  --fp16 --int8 --fuse-conv-bn

Example command 2 (with profile scan):
python tools/convert_bevdet_to_TRT_dynamic.py \
  configs/bevdet/bevdet-r50.py \
  /home/byounghun/workspace/BEVDet/ckpts/bevdet-r50.pth \
  /home/byounghun/workspace/BEVDet/tensorrt \
  --fp16 --int8 --fuse-conv-bn \
  --use-profile-scan --shape-scan-num 0

Example command 3 (for BEVDet4D model):
python tools/convert_bevdet_to_TRT_dynamic.py \
  configs/bevdet/bevdet-r50-4d-depth-cbgs.py \
  /home/byounghun/workspace/BEVDet/ckpts/bevdet-r50-4d-depth-cbgs.pth \
  /home/byounghun/workspace/BEVDet/tensorrt \
  --prefix bevdet-r50-4d-depth-cbgs \
  --fp16 --fuse-conv-bn --max-pad-ratio 1.1 \

### let DepthNet as original FP32 ###
  --force-depth-fp32
'''
[]
class HDF5CalibratorBEVDet(HDF5Calibrator):

    def get_batch(self, names: Sequence[str], **kwargs) -> list:
        """Get batch data."""
        if self.count < self.dataset_length:
            if self.count % 100 == 0:
                print('%d/%d' % (self.count, self.dataset_length))
            ret = []
            for name in names:
                input_group = self.calib_data[name]
                if name in ('img', 'mlp_input'):
                    data_np = input_group[str(self.count)][...].astype(
                        np.float32)
                else:
                    data_np = input_group[str(self.count)][...].astype(
                        np.int32)

                # tile the tensor so we can keep the same distribute
                opt_shape = self.input_shapes[name]['opt_shape']
                data_shape = data_np.shape

                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ]

                data_np = np.tile(data_np, reps)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list]

                data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes)
                cuda.memcpy_htod(data_np_cuda_ptr,
                                 np.ascontiguousarray(data_np))
                self.buffers[name] = data_np_cuda_ptr

                ret.append(self.buffers[name])
            self.count += 1
            return ret
        else:
            return None


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='work dir to save file')
    parser.add_argument(
        '--prefix', default='bevdet', help='prefix of the save file name')
    parser.add_argument(
        '--fp16', action='store_true', help='Whether to use tensorrt fp16')
    parser.add_argument(
        '--int8', action='store_true', help='Whether to use tensorrt int8')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--shape-scan-num',
        type=int,
        default=0,
        help='Number of batches to scan for min/opt/max shapes. 0 means full dataset.'
    )
    parser.add_argument(
        '--use-profile-scan',
        action='store_true',
        help='Scan dataset to compute min/opt/max instead of using defaults.'
    )
    parser.add_argument(
        '--max-pad-ratio',
        type=float,
        default=0.0,
        help='Extra ratio added to max num_points/num_intervals. e.g., 0.2 adds 20%.'
    )
    parser.add_argument(
        '--min-pad-ratio',
        type=float,
        default=0.0,
        help='Ratio to reduce min num_points/num_intervals. e.g., 0.2 reduces min by 20%.'
    )
    parser.add_argument(
        '--profile-multiplier',
        type=float,
        default=1.0,
        help='Multiply min/opt/max num_points/num_intervals by this factor.'
    )
    parser.add_argument(
        '--auto-multiply-num-frame',
        action='store_true',
        help='Multiply profiles by model.num_frame when available (e.g., 4D).'
    )
    parser.add_argument(
        '--points-profile',
        default='178900,179900,180008',
        help='num_points profile as min,opt,max (e.g., 178900,179900,180008).'
    )
    parser.add_argument(
        '--intervals-profile',
        default='11284,11739,12100',
        help='num_intervals profile as min,opt,max (e.g., 11284,11739,12100).'
    )
    parser.add_argument(
        '--force-depth-fp32',
        action='store_true',
        help='Force DepthNet to run in FP32 during export.'
    )
    parser.add_argument(
        '--trt-depth-fp32',
        action='store_true',
        help='Force DepthNet layers to FP32 during TensorRT build.'
    )
    args = parser.parse_args()
    return args


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def create_calib_input_data_impl(calib_file: str,
                                 dataloader: DataLoader,
                                 model_partition: bool = False,
                                 metas: list = []) -> None:
    with h5py.File(calib_file, mode='w') as file:
        calib_data_group = file.create_group('calib_data')
        assert not model_partition
        # create end2end group
        input_data_group = calib_data_group.create_group('end2end')
        input_group_img = input_data_group.create_group('img')
        input_keys = [
            'ranks_bev', 'ranks_depth', 'ranks_feat', 'interval_starts',
            'interval_lengths'
        ]
        if len(metas) == 6:
            input_keys.append('mlp_input')
        input_groups = []
        for input_key in input_keys:
            input_groups.append(input_data_group.create_group(input_key))
        if len(metas) == 6:
            metas = [
                metas[i].int().detach().cpu().numpy() for i in range(5)
            ] + [metas[5].float().detach().cpu().numpy()]
        else:
            metas = [
                metas[i].int().detach().cpu().numpy() for i in range(len(metas))
            ]
        for data_id, input_data in enumerate(tqdm.tqdm(dataloader)):
            # save end2end data
            input_tensor = input_data['img_inputs'][0][0]
            input_ndarray = input_tensor.squeeze(0).detach().cpu().numpy()
            input_group_img.create_dataset(
                str(data_id),
                shape=input_ndarray.shape,
                compression='gzip',
                compression_opts=4,
                data=input_ndarray)
            for kid, input_key in enumerate(input_keys):
                input_groups[kid].create_dataset(
                    str(data_id),
                    shape=metas[kid].shape,
                    compression='gzip',
                    compression_opts=4,
                    data=metas[kid])
            file.flush()


def create_calib_input_data(calib_file: str,
                            deploy_cfg: Union[str, mmcv.Config],
                            model_cfg: Union[str, mmcv.Config],
                            model_checkpoint: Optional[str] = None,
                            dataset_cfg: Optional[Union[str,
                                                        mmcv.Config]] = None,
                            dataset_type: str = 'val',
                            device: str = 'cpu',
                            metas: list = [None]) -> None:
    """Create dataset for post-training quantization."""
    with no_mp():
        if dataset_cfg is None:
            dataset_cfg = model_cfg

        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        if dataset_cfg is None:
            dataset_cfg = model_cfg

        dataset_cfg = load_config(dataset_cfg)[0]

        from mmdeploy.apis.utils import build_task_processor
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)

        dataset = task_processor.build_dataset(dataset_cfg, dataset_type)

        dataloader = task_processor.build_dataloader(
            dataset, 1, 1, dist=False, shuffle=False)

        create_calib_input_data_impl(
            calib_file, dataloader, model_partition=False, metas=metas)


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              int8_mode: bool = False,
              int8_param: Optional[dict] = None,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              force_depth_fp32: bool = False,
              **kwargs) -> trt.ICudaEngine:
    """Create a tensorrt engine from ONNX.

    Modified from mmdeploy.backend.tensorrt.utils.from_onnx
    """

    old_cuda_device = os.environ.get('CUDA_DEVICE', None)
    os.environ['CUDA_DEVICE'] = str(device_id)
    import pycuda.autoinit  # noqa:F401
    if old_cuda_device is not None:
        os.environ['CUDA_DEVICE'] = old_cuda_device
    else:
        os.environ.pop('CUDA_DEVICE')

    load_tensorrt_plugin()
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    cuda_version = search_cuda_version()
    if cuda_version is not None:
        version_major = int(cuda_version.split('.')[0])
        if version_major < 11:
            tactic_source = config.get_tactic_sources() - (
                1 << int(trt.TacticSource.CUBLAS_LT))
            config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = HDF5CalibratorBEVDet(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))
        if version.parse(trt.__version__) < version.parse('8'):
            builder.int8_mode = int8_mode
            builder.int8_calibrator = config.int8_calibrator

    if force_depth_fp32:
        if hasattr(trt.BuilderFlag, 'STRICT_TYPES'):
            config_flag = trt.BuilderFlag.STRICT_TYPES
        else:
            config_flag = None
        if config_flag is not None:
            config.set_flag(config_flag)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if 'depth_net' in layer.name:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)

    engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'

    save(engine, output_file_prefix + '.engine')
    return engine


def scan_profile_shapes(data_loader, model, max_batches, need_mlp_input=False):
    num_points_list = []
    num_intervals_list = []
    first_img = None
    first_metas = None
    first_mlp_input = None

    total = len(data_loader)
    limit = total if max_batches == 0 else min(max_batches, total)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= limit:
                break
            inputs = [t.cuda() for t in data['img_inputs'][0]]
            metas = model.get_bev_pool_input(inputs)

            num_points_list.append(int(metas[0].shape[0]))
            num_intervals_list.append(int(metas[3].shape[0]))

            if first_img is None:
                first_metas = metas
                if need_mlp_input:
                    imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
                        bda, _ = model.prepare_inputs(inputs)
                    first_img = imgs[0].squeeze(0)
                    first_mlp_input = model.img_view_transformer.get_mlp_input(
                        sensor2keyegos[0], ego2globals[0], intrins[0],
                        post_rots[0], post_trans[0], bda)
                else:
                    first_img = inputs[0].squeeze(0)

            if (i + 1) % 100 == 0:
                print(f'shape scan {i + 1}/{limit}')

    if not num_points_list or not num_intervals_list:
        raise RuntimeError('Failed to scan shapes from dataloader')

    def _stats(values):
        v = np.array(values, dtype=np.int64)
        return int(v.min()), int(np.percentile(v, 90)), int(v.max())

    points_stats = _stats(num_points_list)
    intervals_stats = _stats(num_intervals_list)

    return first_img, first_metas, points_stats, intervals_stats, first_mlp_input


def main():
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    load_tensorrt_plugin()
    assert 'bev_pool_v2' in get_plugin_names(), \
        'bev_pool_v2 is not in the plugin list of tensorrt, ' \
        'please install mmdeploy from ' \
        'https://github.com/HuangJunJie2017/mmdeploy.git'

    if args.int8:
        assert args.fp16
    model_prefix = args.prefix + '_dynamic'
    if args.int8:
        model_prefix = model_prefix + '_int8'
    elif args.fp16:
        model_prefix = model_prefix + '_fp16'

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    assert model.img_view_transformer.grid_size[0] == 128
    assert model.img_view_transformer.grid_size[1] == 128
    assert model.img_view_transformer.grid_size[2] == 1
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model.cuda()
    model.eval()
    if args.force_depth_fp32:
        model.force_depth_fp32 = True

    def _parse_profile(value, name):
        if value is None:
            return None
        parts = [p.strip() for p in value.split(',')]
        if len(parts) != 3:
            raise ValueError(f'{name} must be "min,opt,max"')
        return tuple(int(p) for p in parts)

    mlp_input = None
    need_mlp_input = (model.__class__.__name__ == 'BEVDepth4DTRT')
    if args.use_profile_scan:
        img, metas, points_stats, intervals_stats, mlp_input = scan_profile_shapes(
            data_loader, model, args.shape_scan_num, need_mlp_input)
    else:
        points_stats = _parse_profile(args.points_profile, 'points-profile')
        intervals_stats = _parse_profile(args.intervals_profile, 'intervals-profile')
        data_iter = iter(data_loader)
        data = next(data_iter)
        inputs = [t.cuda() for t in data['img_inputs'][0]]
        metas = model.get_bev_pool_input(inputs)
        if need_mlp_input:
            imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
                bda, _ = model.prepare_inputs(inputs)
            img = imgs[0].squeeze(0)
            mlp_input = model.img_view_transformer.get_mlp_input(
                sensor2keyegos[0], ego2globals[0], intrins[0],
                post_rots[0], post_trans[0], bda)
        else:
            img = inputs[0].squeeze(0)

    print('num_points (min/opt/max):', points_stats)
    print('num_intervals (min/opt/max):', intervals_stats)

    onnx_path = os.path.join(args.work_dir, f'{model_prefix}.onnx')
    engine_prefix = os.path.join(args.work_dir, model_prefix)

    if not os.path.exists(onnx_path):
        with torch.no_grad():
            if need_mlp_input and mlp_input is None:
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
                    bda, _ = model.prepare_inputs(inputs)
                img = imgs[0].squeeze(0)
                mlp_input = model.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrins[0],
                    post_rots[0], post_trans[0], bda)
            torch.onnx.export(
                model,
                (img.float().contiguous(),
                 mlp_input.contiguous() if need_mlp_input else torch.empty(0),
                 metas[1].int().contiguous(), metas[2].int().contiguous(),
                 metas[0].int().contiguous(), metas[3].int().contiguous(),
                 metas[4].int().contiguous())
                if need_mlp_input else
                (img.float().contiguous(), metas[1].int().contiguous(),
                 metas[2].int().contiguous(), metas[0].int().contiguous(),
                 metas[3].int().contiguous(), metas[4].int().contiguous()),
                onnx_path,
                opset_version=11,
                input_names=[
                    'img', 'mlp_input', 'ranks_depth', 'ranks_feat',
                    'ranks_bev', 'interval_starts', 'interval_lengths'
                ] if need_mlp_input else [
                    'img', 'ranks_depth', 'ranks_feat', 'ranks_bev',
                    'interval_starts', 'interval_lengths'
                ],
                dynamic_axes={
                    'mlp_input': {0: 'batch', 1: 'num_cams'},
                    'ranks_depth': {0: 'num_points'},
                    'ranks_feat': {0: 'num_points'},
                    'ranks_bev': {0: 'num_points'},
                    'interval_starts': {0: 'num_intervals'},
                    'interval_lengths': {0: 'num_intervals'},
                } if need_mlp_input else {
                    'ranks_depth': {0: 'num_points'},
                    'ranks_feat': {0: 'num_points'},
                    'ranks_bev': {0: 'num_points'},
                    'interval_starts': {0: 'num_intervals'},
                    'interval_lengths': {0: 'num_intervals'},
                },
                output_names=[f'output_{j}' for j in
                              range(6 * len(model.pts_bbox_head.task_heads))])
    else:
        print(f'Skip ONNX export, found {onnx_path}')

    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print('ONNX Model Incorrect')
    else:
        print('ONNX Model Correct')

    img_shape = img.shape
    min_points, opt_points, max_points = points_stats
    min_intervals, opt_intervals, max_intervals = intervals_stats

    if args.max_pad_ratio > 0:
        max_points = int(np.ceil(max_points * (1.0 + args.max_pad_ratio)))
        max_intervals = int(np.ceil(max_intervals * (1.0 + args.max_pad_ratio)))
        opt_points = min(opt_points, max_points)
        opt_intervals = min(opt_intervals, max_intervals)
    if args.min_pad_ratio > 0:
        min_points = max(1, int(np.floor(min_points * (1.0 - args.min_pad_ratio))))
        min_intervals = max(1, int(np.floor(min_intervals * (1.0 - args.min_pad_ratio))))
        opt_points = max(opt_points, min_points)
        opt_intervals = max(opt_intervals, min_intervals)
    if args.auto_multiply_num_frame:
        num_frame = int(getattr(model, 'num_frame', 1))
        if num_frame > 1:
            args.profile_multiplier *= num_frame
    if args.profile_multiplier != 1.0:
        min_points = max(1, int(np.floor(min_points * args.profile_multiplier)))
        opt_points = max(1, int(np.floor(opt_points * args.profile_multiplier)))
        max_points = max(1, int(np.floor(max_points * args.profile_multiplier)))
        min_intervals = max(1, int(np.floor(min_intervals * args.profile_multiplier)))
        opt_intervals = max(1, int(np.floor(opt_intervals * args.profile_multiplier)))
        max_intervals = max(1, int(np.floor(max_intervals * args.profile_multiplier)))
        opt_points = min(opt_points, max_points)
        opt_intervals = min(opt_intervals, max_intervals)
        min_points = min(min_points, opt_points)
        min_intervals = min(min_intervals, opt_intervals)

    input_shapes = dict(
        img=dict(
            min_shape=img_shape, opt_shape=img_shape, max_shape=img_shape),
        ranks_depth=dict(
            min_shape=[min_points],
            opt_shape=[opt_points],
            max_shape=[max_points]),
        ranks_feat=dict(
            min_shape=[min_points],
            opt_shape=[opt_points],
            max_shape=[max_points]),
        ranks_bev=dict(
            min_shape=[min_points],
            opt_shape=[opt_points],
            max_shape=[max_points]),
        interval_starts=dict(
            min_shape=[min_intervals],
            opt_shape=[opt_intervals],
            max_shape=[max_intervals]),
        interval_lengths=dict(
            min_shape=[min_intervals],
            opt_shape=[opt_intervals],
            max_shape=[max_intervals]))
    if need_mlp_input:
        mlp_shape = list(mlp_input.shape)
        input_shapes = {
            'img': input_shapes['img'],
            'mlp_input': dict(
                min_shape=mlp_shape, opt_shape=mlp_shape, max_shape=mlp_shape),
            'ranks_depth': input_shapes['ranks_depth'],
            'ranks_feat': input_shapes['ranks_feat'],
            'ranks_bev': input_shapes['ranks_bev'],
            'interval_starts': input_shapes['interval_starts'],
            'interval_lengths': input_shapes['interval_lengths'],
        }

    deploy_cfg = dict(
        backend_config=dict(
            type='tensorrt',
            common_config=dict(
                fp16_mode=args.fp16,
                max_workspace_size=1073741824,
                int8_mode=args.int8),
            model_inputs=[dict(input_shapes=input_shapes)]),
        codebase_config=dict(
            type='mmdet3d', task='VoxelDetection', model_type='end2end'))

    if args.int8:
        calib_filename = 'calib_data.h5'
        calib_path = os.path.join(args.work_dir, calib_filename)
        calib_metas = metas if not need_mlp_input else list(metas) + [mlp_input]
        create_calib_input_data(
            calib_path,
            deploy_cfg,
            args.config,
            args.checkpoint,
            dataset_cfg=None,
            dataset_type='val',
            device='cuda:0',
            metas=calib_metas)

    from_onnx(
        onnx_path,
        engine_prefix,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        int8_param=dict(
            calib_file=os.path.join(args.work_dir, 'calib_data.h5'),
            model_type='end2end'),
        max_workspace_size=1 << 30,
        input_shapes=input_shapes,
        force_depth_fp32=args.trt_depth_fp32)

    if args.int8:
        os.remove(calib_path)


if __name__ == '__main__':
    main()
