#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU TensorRT evaluation script for BEVDet.
- Uses torch.distributed to shard dataset
- Runs inference with TRT
- Decodes boxes using BEVDetTRT head
- Gathers results by sample_idx and evaluates on rank0

Example usage:
torchrun --nproc_per_node=4 tools/test_trt_bevdet_dist.py \
  configs/bevdet/bevdet-r50.py \
  /home/byounghun/workspace/BEVDet/tensorrt/bevdet_dynamic_int8_fuse.engine \
  --metric bbox \
  --out-dir work_dirs/trt_eval
"""

import argparse
import os
import time

import numpy as np
import torch
import mmcv
from mmcv import Config

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

from mmcv.runner import load_checkpoint
from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.core.bbox import bbox3d2result

'''
torchrun --nproc_per_node=4 tools/test_trt_bevdet_dist.py \
  configs/bevdet/bevdet-r50-4d-depth-cbgs.py \
  /home/byounghun/workspace/BEVDet/tensorrt/bevdet-r50-4d-depth-cbgs_dynamic_fp16_fuse.engine \
  --metric bbox   --out-dir work_dirs/trt_eval 
'''

class HostDeviceMem:
    def __init__(self, host, device, name, dtype, shape):
        self.host = host
        self.device = device
        self.name = name
        self.dtype = dtype
        self.shape = shape


def load_engine(engine_path, logger):
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError('Failed to load TensorRT engine')
    return engine


def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        idx = engine.get_binding_index(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        shape = context.get_binding_shape(idx)
        if shape is None or any(dim < 0 for dim in shape):
            raise RuntimeError(f'Binding {binding} has dynamic shape. Set shapes first.')
        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        mem = HostDeviceMem(host_mem, device_mem, binding, dtype, shape)
        if engine.binding_is_input(binding):
            inputs.append(mem)
        else:
            outputs.append(mem)

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    t0 = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    stream.synchronize()
    t1 = time.time()
    return t1 - t0


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BEVDet TensorRT engine (multi-GPU)')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('engine', help='TensorRT engine file')
    parser.add_argument('--checkpoint', default=None, help='optional pytorch checkpoint for decoder')
    parser.add_argument('--metric', default='bbox')
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--out-dir', default='work_dirs/trt_eval')
    return parser.parse_args()


def init_dist():
    if not torch.distributed.is_available():
        raise RuntimeError('torch.distributed not available')
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return rank, world_size, local_rank


def main():
    args = parse_args()
    rank, world_size, local_rank = init_dist()

    load_tensorrt_plugin()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'

    if not os.path.exists(args.out_dir) and rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    torch.distributed.barrier()

    if hasattr(cfg.data, 'test'):
        test_data_cfg = cfg.data.test
    else:
        raise RuntimeError('Config has no test dataset')

    test_data_cfg.test_mode = True
    dataset = build_dataset(test_data_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    logger = trt.Logger(trt.Logger.ERROR)
    engine = load_engine(args.engine, logger)
    context = engine.create_execution_context()

    input_names = [name for name in engine if engine.binding_is_input(name)]
    output_names = [name for name in engine if not engine.binding_is_input(name)]
    output_names = sorted(output_names, key=lambda n: int(n.split('_')[-1]))

    buffers_ready = False
    cached_shapes = {}
    profile_shapes = {}
    for name in input_names:
        try:
            profile_shapes[name] = engine.get_profile_shape(0, name)
        except Exception:
            profile_shapes[name] = None
    inputs_buf, outputs_buf, bindings, stream = None, None, None, None

    token_to_idx = None
    if hasattr(dataset, 'data_infos'):
        token_to_idx = {info['token']: i for i, info in enumerate(dataset.data_infos)}

    results_by_idx = {}
    times = []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    else:
        prog_bar = None

    for data in data_loader:
        inputs = [t.cuda() for t in data['img_inputs'][0]]
        metas = model.get_bev_pool_input(inputs)
        need_mlp_input = 'mlp_input' in input_names
        mlp_input = None
        if need_mlp_input:
            imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
                bda, _ = model.prepare_inputs(inputs)
            img = imgs[0].squeeze(0).contiguous().float()
            mlp_input = model.img_view_transformer.get_mlp_input(
                sensor2keyegos[0], ego2globals[0], intrins[0],
                post_rots[0], post_trans[0], bda)
        else:
            img = inputs[0].squeeze(0).contiguous().float()
        ranks_depth = metas[1].contiguous().int()
        ranks_feat = metas[2].contiguous().int()
        ranks_bev = metas[0].contiguous().int()
        interval_starts = metas[3].contiguous().int()
        interval_lengths = metas[4].contiguous().int()

        # set binding shapes (dynamic profile only)
        for name in input_names:
            if name == 'img':
                actual = tuple(img.shape)
            elif name == 'mlp_input':
                actual = tuple(mlp_input.shape)
            elif name == 'ranks_depth':
                actual = tuple(ranks_depth.shape)
            elif name == 'ranks_feat':
                actual = tuple(ranks_feat.shape)
            elif name == 'ranks_bev':
                actual = tuple(ranks_bev.shape)
            elif name == 'interval_starts':
                actual = tuple(interval_starts.shape)
            elif name == 'interval_lengths':
                actual = tuple(interval_lengths.shape)
            else:
                raise RuntimeError(f'Unexpected input name: {name}')

            prof = profile_shapes.get(name)
            if prof is not None:
                min_shape, _, max_shape = prof
                if any(a < t for a, t in zip(actual, min_shape)) or any(
                        a > t for a, t in zip(actual, max_shape)):
                    raise RuntimeError(
                        f'Input {name} shape {actual} outside profile '
                        f'min={tuple(min_shape)} max={tuple(max_shape)}. '
                        'Rebuild engine with a wider dynamic profile.')
            shape = actual

            idx = engine.get_binding_index(name)
            context.set_binding_shape(idx, shape)

        current_shapes = {name: tuple(context.get_binding_shape(engine.get_binding_index(name)))
                          for name in input_names}
        if (not buffers_ready) or (current_shapes != cached_shapes):
            inputs_buf, outputs_buf, bindings, stream = allocate_buffers(engine, context)
            cached_shapes = current_shapes
            buffers_ready = True

        for buf in inputs_buf:
            if buf.name == 'img':
                arr = img.cpu().numpy()
            elif buf.name == 'mlp_input':
                arr = mlp_input.cpu().numpy()
            elif buf.name == 'ranks_depth':
                arr = ranks_depth.cpu().numpy()
            elif buf.name == 'ranks_feat':
                arr = ranks_feat.cpu().numpy()
            elif buf.name == 'ranks_bev':
                arr = ranks_bev.cpu().numpy()
            elif buf.name == 'interval_starts':
                arr = interval_starts.cpu().numpy()
            elif buf.name == 'interval_lengths':
                arr = interval_lengths.cpu().numpy()
            else:
                raise RuntimeError(f'Unexpected input name: {buf.name}')
            buf.host[:] = arr.reshape(-1)

        t = do_inference(context, bindings, inputs_buf, outputs_buf, stream)
        times.append(t)

        outputs_np = {}
        for buf in outputs_buf:
            outputs_np[buf.name] = buf.host.reshape(buf.shape)

        outputs_list = []
        for name in output_names:
            outputs_list.append(torch.from_numpy(outputs_np[name]).cuda())

        outs = model.result_deserialize(outputs_list)
        img_metas = data['img_metas'][0].data[0]
        bbox_list = model.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)
        for meta, (bboxes, scores, labels) in zip(img_metas, bbox_list):
            result = bbox3d2result(bboxes, scores, labels)
            if token_to_idx is None:
                raise RuntimeError('token_to_idx not available for this dataset')
            idx = token_to_idx[meta['sample_idx']]
            results_by_idx[idx] = result

        if prog_bar is not None:
            for _ in range(len(img_metas)):
                prog_bar.update()

    # save per-rank results
    result_path = os.path.join(args.out_dir, f'results_rank{rank}.pkl')
    mmcv.dump(results_by_idx, result_path)

    torch.distributed.barrier()

    if rank == 0:
        merged = [None] * len(dataset)
        for r in range(world_size):
            path = os.path.join(args.out_dir, f'results_rank{r}.pkl')
            part = mmcv.load(path)
            for idx, res in part.items():
                merged[idx] = res

        eval_kwargs = dict(metric=args.metric)
        metric = dataset.evaluate(merged, **eval_kwargs)

        print('*' * 50 + ' SUMMARY ' + '*' * 50)
        for key, val in metric.items():
            print(f'{key}: {val}')

        if len(times) > 0:
            latency_ms = sum(times) / len(times) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else float('inf')
            print(f'Latency(ms): {latency_ms:.2f}')
            print(f'FPS: {fps:.2f}')


if __name__ == '__main__':
    main()
