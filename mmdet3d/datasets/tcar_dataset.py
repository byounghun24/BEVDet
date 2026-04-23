import mmcv
import numpy as np
from os import path as osp

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.builder import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from .tcar_eval import NuScenesEval_custom
from mmdet3d.utils.coord_transform import (
    apply_to_boxes,
    get_lidar_to_ego,
    get_lidar_to_global,
    invert_transform,
)


@DATASETS.register_module()
class TcarDataset(NuScenesDataset):
    """TCAR dataset wrapper with nuScenes-style format and TCAR evaluation."""

    def __init__(self, *args, **kwargs):
        self.overfit_eval = kwargs.pop('overfit_eval', False)
        self.gt_box_frame = kwargs.pop('gt_box_frame', 'lidar')
        super().__init__(*args, **kwargs)
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        classes = kwargs.get('classes', None)
        if classes is not None:
            self.eval_detection_configs.class_names = classes
            self.eval_detection_configs.class_range = {cls: 50 for cls in classes}

    def _build_ann_infos(self, info):
        if 'gt_boxes' not in info or 'gt_names' not in info:
            return None
        gt_boxes = info['gt_boxes']
        if gt_boxes.shape[1] == 7 and 'gt_velocity' in info:
            gt_boxes = np.concatenate([gt_boxes, info['gt_velocity']], axis=1)
        elif gt_boxes.shape[1] == 7:
            gt_boxes = np.concatenate(
                [gt_boxes, np.zeros((gt_boxes.shape[0], 2), dtype=gt_boxes.dtype)],
                axis=1)
        gt_names = info['gt_names']
        if self.gt_box_frame != 'lidar':
            # gt_boxes in info are lidar-frame (per tcar_converter). Convert to the requested frame.
            if self.gt_box_frame == 'ego':
                T = get_lidar_to_ego(info)
            elif self.gt_box_frame == 'global':
                T = get_lidar_to_global(info)
            else:
                raise ValueError(f'Unsupported gt_box_frame: {self.gt_box_frame}')
            gt_boxes = apply_to_boxes(T, gt_boxes)
        if 'valid_flag' in info:
            mask = info['valid_flag'].astype(bool)
            gt_boxes = gt_boxes[mask]
            gt_names = gt_names[mask]
        ann_boxes = []
        ann_labels = []
        for i, name in enumerate(gt_names):
            mapped = self.NameMapping.get(name, name)
            if mapped not in self.CLASSES:
                continue
            ann_boxes.append(gt_boxes[i])
            ann_labels.append(self.CLASSES.index(mapped))
        ann_boxes = np.asarray(ann_boxes, dtype=np.float32)
        ann_labels = np.asarray(ann_labels, dtype=np.int64)
        return ann_boxes, ann_labels

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        info = self.data_infos[index]
        ann_infos = info.get('ann_infos', None)
        if ann_infos is None:
            ann_infos = self._build_ann_infos(info)
        if ann_infos is not None:
            input_dict['ann_infos'] = ann_infos
        return input_dict

    def _output_to_nusc_box(self, detection):
        """Convert the output to the nuScenes box class (TCAR convention)."""
        box3d = detection['boxes_3d']
        scores = detection['scores_3d'].numpy()
        labels = detection['labels_3d'].numpy()

        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        box_yaw = box3d.yaw.numpy()
        # ISO lidar -> nuScenes box uses (w, l, h)
        nus_box_dims = box_dims[:, [1, 0, 2]]

        box_list = []
        for i in range(len(box3d)):
            quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*box3d.tensor[i, 7:9], 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
        return box_list

    def _lidar_nusc_box_to_ego(self,
                              info,
                              boxes,
                              classes,
                              eval_configs,
                              eval_version='detection_cvpr_2019'):
        """Convert TCAR boxes from LiDAR to ego and filter by class range."""
        box_list = []
        for box in boxes:
            box.rotate(Quaternion(info['lidar2ego_rotation']))
            box.translate(np.array(info['lidar2ego_translation']))
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
            box_list.append(box)
        return box_list

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert results to the TCAR nuScenes-style format."""
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = self._output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes = self._lidar_nusc_box_to_ego(self.data_infos[sample_id], boxes,
                                                mapped_class_names,
                                                self.eval_detection_configs,
                                                self.eval_version)
            for box in boxes:
                name = mapped_class_names[box.label]
                attr = ''
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in TCAR protocol."""
        from tools.tcar.tcar import TestCar
        print('[TCAR-DEBUG] dataset_type=TcarDataset')
        print(f'[TCAR-DEBUG] data_root={self.data_root}')
        print(f'[TCAR-DEBUG] ann_file={self.ann_file}')
        print(f'[TCAR-DEBUG] num_data_infos={len(self.data_infos)}')
        if len(self.data_infos) > 0:
            print(f"[TCAR-DEBUG] first_data_info_token={self.data_infos[0].get('token', '<missing>')}")
        self.nusc = TestCar(version=self.version, dataroot=self.data_root,
                            verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=False,
            # Always constrain eval to the dataset view that produced predictions.
            # This avoids split-name mismatches between TCAR scene naming and hard-coded splits.
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
