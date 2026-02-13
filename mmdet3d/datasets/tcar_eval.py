import argparse
import copy
import json
import os
import time
from typing import Tuple, Dict, Any
import torch
import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.evaluate import NuScenesEval
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
import tqdm
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from torchvision.transforms.functional import rotate
import pycocotools.mask as mask_util
# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from torchvision.transforms.functional import rotate
import cv2
import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, dist_pr_curve, visualize_sample
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet3d.core.bbox.iou_calculators import BboxOverlaps3D
from IPython import embed
import json
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.constants import TP_METRICS, DETECTION_NAMES, DETECTION_COLORS, TP_METRICS_UNITS, \
    PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricData, DetectionMetricDataList
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.common.data_classes import MetricData, EvalBox


Axis = Any

def class_tp_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.

    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS if not np.isnan(metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1]) for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        if metric == 'trans_err':
            label += f' ({md.max_recall_ind})'  # add recall
            print(f'Recall: {detection_name}: {md.max_recall_ind/100}')
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


class CustomDetectionBox(DetectionBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        # Avoid DetectionBox's class-name assertions (TCAR classes differ) while keeping isinstance(DetectionBox).
        EvalBox.__init__(self, sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])

class DetectionBox_modified(CustomDetectionBox):
    def __init__(self, *args, token=None, visibility=None, index=None, **kwargs):
        '''
        add annotation token
        '''
        super().__init__(*args, **kwargs)
        self.token = token
        self.visibility = visibility
        self.index = index

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'token': self.token,
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name,
            'visibility': self.visibility,
            'index': self.index

        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(
            token=content['token'],
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name'],
            visibility=content['visibility'],
            index=content['index'],
        )


def center_in_image(box, intrinsic: np.ndarray, imsize: Tuple[int, int], vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    center_3d = box.center.reshape(3, 1)
    center_img = view_points(center_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(center_img[0, :] > 0, center_img[0, :] < imsize[0])
    visible = np.logical_and(visible, center_img[1, :] < imsize[1])
    visible = np.logical_and(visible, center_img[1, :] > 0)
    visible = np.logical_and(visible, center_3d[2, :] > 1)

    in_front = center_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def exist_corners_in_image_but_not_all(box, intrinsic: np.ndarray, imsize: Tuple[int, int],
                                       vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible in images but not all corners in image .
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if any(visible) and not all(visible) and all(in_front):
        return True
    else:
        return False


def custom_load_gt(nusc: NuScenes,
                   eval_split: str,
                   box_cls,
                   verbose: bool = False,
                   class_names=None,
                   sample_tokens_override=None):
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    
    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    from tools.tcar import splits

    _splits = splits.val

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'
    index_map = {}
    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        index_map[first_sample_token] = 1
        index = 2
        while sample['next'] != '':
            sample = nusc.get('sample', sample['next'])
            index_map[sample['token']] = index
            index += 1

    if sample_tokens_override is not None:
        sample_tokens = list(sample_tokens_override)
    else:
        sample_tokens = []
        for sample_token in sample_tokens_all:
            scene_token = nusc.get('sample', sample_token)['scene_token']
            scene_record = nusc.get('scene', scene_token)
            if scene_record['name'] in _splits:
                sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    class_name_set = set(class_names) if class_names is not None else None

    # Build quick lookup for datasets where sample_annotation has no category_name.
    inst_to_cat_name = None
    if len(nusc.sample_annotation) > 0 and 'category_name' not in nusc.sample_annotation[0]:
        inst_to_cat = {x['token']: x['category_token'] for x in nusc.instance}
        cat_to_name = {x['token']: x['name'] for x in nusc.category}
        inst_to_cat_name = {
            inst_tok: cat_to_name[cat_tok]
            for inst_tok, cat_tok in inst_to_cat.items()
            if cat_tok in cat_to_name
        }

    def _normalize_category_name(raw_name: str, class_set):
        # 1) nuScenes standard mapping first.
        mapped = category_to_detection_name(raw_name)
        candidates = []
        if mapped is not None:
            candidates.append(mapped)
        # 2) Keep raw for TCAR-native labels.
        candidates.append(raw_name)
        # 3) Add common TCAR alias mappings.
        alias_map = {
            'vehicle.car': 'car',
            'vehicle.truck': 'truck',
            'car': 'car',
            'truck': 'truck',
            'pedestrian': 'ped',
            'ped': 'ped',
            'cyclist': 'cyc',
            'bicycle': 'cyc',
            'motorcycle': 'cyc',
            'cyc': 'cyc',
        }
        # TCAR variants where large-vehicle classes are explicit.
        if class_set is not None and 'truck' in class_set:
            alias_map.update({
                'vehicle.construction': 'truck',
                'vehicle.trailer': 'truck',
                'constrn_veh': 'truck',
                'trailer': 'truck',
            })
        else:
            alias_map.update({
                'vehicle.construction': 'construction_vehicle',
                'vehicle.motorcycle': 'motorcycle',
                'vehicle.trailer': 'trailer',
                'constrn_veh': 'construction_vehicle',
            })
        if class_set is not None and 'cyc' in class_set:
            alias_map['vehicle.motorcycle'] = 'cyc'
            alias_map['vehicle.bicycle'] = 'cyc'
        if raw_name in alias_map:
            candidates.append(alias_map[raw_name])

        # Keep unique order.
        uniq = []
        for c in candidates:
            if c not in uniq:
                uniq.append(c)

        if class_set is None:
            return uniq[0]
        for c in uniq:
            if c in class_set:
                return c
        return None

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox_modified:
                # Get label name in detection task and filter unused labels.
                if 'category_name' in sample_annotation:
                    category_name = sample_annotation['category_name']
                elif inst_to_cat_name is not None:
                    category_name = inst_to_cat_name.get(sample_annotation['instance_token'])
                else:
                    category_name = None

                if category_name is None:
                    continue

                detection_name = _normalize_category_name(category_name, class_name_set)
                if detection_name is None:
                    continue
                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        token=sample_annotation_token,
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'], # wlh
                        rotation=sample_annotation['rotation'],
                        velocity=_safe_velocity(nusc, sample_annotation)[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name,
                        visibility=sample_annotation['visibility_token'],
                        index=index_map[sample_token]
                    )
                )
            elif box_cls == TrackingBox:
                assert False
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


def _safe_velocity(nusc, sample_annotation):
    velocity = sample_annotation.get('velocity', None)
    if velocity is None:
        try:
            velocity = nusc.box_velocity(sample_annotation['token'])
        except Exception:
            velocity = np.zeros(3)
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3)
    return velocity


def filter_eval_boxes_by_id(nusc: NuScenes,
                            eval_boxes: EvalBoxes,
                            id=None,
                            verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param is: the anns token set that used to keep bboxes.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on anns
        total += len(eval_boxes[sample_token])
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.token in id:
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After anns based filtering: %d" % anns_filter)

    return eval_boxes


def filter_eval_boxes_by_visibility(
        ori_eval_boxes: EvalBoxes,
        visibility=None,
        verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param is: the anns token set that used to keep bboxes.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on anns
        total += len(eval_boxes[sample_token])
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.visibility == visibility:
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After visibility based filtering: %d" % anns_filter)

    return eval_boxes


def filter_by_sample_token(ori_eval_boxes, valid_sample_tokens=[],  verbose=False):
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    for sample_token in eval_boxes.sample_tokens:
        if sample_token not in valid_sample_tokens:
            eval_boxes.boxes.pop(sample_token)
    return eval_boxes


def filter_eval_boxes_by_overlap(nusc: NuScenes,
                                 eval_boxes: EvalBoxes,
                                 verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. basedon overlap .
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    cams = ['CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT']

    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on anns
        total += len(eval_boxes[sample_token])
        sample_record = nusc.get('sample', sample_token)
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            count = 0
            for cam in cams:
                '''
                copy-paste form nuscens
                '''
                sample_data_token = sample_record['data'][cam]
                sd_record = nusc.get('sample_data', sample_data_token)
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                imsize = (sd_record['width'], sd_record['height'])
                new_box = Box(box.translation, box.size, Quaternion(box.rotation),
                              name=box.detection_name, token='')

                # Move box to ego vehicle coord system.
                new_box.translate(-np.array(pose_record['translation']))
                new_box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                new_box.translate(-np.array(cs_record['translation']))
                new_box.rotate(Quaternion(cs_record['rotation']).inverse)

                if center_in_image(new_box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                    count += 1
                # if exist_corners_in_image_but_not_all(new_box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                #    count += 1

            if count > 1:
                with open('center_overlap.txt', 'a') as f:
                    try:
                        f.write(box.token + '\n')
                    except:
                        pass
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    verbose = True

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After anns based filtering: %d" % anns_filter)

    return eval_boxes

def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

def custom_add_center_dist(nusc: NuScenes,
                           eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        for box in eval_boxes[sample_token]:
            # TCAR may provide boxes in ego coordinates while the nuScenes devkit assumes global.
            # Evaluate both interpretations and keep the one with smaller xy radius.
            ego_from_ego = np.array(box.translation)
            ego_from_global = np.array(box.translation) - np.array(pose_record['translation'])

            if np.linalg.norm(ego_from_ego[:2]) <= np.linalg.norm(ego_from_global[:2]):
                box.ego_translation = tuple(ego_from_ego.tolist())
            else:
                box.ego_translation = tuple(ego_from_global.tolist())

    return eval_boxes

class NuScenesEval_custom(NuScenesEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 overlap_test=False,
                 eval_mask=False,
                 data_infos=None
                 ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """

        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.overlap_test = overlap_test
        self.eval_mask = eval_mask
        self.data_infos = data_infos
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(
            self.result_path, self.cfg.max_boxes_per_sample,
            CustomDetectionBox, verbose=verbose)
        valid_tokens = None
        if self.data_infos is not None:
            valid_tokens = [info['token'] for info in self.data_infos]
        self.gt_boxes = custom_load_gt(
            self.nusc, self.eval_set, DetectionBox_modified,
            verbose=verbose, class_names=self.cfg.class_names,
            sample_tokens_override=valid_tokens)

        if valid_tokens:
            valid_set = set(valid_tokens)
            self.pred_boxes = filter_by_sample_token(self.pred_boxes, valid_set)
            self.gt_boxes = filter_by_sample_token(self.gt_boxes, valid_set)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = custom_add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = custom_add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).

        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        # if self.overlap_test:
        #     self.pred_boxes = filter_eval_boxes_by_overlap(self.nusc, self.pred_boxes)

        #     self.gt_boxes = filter_eval_boxes_by_overlap(self.nusc, self.gt_boxes, verbose=True)

        self.all_gt = copy.deepcopy(self.gt_boxes)
        self.all_preds = copy.deepcopy(self.pred_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

        self.index_map = {}
        for scene in nusc.scene:
            first_sample_token = scene['first_sample_token']
            sample = nusc.get('sample', first_sample_token)
            self.index_map[first_sample_token] = 1
            index = 2
            while sample['next'] != '':
                sample = nusc.get('sample', sample['next'])
                self.index_map[sample['token']] = index
                index += 1

    def update_gt(self, type_='vis', visibility='1', index=1):
        if type_ == 'vis':
            self.visibility_test = True
            if self.visibility_test:
                '''[{'description': 'visibility of whole object is between 0 and 40%',
                'token': '1',
                'level': 'v0-40'},
                {'description': 'visibility of whole object is between 40 and 60%',
                'token': '2',
                'level': 'v40-60'},
                {'description': 'visibility of whole object is between 60 and 80%',
                'token': '3',
                'level': 'v60-80'},
                {'description': 'visibility of whole object is between 80 and 100%',
                'token': '4',
                'level': 'v80-100'}]'''

                self.gt_boxes = filter_eval_boxes_by_visibility(self.all_gt, visibility, verbose=True)

        elif type_ == 'ord':

            valid_tokens = [key for (key, value) in self.index_map.items() if value == index]
            # from IPython import embed
            # embed()
            self.gt_boxes = filter_by_sample_token(self.all_gt, valid_tokens)
            self.pred_boxes = filter_by_sample_token(self.all_preds, valid_tokens)
        self.sample_tokens = self.gt_boxes.sample_tokens


    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        output_file_path = "./token.txt"
        try:
            with open(output_file_path, "w") as f:
                
                pred_tokens = set(self.pred_boxes.sample_tokens)
                gt_tokens = set(self.gt_boxes.sample_tokens)

                f.write(f"--- PREDICTION TOKENS ({len(pred_tokens)} total) ---\n")
                for token in sorted(list(pred_tokens)):
                    f.write(token + "\n")
                
                f.write(f"\n\n--- GROUND TRUTH TOKENS ({len(gt_tokens)} total) ---\n")
                for token in sorted(list(gt_tokens)):
                    f.write(token + "\n")
                    
            print(f"DEBUG: evaluate() 함수에서 토큰 저장 완료: {output_file_path}")
            
        except Exception as e:
            print(f"DEBUG: evaluate() 함수에서 토큰 저장 실패: {e}")
        print("="*30 + "\n")


        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()

        # print(self.cfg.dist_fcn_callable, self.cfg.dist_ths)
        # self.cfg.dist_ths = [0.3]
        # self.cfg.dist_fcn_callable
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='data/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=0,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = NuScenesEval_custom(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                                    output_dir=output_dir_, verbose=verbose_)
    for vis in ['1', '2', '3', '4']:
        nusc_eval.update_gt(type_='vis', visibility=vis)
        print(f'================ {vis} ===============')
        nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
    #for index in range(1, 41):
    #    nusc_eval.update_gt(type_='ord', index=index)
    #
