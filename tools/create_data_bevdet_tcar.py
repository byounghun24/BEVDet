# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np

from tools.data_converter import tcar_converter as tcar_converter


# TCAR 4-class setup.
classes = ['car', 'truck', 'cyc', 'ped']

map_name_from_general_to_detection = {
    # TCAR native names.
    'car': 'car',
    'truck': 'truck',
    'cyc': 'cyc',
    'ped': 'ped',
    # TCAR variant names.
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.motorcycle': 'cyc',
    'vehicle.bicycle': 'cyc',
    'motorcycle': 'cyc',
    'bicycle': 'cyc',
    'pedestrian': 'ped',
    # Optional coarse mapping to keep compatibility across TCAR variants.
    'vehicle.construction': 'truck',
    'vehicle.trailer': 'truck',
    'constrn_veh': 'truck',
    'trailer': 'truck',
    'human.pedestrian.adult': 'ped',
    'human.pedestrian.child': 'ped',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'ped',
    'human.pedestrian.construction_worker': 'ped',
    'animal': 'ignore',
    'vehicle.bus.bendy': 'ignore',
    'vehicle.bus.rigid': 'ignore',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'movable_object.barrier': 'ignore',
    'movable_object.trafficcone': 'ignore',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


def _build_ann_infos(info):
    """Build ann_infos for BEVDet LoadAnnotations from TCAR info."""
    if 'gt_boxes' not in info or 'gt_names' not in info:
        return None

    gt_boxes = info['gt_boxes']
    if gt_boxes is None or len(gt_boxes) == 0:
        return [], []

    # Append velocity if available (expects 9D boxes in BEVDet pipeline).
    if 'gt_velocity' in info and gt_boxes.shape[1] == 7:
        gt_boxes = np.concatenate([gt_boxes, info['gt_velocity']], axis=1)
    elif gt_boxes.shape[1] == 7:
        zeros = np.zeros((gt_boxes.shape[0], 2), dtype=gt_boxes.dtype)
        gt_boxes = np.concatenate([gt_boxes, zeros], axis=1)

    # Valid mask
    if 'valid_flag' in info:
        mask = info['valid_flag'].astype(bool)
    elif 'num_lidar_pts' in info:
        mask = info['num_lidar_pts'] > 0
    else:
        mask = np.ones((gt_boxes.shape[0],), dtype=bool)

    gt_boxes = gt_boxes[mask]
    gt_names = np.array(info['gt_names'])[mask]

    ann_boxes = []
    ann_labels = []
    for i, name in enumerate(gt_names):
        mapped = map_name_from_general_to_detection.get(name, name)
        if mapped not in classes:
            continue
        ann_boxes.append(gt_boxes[i])
        ann_labels.append(classes.index(mapped))
    return ann_boxes, ann_labels


def tcar_data_prep(root_path, out_path, info_prefix, version, max_sweeps=10):
    """Prepare BEVDet-style infos for TCAR."""
    # can_bus_root_path is unused in TCAR converter, pass empty string.
    tcar_converter.create_nuscenes_infos(
        root_path=root_path,
        out_path=out_path,
        can_bus_root_path='',
        info_prefix=info_prefix,
        version=version,
        max_sweeps=max_sweeps,
        convert_yaw=False)


def add_ann_infos(info_path):
    """Inject ann_infos into the TCAR info file for BEVDet pipeline."""
    dataset = pickle.load(open(info_path, 'rb'))
    infos = dataset['infos']
    for i, info in enumerate(infos):
        if i % 2000 == 0:
            print(f'[{i}/{len(infos)}] adding ann_infos')
        ann_infos = _build_ann_infos(info)
        if ann_infos is not None:
            info['ann_infos'] = ann_infos
    with open(info_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    version = 'v1.0-trainval'
    root_path = './data/tcar'
    out_path = root_path
    extra_tag = 'bevdet-tcar'

    tcar_data_prep(
        root_path=root_path,
        out_path=out_path,
        info_prefix=extra_tag,
        version=version,
        max_sweeps=10)

    train_info = f'{out_path}/{extra_tag}_infos_temporal_train.pkl'
    val_info = f'{out_path}/{extra_tag}_infos_temporal_val.pkl'

    add_ann_infos(train_info)
    add_ann_infos(val_info)

    # GT database is optional for BEVDet and current helper assumes nuScenes 10 classes.
    # Skip by default to avoid mismatched class names in dbinfos.
    print('Skip create_groundtruth_database by default.')
