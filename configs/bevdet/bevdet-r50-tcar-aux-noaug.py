_base_ = ['./bevdet-r50-tcar-aux.py']

# Debug config: disable train-time image/BEV augmentations to inspect
# whether aux 2D GT projection quality improves without augmentation.

# TCAR classes and point cloud range from tcar base config.
class_names = ['car', 'truck', 'cyc', 'ped']
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Turn off image-view augmentations.
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(1080, 1920),
    resize=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    crop_h=(0.0, 0.0),
    resize_test=0.0,
)

# Turn off BEV augmentations.
bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0,
)

train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
    ))

# Keep outputs separate from the default aux config runs.
import time as _time
_run_id = _time.strftime('%Y%m%d_%H%M%S')
work_dir = f'./outputs_bh/bevdet-r50-tcar-aux-noaug/run_{_run_id}'
