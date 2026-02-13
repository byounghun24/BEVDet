_base_ = ['./bevdet-r50.py']

dataset_type = 'TcarDataset'
# Update this if your TCAR dataset is stored elsewhere.
data_root = 'data/tcar/'

# Save checkpoints/logs under ./outputs instead of work_dirs.
# Use a timestamped subdir to avoid overwriting previous runs.
import time as _time
_run_id = _time.strftime('%Y%m%d_%H%M%S')
work_dir = f'./outputs_bh/bevdet-r50-tcar/run_{_run_id}'

# Point cloud range is referenced in the train pipeline.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Copy base data configs used in pipelines.
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (1080, 1920),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# TCAR classes (from gt_names mapping).
class_names = ['car', 'truck', 'cyc', 'ped']

# Override dataset settings to use TCAR infos.
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdet-tcar_infos_temporal_train.pkl',
        classes=class_names),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdet-tcar_infos_temporal_val.pkl',
        classes=class_names),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdet-tcar_infos_temporal_val.pkl',
        classes=class_names),
)

# TCAR train pipeline (match BEVDet train behavior).
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d']),
]

# TCAR test pipeline (robust point loading).
test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(
        type='LoadPointsFromFileTCAR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

# Apply TCAR pipelines.
data['train'].update(dict(pipeline=train_pipeline))
data['val'].update(dict(pipeline=test_pipeline))
data['test'].update(dict(pipeline=test_pipeline))

# Update model head for TCAR classes.
model = dict(
    pts_bbox_head=dict(
        tasks=[
            dict(num_class=4, class_names=class_names),
        ],
    ),
    test_cfg=dict(
        pts=dict(
            min_radius=[4, 10, 1.1, 0.85],
            nms_rescale_factor=[[1.0, 0.7, 1.0, 0.55]],
        )
    )
)

# Enable Weights & Biases logging.
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='bevdet',
                name='bevdet-r50-tcar'))
    ])

# Save checkpoints only at specific epochs.
checkpoint_config = None
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SaveSelectedEpochsHook',
        epochs=[1, 5, 20, 24],
        save_optimizer=True,
        save_last=False,
    ),
]
