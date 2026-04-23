_base_ = ['./bevdet-r50.py']

dataset_type = 'TcarDataset'
data_root = 'data/tcar/'

import time as _time
_run_id = _time.strftime('%Y%m%d_%H%M%S')
work_dir = f'./outputs_bh/bevdet-r50-tcar-5030/run_{_run_id}'

# BEV range: x(front/back) +-50m, y(left/right) +-30m.
point_cloud_range = [-50.0, -30.0, -5.0, 50.0, 30.0, 3.0]
grid_config = {
    'x': [-50.0, 50.0, 0.8],
    'y': [-30.0, 30.0, 0.8],
    'z': [-5.0, 3.0, 8.0],
    'depth': [1.0, 60.0, 1.0],
}
voxel_size = [0.1, 0.1, 0.2]

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

class_names = ['car', 'truck', 'cyc', 'ped']

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

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
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

data['train'].update(dict(pipeline=train_pipeline))
data['val'].update(dict(pipeline=test_pipeline))
data['test'].update(dict(pipeline=test_pipeline))

model = dict(
    img_view_transformer=dict(grid_config=grid_config),
    pts_bbox_head=dict(
        tasks=[
            dict(num_class=4, class_names=class_names),
        ],
        bbox_coder=dict(
            pc_range=point_cloud_range[:2],
            post_center_range=[-50.0, -30.0, -10.0, 50.0, 30.0, 10.0])),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1000, 600, 40],
            voxel_size=voxel_size)),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-50.0, -30.0, -10.0, 50.0, 30.0, 10.0],
            voxel_size=voxel_size[:2],
            min_radius=[4, 10, 1.1, 0.85],
            nms_rescale_factor=[[1.0, 0.7, 1.0, 0.55]],
        )))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='bevdet',
                name='bevdet-r50-tcar-5030'))
    ])

checkpoint_config = None
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        save_epochs=[1, 20, 24],
        max_keep_ckpts=1,
    ),
    dict(
        type='SaveSelectedEpochsHook',
        epochs=[1, 20, 24],
        save_optimizer=True,
        save_last=False,
    ),
]
