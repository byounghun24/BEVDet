_base_ = ['./bevdet-r50-4d-depth-cbgs-tcar-depthkd.py']

import time as _time
_run_id = _time.strftime('%Y%m%d_%H%M%S')
work_dir = (
    f'./outputs_bh/bevdet-r50-4d-depth-cbgs-tcar-depthkd-1056x384/'
    f'run_{_run_id}'
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='bevdet',
                name='bevdet-r50-4d-depth-cbgs-tcar-depthkd-1056x384'))
    ])

# Requested input resolution: 1056 x 384 (W x H)
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(384, 1056),  # (H, W)
    src_size=(1080, 1920),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)

class_names = ['car', 'truck', 'cyc', 'ped']
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(
        type='LoadPointsFromFileTCAR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=grid_config),
    dict(
        type='LoadTeacherDepthFromNPZ',
        depth_root=teacher_depth_root,
        index_file=teacher_depth_root + '/index_by_token.pkl',
        depth_key='depth',
        output_key='teacher_depth',
        apply_img_aug=True,
        ignore_missing=False,
        fill_value=0.0),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth',
            'teacher_depth'
        ])
]

data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))

# Keep view transformer frustum definition aligned with resized input.
model = dict(
    img_view_transformer=dict(
        input_size=data_config['input_size'],
    ),
)
