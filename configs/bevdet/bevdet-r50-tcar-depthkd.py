_base_ = ['./bevdet-r50-4d-depth-cbgs-tcar.py']

import time as _time
_run_id = _time.strftime('%Y%m%d_%H%M%S')
work_dir = f'./outputs_bh/bevdet-r50-tcar-depthkd/run_{_run_id}'
multi_adj_frame_id_cfg = (1, 1, 1)  # No adjacent frame.

teacher_depth_mode = 'metric'  # 'metric' or 'relative'
if teacher_depth_mode == 'relative':
    teacher_depth_root = 'data/tcar/da3_relative_depth_stage1'
    kd_loss_weight = 0.0
    kd_relative_loss_weight = 0.5
elif teacher_depth_mode == 'metric':
    teacher_depth_root = 'data/tcar/da3_metric_depth_stage1'
    kd_loss_weight = 0.5
    kd_relative_loss_weight = 0.0
else:
    raise ValueError('teacher_depth_mode must be "metric" or "relative"')

class_names = ['car', 'truck', 'cyc', 'ped']
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(1080, 1920),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
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

data = dict(
    train=dict(
        dataset=dict(
            pipeline=train_pipeline,
            multi_adj_frame_id_cfg=multi_adj_frame_id_cfg)),
    val=dict(multi_adj_frame_id_cfg=multi_adj_frame_id_cfg),
    test=dict(multi_adj_frame_id_cfg=multi_adj_frame_id_cfg))

model = dict(
    type='BEVDepthDepthKD',
    num_adj=0,
    img_bev_encoder_backbone=dict(numC_input=80),
    depth_kd_cfg=dict(
        enabled=True,
        teacher_source='tensor',
        teacher_depth_mode=teacher_depth_mode,
        loss_type='kl',
        loss_weight=kd_loss_weight,
        temperature=1.0,
        teacher_key='teacher_depth',
        teacher_is_logits=False,
        ignore_if_missing=False,
        use_fg_mask=True,
        dense_spatial=True,
        dense_base_loss=True,
        dense_interp='bilinear',
        grad_loss_weight=0.0,
        relative_loss_weight=kd_relative_loss_weight,
        relative_temperature=1.0,
        relative_min_diff=0.0,
        relative_use_log_depth=False,
        ray_loss_weight=0.0,
        ray_loss_type='l1',
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='bevdet',
                name='bevdet-r50-tcar-depthkd'))
    ])

# Save checkpoints only at selected epochs to reduce disk usage.
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
