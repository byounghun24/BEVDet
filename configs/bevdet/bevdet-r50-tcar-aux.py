_base_ = ['./bevdet-r50-tcar.py']

# Save auxiliary-run checkpoints/logs under a separate timestamped directory.
import time as _time
_run_id = _time.strftime('%Y%m%d_%H%M%S')
work_dir = f'./outputs_bh/bevdet-r50-tcar-aux/run_{_run_id}'

model = dict(
    type='BEVDetAux',
    aux_2d_head=dict(
        type='SeparateHead',
        in_channels=256,
        heads=dict(
            heatmap=(4, 2),
            offset=(2, 2),
        ),
        head_conv=64,
        final_kernel=3,
        init_bias=-2.19),
    aux_2d_train_cfg=dict(
        max_objs=150,
        min_overlap=0.3,
        min_radius=2,
        min_box_size=2.0,
        min_center_depth=0.5,
        max_center_depth=80.0,
        edge_margin=2.0,
    ),
    aux_2d_loss_cls=dict(
        type='GaussianFocalLoss',
        reduction='mean',
    ),
    aux_2d_loss_bbox=dict(
        type='L1Loss',
        reduction='mean',
        loss_weight=1.0,
    ),
    aux_2d_loss_weight=0.1,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='bevdet',
                name='bevdet-r50-tcar-aux'))
    ])
