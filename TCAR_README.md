# TCAR Integration

This repo now includes a TCAR dataset/eval integration that mirrors the TCAR customization used in `TcarBEVFormer/`.

## Expected folder layout

```
<data_root>/
  v1.0-trainval/
    category.json
    instance.json
    sensor.json
    calibrated_sensor.json
    ego_pose.json
    scene.json
    sample.json
    sample_data.json
    sample_annotation.json
    ...
  samples/
    LIDAR_TOP/...
    CAM_FRONT/...
    ...
```

TCAR uses a nuScenes-style table layout and sample_data layout. Paths inside the tables should be relative to `<data_root>`.

## TCAR info files

Generate info files that match the TCAR split and annotation format:

```
python tools/create_data_tcar.py \
  --root-path data/tcar \
  --out-dir data/tcar \
  --info-prefix tcar \
  --version v1.0-trainval \
  --max-sweeps 10
```

This produces:
- `data/tcar/tcar_infos_temporal_train.pkl`
- `data/tcar/tcar_infos_temporal_val.pkl`

## Training / evaluation

Use the TCAR config to switch dataset + evaluator:

```
# BEVDet example
python tools/train.py configs/bevdet/bevdet-r50-tcar.py
python tools/test.py configs/bevdet/bevdet-r50-tcar.py <checkpoint>
```

NuScenes configs remain unchanged.

## Metrics

Evaluation uses TCAR’s nuScenes-style metrics (via `NuScenesEval_custom`) and reports:
- per-class AP / TP error metrics
- `NDS`, `mAP`

## Notes

- Splits are defined in `tools/tcar/splits.py` and are used for evaluation.
- TCAR boxes are defined in LiDAR coordinates and converted to ego coordinates during evaluation.
- If your class names differ from the default 10 nuScenes classes, update `class_names` in the config.
