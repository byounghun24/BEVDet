# TCAR Visualization Coordinate Relationships (tcar_gt_viz.ipynb)

This note documents the **exact coordinate/transform chain used in `tcar_gt_viz.ipynb`** after fixing lidar loading to use the TCAR utilities. Use it as the reference for how lidar, camera, and GT boxes are related in the visualization.

**Scope**
- Source of truth: `tcar_gt_viz.ipynb` and `tools/tcar/tcar.py`.
- This is a visualization chain, not a training pipeline.

## Frames Used
- **Lidar frame (L)**: raw point cloud from `sample_data['filename']` for `LIDAR_TOP`.
- **Ego frame (E)**: vehicle frame at the lidar timestamp.
- **Camera frame (C)**: camera sensor frame defined by `calibrated_sensor`.
- **Image plane (I)**: pixel coordinates obtained by camera intrinsics.

## Lidar Point Cloud Flow (Image Projection)
1. **Load lidar points**
   - Source: `sample['data'][LIDAR_TOP]` → `sample_data['filename']`.
   - Load with `LidarPointCloud.from_file`.
   - Remove NaNs: `pc_raw.points = pc_raw.points[:, ~np.isnan(pc_raw.points).any(axis=0)]`.

2. **Lidar → Ego**
   - Transform from lidar to ego using the **inverse** of the calibrated sensor pose:
     - `lidar2ego = transform_matrix(lidar_cs.translation, Quaternion(lidar_cs.rotation), inverse=True)`
     - `pc_cam.transform(lidar2ego)`

3. **Ego → Camera**
   - Use the camera calibrated sensor pose directly:
     - `ego2cam = transform_matrix(cam_cs.translation, Quaternion(cam_cs.rotation), inverse=False)`
     - `pc_cam.transform(ego2cam)`

4. **Project to image**
   - Depth is **camera Z**: `depths = pc_cam.points[2, :]`.
   - Perspective projection:
     - `points = K @ (pc_cam.points[:3, :] / depths)`
   - Mask:
     - `depths > min_dist`
     - `1 < u < w-1` and `1 < v < h-1`

5. **Render**
   - `matplotlib.scatter(u, v, c=depths, cmap='jet')` with optional colorbar.

## Lidar Point Cloud Flow (BEV)
- Use the same `lidar2ego` transform:
  - `pc_ego = LidarPointCloud(pc_raw.points.copy())`
  - `pc_ego.transform(lidar2ego)`
- Render in BEV using the **ego frame** points.
- BEV axis mapping in the notebook:
  - `bev_mode = "lidar"`: x → right, y → up
  - `bev_mode = "map"`: x → up, y → left

## GT Boxes in tcar_gt_viz
- Boxes come from `nusc.get_box(sample_annotation_token)` in `tools/tcar/tcar.py`.
- `get_box()` uses `sample_annotation['translation']` and `['rotation']` directly and **does not apply pose/sensor transforms**.
- In the notebook, boxes are treated as **ego-frame boxes** and are:
  - rendered directly in BEV,
  - transformed only by **ego → camera** for image rendering.

## Summary Diagram
- **Lidar points to image**: `L → E → C → I`
- **Lidar points to BEV**: `L → E`
- **GT boxes to image**: `E → C → I`
- **GT boxes to BEV**: `E`

## Practical Notes
- If lidar appears as vertical streaks, the usual causes are **time offsets**, **motion distortion**, or **depth axis mismatch**, not the rendering step itself.
- The flow above assumes the TCAR dataset stores **calibrated_sensor** in the same convention as used by `transform_matrix`.

