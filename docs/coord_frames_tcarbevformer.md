# TCarBEVFormer Coordinate Frames & Transforms (Evidence-Based)

This document summarizes the coordinate-frame conventions and transform chains **as implemented in TCarBEVFormer**, with evidence points to specific code locations. All statements below are traceable to variables, comments, and formulas in the repository.

## 1) Frames and where they appear

**Lidar frame (L)**  
- Used as the canonical frame for GT 3D boxes in the info files.  
  Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` builds `gt_boxes` after converting from *ego → lidar* (comment and formula).

**Ego frame (E)**  
- Lidar-to-ego stored as `lidar2ego_rotation/translation` in info.  
  Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 214–233 define `q_l2e` and `t_l2e` and store them as `lidar2ego_*`.

**Global frame (G)**  
- Ego-to-global stored as `ego2global_rotation/translation` in info.  
  Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 231–235 store `ego2global_*`.

**Camera/Sensor frame (C / S)**  
- Camera extrinsics are stored as `sensor2ego_*` and `sensor2lidar_*` inside `cams` in each info.  
  Evidence: `obtain_sensor2top()` in `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 347–379 creates `sensor2ego_*` and `sensor2lidar_*`.

> **Axis directions** are not explicitly documented in TCarBEVFormer. No authoritative axis directions were found in code comments. The only explicit axis assumption appears in yaw conversions (see below).

## 2) Transform construction in TCarBEVFormer

### 2.1 Lidar ↔ Ego
- `q_e2l = Quaternion(cs_record['rotation'])` and `t_e2l = cs_record['translation']`.  
  Then `q_l2e = q_e2l.inverse`, `t_l2e = - (q_l2e.rotation_matrix @ t_e2l)`.  
  Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 214–217.

### 2.2 Ego ↔ Global
- `ego2global_translation` and `ego2global_rotation` are stored directly from `pose_record`.  
  Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 231–235.

### 2.3 Sensor ↔ Ego and Sensor ↔ Lidar
- `sensor2ego_rotation/translation` are computed as the inverse of `cs_record` (ego→sensor).  
  Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 347–357.
- `sensor2lidar_rotation/translation` are computed in `obtain_sensor2top()` with the comment:
  **“points @ R.T + T”**, which implies the row-vector convention.  
  Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 368–379.

### 2.4 Lidar → Camera / Lidar → Image
TCarBEVFormer’s dataset uses camera info stored in `cams[cam]`:
- `lidar2cam_r = inv(sensor2lidar_rotation)`  
- `lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T`  
- `lidar2cam_rt[:3,:3] = lidar2cam_r.T; lidar2cam_rt[3,:3] = -lidar2cam_t`  
- `lidar2img = K @ lidar2cam_rt.T`  
Evidence: `TcarBEVFormer/projects/mmdet3d_plugin/datasets/tcar_dataset.py` lines 135–145.

## 3) GT boxes: source frame and conversion

### 3.1 GT boxes are converted to **lidar** in the info
In `tools/data_converter/tcar_converter.py`:
- **Comment:** “from ego coordinate to lidar coordinate”  
- **Formula:** `locs = (center - l2e_t) @ l2e_r_mat.T`  
Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 279–281.

### 3.2 Yaw conversion to SECOND format
GT yaw is converted as:
```
gt_boxes = [locs, dims, -rots - pi/2]
```
Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 294–295.

### 3.3 Evaluation uses lidar → ego
The evaluation path explicitly treats TCAR boxes as lidar-frame:
```
our tcar bbox coordinate is defined as the lidar coordinate.
so we should change the ego from lidar!
```
Then it applies `lidar2ego` to each box.  
Evidence: `TcarBEVFormer/projects/mmdet3d_plugin/datasets/tcar_dataset.py` lines 232–246.

### 3.4 Output yaw conversion
Detection outputs apply:
```
box_yaw = -box_yaw - pi/2
```
Evidence: `TcarBEVFormer/projects/mmdet3d_plugin/datasets/tcar_dataset.py` lines 207–212.

## 4) TCarBEVFormer visualization evidence

### 4.1 BEV rendering style
TCarBEVFormer BEV visualization:
- Renders GT in lidar frame (`get_sample_data` on `LIDAR_TOP`).  
- Applies a 90° CCW view matrix for visualization.  
- Uses fixed colors and a legend.  
Evidence: `TcarBEVFormer/tools/analysis_tools/tcar_visual.py` lines 327–360.

## 5) Notes on `tools/tcar/tcar.py`

In TCarBEVFormer, `get_sample_data()` returns GT boxes **without applying transforms**:
- Code builds `boxes` and appends them directly without pose/sensor transforms.  
Evidence: `TcarBEVFormer/tools/tcar/tcar.py` lines 231–273.

The docstring says boxes are transformed into the current sensor frame, but the implementation does not do the transform.

## 6) Example transform chains (from evidence)

**GT boxes (as stored in info):**
```
ego -> lidar
locs = (center - l2e_t) @ l2e_r_mat.T
```
Evidence: `TcarBEVFormer/tools/data_converter/tcar_converter.py` lines 279–281.

**Lidar → Camera and Lidar → Image (in dataset input):**
```
lidar2cam_r = inv(sensor2lidar_rotation)
lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
lidar2cam_rt[:3,:3] = lidar2cam_r.T
lidar2cam_rt[3,:3] = -lidar2cam_t
lidar2img = K @ lidar2cam_rt.T
```
Evidence: `TcarBEVFormer/projects/mmdet3d_plugin/datasets/tcar_dataset.py` lines 135–145.

**Evaluation range filtering:**
```
lidar -> ego
```
Evidence: `TcarBEVFormer/projects/mmdet3d_plugin/datasets/tcar_dataset.py` lines 232–246.
