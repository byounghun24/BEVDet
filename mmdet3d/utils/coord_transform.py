import numpy as np
from pyquaternion import Quaternion


def _as_np(x):
    return np.asarray(x, dtype=np.float32)


def make_transform(rotation, translation):
    """Create a row-vector transform matrix.

    Uses the convention: p' = p @ R.T + t
    Where rotation is a standard column-vector rotation matrix R.
    """
    rot = _as_np(rotation)
    trans = _as_np(translation).reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot.T
    T[:3, 3] = trans
    return T


def invert_transform(T):
    """Invert a row-vector transform matrix."""
    rot_t = _as_np(T[:3, :3])
    rot = rot_t.T
    trans = _as_np(T[:3, 3])
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = rot
    inv[:3, 3] = -trans @ rot
    return inv


def compose(*Ts):
    """Compose row-vector transforms (right-multiply)."""
    out = np.eye(4, dtype=np.float32)
    for T in Ts:
        out = out @ _as_np(T)
    return out


def apply_to_points(T, points):
    """Apply row-vector transform to Nx3 points."""
    pts = _as_np(points)
    return pts @ _as_np(T[:3, :3]) + _as_np(T[:3, 3])


def rotation_yaw(rot):
    """Return yaw (radians) from a rotation matrix (column-vector convention)."""
    rot = _as_np(rot)
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def wrap_yaw(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def apply_to_boxes(T, boxes, with_vel=True):
    """Apply row-vector transform to boxes.

    boxes: (N, 7) or (N, 9) with format (x, y, z, w, l, h, yaw[, vx, vy])
    """
    if boxes is None or len(boxes) == 0:
        return boxes
    b = _as_np(boxes)
    centers = b[:, :3]
    centers_t = apply_to_points(T, centers)
    rot_col = _as_np(T[:3, :3]).T
    yaw_add = rotation_yaw(rot_col)
    yaws = wrap_yaw(b[:, 6] + yaw_add)
    out = b.copy()
    out[:, :3] = centers_t
    out[:, 6] = yaws
    if with_vel and out.shape[1] >= 9:
        vel = out[:, 7:9]
        vel_t = vel @ rot_col[:2, :2].T
        out[:, 7:9] = vel_t
    return out


def get_lidar_to_ego(meta):
    rot = Quaternion(meta['lidar2ego_rotation']).rotation_matrix
    trans = meta['lidar2ego_translation']
    return make_transform(rot, trans)


def get_ego_to_global(meta):
    rot = Quaternion(meta['ego2global_rotation']).rotation_matrix
    trans = meta['ego2global_translation']
    return make_transform(rot, trans)


def get_lidar_to_global(meta):
    return compose(get_lidar_to_ego(meta), get_ego_to_global(meta))


def get_lidar_to_cam(meta, cam_id):
    cam_info = meta['cams'][cam_id]
    s2l_r = _as_np(cam_info['sensor2lidar_rotation'])
    s2l_t = _as_np(cam_info['sensor2lidar_translation'])
    l2c_r = np.linalg.inv(s2l_r)
    l2c_t = s2l_t @ l2c_r.T
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = l2c_r.T
    T[:3, 3] = -l2c_t
    return T


def get_cam_intrinsic(meta, cam_id):
    return _as_np(meta['cams'][cam_id]['cam_intrinsic'])
