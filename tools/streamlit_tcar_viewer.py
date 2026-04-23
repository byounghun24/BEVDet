#!/usr/bin/env python3
"""Streamlit app for TCAR sample visualization and scene video generation.

Usage:
    streamlit run tools/streamlit_tcar_viewer.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import compat_cfg
from tools.tcar.tcar import TestCar
from tools.tcar.utils import LidarPointCloud


DEFAULT_CAM_CHANNELS_STORE = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

DEFAULT_CAM_CHANNELS_DISPLAY = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def _clone_box(box: Box) -> Box:
    return Box(
        box.center.copy(),
        box.wlh.copy(),
        Quaternion(box.orientation),
        name=box.name,
        token=box.token,
        velocity=box.velocity,
    )


def _maybe_fix_z_origin(box: Box, z_is_bottom: bool) -> Box:
    if not z_is_bottom:
        return box
    b = _clone_box(box)
    b.center[2] += b.wlh[2] / 2.0
    return b


def _transform_box(box: Box, translation, rotation, inverse: bool = False) -> Box:
    b = _clone_box(box)
    q = Quaternion(rotation)
    if inverse:
        b.translate(-np.array(translation))
        b.rotate(q.inverse)
    else:
        b.rotate(q)
        b.translate(np.array(translation))
    return b


def _render_box_2d(ax, box: Box, cam_k: np.ndarray, im_size, color: str = "r", lw: float = 1.0):
    width, height = im_size
    corners = box.corners()
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    def _clip_line_to_rect(x0, y0, x1, y1, w, h):
        inside, left, right, bottom, top = 0, 1, 2, 4, 8

        def _outcode(x, y):
            code = inside
            if x < 0:
                code |= left
            elif x > w:
                code |= right
            if y < 0:
                code |= top
            elif y > h:
                code |= bottom
            return code

        out0 = _outcode(x0, y0)
        out1 = _outcode(x1, y1)
        while True:
            if (out0 | out1) == 0:
                return x0, y0, x1, y1
            if (out0 & out1) != 0:
                return None

            out = out0 if out0 != 0 else out1
            if out & top:
                if y1 == y0:
                    return None
                x = x0 + (x1 - x0) * (0 - y0) / (y1 - y0)
                y = 0
            elif out & bottom:
                if y1 == y0:
                    return None
                x = x0 + (x1 - x0) * ((h - 1) - y0) / (y1 - y0)
                y = h - 1
            elif out & right:
                if x1 == x0:
                    return None
                y = y0 + (y1 - y0) * ((w - 1) - x0) / (x1 - x0)
                x = w - 1
            else:
                if x1 == x0:
                    return None
                y = y0 + (y1 - y0) * (0 - x0) / (x1 - x0)
                x = 0

            if out == out0:
                x0, y0 = x, y
                out0 = _outcode(x0, y0)
            else:
                x1, y1 = x, y
                out1 = _outcode(x1, y1)

    for i0, i1 in edges:
        p0, p1 = corners[:, i0], corners[:, i1]
        if p0[2] <= 0 or p1[2] <= 0:
            continue

        pts = np.stack([p0, p1], axis=1)
        proj = cam_k @ pts
        proj[:2, :] /= proj[2:3, :]

        clipped = _clip_line_to_rect(proj[0, 0], proj[1, 0], proj[0, 1], proj[1, 1], width - 1, height - 1)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw)


def _render_bev_boxes(
    ax,
    boxes: Sequence[Box],
    color: str = "r",
    lw: float = 1.0,
    draw_heading: bool = False,
    heading_scale: float = 2.5,
):
    for box in boxes:
        corners = box.bottom_corners()
        corners = np.concatenate([corners, corners[:, :1]], axis=1)
        ax.plot(corners[0, :], corners[1, :], color=color, linewidth=lw)

        if draw_heading:
            center = box.center[:2]
            heading_vec = box.orientation.rotation_matrix[:2, 0]
            end = center + heading_vec * heading_scale
            ax.arrow(
                center[0],
                center[1],
                end[0] - center[0],
                end[1] - center[1],
                color=color,
                width=0.03,
                head_width=0.35,
                head_length=0.5,
                length_includes_head=True,
                alpha=0.9,
            )


def _render_ego_vehicle_bev(ax, color: str = "blue", lw: float = 5.0, alpha: float = 0.9):
    # Simple ego footprint in BEV (x: forward, y: left).
    ego_poly = np.array(
        [
            [2.2, 0.0],
            [1.2, 0.95],
            [-1.2, 0.95],
            [-1.2, -0.95],
            [1.2, -0.95],
            [2.2, 0.0],
        ],
        dtype=np.float32,
    )
    ax.plot(ego_poly[:, 0], ego_poly[:, 1], color=color, linewidth=lw, alpha=alpha)


def _build_model_and_dataset(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda:0",
    split: str = "val",
):
    cfg = Config.fromfile(config_path)
    cfg = compat_cfg(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    ds_cfg = cfg.data.val if split == "val" else cfg.data.test
    dataset = build_dataset(ds_cfg)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model.to(device)
    model.eval()
    return cfg, model, dataset


def _build_scene_index(dataset) -> List[Tuple[str, List[int]]]:
    scene_to_indices: Dict[str, List[int]] = {}
    for idx, info in enumerate(dataset.data_infos):
        scene = info.get("scene_token", "unknown_scene")
        scene_to_indices.setdefault(scene, []).append(idx)

    ordered: List[Tuple[str, List[int]]] = []
    for scene, ids in scene_to_indices.items():
        ids.sort(key=lambda i: dataset.data_infos[i].get("timestamp", 0))
        ordered.append((scene, ids))

    ordered.sort(key=lambda item: dataset.data_infos[item[1][0]].get("timestamp", 0))
    return ordered


def _inference_single(model, dataset, idx: int, device: str = "cuda:0"):
    data = dataset[idx]
    data = collate([data], samples_per_gpu=1)
    if str(device).startswith("cuda"):
        gpu_id = torch.device(device).index
        if gpu_id is None:
            gpu_id = 0
        data = scatter(data, [gpu_id])[0]

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result[0]


def _inference_single_with_nms_stats(model, dataset, idx: int, device: str = "cuda:0"):
    from mmdet3d.models.dense_heads import centerpoint_head as cp_head_mod

    stats = {
        "rotate_calls": 0,
        "rotate_before_total": 0,
        "rotate_keep_total": 0,
        "circle_calls": 0,
        "circle_before_total": 0,
        "circle_keep_total": 0,
    }

    orig_nms_bev = cp_head_mod.nms_bev
    orig_circle_nms = cp_head_mod.circle_nms

    def _wrapped_nms_bev(boxes, scores, *args, **kwargs):
        keep = orig_nms_bev(boxes, scores, *args, **kwargs)
        stats["rotate_calls"] += 1
        stats["rotate_before_total"] += int(scores.shape[0])
        stats["rotate_keep_total"] += int(keep.shape[0] if hasattr(keep, "shape") else len(keep))
        return keep

    def _wrapped_circle_nms(dets, thresh, post_max_size=83):
        keep = orig_circle_nms(dets, thresh, post_max_size=post_max_size)
        stats["circle_calls"] += 1
        stats["circle_before_total"] += int(dets.shape[0])
        stats["circle_keep_total"] += int(len(keep))
        return keep

    cp_head_mod.nms_bev = _wrapped_nms_bev
    cp_head_mod.circle_nms = _wrapped_circle_nms
    try:
        result = _inference_single(model, dataset, idx, device=device)
    finally:
        cp_head_mod.nms_bev = orig_nms_bev
        cp_head_mod.circle_nms = orig_circle_nms
    return result, stats


def _extract_pred(result):
    out = result["pts_bbox"] if isinstance(result, dict) and "pts_bbox" in result else result
    boxes_3d = out["boxes_3d"]
    scores_3d = out["scores_3d"].detach().cpu().numpy()
    labels_3d = out["labels_3d"].detach().cpu().numpy()
    box_tensor = boxes_3d.tensor.detach().cpu().numpy()
    return box_tensor, scores_3d, labels_3d


def _compute_pred_bev_overlap_debug(
    result,
    class_names: Sequence[str],
    score_thr: float = 0.3,
    class_score_thr: Optional[Dict[str, float]] = None,
    iou_thr: float = 0.2,
    max_pairs: int = 20,
):
    out = result["pts_bbox"] if isinstance(result, dict) and "pts_bbox" in result else result
    boxes_3d = out["boxes_3d"]
    scores = out["scores_3d"].detach().cpu()
    labels = out["labels_3d"].detach().cpu()

    if scores.numel() == 0:
        return {
            "num_preds": 0,
            "max_iou": 0.0,
            "num_pairs_over_thr": 0,
            "pairs": [],
            "error": None,
        }

    # Apply the same score filtering used for visualization.
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for i in range(scores.shape[0]):
        li = int(labels[i].item())
        cls_name = class_names[li] if li < len(class_names) else str(li)
        thr = float(score_thr)
        if isinstance(class_score_thr, dict):
            thr = float(class_score_thr.get(cls_name, score_thr))
        keep_mask[i] = scores[i] >= thr

    boxes_bev = boxes_3d.bev[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]

    num_preds = int(scores.numel())
    info = {
        "num_preds": num_preds,
        "max_iou": 0.0,
        "num_pairs_over_thr": 0,
        "pairs": [],
        "error": None,
    }
    if num_preds < 2:
        return info

    try:
        from mmcv.ops import box_iou_rotated

        iou_mat = box_iou_rotated(boxes_bev, boxes_bev).detach().cpu()
    except Exception as exc:
        info["error"] = f"box_iou_rotated failed: {exc}"
        return info

    iou_mat.fill_diagonal_(0.0)
    info["max_iou"] = float(iou_mat.max().item())

    pairs = []
    nz = torch.nonzero(iou_mat > float(iou_thr), as_tuple=False)
    for row in nz:
        i = int(row[0].item())
        j = int(row[1].item())
        if i >= j:
            continue
        iou_val = float(iou_mat[i, j].item())
        pairs.append((iou_val, i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)
    info["num_pairs_over_thr"] = len(pairs)

    top_pairs = []
    for iou_val, i, j in pairs[: max(1, int(max_pairs))]:
        li = int(labels[i].item())
        lj = int(labels[j].item())
        top_pairs.append(
            {
                "iou_bev": round(iou_val, 4),
                "i": i,
                "j": j,
                "score_i": round(float(scores[i].item()), 4),
                "score_j": round(float(scores[j].item()), 4),
                "label_i": class_names[li] if li < len(class_names) else str(li),
                "label_j": class_names[lj] if lj < len(class_names) else str(lj),
            }
        )
    info["pairs"] = top_pairs
    return info


def _parse_float_list_csv(raw_text: str) -> List[float]:
    vals = []
    for chunk in raw_text.split(","):
        s = chunk.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def _normalize_rescale_factors(
    factors: Sequence[float],
    num_classes: int,
) -> List[float]:
    out = [1.0] * int(num_classes)
    for i, v in enumerate(list(factors)[:num_classes]):
        out[i] = float(v)
    return out


def _default_rescale_factors_from_test_cfg(test_cfg_pts: dict, num_classes: int) -> List[float]:
    raw = test_cfg_pts.get("nms_rescale_factor", None)
    if raw is None:
        return [1.0] * int(num_classes)

    vals: List[float]
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return [1.0] * int(num_classes)
        first = raw[0]
        if isinstance(first, (list, tuple)):
            vals = [float(x) for x in first]
        else:
            vals = [float(x) for x in raw]
    else:
        return [1.0] * int(num_classes)
    return _normalize_rescale_factors(vals, num_classes=num_classes)


def _apply_visualization_rotated_nms(
    pred_boxes_ego: Sequence[Tuple[Box, float, int]],
    nms_thr: float,
    rescale_factors: Sequence[float],
) -> Tuple[List[Tuple[Box, float, int]], Optional[str]]:
    if len(pred_boxes_ego) <= 1:
        return list(pred_boxes_ego), None

    try:
        from mmcv.ops import nms_rotated
    except Exception as exc:
        return list(pred_boxes_ego), f"Visualization NMS unavailable: {exc}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    boxes_xywhr = []
    scores = []
    labels = []
    for box, score, label in pred_boxes_ego:
        yaw = float(box.orientation.yaw_pitch_roll[0])
        boxes_xywhr.append([float(box.center[0]), float(box.center[1]), float(box.wlh[0]), float(box.wlh[1]), yaw])
        scores.append(float(score))
        labels.append(int(label))

    boxes_xywhr_t = torch.tensor(boxes_xywhr, dtype=torch.float32, device=device)
    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    scaled_boxes = boxes_xywhr_t.clone()
    if len(rescale_factors) > 0:
        for cid, factor in enumerate(rescale_factors):
            mask = labels_t == int(cid)
            if mask.any():
                scaled_boxes[mask, 2:4] = scaled_boxes[mask, 2:4] * float(factor)

    try:
        keep = nms_rotated(scaled_boxes, scores_t, float(nms_thr))[1]
    except Exception as exc:
        return list(pred_boxes_ego), f"Visualization NMS failed: {exc}"

    keep_idx = keep.detach().cpu().numpy().tolist()
    return [pred_boxes_ego[i] for i in keep_idx], None


def _pred_to_lidar_box_list(
    box_tensor,
    scores,
    labels,
    classes: Sequence[str],
    score_thr: float = 0.3,
    size_mode: str = "dx_dy",
    yaw_mode: str = "raw",
    z_is_bottom: bool = True,
    class_score_thr: Optional[Dict[str, float]] = None,
):
    boxes = []
    for i in range(box_tensor.shape[0]):
        score = float(scores[i])
        label = int(labels[i])
        cls_name = classes[label] if label < len(classes) else str(label)

        threshold = float(score_thr)
        if isinstance(class_score_thr, dict):
            threshold = float(class_score_thr.get(cls_name, score_thr))
        if score < threshold:
            continue

        x, y, z, dx, dy, dz, yaw = box_tensor[i, :7]
        if z_is_bottom:
            z = z + dz * 0.5

        if yaw_mode == "raw":
            yaw_vis = float(yaw)
        elif yaw_mode == "neg":
            yaw_vis = float(-yaw)
        elif yaw_mode == "p90":
            yaw_vis = float(yaw + np.pi / 2)
        elif yaw_mode == "m90":
            yaw_vis = float(yaw - np.pi / 2)
        else:
            raise ValueError(f"Unsupported yaw_mode: {yaw_mode}")

        if size_mode == "dx_dy":
            size = np.array([dx, dy, dz], dtype=np.float32)
        elif size_mode == "dy_dx":
            size = np.array([dy, dx, dz], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported size_mode: {size_mode}")

        box = Box(
            center=np.array([x, y, z], dtype=np.float32),
            size=size,
            orientation=Quaternion(axis=[0, 0, 1], radians=yaw_vis),
            name=cls_name,
            token=f"pred_{i}",
        )
        boxes.append((box, score, label))
    return boxes


def _resize_panel_to_hw(panel: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    h, w = panel.shape[:2]
    if h == target_h and w == target_w:
        return panel
    return cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _pad_to_even_hw(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame_bgr
    return cv2.copyMakeBorder(frame_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _resize_frame_max_width(frame: np.ndarray, max_width: Optional[int]) -> np.ndarray:
    if max_width is None or max_width <= 0:
        return frame

    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    new_w = int(max_width)
    new_h = max(2, int(round(h * (new_w / float(w)))))
    if (new_w % 2) != 0:
        new_w -= 1
    if (new_h % 2) != 0:
        new_h -= 1
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _open_video_writer(path: str, fps: float, size: Tuple[int, int]):
    codec = "mp4v"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter with codec={codec} for {path}")
    return writer


def _ffmpeg_h264_candidates() -> List[str]:
    if shutil.which("ffmpeg") is None:
        return []
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    enc_text = (probe.stdout or "") + (probe.stderr or "")
    candidates = []
    for enc in ["libx264", "libopenh264"]:
        if enc in enc_text:
            candidates.append(enc)
    return candidates


def _make_h264_preview_bytes(video_bytes: bytes) -> Tuple[Optional[bytes], Optional[str]]:
    candidates = _ffmpeg_h264_candidates()
    if not candidates:
        return None, "H.264 preview unavailable: ffmpeg/libx264/libopenh264 not found. Download video and play locally."

    in_tmp = tempfile.NamedTemporaryFile(prefix="tcar_in_", suffix=".mp4", delete=False)
    in_path = Path(in_tmp.name)
    in_tmp.close()
    out_tmp = tempfile.NamedTemporaryFile(prefix="tcar_h264_", suffix=".mp4", delete=False)
    out_path = Path(out_tmp.name)
    out_tmp.close()

    try:
        in_path.write_bytes(video_bytes)
        last_err = ""
        for encoder in candidates:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(in_path),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                encoder,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(out_path),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return out_path.read_bytes(), None
            last_err = (proc.stderr or "").strip()
        if last_err:
            return None, f"H.264 preview transcode failed. Last ffmpeg error: {last_err[-300:]}"
        return None, "H.264 preview transcode failed."
    finally:
        try:
            in_path.unlink()
        except FileNotFoundError:
            pass
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass


def _render_cam_panel_frame(
    channel: str,
    cam_infos: Dict[str, dict],
    pc_raw: LidarPointCloud,
    lidar2ego: np.ndarray,
    boxes_ego: Sequence[Box],
    pred_boxes_ego: Sequence[Tuple[Box, float, int]],
    min_dist: float,
    z_is_bottom: bool,
    do_hflip: bool,
    draw_cam_lidar_points: bool,
    cam_point_size: float,
    cam_point_alpha: float,
    gt_lw: float,
    pred_lw: float,
):
    info = cam_infos[channel]
    cam_cs = info["cs"]
    cam_k = info["K"]
    img = info["img"]

    pc_cam = LidarPointCloud(pc_raw.points.copy())
    pc_cam.transform(lidar2ego)
    ego2cam = transform_matrix(cam_cs["translation"], Quaternion(cam_cs["rotation"]), inverse=False)
    pc_cam.transform(ego2cam)

    depths = pc_cam.points[2, :]
    points = cam_k @ (pc_cam.points[:3, :] / depths)

    w, h = img.size
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < w - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < h - 1)
    points = points[:, mask]
    depths = depths[mask]

    dpi = 100
    fig = plt.Figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)

    if draw_cam_lidar_points and points.shape[1] > 0:
        vmin = float(np.percentile(depths, 5))
        vmax = float(np.percentile(depths, 95))
        ax.scatter(
            points[0, :],
            points[1, :],
            s=cam_point_size,
            c=depths,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            alpha=cam_point_alpha,
        )

    for box in boxes_ego:
        box_gt = _maybe_fix_z_origin(box, z_is_bottom)
        box_gt = _transform_box(box_gt, cam_cs["translation"], cam_cs["rotation"], inverse=False)
        _render_box_2d(ax, box_gt, cam_k, img.size, color="r", lw=gt_lw)

    for pred_box, _, _ in pred_boxes_ego:
        box_pred = _transform_box(pred_box, cam_cs["translation"], cam_cs["rotation"], inverse=False)
        _render_box_2d(ax, box_pred, cam_k, img.size, color="lime", lw=pred_lw)

    ax.axis("off")
    canvas.draw()
    panel = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    if do_hflip:
        panel = np.fliplr(panel)
    return panel


def _render_bev_panel_frame(
    target_h: int,
    points_ego: np.ndarray,
    boxes_ego: Sequence[Box],
    pred_boxes_ego: Sequence[Tuple[Box, float, int]],
    token: str,
    bev_point_size: float,
    bev_point_alpha: float,
    draw_ego_vehicle: bool,
    axes_limit: float,
    draw_heading: bool,
    heading_scale: float,
    gt_lw: float,
    pred_lw: float,
):
    bev_w = int(target_h * 1.10)
    bev_h = int(target_h)
    dpi = 100

    fig = plt.Figure(figsize=(bev_w / dpi, bev_h / dpi), dpi=dpi, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.scatter(points_ego[0, :], points_ego[1, :], s=bev_point_size, c="gray", alpha=bev_point_alpha)
    if draw_ego_vehicle:
        _render_ego_vehicle_bev(ax, color="blue", lw=5.0, alpha=0.95)
    _render_bev_boxes(ax, boxes_ego, color="r", lw=gt_lw, draw_heading=draw_heading, heading_scale=heading_scale)
    _render_bev_boxes(
        ax,
        [box for box, _, _ in pred_boxes_ego],
        color="lime",
        lw=pred_lw,
        draw_heading=draw_heading,
        heading_scale=heading_scale,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_xlabel("x forward")
    ax.set_ylabel("y left")
    ax.set_title(f"Token: {token[-8:]} | GT:red Pred:lime")

    fig.subplots_adjust(0, 0, 1, 1)
    canvas.draw()
    bev = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return bev


def _build_sample_context(ctx: dict, token: str):
    dataset = ctx["dataset"]
    nusc = ctx["nusc"]
    lidar_channel = ctx["lidar_channel"]

    dataset_idx = ctx["token_to_idx"].get(token)
    if dataset_idx is None:
        raise ValueError(f"sample token not found in dataset: {token}")

    sample = nusc.get("sample", token)
    lidar_sd_token = sample["data"][lidar_channel]
    lidar_sd = nusc.get("sample_data", lidar_sd_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])

    cam_infos = {}
    for channel in ctx["cam_channels_store"]:
        cam_sd_token = sample["data"][channel]
        cam_sd = nusc.get("sample_data", cam_sd_token)
        cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_k = np.array(cam_cs["camera_intrinsic"])
        img_path = os.path.join(nusc.dataroot, cam_sd["filename"])
        with Image.open(img_path) as img_src:
            img = img_src.convert("RGB").copy()
        cam_infos[channel] = {"sd": cam_sd, "cs": cam_cs, "K": cam_k, "img": img}

    boxes_ego = [nusc.get_box(t) for t in sample["anns"]]

    nms_stats = None
    if ctx.get("collect_nms_stats", False):
        result, nms_stats = _inference_single_with_nms_stats(
            ctx["model"], dataset, dataset_idx, device=ctx["device"]
        )
    else:
        result = _inference_single(ctx["model"], dataset, dataset_idx, device=ctx["device"])
    overlap_debug = None
    if ctx.get("collect_overlap_debug", False):
        overlap_debug = _compute_pred_bev_overlap_debug(
            result=result,
            class_names=list(ctx["model"].CLASSES),
            score_thr=float(ctx["score_thr"]),
            class_score_thr=(ctx["class_score_thr"] if ctx["use_classwise_thr"] else None),
            iou_thr=float(ctx.get("overlap_debug_iou_thr", 0.2)),
            max_pairs=int(ctx.get("overlap_debug_max_pairs", 20)),
        )
    box_tensor, scores, labels = _extract_pred(result)

    pred_boxes_lidar = _pred_to_lidar_box_list(
        box_tensor,
        scores,
        labels,
        classes=list(ctx["model"].CLASSES),
        score_thr=ctx["score_thr"],
        size_mode=ctx["size_mode"],
        yaw_mode=ctx["yaw_mode"],
        z_is_bottom=ctx["pred_z_is_bottom"],
        class_score_thr=(ctx["class_score_thr"] if ctx["use_classwise_thr"] else None),
    )

    pc_raw = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_sd["filename"]))
    nan_mask = ~np.isnan(pc_raw.points).any(axis=0)
    pc_raw.points = pc_raw.points[:, nan_mask]

    lidar2ego = transform_matrix(lidar_cs["translation"], Quaternion(lidar_cs["rotation"]), inverse=True)
    pc_ego = LidarPointCloud(pc_raw.points.copy())
    pc_ego.transform(lidar2ego)
    points_ego = pc_ego.points[:3, :].copy()

    pred_boxes_ego = []
    for box, score, label in pred_boxes_lidar:
        box_ego = _transform_box(box, lidar_cs["translation"], lidar_cs["rotation"], inverse=True)
        pred_boxes_ego.append((box_ego, score, label))

    return {
        "token": token,
        "cam_infos": cam_infos,
        "pc_raw": pc_raw,
        "lidar2ego": lidar2ego,
        "boxes_ego": boxes_ego,
        "pred_boxes_ego": pred_boxes_ego,
        "points_ego": points_ego,
        "nms_stats": nms_stats,
        "overlap_debug": overlap_debug,
    }


def _render_canvas_from_context(
    ctx: dict,
    sample_ctx: dict,
    pred_boxes_ego_override: Optional[Sequence[Tuple[Box, float, int]]] = None,
) -> np.ndarray:
    pred_boxes_ego = (
        list(pred_boxes_ego_override)
        if pred_boxes_ego_override is not None
        else sample_ctx["pred_boxes_ego"]
    )
    display_order = list(ctx["cam_channels_display"])
    if (
        ctx["swap_rear_positions"]
        and "CAM_BACK_LEFT" in display_order
        and "CAM_BACK_RIGHT" in display_order
    ):
        idx_bl = display_order.index("CAM_BACK_LEFT")
        idx_br = display_order.index("CAM_BACK_RIGHT")
        display_order[idx_bl], display_order[idx_br] = display_order[idx_br], display_order[idx_bl]

    cam_panels = []
    for channel in display_order:
        do_hflip = ctx["rear_flip_for_display"] and (channel in ctx["rear_flip_cams"])
        panel = _render_cam_panel_frame(
            channel=channel,
            cam_infos=sample_ctx["cam_infos"],
            pc_raw=sample_ctx["pc_raw"],
            lidar2ego=sample_ctx["lidar2ego"],
            boxes_ego=sample_ctx["boxes_ego"],
            pred_boxes_ego=pred_boxes_ego,
            min_dist=ctx["min_dist"],
            z_is_bottom=ctx["z_is_bottom"],
            do_hflip=do_hflip,
            draw_cam_lidar_points=ctx["draw_cam_lidar_points"],
            cam_point_size=ctx["cam_point_size"],
            cam_point_alpha=ctx["cam_point_alpha"],
            gt_lw=ctx["cam_gt_lw"],
            pred_lw=ctx["cam_pred_lw"],
        )
        cam_panels.append(panel)

    target_cam_hw = cam_panels[0].shape[:2]
    cam_panels = [_resize_panel_to_hw(panel, target_cam_hw) for panel in cam_panels]

    row_top = np.concatenate(cam_panels[:3], axis=1)
    row_bottom = np.concatenate(cam_panels[3:], axis=1)
    cams_canvas = np.concatenate([row_top, row_bottom], axis=0)

    bev_panel = _render_bev_panel_frame(
        target_h=cams_canvas.shape[0],
        points_ego=sample_ctx["points_ego"],
        boxes_ego=sample_ctx["boxes_ego"],
        pred_boxes_ego=pred_boxes_ego,
        token=sample_ctx["token"],
        bev_point_size=ctx["bev_point_size"],
        bev_point_alpha=ctx["bev_point_alpha"],
        draw_ego_vehicle=ctx["draw_ego_vehicle"],
        axes_limit=ctx["axes_limit"],
        draw_heading=ctx["draw_heading"],
        heading_scale=ctx["heading_scale"],
        gt_lw=ctx["bev_gt_lw"],
        pred_lw=ctx["bev_pred_lw"],
    )
    if bev_panel.shape[0] != cams_canvas.shape[0]:
        bev_panel = np.array(
            Image.fromarray(bev_panel).resize((bev_panel.shape[1], cams_canvas.shape[0]), Image.BILINEAR)
        )

    canvas = np.concatenate([cams_canvas, bev_panel], axis=1)
    canvas = _resize_frame_max_width(canvas, ctx["canvas_max_width"])
    return canvas


def render_sample_canvas(
    ctx: dict,
    token: str,
) -> Tuple[np.ndarray, int, Optional[dict], Optional[dict], Optional[np.ndarray], Optional[dict]]:
    sample_ctx = _build_sample_context(ctx, token)
    base_pred_boxes_ego = list(sample_ctx["pred_boxes_ego"])
    use_custom_main = bool(ctx.get("apply_custom_vis_nms_sample", True))
    compare_enabled = bool(ctx.get("enable_vis_nms_compare", False))

    custom_pred_boxes_ego = base_pred_boxes_ego
    custom_nms_err = None
    if use_custom_main or compare_enabled:
        custom_pred_boxes_ego, custom_nms_err = _apply_visualization_rotated_nms(
            pred_boxes_ego=base_pred_boxes_ego,
            nms_thr=float(ctx["vis_nms_thr"]),
            rescale_factors=ctx["vis_nms_rescale_factors"],
        )

    primary_pred_boxes_ego = custom_pred_boxes_ego if use_custom_main else base_pred_boxes_ego
    canvas = _render_canvas_from_context(
        ctx,
        sample_ctx,
        pred_boxes_ego_override=primary_pred_boxes_ego,
    )
    compare_canvas = None
    compare_meta = None

    if compare_enabled:
        if use_custom_main:
            compare_pred_boxes_ego = base_pred_boxes_ego
            primary_label = "Custom visualization NMS"
            compare_label = "Baseline (model output)"
        else:
            compare_pred_boxes_ego = custom_pred_boxes_ego
            primary_label = "Baseline (model output)"
            compare_label = "Custom visualization NMS"
        compare_canvas = _render_canvas_from_context(
            ctx, sample_ctx, pred_boxes_ego_override=compare_pred_boxes_ego
        )
        compare_meta = {
            "before": len(base_pred_boxes_ego),
            "after": len(custom_pred_boxes_ego),
            "removed": len(base_pred_boxes_ego) - len(custom_pred_boxes_ego),
            "thr": float(ctx["vis_nms_thr"]),
            "rescale_factors": list(ctx["vis_nms_rescale_factors"]),
            "error": custom_nms_err,
            "primary_label": primary_label,
            "compare_label": compare_label,
        }

    return (
        canvas,
        len(primary_pred_boxes_ego),
        sample_ctx.get("nms_stats"),
        sample_ctx.get("overlap_debug"),
        compare_canvas,
        compare_meta,
    )


def generate_scene_video(
    ctx: dict,
    scene_idx: int,
    fps: float,
    stride: int,
    max_frames: Optional[int],
    make_web_preview: bool,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[bytes, Optional[bytes], int, Optional[str]]:
    dataset = ctx["dataset"]
    scene_token, scene_indices = ctx["scene_index"][scene_idx]

    frame_indices = scene_indices[:: max(1, stride)]
    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[:max_frames]

    tmp_file = tempfile.NamedTemporaryFile(prefix="tcar_scene_", suffix=".mp4", delete=False)
    tmp_file_path = Path(tmp_file.name)
    tmp_file.close()

    writer = None
    target_size = None
    written = 0
    vis_nms_video_before_total = 0
    vis_nms_video_after_total = 0
    vis_nms_video_errors = 0
    vis_nms_video_applied = bool(ctx.get("apply_custom_vis_nms_video", False))
    prev_collect_nms_stats = bool(ctx.get("collect_nms_stats", False))
    prev_collect_overlap_debug = bool(ctx.get("collect_overlap_debug", False))
    ctx["collect_nms_stats"] = False
    ctx["collect_overlap_debug"] = False

    try:
        for i, ds_idx in enumerate(frame_indices):
            token = dataset.data_infos[ds_idx]["token"]
            sample_ctx = _build_sample_context(ctx, token)
            pred_override = None
            if vis_nms_video_applied:
                vis_nms_video_before_total += len(sample_ctx["pred_boxes_ego"])
                pred_override, nms_err = _apply_visualization_rotated_nms(
                    pred_boxes_ego=sample_ctx["pred_boxes_ego"],
                    nms_thr=float(ctx["vis_nms_thr"]),
                    rescale_factors=ctx["vis_nms_rescale_factors"],
                )
                vis_nms_video_after_total += len(pred_override)
                if nms_err:
                    vis_nms_video_errors += 1

            canvas = _render_canvas_from_context(
                ctx,
                sample_ctx,
                pred_boxes_ego_override=pred_override,
            )

            frame_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            frame_bgr = _pad_to_even_hw(frame_bgr)

            if writer is None:
                h, w = frame_bgr.shape[:2]
                target_size = (w, h)
                writer = _open_video_writer(str(tmp_file_path), fps, target_size)
            elif frame_bgr.shape[1] != target_size[0] or frame_bgr.shape[0] != target_size[1]:
                frame_bgr = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_LINEAR)

            writer.write(frame_bgr)
            written += 1

            if progress_cb is not None:
                progress_cb(i + 1, len(frame_indices), token)
    finally:
        if writer is not None:
            writer.release()
        ctx["collect_nms_stats"] = prev_collect_nms_stats
        ctx["collect_overlap_debug"] = prev_collect_overlap_debug

    if written == 0:
        try:
            tmp_file_path.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(f"No frames rendered for scene_idx={scene_idx}, scene_token={scene_token}")

    try:
        video_bytes = tmp_file_path.read_bytes()
    finally:
        try:
            tmp_file_path.unlink()
        except FileNotFoundError:
            pass

    preview_bytes = None
    preview_note = None
    if make_web_preview:
        preview_bytes, preview_note = _make_h264_preview_bytes(video_bytes)

    if vis_nms_video_applied:
        vis_note = (
            "Video custom visualization NMS applied | "
            f"before_total={vis_nms_video_before_total} "
            f"after_total={vis_nms_video_after_total} "
            f"removed_total={vis_nms_video_before_total - vis_nms_video_after_total} "
            f"thr={float(ctx['vis_nms_thr']):.2f} "
            f"rescale={list(ctx['vis_nms_rescale_factors'])}"
        )
        if vis_nms_video_errors > 0:
            vis_note += f" | warnings={vis_nms_video_errors}"
        if preview_note:
            preview_note = f"{preview_note}\n{vis_note}"
        else:
            preview_note = vis_note

    return video_bytes, preview_bytes, written, preview_note


@st.cache_resource(show_spinner=False)
def load_runtime(
    dataroot: str,
    version: str,
    config_path: str,
    checkpoint_path: str,
    device: str,
    split: str,
):
    nusc = TestCar(version=version, dataroot=dataroot, verbose=True)
    cfg, model, dataset = _build_model_and_dataset(config_path, checkpoint_path, device=device, split=split)

    scene_index = _build_scene_index(dataset)
    token_to_idx = {info["token"]: i for i, info in enumerate(dataset.data_infos)}

    return {
        "cfg": cfg,
        "model": model,
        "dataset": dataset,
        "nusc": nusc,
        "scene_index": scene_index,
        "token_to_idx": token_to_idx,
    }


def _parse_class_thr(raw_text: str) -> Dict[str, float]:
    txt = raw_text.strip()
    if not txt:
        return {}
    data = json.loads(txt)
    if not isinstance(data, dict):
        raise ValueError("Classwise threshold JSON must be an object, e.g. {\"car\": 0.3}")
    parsed = {}
    for key, val in data.items():
        parsed[str(key)] = float(val)
    return parsed


def _scene_labels(scene_index: List[Tuple[str, List[int]]]) -> List[str]:
    labels = []
    for i, (scene_token, ids) in enumerate(scene_index):
        labels.append(f"{i:03d} | {scene_token} ({len(ids)} samples)")
    return labels


def main():
    st.set_page_config(page_title="TCAR GT/Pred Streamlit", layout="wide")
    st.title("TCAR GT + Pred Visualizer")

    with st.sidebar:
        st.header("Runtime")
        dataroot = st.text_input("Data root", value="data/tcar")
        version = st.text_input("Version", value="v1.0-trainval")
        config_path = st.text_input("Config path", value="configs/bevdet/bevdet-r50-tcar.py")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="/data2/TCAR_DATA/outputs_bh/bevdet-r50-tcar/run_20260213_054926/latest.pth",
        )
        split = st.selectbox("Dataset split", options=["val", "test"], index=0)
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = st.text_input("Device", value=default_device)

        st.divider()
        st.header("Inference / Decode")
        score_thr = st.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        size_mode = st.selectbox("Size mode", options=["dx_dy", "dy_dx"], index=0)
        yaw_mode = st.selectbox("Yaw mode", options=["raw", "neg", "p90", "m90"], index=0)
        pred_z_is_bottom = st.checkbox("Pred z is bottom", value=True)

        use_classwise_thr = st.checkbox("Use classwise threshold JSON", value=False)
        class_thr_text = st.text_area(
            "Classwise threshold JSON",
            value="",
            help='Example: {"car": 0.35, "truck": 0.25, "pedestrian": 0.20}',
            height=80,
        )
        collect_nms_stats = st.checkbox(
            "Collect NMS stats on sample render",
            value=True,
            help="Wrap NMS call to report before/after candidate counts for this sample.",
        )
        collect_overlap_debug = st.checkbox(
            "Collect final pred-pred BEV IoU debug",
            value=True,
            help="Compute BEV IoU among final predictions and list high-overlap pairs.",
        )
        overlap_debug_iou_thr = st.slider(
            "Overlap debug IoU threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
        )
        overlap_debug_max_pairs = st.number_input(
            "Overlap debug max pairs",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
        )
        enable_vis_nms_compare = st.checkbox(
            "Compare baseline vs custom visualization NMS",
            value=False,
            help="Apply extra rotated NMS only for visualization and compare side-by-side.",
        )
        apply_custom_vis_nms_sample = st.checkbox(
            "Apply custom visualization NMS to sample",
            value=True,
            help="Use custom IoU threshold/rescale factors for the primary sample canvas.",
        )
        vis_nms_thr = st.slider(
            "Custom visualization NMS IoU threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
        )
        vis_nms_rescale_text = st.text_input(
            "Custom visualization NMS rescale factors (csv)",
            value="1.0,1.0,1.0,1.0",
            help="Per-class factors, e.g. 1.0,0.7,1.0,0.55",
        )

        st.divider()
        st.header("Canvas")
        min_dist = st.number_input("Min lidar depth", min_value=0.0, value=1.0, step=0.1)
        draw_cam_lidar_points = st.checkbox("Draw lidar points on camera panels", value=False)
        cam_point_size = st.number_input("Camera lidar point size", min_value=0.1, value=0.5, step=0.1)
        cam_point_alpha = st.slider("Camera lidar point alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        bev_point_size = st.number_input("BEV point size", min_value=0.01, value=0.2, step=0.05)
        bev_point_alpha = st.slider("BEV lidar point alpha", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
        draw_ego_vehicle = st.checkbox("Draw ego vehicle in BEV", value=True)
        axes_limit = st.number_input("BEV axis limit", min_value=1.0, value=50.0, step=1.0)
        z_is_bottom = st.checkbox("GT z is bottom", value=False)
        draw_heading = st.checkbox("Draw heading in BEV", value=True)
        heading_scale = st.number_input("Heading scale", min_value=0.1, value=2.5, step=0.1)

        swap_rear_positions = st.checkbox("Swap CAM_BACK_LEFT/RIGHT positions", value=True)
        rear_flip_for_display = st.checkbox("Flip rear cams for display", value=False)
        rear_flip_cams = st.multiselect(
            "Rear flip cams",
            options=DEFAULT_CAM_CHANNELS_STORE,
            default=["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
        )

        st.subheader("Line Width")
        cam_gt_lw = st.number_input("Cam GT line width", min_value=0.5, value=3.0, step=0.5)
        cam_pred_lw = st.number_input("Cam Pred line width", min_value=0.5, value=3.0, step=0.5)
        bev_gt_lw = st.number_input("BEV GT line width", min_value=0.5, value=4.0, step=0.5)
        bev_pred_lw = st.number_input("BEV Pred line width", min_value=0.5, value=5.0, step=0.5)

        canvas_max_width = st.number_input(
            "Canvas max width(px)",
            min_value=0,
            value=2400,
            step=100,
            help="0 means no width cap.",
        )

        st.divider()
        st.header("Video")
        video_fps = st.number_input("Video FPS", min_value=1, max_value=60, value=5, step=1)
        video_stride = st.number_input("Video stride", min_value=1, value=1, step=1)
        video_max_frames = st.number_input(
            "Video max frames (0 = full scene)",
            min_value=0,
            value=0,
            step=1,
        )
        video_make_web_preview = st.checkbox(
            "Make web-preview (H.264) for browser playback",
            value=True,
            help="If ffmpeg H.264 encoder exists, Streamlit preview uses web-compatible video.",
        )
        apply_custom_vis_nms_video = st.checkbox(
            "Apply custom visualization NMS to video",
            value=True,
            help="Use custom IoU threshold/rescale factors when rendering scene video frames.",
        )

    if use_classwise_thr:
        try:
            class_score_thr = _parse_class_thr(class_thr_text)
        except Exception as exc:
            st.error(f"Invalid class threshold JSON: {exc}")
            st.stop()
    else:
        class_score_thr = {}

    with st.spinner("Loading model + dataset + TCAR tables..."):
        try:
            runtime = load_runtime(
                dataroot=dataroot,
                version=version,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=device,
                split=split,
            )
        except Exception as exc:
            st.error(f"Failed to initialize runtime: {exc}")
            st.stop()

    dataset = runtime["dataset"]
    scene_index = runtime["scene_index"]
    test_cfg_pts = runtime["model"].pts_bbox_head.test_cfg
    nms_type_cfg = test_cfg_pts.get("nms_type", None)
    nms_thr_cfg = test_cfg_pts.get("nms_thr", None)
    default_rescale_factors = _default_rescale_factors_from_test_cfg(
        test_cfg_pts=test_cfg_pts,
        num_classes=len(getattr(runtime["model"], "CLASSES", [])),
    )
    custom_default_rescale_factors = _normalize_rescale_factors(
        [1.0, 1.0, 1.0, 1.0],
        num_classes=len(getattr(runtime["model"], "CLASSES", [])),
    )
    if not vis_nms_rescale_text.strip():
        vis_nms_rescale_factors = custom_default_rescale_factors
    else:
        try:
            vis_nms_rescale_factors = _normalize_rescale_factors(
                _parse_float_list_csv(vis_nms_rescale_text),
                num_classes=len(getattr(runtime["model"], "CLASSES", [])),
            )
        except Exception as exc:
            st.error(f"Invalid custom visualization rescale factors: {exc}")
            st.stop()

    render_ctx = {
        **runtime,
        "device": device,
        "cam_channels_store": DEFAULT_CAM_CHANNELS_STORE,
        "cam_channels_display": DEFAULT_CAM_CHANNELS_DISPLAY,
        "lidar_channel": "LIDAR_TOP",
        "score_thr": score_thr,
        "size_mode": size_mode,
        "yaw_mode": yaw_mode,
        "pred_z_is_bottom": pred_z_is_bottom,
        "use_classwise_thr": use_classwise_thr,
        "class_score_thr": class_score_thr,
        "collect_nms_stats": bool(collect_nms_stats),
        "collect_overlap_debug": bool(collect_overlap_debug),
        "overlap_debug_iou_thr": float(overlap_debug_iou_thr),
        "overlap_debug_max_pairs": int(overlap_debug_max_pairs),
        "enable_vis_nms_compare": bool(enable_vis_nms_compare),
        "apply_custom_vis_nms_sample": bool(apply_custom_vis_nms_sample),
        "vis_nms_thr": float(vis_nms_thr),
        "vis_nms_rescale_factors": vis_nms_rescale_factors,
        "apply_custom_vis_nms_video": bool(apply_custom_vis_nms_video),
        "min_dist": min_dist,
        "draw_cam_lidar_points": bool(draw_cam_lidar_points),
        "cam_point_size": float(cam_point_size),
        "cam_point_alpha": float(cam_point_alpha),
        "bev_point_size": bev_point_size,
        "bev_point_alpha": float(bev_point_alpha),
        "draw_ego_vehicle": bool(draw_ego_vehicle),
        "axes_limit": axes_limit,
        "z_is_bottom": z_is_bottom,
        "draw_heading": draw_heading,
        "heading_scale": heading_scale,
        "swap_rear_positions": swap_rear_positions,
        "rear_flip_for_display": rear_flip_for_display,
        "rear_flip_cams": set(rear_flip_cams),
        "cam_gt_lw": float(cam_gt_lw),
        "cam_pred_lw": float(cam_pred_lw),
        "bev_gt_lw": float(bev_gt_lw),
        "bev_pred_lw": float(bev_pred_lw),
        "canvas_max_width": int(canvas_max_width) if int(canvas_max_width) > 0 else None,
    }

    st.subheader("Dataset Summary")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Samples", len(dataset.data_infos))
    col_b.metric("Scenes", len(scene_index))
    col_c.metric("Classes", len(getattr(runtime["model"], "CLASSES", [])))

    if not scene_index:
        st.warning("No scenes found in dataset.")
        st.stop()

    st.subheader("Sample Visualization")
    scene_labels = _scene_labels(scene_index)
    selected_scene_label = st.selectbox("Scene", options=scene_labels, index=0)
    selected_scene_idx = int(selected_scene_label.split("|")[0].strip())

    scene_token, sample_indices = scene_index[selected_scene_idx]
    sample_pos = st.slider(
        "Sample position in scene",
        min_value=0,
        max_value=max(0, len(sample_indices) - 1),
        value=0,
        step=1,
    )

    selected_dataset_idx = sample_indices[sample_pos]
    selected_info = dataset.data_infos[selected_dataset_idx]
    selected_token = selected_info["token"]

    st.caption(
        f"scene_idx={selected_scene_idx}, scene_token={scene_token}, "
        f"dataset_idx={selected_dataset_idx}, token={selected_token}, "
        f"timestamp={selected_info.get('timestamp', 'N/A')}"
    )
    st.caption(f"NMS config: nms_type={nms_type_cfg}, nms_thr={nms_thr_cfg}")
    st.caption(f"NMS default rescale factors: {default_rescale_factors}")

    col_render, col_video = st.columns(2)
    with col_render:
        render_now = st.button("Render Selected Sample", use_container_width=True)
    with col_video:
        video_file_name = st.text_input(
            "Video filename",
            value=f"tcar_scene_val{selected_scene_idx}.mp4",
            help="Used as downloaded file name.",
        )
        generate_now = st.button("Generate Scene Video", use_container_width=True)

    if render_now:
        with st.spinner("Running inference and rendering sample canvas..."):
            try:
                canvas, num_pred, nms_stats, overlap_debug, compare_canvas, compare_meta = render_sample_canvas(
                    render_ctx, selected_token
                )
            except Exception as exc:
                st.error(f"Failed to render sample: {exc}")
            else:
                st.session_state["last_canvas"] = canvas
                st.session_state["last_canvas_compare"] = compare_canvas
                st.session_state["last_canvas_meta"] = {
                    "token": selected_token,
                    "num_pred": num_pred,
                    "scene_idx": selected_scene_idx,
                    "dataset_idx": selected_dataset_idx,
                    "file_name": f"tcar_sample_scene{selected_scene_idx:03d}_{selected_token[-8:]}.png",
                    "nms_stats": nms_stats,
                    "overlap_debug": overlap_debug,
                    "compare_meta": compare_meta,
                    "sample_vis_mode": (
                        "custom"
                        if bool(render_ctx.get("apply_custom_vis_nms_sample", True))
                        else "baseline"
                    ),
                }

    if "last_canvas" in st.session_state:
        meta = st.session_state.get("last_canvas_meta", {})
        st.success(
            f"Rendered token={meta.get('token', 'N/A')} "
            f"(pred boxes={meta.get('num_pred', 'N/A')})"
        )
        nms_stats = meta.get("nms_stats")
        if isinstance(nms_stats, dict):
            rotate_calls = int(nms_stats.get("rotate_calls", 0))
            circle_calls = int(nms_stats.get("circle_calls", 0))
            if rotate_calls > 0 or circle_calls > 0:
                st.info(
                    "NMS stats | "
                    f"rotate: {nms_stats.get('rotate_before_total', 0)} -> {nms_stats.get('rotate_keep_total', 0)} "
                    f"(calls={rotate_calls}), "
                    f"circle: {nms_stats.get('circle_before_total', 0)} -> {nms_stats.get('circle_keep_total', 0)} "
                    f"(calls={circle_calls})"
                )
            else:
                st.info("NMS stats: no centerpoint NMS call captured for this sample.")
        overlap_debug = meta.get("overlap_debug")
        if isinstance(overlap_debug, dict):
            err = overlap_debug.get("error")
            if err:
                st.warning(f"Overlap debug failed: {err}")
            else:
                st.info(
                    "Final pred-pred BEV IoU (after NMS + visualization score filtering) | "
                    f"num_preds={overlap_debug.get('num_preds', 0)}, "
                    f"max_iou={overlap_debug.get('max_iou', 0.0):.4f}, "
                    f"pairs_over_thr={overlap_debug.get('num_pairs_over_thr', 0)}"
                )
                pairs = overlap_debug.get("pairs", [])
                if pairs:
                    st.dataframe(pairs, use_container_width=True)
        canvas = st.session_state["last_canvas"]
        compare_canvas = st.session_state.get("last_canvas_compare")
        compare_meta = meta.get("compare_meta")
        if compare_canvas is not None:
            primary_label = "Primary"
            compare_label = "Compared"
            if isinstance(compare_meta, dict):
                primary_label = str(compare_meta.get("primary_label", primary_label))
                compare_label = str(compare_meta.get("compare_label", compare_label))
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(canvas, caption=primary_label, use_container_width=True)
            with col_b:
                st.image(compare_canvas, caption=compare_label, use_container_width=True)
            if isinstance(compare_meta, dict):
                st.info(
                    "Visualization NMS compare | "
                    f"before={compare_meta.get('before', 0)} "
                    f"after={compare_meta.get('after', 0)} "
                    f"removed={compare_meta.get('removed', 0)} "
                    f"thr={compare_meta.get('thr', 0.0):.2f} "
                    f"rescale={compare_meta.get('rescale_factors', [])}"
                )
                if compare_meta.get("error"):
                    st.warning(compare_meta["error"])
        else:
            sample_vis_mode = str(meta.get("sample_vis_mode", "baseline"))
            if sample_vis_mode == "custom":
                caption = "GT(red) / Pred(lime) Canvas | custom visualization NMS"
            else:
                caption = "GT(red) / Pred(lime) Canvas | baseline (model output)"
            st.image(canvas, caption=caption, use_container_width=True)
        png_name = meta.get("file_name", "tcar_sample.png")
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        st.download_button(
            label="Download Sample Image (PNG)",
            data=buf.getvalue(),
            file_name=png_name,
            mime="image/png",
            use_container_width=True,
        )
        if compare_canvas is not None:
            compare_png_name = png_name.rsplit(".", 1)[0] + "_compared.png"
            buf_cmp = io.BytesIO()
            Image.fromarray(compare_canvas).save(buf_cmp, format="PNG")
            st.download_button(
                label="Download Compared Image (PNG)",
                data=buf_cmp.getvalue(),
                file_name=compare_png_name,
                mime="image/png",
                use_container_width=True,
            )

    if generate_now:
        max_frames = int(video_max_frames)
        max_frames = None if max_frames <= 0 else max_frames

        progress = st.progress(0)
        status = st.empty()

        def _progress_cb(done: int, total: int, token: str):
            ratio = 0 if total <= 0 else int(done * 100 / total)
            progress.progress(min(100, max(0, ratio)))
            status.text(f"Rendering frame {done}/{total} | token={token[-8:]}")

        with st.spinner("Generating scene video..."):
            try:
                video_bytes, preview_bytes, frame_count, preview_note = generate_scene_video(
                    ctx=render_ctx,
                    scene_idx=selected_scene_idx,
                    fps=float(video_fps),
                    stride=int(video_stride),
                    max_frames=max_frames,
                    make_web_preview=bool(video_make_web_preview),
                    progress_cb=_progress_cb,
                )
            except Exception as exc:
                st.error(f"Failed to generate video: {exc}")
            else:
                progress.progress(100)
                status.text(f"Completed: {frame_count} frames")
                st.session_state["last_video_bytes"] = video_bytes
                st.session_state["last_video_preview_bytes"] = preview_bytes
                st.session_state["last_video_meta"] = {
                    "scene_idx": selected_scene_idx,
                    "scene_token": scene_token,
                    "frames": frame_count,
                    "file_name": video_file_name,
                    "preview_note": preview_note,
                }

    if "last_video_bytes" in st.session_state:
        meta = st.session_state.get("last_video_meta", {})
        st.subheader("Generated Scene Video")
        st.success(
            f"scene_idx={meta.get('scene_idx', 'N/A')}, "
            f"frames={meta.get('frames', 'N/A')}"
        )
        video_bytes = st.session_state["last_video_bytes"]
        preview_bytes = st.session_state.get("last_video_preview_bytes")
        file_name = meta.get("file_name", "tcar_scene.mp4")
        preview_note = meta.get("preview_note")
        st.video(preview_bytes if preview_bytes is not None else video_bytes)
        if preview_note:
            st.info(preview_note)
        st.download_button(
            label="Download Original Video (MP4)",
            data=video_bytes,
            file_name=file_name,
            mime="video/mp4",
            use_container_width=True,
        )
        if preview_bytes is not None:
            h264_name = file_name.rsplit(".", 1)[0] + "_web.mp4"
            st.download_button(
                label="Download Web Preview (H.264 MP4)",
                data=preview_bytes,
                file_name=h264_name,
                mime="video/mp4",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
