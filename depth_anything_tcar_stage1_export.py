import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


PROJECT_ROOT = Path(".").resolve()
TCAR_ROOT = PROJECT_ROOT / "data" / "tcar"
INFO_FILES = [
    TCAR_ROOT / "bevdet-tcar_infos_temporal_train.pkl",
    # TCAR_ROOT / "tcar_infos_temporal_val.pkl",
]

DA3_SRC = PROJECT_ROOT / "Depth-Anything-3" / "src"
USE_RAY_POSE = True
DEPTH_EXPORT_MODE = os.environ.get("DEPTH_EXPORT_MODE", "relative")  # "metric" or "relative"
print(f"Using DEPTH_EXPORT_MODE={DEPTH_EXPORT_MODE} (set env var to override)")
if DEPTH_EXPORT_MODE not in {"metric", "relative"}:
    raise ValueError("DEPTH_EXPORT_MODE must be 'metric' or 'relative'")
DA3_MODEL_METRIC = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_MODEL_RELATIVE = "depth-anything/DA3-GIANT-1.1"
DA3_MODEL = DA3_MODEL_RELATIVE if DEPTH_EXPORT_MODE == "relative" else DA3_MODEL_METRIC

STORAGE_ROOT = Path("/data/byounghun/tcar")
if DEPTH_EXPORT_MODE == "relative":
    OUT_TAG = "da3_relative_depth_stage1"
else:
    OUT_TAG = "da3_metric_depth_stage1"
OUT_ROOT = STORAGE_ROOT / OUT_TAG
LINK_PATH = TCAR_ROOT / OUT_TAG
DEPTH_DIR = OUT_ROOT / "depth_npz"
MANIFEST_PATH = OUT_ROOT / "manifest.jsonl"
INDEX_PATH = OUT_ROOT / "index_by_token.pkl"

CAM_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

MAX_SAMPLES = None
SKIP_EXISTING = True
OVERWRITE_MANIFEST = False
SAVE_CONF = True
SAVE_DTYPE = np.float16
FAIL_LOG_LIMIT = 20
FAIL_FAST = bool(int(os.environ.get("FAIL_FAST", "0")))


def convert_to_relative_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Make depth scale-invariant by per-image median normalization."""
    depth = np.asarray(depth, dtype=np.float32)
    rel = np.zeros_like(depth, dtype=np.float32)
    for i in range(depth.shape[0]):
        d = depth[i]
        valid = np.isfinite(d) & (d > 0.0)
        if valid.any():
            med = float(np.median(d[valid]))
            scale = med if med > eps else 1.0
            rel_i = d / scale
            rel_i[~np.isfinite(rel_i)] = 0.0
            rel_i = np.maximum(rel_i, 0.0)
            rel[i] = rel_i
        else:
            rel[i] = np.zeros_like(d, dtype=np.float32)
    return rel


def _infer_split_name(path: Path) -> str:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return path.stem


def load_info_records(info_files):
    records = []
    for info_path in info_files:
        info_path = Path(info_path)
        if not info_path.exists():
            raise FileNotFoundError(info_path)
        split = _infer_split_name(info_path)
        with open(info_path, "rb") as f:
            payload = pickle.load(f)
        infos = payload["infos"] if isinstance(payload, dict) and "infos" in payload else payload
        for info in infos:
            records.append((split, info))
    return records


def main():
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 0):
        raise RuntimeError(f"Need torch>=2.0. Current: {torch.__version__}")

    sys.path.insert(0, str(DA3_SRC.resolve()))
    from depth_anything_3.api import DepthAnything3

    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("out:", OUT_ROOT)
    print("link:", LINK_PATH)
    print("use_ray_pose:", USE_RAY_POSE)
    print("depth_export_mode:", DEPTH_EXPORT_MODE)

    records = load_info_records(INFO_FILES)
    if MAX_SAMPLES is not None:
        records = records[:MAX_SAMPLES]

    print("num records:", len(records))
    print("example token:", records[0][1]["token"] if records else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(DA3_MODEL).to(device)
    model.eval()
    print("Loaded model:", DA3_MODEL)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    if not LINK_PATH.exists() and not LINK_PATH.is_symlink():
        LINK_PATH.parent.mkdir(parents=True, exist_ok=True)
        LINK_PATH.symlink_to(OUT_ROOT, target_is_directory=True)
    elif LINK_PATH.is_symlink() and LINK_PATH.resolve() == OUT_ROOT.resolve():
        pass
    else:
        print(f"[WARN] Keep existing path without override: {LINK_PATH}")

    if OVERWRITE_MANIFEST and MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    open_mode = "a" if MANIFEST_PATH.exists() else "w"
    written = 0
    skipped = 0
    failed = 0
    fail_logged = 0

    with open(MANIFEST_PATH, open_mode, encoding="utf-8") as mf:
        for split, info in tqdm(records, desc="DA3 depth export"):
            cams = info.get("cams", {})
            sample_token = info.get("token", "")

            image_paths = []
            rows = []
            missing_cam = False

            for cam_name in CAM_ORDER:
                if cam_name not in cams:
                    missing_cam = True
                    break
                cam = cams[cam_name]
                sd_token = cam["sample_data_token"]
                img_rel = cam["data_path"]
                img_abs = PROJECT_ROOT / img_rel
                if not img_abs.exists():
                    missing_cam = True
                    break
                npz_rel = Path("depth_npz") / sd_token[:2] / f"{sd_token}.npz"
                npz_abs = OUT_ROOT / npz_rel
                image_paths.append(str(img_abs))
                rows.append((cam_name, cam, sd_token, img_rel, npz_rel, npz_abs))

            if missing_cam:
                failed += 1
                if fail_logged < FAIL_LOG_LIMIT:
                    print(f"[FAIL][missing_cam] sample={sample_token}")
                    fail_logged += 1
                continue

            if SKIP_EXISTING and all(r[-1].exists() for r in rows):
                skipped += len(rows)
                continue

            try:
                pred = model.inference(
                    image_paths,
                    use_ray_pose=USE_RAY_POSE,
                )
                depth = np.asarray(pred.depth, dtype=np.float32)
                source_is_metric = bool(getattr(pred, "is_metric", 0))
                if DEPTH_EXPORT_MODE == "relative":
                    if source_is_metric:
                        depth = convert_to_relative_depth(depth)
                    else:
                        # Already relative depth from model output.
                        depth = np.where(np.isfinite(depth), depth, 0.0)
                        depth = np.maximum(depth, 0.0)
                else:
                    depth = np.where(np.isfinite(depth), depth, 0.0)
                    depth = np.maximum(depth, 0.0)
                conf = (
                    np.asarray(pred.conf)
                    if SAVE_CONF and getattr(pred, "conf", None) is not None
                    else None
                )
            except Exception as exc:
                failed += 1
                if fail_logged < FAIL_LOG_LIMIT:
                    print(f"[FAIL][inference] sample={sample_token} err={repr(exc)}")
                    print(traceback.format_exc())
                    fail_logged += 1
                if FAIL_FAST:
                    raise
                continue

            for i, (cam_name, cam, sd_token, img_rel, npz_rel, npz_abs) in enumerate(rows):
                if SKIP_EXISTING and npz_abs.exists():
                    skipped += 1
                    continue

                npz_abs.parent.mkdir(parents=True, exist_ok=True)
                save_dict = {"depth": depth[i].astype(SAVE_DTYPE)}
                if conf is not None:
                    save_dict["conf"] = conf[i].astype(SAVE_DTYPE)
                np.savez_compressed(npz_abs, **save_dict)

                rec = {
                    "sample_data_token": sd_token,
                    "sample_token": sample_token,
                    "scene_token": info.get("scene_token", ""),
                    "split": split,
                    "cam_name": cam_name,
                    "timestamp": int(cam.get("timestamp", 0)),
                    "image_relpath": img_rel,
                    "depth_relpath": str(npz_rel),
                    "shape_hw": list(depth[i].shape),
                    "dtype": str(SAVE_DTYPE),
                    "model": DA3_MODEL,
                    "use_ray_pose": bool(USE_RAY_POSE),
                    "depth_mode": DEPTH_EXPORT_MODE,
                    "source_is_metric": source_is_metric,
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(
        {
            "written_npz": written,
            "skipped_npz": skipped,
            "failed_samples": failed,
            "manifest": str(MANIFEST_PATH),
        }
    )

    index = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            index[rec["sample_data_token"]] = rec

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("index size:", len(index))
    print("index path:", INDEX_PATH)

    if index:
        one = next(iter(index.values()))
        npz_abs = OUT_ROOT / one["depth_relpath"]
        arr = np.load(npz_abs)
        print("example token:", one["sample_data_token"])
        print("depth shape:", arr["depth"].shape, "dtype:", arr["depth"].dtype)
        if "conf" in arr:
            print("conf shape:", arr["conf"].shape, "dtype:", arr["conf"].dtype)


if __name__ == "__main__":
    main()
