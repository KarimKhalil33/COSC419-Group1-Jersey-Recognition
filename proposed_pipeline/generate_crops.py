#!/usr/bin/env python3
"""
generate_srt_crops.py

Goal:
- Use YOLO Pose estimation to crop jersey-number regions
- Generate ~50,000 crops total for SRT training
- Uses cheap quality filters (stride + sharpness) and pose-based torso crop
- Builds tracklet-level splits (train/val/test) to avoid leakage

Assumptions about data layout (generic SoccerNet-like):
- images_root/
    <tracklet_id>/
        *.jpg / *.png  (frames)
  (Script will also work if frames are nested deeper; it recursively finds images under each tracklet folder.)

- gt_json is a dict mapping: { "tracklet_id": jersey_number_int, ... }
  (jersey_number can be int or string; script normalizes to string label)

Outputs:
- out_dir/
    crops/<tracklet_id>/<frame_basename>__crop.jpg
    splits/train.tsv
    splits/val.tsv
    splits/test.tsv
    stats.json

TSV format (SRT-friendly, common):
relative_path<TAB>label

Dependencies:
- ultralytics
- opencv-python
- numpy
- tqdm

Example:
python generate_srt_crops.py \
  --images_root /path/to/test \
  --gt_json /path/to/test_gt.json \
  --out_dir /path/to/out_srt \
  --pose_weights yolov8s-pose.pt \
  --target_total 50000 \
  --frame_stride 8 \
  --sharpness_thresh 80 \
  --max_per_tracklet 40
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ultralytics is required
try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: ultralytics is not installed or failed to import.")
    print("Install with: pip install ultralytics")
    raise


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CropRecord:
    rel_path: str
    label: str
    tracklet_id: str


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--images_root", type=str, required=True,
                    help="Root directory containing tracklet folders.")
    ap.add_argument("--gt_json", type=str, required=True,
                    help="JSON file with {tracklet_id: jersey_number} mapping.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for crops and splits.")

    ap.add_argument("--pose_weights", type=str, default="yolov8s-pose.pt",
                    help="Ultralytics YOLO pose weights (e.g., yolov8n-pose.pt, yolov8s-pose.pt).")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size for YOLO pose.")

    ap.add_argument("--target_total", type=int, default=50000, help="Total crops to generate.")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--frame_stride", type=int, default=8,
                    help="Only process every Nth frame within each tracklet (cheap subsampling).")
    ap.add_argument("--max_per_tracklet", type=int, default=40,
                    help="Hard cap crops per tracklet to avoid domination by long tracklets.")
    ap.add_argument("--sharpness_thresh", type=float, default=80.0,
                    help="Variance of Laplacian threshold. Increase to be stricter.")

    ap.add_argument("--pose_conf", type=float, default=0.4, help="Pose detection confidence threshold.")
    ap.add_argument("--kp_conf", type=float, default=0.3,
                    help="Minimum keypoint confidence to consider a keypoint valid.")
    ap.add_argument("--pad_scale", type=float, default=1.35,
                    help="Padding scale on the torso crop (bigger => looser crop).")

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--write_jpg_quality", type=int, default=95)

    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_gt(gt_json_path: str) -> Dict[str, str]:
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    # Normalize keys/values to strings
    out = {}
    for k, v in gt.items():
        if v is None:
            out[str(k)] = "-1"
        else:
            out[str(k)] = str(v)
    return out


def list_tracklet_dirs(images_root: str) -> List[str]:
    # Tracklet dirs are immediate children; if your structure differs,
    # adjust this function.
    dirs = []
    for name in os.listdir(images_root):
        p = os.path.join(images_root, name)
        if os.path.isdir(p):
            dirs.append(p)
    dirs.sort()
    return dirs


def list_images_recursive(tracklet_dir: str) -> List[str]:
    imgs = []
    for root, _, files in os.walk(tracklet_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                imgs.append(os.path.join(root, fn))
    imgs.sort()
    return imgs


def var_laplacian_sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clamp_box(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def pick_best_instance_pose(result, kp_conf: float):
    """
    Pick the best pose instance from an Ultralytics pose Result:
    Prefer instance with highest mean confidence over keypoints.
    Fallback to highest box confidence if keypoints are missing.
    """
    if result is None:
        return None

    # Boxes + keypoints are aligned by instance index.
    kps = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return None

    n = len(boxes)

    best_i = 0
    best_score = -1.0

    if kps is not None and hasattr(kps, "conf") and kps.conf is not None:
        # kps.conf shape: (n, num_kp)
        conf = kps.conf.detach().cpu().numpy()
        for i in range(n):
            valid = conf[i] >= kp_conf
            if valid.any():
                score = float(conf[i][valid].mean())
            else:
                score = 0.0
            # tie-break by box conf
            try:
                bconf = float(boxes.conf[i].item())
            except Exception:
                bconf = 0.0
            score = score + 0.05 * bconf
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    # Fallback: highest box conf
    try:
        confs = boxes.conf.detach().cpu().numpy()
        best_i = int(np.argmax(confs))
    except Exception:
        best_i = 0
    return best_i


def torso_crop_from_keypoints(
    bgr: np.ndarray,
    kpxy: np.ndarray,   # (num_kp, 2)
    kpconf: np.ndarray, # (num_kp,)
    kp_conf: float,
    pad_scale: float
) -> Optional[np.ndarray]:
    """
    Build a torso-based crop using shoulders + hips (COCO indices):
    5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
    Returns cropped BGR region or None if insufficient keypoints.
    """
    h, w = bgr.shape[:2]
    idx = [5, 6, 11, 12]
    pts = []
    for i in idx:
        if i < len(kpconf) and kpconf[i] >= kp_conf:
            pts.append(kpxy[i])
    if len(pts) < 2:
        return None

    pts = np.array(pts, dtype=np.float32)

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    # Expand to include more jersey region (digits often on chest/back)
    # and be robust to missing one of hips/shoulders.
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    bw = (x_max - x_min) * pad_scale
    bh = (y_max - y_min) * (pad_scale * 1.25)

    # Bias crop downward slightly (numbers often mid-torso)
    cy = cy + 0.08 * bh

    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh

    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def bbox_crop(bgr: np.ndarray, xyxy: np.ndarray, pad_scale: float) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy.tolist())
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * pad_scale
    bh = (y2 - y1) * pad_scale

    # Focus more on upper body region for jersey number
    cy = cy - 0.1 * bh

    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_tsv(path: str, rows: List[CropRecord]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r.rel_path}\t{r.label}\n")


def split_by_tracklet(
    records: List[CropRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
):
    # Tracklet-level split
    tracklets = sorted({r.tracklet_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(tracklets)

    n = len(tracklets)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = n - n_train - n_val

    train_t = set(tracklets[:n_train])
    val_t = set(tracklets[n_train:n_train + n_val])
    test_t = set(tracklets[n_train + n_val:])

    train_rows, val_rows, test_rows = [], [], []
    for r in records:
        if r.tracklet_id in train_t:
            train_rows.append(r)
        elif r.tracklet_id in val_t:
            val_rows.append(r)
        else:
            test_rows.append(r)

    return train_rows, val_rows, test_rows, {
        "num_tracklets_total": n,
        "num_tracklets_train": len(train_t),
        "num_tracklets_val": len(val_t),
        "num_tracklets_test": len(test_t),
        "num_crops_train": len(train_rows),
        "num_crops_val": len(val_rows),
        "num_crops_test": len(test_rows),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if not (abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6):
        print("ERROR: train/val/test ratios must sum to 1.0")
        sys.exit(1)

    gt = load_gt(args.gt_json)

    # Only "valid" labels (exclude -1) for SRT training
    valid_labels = sorted({v for v in gt.values() if v != "-1"})
    if len(valid_labels) == 0:
        print("ERROR: No valid labels found (all are -1?)")
        sys.exit(1)

    # Uniform per-label quota (simple + effective)
    per_label_quota = int(math.ceil(args.target_total / len(valid_labels)))

    out_crops_root = os.path.join(args.out_dir, "crops")
    out_splits_root = os.path.join(args.out_dir, "splits")
    ensure_dir(out_crops_root)
    ensure_dir(out_splits_root)

    # Load YOLO Pose
    model = YOLO(args.pose_weights)

    # Stats + bookkeeping
    counts_by_label = {lab: 0 for lab in valid_labels}
    counts_by_tracklet = {}
    records: List[CropRecord] = []

    rejects = {
        "no_gt": 0,
        "gt_is_-1": 0,
        "no_images": 0,
        "cv2_read_fail": 0,
        "sharpness_low": 0,
        "pose_no_det": 0,
        "pose_conf_low": 0,
        "crop_failed": 0,
        "crop_too_small": 0,
        "label_quota_full": 0,
        "tracklet_cap_full": 0,
        "already_have_enough": 0,
    }

    tracklet_dirs = list_tracklet_dirs(args.images_root)

    # First pass: enforce per-label quota (uniform)
    pbar = tqdm(tracklet_dirs, desc="Pass1 tracklets", unit="trk")
    for trk_dir in pbar:
        trk_id = os.path.basename(trk_dir)

        if trk_id not in gt:
            rejects["no_gt"] += 1
            continue

        label = gt[trk_id]
        if label == "-1":
            rejects["gt_is_-1"] += 1
            continue

        if label not in counts_by_label:
            # In case label set changed after valid_labels computed (rare)
            counts_by_label[label] = 0

        # If this label quota already full, skip this tracklet in pass1
        if counts_by_label[label] >= per_label_quota:
            rejects["label_quota_full"] += 1
            continue

        imgs = list_images_recursive(trk_dir)
        if not imgs:
            rejects["no_images"] += 1
            continue

        # Init tracklet cap
        counts_by_tracklet.setdefault(trk_id, 0)
        if counts_by_tracklet[trk_id] >= args.max_per_tracklet:
            rejects["tracklet_cap_full"] += 1
            continue

        # Process frames with stride
        for idx in range(0, len(imgs), args.frame_stride):
            if len(records) >= args.target_total:
                rejects["already_have_enough"] += 1
                break

            if counts_by_label[label] >= per_label_quota:
                rejects["label_quota_full"] += 1
                break

            if counts_by_tracklet[trk_id] >= args.max_per_tracklet:
                rejects["tracklet_cap_full"] += 1
                break

            img_path = imgs[idx]
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                rejects["cv2_read_fail"] += 1
                continue

            sharp = var_laplacian_sharpness(bgr)
            if sharp < args.sharpness_thresh:
                rejects["sharpness_low"] += 1
                continue

            # YOLO pose inference
            res_list = model.predict(
                source=bgr,
                imgsz=args.imgsz,
                conf=args.pose_conf,
                verbose=False
            )
            if not res_list or res_list[0] is None:
                rejects["pose_no_det"] += 1
                continue

            res = res_list[0]
            if res.boxes is None or len(res.boxes) == 0:
                rejects["pose_no_det"] += 1
                continue

            # pick best instance
            best_i = pick_best_instance_pose(res, args.kp_conf)
            if best_i is None:
                rejects["pose_no_det"] += 1
                continue

            # confirm confidence
            try:
                bconf = float(res.boxes.conf[best_i].item())
            except Exception:
                bconf = 0.0
            if bconf < args.pose_conf:
                rejects["pose_conf_low"] += 1
                continue

            crop = None
            # Use keypoints if available
            if res.keypoints is not None and hasattr(res.keypoints, "xy") and res.keypoints.xy is not None:
                try:
                    kpxy = res.keypoints.xy[best_i].detach().cpu().numpy()  # (num_kp,2)
                    kpconf = res.keypoints.conf[best_i].detach().cpu().numpy()  # (num_kp,)
                    crop = torso_crop_from_keypoints(bgr, kpxy, kpconf, args.kp_conf, args.pad_scale)
                except Exception:
                    crop = None

            # Fallback to bbox if keypoints insufficient
            if crop is None:
                try:
                    xyxy = res.boxes.xyxy[best_i].detach().cpu().numpy()
                    crop = bbox_crop(bgr, xyxy, args.pad_scale)
                except Exception:
                    crop = None

            if crop is None:
                rejects["crop_failed"] += 1
                continue

            ch, cw = crop.shape[:2]
            if ch < 64 or cw < 64:
                rejects["crop_too_small"] += 1
                continue

            # Write crop
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_dir_trk = os.path.join(out_crops_root, trk_id)
            ensure_dir(out_dir_trk)
            out_name = f"{base}__crop.jpg"
            out_abs = os.path.join(out_dir_trk, out_name)

            ok = cv2.imwrite(out_abs, crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.write_jpg_quality)])
            if not ok:
                rejects["crop_failed"] += 1
                continue

            rel = os.path.relpath(out_abs, args.out_dir).replace("\\", "/")
            records.append(CropRecord(rel_path=rel, label=label, tracklet_id=trk_id))
            counts_by_label[label] += 1
            counts_by_tracklet[trk_id] += 1

        if len(records) >= args.target_total:
            break

        pbar.set_postfix({
            "crops": len(records),
            "labels_filled": sum(1 for l in valid_labels if counts_by_label.get(l, 0) >= per_label_quota)
        })

    # Second pass: fill remaining up to target_total without label quotas
    if len(records) < args.target_total:
        need = args.target_total - len(records)
        pbar2 = tqdm(tracklet_dirs, desc=f"Pass2 fill remaining ({need})", unit="trk")
        for trk_dir in pbar2:
            if len(records) >= args.target_total:
                break
            trk_id = os.path.basename(trk_dir)

            if trk_id not in gt:
                continue
            label = gt[trk_id]
            if label == "-1":
                continue

            imgs = list_images_recursive(trk_dir)
            if not imgs:
                continue

            counts_by_tracklet.setdefault(trk_id, 0)
            if counts_by_tracklet[trk_id] >= args.max_per_tracklet:
                continue

            for idx in range(0, len(imgs), args.frame_stride):
                if len(records) >= args.target_total:
                    break
                if counts_by_tracklet[trk_id] >= args.max_per_tracklet:
                    break

                img_path = imgs[idx]
                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                sharp = var_laplacian_sharpness(bgr)
                if sharp < args.sharpness_thresh:
                    continue

                res_list = model.predict(
                    source=bgr,
                    imgsz=args.imgsz,
                    conf=args.pose_conf,
                    verbose=False
                )
                if not res_list or res_list[0] is None:
                    continue
                res = res_list[0]
                if res.boxes is None or len(res.boxes) == 0:
                    continue

                best_i = pick_best_instance_pose(res, args.kp_conf)
                if best_i is None:
                    continue

                crop = None
                if res.keypoints is not None and hasattr(res.keypoints, "xy") and res.keypoints.xy is not None:
                    try:
                        kpxy = res.keypoints.xy[best_i].detach().cpu().numpy()
                        kpconf = res.keypoints.conf[best_i].detach().cpu().numpy()
                        crop = torso_crop_from_keypoints(bgr, kpxy, kpconf, args.kp_conf, args.pad_scale)
                    except Exception:
                        crop = None

                if crop is None:
                    try:
                        xyxy = res.boxes.xyxy[best_i].detach().cpu().numpy()
                        crop = bbox_crop(bgr, xyxy, args.pad_scale)
                    except Exception:
                        crop = None

                if crop is None:
                    continue

                ch, cw = crop.shape[:2]
                if ch < 64 or cw < 64:
                    continue

                base = os.path.splitext(os.path.basename(img_path))[0]
                out_dir_trk = os.path.join(out_crops_root, trk_id)
                ensure_dir(out_dir_trk)
                out_name = f"{base}__crop.jpg"
                out_abs = os.path.join(out_dir_trk, out_name)

                ok = cv2.imwrite(out_abs, crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.write_jpg_quality)])
                if not ok:
                    continue

                rel = os.path.relpath(out_abs, args.out_dir).replace("\\", "/")
                records.append(CropRecord(rel_path=rel, label=label, tracklet_id=trk_id))
                counts_by_label[label] = counts_by_label.get(label, 0) + 1
                counts_by_tracklet[trk_id] += 1

            pbar2.set_postfix({"crops": len(records)})

    # If we overshot (shouldn't), trim deterministically
    if len(records) > args.target_total:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[:args.target_total]

    # Tracklet-level split
    train_rows, val_rows, test_rows, split_stats = split_by_tracklet(
        records, args.seed, args.train_ratio, args.val_ratio, args.test_ratio
    )

    write_tsv(os.path.join(out_splits_root, "train.tsv"), train_rows)
    write_tsv(os.path.join(out_splits_root, "val.tsv"), val_rows)
    write_tsv(os.path.join(out_splits_root, "test.tsv"), test_rows)

    # Build label distribution
    final_counts = {}
    for r in records:
        final_counts[r.label] = final_counts.get(r.label, 0) + 1

    stats = {
        "args": vars(args),
        "target_total": args.target_total,
        "generated_total": len(records),
        "num_valid_labels": len(valid_labels),
        "per_label_quota_pass1": per_label_quota,
        "final_label_counts": dict(sorted(final_counts.items(), key=lambda x: (-x[1], x[0]))),
        "rejects": rejects,
        "split_stats": split_stats,
        "notes": [
            "Pass1 uses uniform per-label quota to encourage balance.",
            "Pass2 fills remaining without label quotas.",
            "Splits are by tracklet_id to avoid leakage."
        ],
    }

    ensure_dir(args.out_dir)
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDone.")
    print(f"Output root: {args.out_dir}")
    print(f"Total crops: {len(records)}")
    print(f"TSV: {os.path.join(out_splits_root, 'train.tsv')}, val.tsv, test.tsv")
    print(f"Stats: {os.path.join(args.out_dir, 'stats.json')}")


if __name__ == "__main__":
    main()