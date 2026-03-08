#!/usr/bin/env python3
"""
make_legibility_dataset.py

Create a balanced legibility fine-tuning dataset (cropped jersey images) from:
  1) STR per-crop predictions (jersey_id_results.json or .txt containing JSON)
  2) cropped jersey images directory (crops/)
  3) tracklet-level GT mapping (train_gt.txt-style JSON: {tracklet_id: jersey_number})

Outputs a folder compatible with jersey_number_dataset.JerseyNumberLegibilityDataset:
  OUT/
    train/images/*.jpg
    train/train_gt.csv     (two columns: filename,label)
    val/images/*.jpg
    val/val_gt.csv

Labeling policy (high precision, low noise):
  Positive (1): pred is valid 1..99 AND pred == GT AND confidence_product >= pos_quantile threshold
  Negative (0): pred invalid (not 1..99) OR confidence_product <= neg_quantile threshold
  Ambiguous: dropped (not used)

Balancing:
  Keeps equal positives and negatives within each split by dropping extras.

Splitting:
  Split by TRACKLET id (not by image) to avoid leakage. Use --val-frac.

Notes:
  - This script does NOT require running the full pipeline. It just needs the 3 artifacts above.
  - It assumes STR JSON keys correspond to crop filenames in crops_dir (e.g., "547_12.jpg").
"""

import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple, List

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def is_valid_number_1_99(s: str) -> bool:
    try:
        v = int(str(s))
        return 1 <= v <= 99
    except Exception:
        return False

def token_conf_product(conf_list) -> float:
    """
    Multiply token confidences, ignoring the last "end" token (matches repo logic).
    If conf_list is missing or malformed, returns 0.
    """
    if not isinstance(conf_list, (list, tuple)) or len(conf_list) == 0:
        return 0.0
    prod = 1.0
    # ignore last token (EOS)
    for c in conf_list[:-1]:
        try:
            prod *= float(c)
        except Exception:
            return 0.0
    return float(prod)

def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def load_tracklet_gt(path: Path) -> Dict[str, int]:
    """
    Expected format: dict {tracklet_id: jersey_number}
    Values can be int or str; we normalize to int when possible.
    """
    gt = load_json(path)
    if not isinstance(gt, dict):
        raise ValueError(f"GT file must be a JSON dict {{tracklet: number}}. Got: {type(gt)}")
    out = {}
    for k, v in gt.items():
        out[str(k)] = safe_int(v, default=-1)
    return out

def parse_tracklet_from_crop_name(crop_name: str) -> str:
    """
    Most common: "<tracklet>_<something>.jpg"
    Fallback: try to find a leading integer.
    """
    base = os.path.basename(crop_name)
    if "_" in base:
        return base.split("_", 1)[0]
    m = re.match(r"^(\d+)", base)
    return m.group(1) if m else "unknown"

def quantile(values: List[float], q: float) -> float:
    """
    Simple quantile without numpy (works for big lists).
    """
    if not values:
        return 0.0
    v = sorted(values)
    q = min(max(q, 0.0), 1.0)
    idx = int(round((len(v) - 1) * q))
    return float(v[idx])

def count_tracklets_from_pose_input(pose_input_path: Path) -> Tuple[int, int]:
    """
    If you have pose_input.json (COCO-like with 'images'), count unique tracklets.
    Tracklet id inferred from /images/<tracklet_id>/ in file_name.
    """
    data = load_json(pose_input_path)
    if not isinstance(data, dict) or "images" not in data:
        raise ValueError("pose_input must be a dict with an 'images' list.")
    tracklets = set()
    for img in data["images"]:
        fn = str(img.get("file_name", ""))
        m = re.search(r"/images/([^/]+)/", fn)
        if m:
            tracklets.add(m.group(1))
        else:
            # try last folder name
            parts = Path(fn).parts
            if len(parts) >= 2:
                tracklets.add(parts[-2])
    return len(data["images"]), len(tracklets)

def build_records(str_json: Dict[str, Any], crops_dir: Path, gt_map: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Build per-crop records with pred, prob, gt, and existence.
    """
    records = []
    probs = []
    missing = 0
    for crop_name, payload in str_json.items():
        crop_name = str(crop_name)
        tracklet = parse_tracklet_from_crop_name(crop_name)
        pred = str(payload.get("label", "")).strip()
        conf = payload.get("confidence", [])
        prob = token_conf_product(conf)
        gt_val = gt_map.get(str(tracklet), -1)

        img_path = crops_dir / crop_name
        if not img_path.exists():
            missing += 1

        rec = {
            "crop_name": crop_name,
            "tracklet": str(tracklet),
            "pred": pred,
            "prob": prob,
            "gt": gt_val,
            "exists": img_path.exists(),
            "src_path": str(img_path),
        }
        records.append(rec)
        probs.append(prob)

    return records

def label_and_select(
    records: List[Dict[str, Any]],
    pos_thr: float,
    neg_thr: float,
    max_pos_per_tracklet: int,
    max_neg_per_tracklet: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """
    Returns (pos_by_tracklet, neg_by_tracklet, stats)
    """
    pos_by = defaultdict(list)
    neg_by = defaultdict(list)
    stats = {"missing_images": 0, "ambiguous_dropped": 0, "pos_raw": 0, "neg_raw": 0}

    for r in records:
        if not r["exists"]:
            stats["missing_images"] += 1
            continue

        pred = r["pred"]
        gt = r["gt"]
        prob = r["prob"]
        t = r["tracklet"]

        is_pos = (gt != -1) and is_valid_number_1_99(pred) and (safe_int(pred, None) == safe_int(gt, None)) and (prob >= pos_thr)
        is_neg = (not is_valid_number_1_99(pred)) or (prob <= neg_thr)

        if is_pos:
            pos_by[t].append(r)
            stats["pos_raw"] += 1
        elif is_neg:
            neg_by[t].append(r)
            stats["neg_raw"] += 1
        else:
            stats["ambiguous_dropped"] += 1

    # cap per tracklet to avoid duplicates dominating
    rng = random.Random(0)
    for t, lst in pos_by.items():
        if len(lst) > max_pos_per_tracklet:
            rng.shuffle(lst)
            pos_by[t] = lst[:max_pos_per_tracklet]
    for t, lst in neg_by.items():
        if len(lst) > max_neg_per_tracklet:
            rng.shuffle(lst)
            neg_by[t] = lst[:max_neg_per_tracklet]

    return pos_by, neg_by, stats

def split_tracklets(tracklets: List[str], val_frac: float, seed: int) -> Tuple[set, set]:
    rng = random.Random(seed)
    t = list(tracklets)
    rng.shuffle(t)
    n_val = int(round(len(t) * val_frac))
    val = set(t[:n_val])
    train = set(t[n_val:])
    return train, val

def balance_entries(entries: List[Tuple[Dict[str, Any], int]], seed: int) -> List[Tuple[Dict[str, Any], int]]:
    """
    entries: list of (record,label)
    Makes labels balanced by dropping extras.
    """
    pos = [e for e in entries if e[1] == 1]
    neg = [e for e in entries if e[1] == 0]
    n = min(len(pos), len(neg))
    rng = random.Random(seed)
    rng.shuffle(pos); rng.shuffle(neg)
    out = pos[:n] + neg[:n]
    rng.shuffle(out)
    return out

def write_split(out_split_dir: Path, crops_dir: Path, entries: List[Tuple[Dict[str, Any], int]], copy_images: bool):
    img_dir = out_split_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out_split_dir / (out_split_dir.name + "_gt.csv")

    with open(ann_path, "w") as f:
        for rec, label in entries:
            crop_name = rec["crop_name"]
            src = crops_dir / crop_name
            dst = img_dir / crop_name
            if copy_images:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                f.write(f"{crop_name},{label}\n")
            else:
                # If not copying, the dataset class (as written) expects images inside img_dir.
                # So "no-copy" is not compatible unless you modify the dataset class.
                raise ValueError("copy_images=False is not supported with the current JerseyNumberLegibilityDataset implementation.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--str-json", required=True, help="Path to jersey_id_results.json (or .txt containing JSON)")
    ap.add_argument("--crops-dir", required=True, help="Directory containing crop images referenced by STR JSON keys")
    ap.add_argument("--gt-json", required=True, help="Path to tracklet GT mapping JSON (e.g., train_gt.txt)")
    ap.add_argument("--out-dir", required=True, help="Output dataset folder")
    ap.add_argument("--pos-quantile", type=float, default=0.7)
    ap.add_argument("--neg-quantile", type=float, default=0.3)
    ap.add_argument("--max-pos-per-tracklet", type=int, default=5)
    ap.add_argument("--max-neg-per-tracklet", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2, help="Split fraction by tracklet")
    ap.add_argument("--seed", type=int, default=519)
    ap.add_argument("--balance", action="store_true", help="Balance pos/neg by dropping extras (recommended)")
    ap.add_argument("--pose-input", default=None, help="Optional: pose_input.json to count tracklets/images")
    args = ap.parse_args()

    str_path = Path(args.str_json)
    crops_dir = Path(args.crops_dir)
    gt_path = Path(args.gt_json)
    out_dir = Path(args.out_dir)

    if args.pose_input:
        n_imgs, n_trk = count_tracklets_from_pose_input(Path(args.pose_input))
        print(f"[INFO] pose_input images={n_imgs}, unique_tracklets={n_trk}")

    str_json = load_json(str_path)
    if not isinstance(str_json, dict):
        raise ValueError(f"STR json must be dict. Got: {type(str_json)}")

    gt_map = load_tracklet_gt(gt_path)
    records = build_records(str_json, crops_dir, gt_map)

    probs = [r["prob"] for r in records]
    pos_thr = quantile(probs, args.pos_quantile)
    neg_thr = quantile(probs, args.neg_quantile)
    if neg_thr > pos_thr:
        # pathological; swap
        neg_thr, pos_thr = pos_thr, neg_thr

    pos_by, neg_by, stats = label_and_select(
        records,
        pos_thr=pos_thr,
        neg_thr=neg_thr,
        max_pos_per_tracklet=args.max_pos_per_tracklet,
        max_neg_per_tracklet=args.max_neg_per_tracklet,
    )

    all_tracklets = sorted(set(pos_by.keys()) | set(neg_by.keys()))
    train_trk, val_trk = split_tracklets(all_tracklets, args.val_frac, args.seed)

    # build entries per split
    train_entries = []
    val_entries = []
    for t, lst in pos_by.items():
        for rec in lst:
            (train_entries if t in train_trk else val_entries).append((rec, 1))
    for t, lst in neg_by.items():
        for rec in lst:
            (train_entries if t in train_trk else val_entries).append((rec, 0))

    if args.balance:
        train_entries = balance_entries(train_entries, seed=args.seed)
        val_entries = balance_entries(val_entries, seed=args.seed + 1)

    # write
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    write_split(out_dir / "train", crops_dir, train_entries, copy_images=True)
    write_split(out_dir / "val", crops_dir, val_entries, copy_images=True)

    # stats
    report = {
        "pos_quantile": args.pos_quantile,
        "neg_quantile": args.neg_quantile,
        "pos_threshold": pos_thr,
        "neg_threshold": neg_thr,
        "stats_raw": stats,
        "tracklets_total": len(all_tracklets),
        "tracklets_train": len(train_trk),
        "tracklets_val": len(val_trk),
        "samples_train": len(train_entries),
        "samples_val": len(val_entries),
        "balance": bool(args.balance),
        "max_pos_per_tracklet": args.max_pos_per_tracklet,
        "max_neg_per_tracklet": args.max_neg_per_tracklet,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[DONE] dataset written to:", str(out_dir))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
