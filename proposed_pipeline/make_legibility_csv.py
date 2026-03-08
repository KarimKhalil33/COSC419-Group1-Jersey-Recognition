#!/usr/bin/env python3
import os
import csv
from typing import Any, Tuple, List
from PIL import Image, UnidentifiedImageError

# ============================================================
# HARD-CODED PATHS (DO NOT TOUCH)
# ============================================================
BASE_DIR = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline"

TRAIN_CSV = os.path.join(BASE_DIR, "splits", "train_split.csv")
VAL_CSV   = os.path.join(BASE_DIR, "splits", "val_split.csv")

OUT_DIR   = os.path.join(BASE_DIR, "splits_legibility")

TRAIN_OUT = os.path.join(OUT_DIR, "train_legibility.csv")
VAL_OUT   = os.path.join(OUT_DIR, "val_legibility.csv")

BAD_TRAIN = os.path.join(OUT_DIR, "bad_train_images.txt")
BAD_VAL   = os.path.join(OUT_DIR, "bad_val_images.txt")
# ============================================================


PATH_COLS  = ["image_path", "path", "img_path", "filepath", "file_path", "image"]
LABEL_COLS = ["label", "jersey", "jersey_number", "jerseynumber", "number", "target", "y"]


def detect_columns(fieldnames: List[str]) -> Tuple[str, str]:
    lower = [f.lower() for f in fieldnames]

    path_col = next((fieldnames[lower.index(c)] for c in PATH_COLS if c in lower), None)
    label_col = next((fieldnames[lower.index(c)] for c in LABEL_COLS if c in lower), None)

    if path_col is None or label_col is None:
        raise RuntimeError(f"Cannot detect columns. Found columns: {fieldnames}")

    return path_col, label_col


def parse_label(x: Any) -> int:
    return int(float(str(x).strip()))


def image_ok(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def convert(in_csv: str, out_csv: str, bad_txt: str):
    with open(in_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"No header in {in_csv}")

        path_col, label_col = detect_columns(reader.fieldnames)

        out_rows = []
        bad_paths = []

        for row in reader:
            img_path = str(row[path_col]).strip()
            y = parse_label(row[label_col])

            # -------------------------------
            # MAP LABELS:
            #   -1 -> -1
            #   others -> 0
            # -------------------------------
            y_leg = -1 if y == -1 else 0

            if not image_ok(img_path):
                bad_paths.append(img_path)
                continue

            out_rows.append({
                "image_path": img_path,
                "label": y_leg
            })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerows(out_rows)

    with open(bad_txt, "w") as f:
        for p in bad_paths:
            f.write(p + "\n")

    print(f"[OK] {in_csv}")
    print(f"     -> {out_csv} (kept={len(out_rows)})")
    print(f"     bad images: {len(bad_paths)} -> {bad_txt}")


def main():
    ensure_dir(OUT_DIR)

    convert(TRAIN_CSV, TRAIN_OUT, BAD_TRAIN)
    convert(VAL_CSV,   VAL_OUT,   BAD_VAL)

    print("\n[DONE] Legibility CSVs created.")
    print(f"  Train: {TRAIN_OUT}")
    print(f"  Val  : {VAL_OUT}")


if __name__ == "__main__":
    main()
