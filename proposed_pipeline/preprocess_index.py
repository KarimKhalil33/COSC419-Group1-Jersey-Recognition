import os
import json
import csv
import random
from collections import defaultdict

# =======================
# HARD-CODED PATHS
# =======================
DATA_ROOT = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023"

TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "train", "images")
TRAIN_GT_JSON  = os.path.join(DATA_ROOT, "train", "train_gt.json")

OUT_DIR = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline/splits"
INDEX_CSV = os.path.join(OUT_DIR, "train_index.csv")
TRAIN_SPLIT_CSV = os.path.join(OUT_DIR, "train_split.csv")
VAL_SPLIT_CSV   = os.path.join(OUT_DIR, "val_split.csv")

VAL_RATIO = 0.2
SEED = 42

# If dataset has jersey number "00" you want to treat as two-digit,
# note: string "00" becomes int 0 after normalization, so we can't distinguish "0" vs "00".
# This flag chooses the interpretation for label == 0.
ALLOW_ZERO_ZERO = True
# =======================

IMG_EXTS = (".jpg", ".jpeg", ".png")

# Digit-head label conventions
ONES_UNK = 10            # ones: 0..9 + UNK(10)
TENS_BLANK = 10          # tens: 0..9 + BLANK(10) + UNK(11)
TENS_UNK = 11
NUM_UNK = 0              # num_digits: UNK(0), 1, 2


def list_images_under_tracklet(tracklet_dir: str):
    files = []
    for name in os.listdir(tracklet_dir):
        if name.lower().endswith(IMG_EXTS):
            files.append(os.path.join(tracklet_dir, name))
    return sorted(files)


def normalize_label(label):
    """
    Make label an int jersey number.
    Supports:
      - int
      - str digits like "07", "10", "0", "00"
      - "-1" for unknown
    """
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        s = label.strip()
        if s == "-1":
            return -1
        if s.isdigit():
            return int(s)
    raise ValueError(f"Unrecognized label format: {label} ({type(label)})")


def load_gt_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mapping(gt):
    """
    SoccerNet JSON formats vary.
    Supports:
      Pattern A (dict): { "tracklet_id": label, ... }
      Pattern B (list of dict): [{ "tracklet_id": ..., "label": ...}, ...]
      Pattern C (dict): { "annotations": [ {...}, ... ] }

    Returns: dict tracklet_id(str) -> jersey_number(int)
    """
    mapping = {}

    if isinstance(gt, dict):
        if "annotations" in gt and isinstance(gt["annotations"], list):
            for item in gt["annotations"]:
                tid = item.get("tracklet_id") or item.get("tracklet") or item.get("id")
                lab = item.get("label") or item.get("jersey_number") or item.get("number")
                if tid is None or lab is None:
                    continue
                mapping[str(tid)] = normalize_label(lab)
            return mapping

        # Direct mapping case: {"1069": 12, ...} or {"1069": "-1", ...}
        all_ok = True
        for _, v in gt.items():
            try:
                _ = normalize_label(v)
            except Exception:
                all_ok = False
                break
        if all_ok:
            for k, v in gt.items():
                mapping[str(k)] = normalize_label(v)
            return mapping

    if isinstance(gt, list):
        for item in gt:
            if not isinstance(item, dict):
                continue
            tid = item.get("tracklet_id") or item.get("tracklet") or item.get("id")
            lab = item.get("label") or item.get("jersey_number") or item.get("number")
            if tid is None or lab is None:
                continue
            mapping[str(tid)] = normalize_label(lab)
        return mapping

    raise ValueError("Unsupported JSON format for gt. Please inspect the json structure.")


def jersey_to_digits(jersey_number: int):
    """
    Encode jersey_number into digit-wise targets, keeping -1 as a VALID class.

    Returns:
      num_digits: {0(UNK), 1, 2}
      tens: {0..9, 10(BLANK), 11(UNK)}
      ones: {0..9, 10(UNK)}
    """
    # Unknown/unreadable
    if jersey_number == -1:
        return NUM_UNK, TENS_UNK, ONES_UNK

    if jersey_number < -1:
        raise ValueError(f"Jersey number must be -1 or non-negative, got: {jersey_number}")

    # 0 is ambiguous ("0" vs "00") after int conversion; choose via flag.
    if jersey_number == 0:
        if ALLOW_ZERO_ZERO:
            # treat as "00"
            return 2, 0, 0
        else:
            # treat as single digit "0"
            return 1, TENS_BLANK, 0

    if jersey_number < 10:
        return 1, TENS_BLANK, jersey_number

    if jersey_number <= 99:
        tens = jersey_number // 10
        ones = jersey_number % 10
        return 2, tens, ones

    raise ValueError(f"Jersey number out of supported range (-1, 0-99): {jersey_number}")


def grouped_stratified_split(tracklets, tid_to_label, val_ratio, seed):
    """
    Grouped by tracklet_id, stratified by jersey_number.
    Constraint: for every jersey_number, keep at least 1 tracklet in TRAIN
    so that validation never contains labels unseen during training.
    """
    rnd = random.Random(seed)

    label_to_tids = defaultdict(list)
    for tid in tracklets:
        label_to_tids[tid_to_label[tid]].append(tid)

    # shuffle within each label
    for lab in label_to_tids:
        rnd.shuffle(label_to_tids[lab])

    total = len(tracklets)
    target_val = int(round(total * val_ratio))
    val_tids = set()

    # initial allocation per label
    for lab, tids in label_to_tids.items():
        n = len(tids)
        if n <= 1:
            k = 0  # must keep it in train
        else:
            k = int(round(n * val_ratio))
            k = max(0, min(k, n - 1))  # ensure >=1 remains in train
        val_tids.update(tids[:k])

    # adjust to hit target_val (greedy), respecting per-label constraint
    def current_val_count(lab):
        return sum(1 for t in label_to_tids[lab] if t in val_tids)

    def can_add(lab):
        n = len(label_to_tids[lab])
        return current_val_count(lab) < (n - 1)

    def can_remove(lab):
        return current_val_count(lab) > 0

    def pick_add_tid(lab):
        for t in label_to_tids[lab]:
            if t not in val_tids:
                return t
        return None

    def pick_remove_tid(lab):
        for t in reversed(label_to_tids[lab]):
            if t in val_tids:
                return t
        return None

    while len(val_tids) < target_val:
        candidates = []
        for lab in label_to_tids:
            if can_add(lab):
                n = len(label_to_tids[lab])
                cur = current_val_count(lab)
                remaining_add = (n - 1) - cur
                candidates.append((remaining_add, lab))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, lab = candidates[0]
        t = pick_add_tid(lab)
        if t is None:
            break
        val_tids.add(t)

    while len(val_tids) > target_val:
        candidates = []
        for lab in label_to_tids:
            if can_remove(lab):
                candidates.append((current_val_count(lab), lab))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, lab = candidates[0]
        t = pick_remove_tid(lab)
        if t is None:
            break
        val_tids.remove(t)

    train_tids = set(tracklets) - val_tids
    return train_tids, val_tids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)

    if not os.path.isdir(TRAIN_IMG_ROOT):
        raise FileNotFoundError(f"Missing train images folder: {TRAIN_IMG_ROOT}")
    if not os.path.isfile(TRAIN_GT_JSON):
        raise FileNotFoundError(f"Missing train gt json: {TRAIN_GT_JSON}")

    gt = load_gt_json(TRAIN_GT_JSON)
    tid_to_label = build_mapping(gt)

    # tracklets present on disk
    tracklet_dirs = sorted([
        d for d in os.listdir(TRAIN_IMG_ROOT)
        if os.path.isdir(os.path.join(TRAIN_IMG_ROOT, d))
    ])

    print(f"[INFO] Tracklet folders on disk: {len(tracklet_dirs)}")
    print(f"[INFO] Labels in gt json:        {len(tid_to_label)}")

    # Keep only tracklets that have labels
    tracklets = [tid for tid in tracklet_dirs if tid in tid_to_label]
    dropped = [tid for tid in tracklet_dirs if tid not in tid_to_label]
    if dropped:
        print(f"[WARN] {len(dropped)} tracklet folders have no label in train_gt.json (dropping). Example: {dropped[:5]}")

    if not tracklets:
        raise RuntimeError("No labeled tracklets found. Likely gt json keys don't match folder names.")

    # Build full per-frame index rows (do NOT skip -1)
    rows = []
    bad_labels = 0

    for tid in tracklets:
        jersey = tid_to_label[tid]
        try:
            num_digits, tens, ones = jersey_to_digits(jersey)
        except Exception as e:
            bad_labels += 1
            print(f"[WARN] Skipping tracklet={tid} due to label issue: {e}")
            continue

        img_dir = os.path.join(TRAIN_IMG_ROOT, tid)
        imgs = list_images_under_tracklet(img_dir)
        for p in imgs:
            rows.append((p, jersey, num_digits, tens, ones, tid))

    if not rows:
        raise RuntimeError("No frames indexed. Check image extensions and directory structure.")

    # Write full index
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "jersey_number", "num_digits", "tens", "ones", "tracklet_id"])
        w.writerows(rows)

    # Grouped + stratified split by jersey_number
    # Use only tracklets that survived label encoding
    valid_tracklets = sorted(set(tid for *_, tid in rows))
    train_tids, val_tids = grouped_stratified_split(
        tracklets=valid_tracklets,
        tid_to_label=tid_to_label,
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    def write_split(out_csv, keep_tids):
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "jersey_number", "num_digits", "tens", "ones", "tracklet_id"])
            for (p, jersey, num_digits, tens, ones, tid) in rows:
                if tid in keep_tids:
                    w.writerow([p, jersey, num_digits, tens, ones, tid])

    write_split(TRAIN_SPLIT_CSV, train_tids)
    write_split(VAL_SPLIT_CSV, val_tids)

    # Sanity checks: label coverage
    train_labels = set(tid_to_label[tid] for tid in train_tids)
    val_labels = set(tid_to_label[tid] for tid in val_tids)
    missing_in_train = sorted(val_labels - train_labels)

    print(f"[OK] Wrote index:      {INDEX_CSV} ({len(rows)} frames)")
    print(f"[OK] Wrote train split:{TRAIN_SPLIT_CSV} | tracklets={len(train_tids)}")
    print(f"[OK] Wrote val split:  {VAL_SPLIT_CSV} | tracklets={len(val_tids)}")
    print(f"[INFO] labels_in_val_but_not_train = {missing_in_train}")
    if bad_labels:
        print(f"[WARN] Skipped {bad_labels} tracklets due to invalid labels/encoding.")


if __name__ == "__main__":
    main()
