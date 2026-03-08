#!/usr/bin/env python3
import os
import csv
import random
from collections import Counter

# ============================================================
# HARD-CODED PATHS (MATCH YOUR PIPELINE)
# ============================================================
BASE_DIR = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline"

TRAIN_IN = os.path.join(BASE_DIR, "splits_legibility", "train_legibility.csv")
VAL_IN   = os.path.join(BASE_DIR, "splits_legibility", "val_legibility.csv")

OUT_DIR  = os.path.join(BASE_DIR, "splits_legibility_balanced")

TRAIN_OUT = os.path.join(OUT_DIR, "train_legibility_balanced.csv")
VAL_OUT   = os.path.join(OUT_DIR, "val_legibility.csv")  # copied, not modified
# ============================================================

random.seed(1337)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def label_stats(rows):
    return Counter(int(row["label"]) for row in rows)


def balance_train(rows):
    """
    Keep all -1 samples.
    Randomly downsample 0 samples to match count(-1).
    """
    neg = [r for r in rows if int(r["label"]) == -1]
    pos = [r for r in rows if int(r["label"]) == 0]

    if len(neg) == 0 or len(pos) == 0:
        raise RuntimeError("Cannot balance: one class is empty.")

    target = min(len(neg), len(pos))
    pos_bal = random.sample(pos, target)

    balanced = neg + pos_bal
    random.shuffle(balanced)
    return balanced


def main():
    ensure_dir(OUT_DIR)

    # ------------------------
    # TRAIN: balance
    # ------------------------
    train_rows, train_fields = read_csv(TRAIN_IN)

    print("[TRAIN] original distribution:", label_stats(train_rows))

    train_bal = balance_train(train_rows)

    print("[TRAIN] balanced distribution:", label_stats(train_bal))

    write_csv(TRAIN_OUT, train_bal, train_fields)

    # ------------------------
    # VAL: copy as-is
    # ------------------------
    val_rows, val_fields = read_csv(VAL_IN)

    print("[VAL] untouched distribution:", label_stats(val_rows))

    write_csv(VAL_OUT, val_rows, val_fields)

    print("\n[DONE]")
    print(" Balanced train CSV:", TRAIN_OUT)
    print(" Validation CSV   :", VAL_OUT)


if __name__ == "__main__":
    main()
