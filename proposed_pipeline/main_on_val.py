#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_on_val.py

Run your FINAL pipeline (sharpness -> legibility -> digit model -> aggregation)
on the VALIDATION SET, then compute accuracy vs the validation GT dict.

- NO internet (loads checkpoints locally)
- Robustly loads GT from either JSON or a Python-dict text file
- Tries a few common SoccerNet jersey-2023 validation paths; edit VAL_IMG_ROOT / VAL_GT_PATH if needed.

Outputs:
  - val_pred_final.json  (dict: {tracklet_id: jersey_number_or_-1})
  - prints accuracy + extra breakdown stats
"""

import os
import json
import ast
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, models

# Robust PIL for truncated JPGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# HARD-CODED PATHS (EDIT ONLY THESE IF NEEDED)
# ============================================================
BASE_DIR = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline"

# --- Validation image root: should contain tracklet folders, each with JPGs ---
# If your val images are elsewhere, set VAL_IMG_ROOT directly to the correct path.
VAL_IMG_ROOT_CANDIDATES = [
    # common patterns
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/val/images",
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/valid/images",
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/train/images",  # if you use a val split inside train
]
VAL_IMG_ROOT = None  # leave None to auto-pick first existing from candidates

# --- Validation GT dict path: dict {tracklet_id(str or int): jersey(int)} ---
# Put your actual val GT file here if you have it.
VAL_GT_PATH_CANDIDATES = [
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/val/gt.json",
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/val/gt.txt",
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/valid/gt.json",
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/valid/gt.txt",
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/train/gt.json",
    "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/train/gt.txt",
]
VAL_GT_PATH = None  # leave None to auto-pick first existing from candidates

# Checkpoints (as you trained)
LEGIBILITY_CKPT = os.path.join(BASE_DIR, "runs_legibility", "best.pt")
DIGIT_CKPT      = os.path.join(BASE_DIR, "runs_finetune_digitwise", "best.pt")

OUT_DIR  = os.path.join(BASE_DIR, "runs_final_pipeline")
OUT_JSON = os.path.join(OUT_DIR, "val_pred_final.json")
os.makedirs(OUT_DIR, exist_ok=True)
# ============================================================


# ============================================================
# THRESHOLDS (TUNE IF NEEDED)
# ============================================================
IMAGE_SIZE = 224

# 1) Sharpness gate (variance of Laplacian on resized grayscale)
SHARPNESS_THRESH = 0.0

# 2) Legibility gate: keep frame if P(legible) >= thresh
LEGIBILITY_PROB_THRESH = 0.5

# 3) Digit confidence gate (optional)
DIGIT_CONF_THRESH: Optional[float] = None  # e.g. 0.50

# Tracklet decision: if -1 score is "close enough" to best score, output -1
# NOTE: your previous runs used a very aggressive value; for debugging,
# keep it lenient (small) or even set to 0.0 to almost never force -1.
NEG1_DOMINATE_RATIO = 0.1

# Batch sizes
LEGIBILITY_BS = 256
DIGIT_BS = 256
# ============================================================


# ============================================================
# LABEL CONVENTIONS (must match your digit-wise training)
# ============================================================
ONES_UNK   = 10  # ones: 0..9 + UNK(10) => 11 classes
TENS_BLANK = 10  # tens: 0..9 + BLANK(10) + UNK(11) => 12 classes
TENS_UNK   = 11
NUM_UNK    = 0   # num_digits: UNK(0), 1, 2 => 3 classes
# ============================================================


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Utilities: paths / IO
# ----------------------------
def pick_first_existing(cands: List[str], kind: str) -> str:
    for p in cands:
        if os.path.exists(p):
            return p
    msg = [f"[ERROR] Could not find {kind}. Tried:"]
    msg += [f"  - {p}" for p in cands]
    raise FileNotFoundError("\n".join(msg))


def list_tracklets(img_root: str) -> List[str]:
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"IMG_ROOT not found: {img_root}")
    tids = []
    for d in os.listdir(img_root):
        p = os.path.join(img_root, d)
        if os.path.isdir(p):
            tids.append(d)
    return sorted(tids)


def list_images(tracklet_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    out = []
    for n in os.listdir(tracklet_dir):
        if n.lower().endswith(exts):
            out.append(os.path.join(tracklet_dir, n))
    return sorted(out)


def safe_open_rgb(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return None


def load_gt_dict(path: str) -> Dict[str, int]:
    """
    Supports:
      - JSON file containing dict
      - TXT file containing Python literal dict (via ast.literal_eval)
    Returns keys as str, values as int.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GT file not found: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        d = ast.literal_eval(txt)

    out: Dict[str, int] = {}
    for k, v in d.items():
        out[str(k)] = int(v)
    return out


# ----------------------------
# 1) Sharpness: variance of Laplacian (numpy only)
# ----------------------------
def _laplacian_var(gray_u8: np.ndarray) -> float:
    g = gray_u8.astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0

    center = g[1:-1, 1:-1]
    up     = g[:-2,  1:-1]
    down   = g[2:,   1:-1]
    left   = g[1:-1, :-2]
    right  = g[1:-1, 2:]

    lap = (up + down + left + right) - 4.0 * center
    return float(np.var(lap))


def sharpness_score(img_rgb: Image.Image, resize_to: int = 224) -> float:
    im = img_rgb.resize((resize_to, resize_to))
    gray = np.array(im.convert("L"), dtype=np.uint8)
    return _laplacian_var(gray)


# ----------------------------
# 2) Legibility model
# ----------------------------
class LegibilityResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)  # class0=illegible, class1=legible
        self.net = m

    def forward(self, x):
        return self.net(x)


def load_legibility_model(ckpt_path: str, device: str) -> nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Legibility checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = None
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            if any(k.startswith("net.") or k.startswith("module.") for k in ckpt.keys()):
                state = ckpt
    if state is None:
        raise RuntimeError("Unrecognized legibility checkpoint format.")

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v

    model = LegibilityResNet50()
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        print(f"[WARN] Legibility unexpected keys (showing 5): {unexpected[:5]}", flush=True)
    if missing:
        print(f"[WARN] Legibility missing keys (showing 5): {missing[:5]}", flush=True)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def legibility_prob_legible(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[:, 1]  # class1 = legible
    return prob


# ----------------------------
# 3) Digit-wise jersey model
# ----------------------------
class DigitWiseHeadNet(nn.Module):
    """
    Backbone + 3 heads:
      - num_digits: 3 classes (UNK=0, 1, 2)
      - tens: 12 classes (0..9, BLANK=10, UNK=11)
      - ones: 11 classes (0..9, UNK=10)
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        m = models.resnet50(weights=None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.drop = nn.Dropout(dropout)
        self.head_num = nn.Linear(feat_dim, 3)
        self.head_tens = nn.Linear(feat_dim, 12)
        self.head_ones = nn.Linear(feat_dim, 11)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.drop(feat)
        return self.head_num(feat), self.head_tens(feat), self.head_ones(feat)


def load_digit_model(ckpt_path: str, device: str) -> nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Digit checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            if any(k.startswith("backbone.") or k.startswith("module.") for k in ckpt.keys()):
                state = ckpt
    if state is None:
        raise RuntimeError("Unrecognized digit checkpoint format.")

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v

    model = DigitWiseHeadNet(dropout=0.0)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        print(f"[WARN] Digit unexpected keys (showing 5): {unexpected[:5]}", flush=True)
    if missing:
        print(f"[WARN] Digit missing keys (showing 5): {missing[:5]}", flush=True)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def decode_frame(num_logits: torch.Tensor,
                 tens_logits: torch.Tensor,
                 ones_logits: torch.Tensor,
                 conf_thresh: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode logits -> jersey in {-1, 0..99}. Returns (jersey_pred[B], conf[B]).

    IMPORTANT: For 1-digit jerseys, we DO NOT require tens_pred == BLANK.
    We only require ones digit is valid and not UNK.
    """
    num_prob  = torch.softmax(num_logits, dim=1)
    tens_prob = torch.softmax(tens_logits, dim=1)
    ones_prob = torch.softmax(ones_logits, dim=1)

    num_pred  = torch.argmax(num_prob,  dim=1)
    tens_pred = torch.argmax(tens_prob, dim=1)
    ones_pred = torch.argmax(ones_prob, dim=1)

    num_conf  = torch.max(num_prob,  dim=1).values
    tens_conf = torch.max(tens_prob, dim=1).values
    ones_conf = torch.max(ones_prob, dim=1).values

    conf = ones_conf * num_conf
    conf = torch.where(num_pred == 2, conf * tens_conf, conf)

    B = num_pred.shape[0]
    jersey = torch.full((B,), -1, dtype=torch.long, device=num_pred.device)

    if conf_thresh is not None:
        low = conf < conf_thresh
        num_pred = num_pred.clone()
        tens_pred = tens_pred.clone()
        ones_pred = ones_pred.clone()
        num_pred[low] = NUM_UNK
        tens_pred[low] = TENS_UNK
        ones_pred[low] = ONES_UNK

    unk = (num_pred == NUM_UNK) | (ones_pred == ONES_UNK)

    # 1-digit: just use ones (ignore tens head)
    one_ok = (num_pred == 1) & (~unk) & (ones_pred <= 9)
    jersey[one_ok] = ones_pred[one_ok]

    # 2-digit: tens and ones must be valid 0..9
    two_ok = (num_pred == 2) & (~unk) & (tens_pred <= 9) & (ones_pred <= 9)
    jersey[two_ok] = tens_pred[two_ok] * 10 + ones_pred[two_ok]

    return jersey, conf


# ----------------------------
# Transforms
# ----------------------------
TF_EVAL = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ----------------------------
# 4) Tracklet aggregation
# ----------------------------
def aggregate_scores(preds: List[int], confs: List[float]) -> int:
    if not preds:
        return -1

    score: Dict[int, float] = {}
    for j, c in zip(preds, confs):
        score[j] = score.get(j, 0.0) + float(c)

    neg1 = score.get(-1, 0.0)
    best_j, best_s = max(score.items(), key=lambda kv: kv[1])

    if best_j == -1:
        return -1
    if neg1 >= NEG1_DOMINATE_RATIO * best_s:
        return -1
    return int(best_j)


# ----------------------------
# Full per-tracklet pipeline
# ----------------------------
@torch.no_grad()
def process_tracklet(tracklet_dir: str,
                     leg_model: nn.Module,
                     digit_model: nn.Module) -> int:
    img_paths = list_images(tracklet_dir)
    if not img_paths:
        return -1

    # Stage A: sharpness filter
    keep_paths: List[str] = []
    for p in img_paths:
        im = safe_open_rgb(p)
        if im is None:
            continue
        s = sharpness_score(im, resize_to=224)
        if s >= SHARPNESS_THRESH:
            keep_paths.append(p)
    if not keep_paths:
        return -1

    # Stage B: legibility filter
    legible_paths: List[str] = []
    for i in range(0, len(keep_paths), LEGIBILITY_BS):
        batch_paths = keep_paths[i:i + LEGIBILITY_BS]
        xs = []
        ok_paths = []
        for p in batch_paths:
            im = safe_open_rgb(p)
            if im is None:
                continue
            xs.append(TF_EVAL(im))
            ok_paths.append(p)
        if not xs:
            continue

        x = torch.stack(xs, dim=0).to(DEVICE)
        prob_leg = legibility_prob_legible(leg_model, x).detach().cpu().numpy().tolist()

        for p, pr in zip(ok_paths, prob_leg):
            if float(pr) >= LEGIBILITY_PROB_THRESH:
                legible_paths.append(p)
    if not legible_paths:
        return -1

    # Stage C: digit model
    frame_preds: List[int] = []
    frame_confs: List[float] = []

    for i in range(0, len(legible_paths), DIGIT_BS):
        batch_paths = legible_paths[i:i + DIGIT_BS]
        xs = []
        for p in batch_paths:
            im = safe_open_rgb(p)
            if im is None:
                continue
            xs.append(TF_EVAL(im))
        if not xs:
            continue

        x = torch.stack(xs, dim=0).to(DEVICE)
        num_logits, tens_logits, ones_logits = digit_model(x)

        jersey_pred, conf = decode_frame(num_logits, tens_logits, ones_logits, conf_thresh=DIGIT_CONF_THRESH)
        jersey_pred = jersey_pred.detach().cpu().numpy().tolist()
        conf = conf.detach().cpu().numpy().tolist()

        for j, c in zip(jersey_pred, conf):
            frame_preds.append(int(j))
            frame_confs.append(float(c))

    # Stage D: aggregate
    return aggregate_scores(frame_preds, frame_confs)


def evaluate_predictions(gt: Dict[str, int], pred: Dict[str, int]) -> None:
    keys = sorted(set(gt.keys()) & set(pred.keys()), key=lambda x: int(x) if x.isdigit() else x)

    total = 0
    correct = 0

    # extra breakdown
    gt_valid = 0          # gt != -1
    correct_valid = 0
    gt_invalid = 0        # gt == -1
    correct_invalid = 0   # pred == -1 when gt == -1

    for k in keys:
        g = int(gt[k])
        p = int(pred[k])
        total += 1
        if g == p:
            correct += 1
        if g != -1:
            gt_valid += 1
            if g == p:
                correct_valid += 1
        else:
            gt_invalid += 1
            if p == -1:
                correct_invalid += 1

    acc = correct / max(1, total)
    acc_valid = correct_valid / max(1, gt_valid)
    acc_invalid = correct_invalid / max(1, gt_invalid)

    print("\n================= VAL EVAL =================")
    print(f"Total tracklets evaluated : {total}")
    print(f"Overall accuracy          : {acc*100:.2f}%  ({correct}/{total})")
    print(f"GT valid (!= -1)          : {gt_valid} | accuracy on valid : {acc_valid*100:.2f}%  ({correct_valid}/{gt_valid})")
    print(f"GT invalid (== -1)        : {gt_invalid} | accuracy on invalid: {acc_invalid*100:.2f}%  ({correct_invalid}/{gt_invalid})")
    print("===========================================\n")


def main():
    global VAL_IMG_ROOT, VAL_GT_PATH

    if VAL_IMG_ROOT is None:
        VAL_IMG_ROOT = pick_first_existing(VAL_IMG_ROOT_CANDIDATES, "VAL_IMG_ROOT")
    if VAL_GT_PATH is None:
        VAL_GT_PATH = pick_first_existing(VAL_GT_PATH_CANDIDATES, "VAL_GT_PATH")

    print("[INFO] device:", DEVICE, flush=True)
    print("[INFO] val root:", VAL_IMG_ROOT, flush=True)
    print("[INFO] val gt  :", VAL_GT_PATH, flush=True)
    print("[INFO] leg ckpt:", LEGIBILITY_CKPT, flush=True)
    print("[INFO] digit ckpt:", DIGIT_CKPT, flush=True)
    print("[INFO] thresholds:",
          f"sharp={SHARPNESS_THRESH} leg_prob={LEGIBILITY_PROB_THRESH} digit_conf={DIGIT_CONF_THRESH} neg1_ratio={NEG1_DOMINATE_RATIO}",
          flush=True)

    gt = load_gt_dict(VAL_GT_PATH)

    leg_model = load_legibility_model(LEGIBILITY_CKPT, DEVICE)
    digit_model = load_digit_model(DIGIT_CKPT, DEVICE)

    tids = list_tracklets(VAL_IMG_ROOT)

    out: Dict[str, int] = {}
    for tid in tqdm(tids, desc="final pipeline (VAL)"):
        tdir = os.path.join(VAL_IMG_ROOT, tid)
        pred = process_tracklet(tdir, leg_model, digit_model)
        out[str(tid)] = int(pred)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[OK] wrote -> {OUT_JSON}", flush=True)

    # Evaluate on intersection of keys (in case your val root contains subset)
    evaluate_predictions(gt, out)


if __name__ == "__main__":
    main()
