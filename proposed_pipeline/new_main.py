#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
final_pipeline_infer.py

Per-tracklet inference pipeline (NO internet, all weights loaded locally):

For each tracklet:
  1) Sharpness detection (variance of Laplacian) -> drop blurry frames
  2) Legibility model (binary) -> compute per-frame P(legible)
     2a) Tracklet-level legibility gate -> if clearly illegible, output -1
     2b) Select best frames for digit model (top-K by legibility, plus any above threshold)
  3) Jersey-number digit-wise model on selected frames -> per-frame jersey + confidence
  4) Tracklet aggregation (confidence-weighted voting) -> final jersey (or -1)

Outputs:
  - test_pred_final.json   (dict: {tracklet_id: jersey_number_or_-1})

Hard-coded for your UBC ARC paths.
"""

import os
import json
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
# HARD-CODED PATHS (YOUR SETUP)  (KEEP THESE)
# ============================================================
BASE_DIR = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline"

TEST_IMG_ROOT = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/test/images"

LEGIBILITY_CKPT = os.path.join(BASE_DIR, "runs_legibility", "best.pt")
DIGIT_CKPT      = os.path.join(BASE_DIR, "runs_finetune_digitwise", "best.pt")

OUT_JSON = os.path.join(BASE_DIR, "runs_final_pipeline", "test_new_pred_final.json")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
# ============================================================


# ============================================================
# THRESHOLDS (TUNE IF NEEDED)
# ============================================================
IMAGE_SIZE = 224

# 1) Sharpness gate: variance of Laplacian on grayscale
# Typical working range depends on resize; start with 25~80 and tune.
SHARPNESS_THRESH = 0.0

# 2) Frame-level legibility threshold (used for "good frame" counting)
LEGIBILITY_PROB_THRESH = 0.5

# Tracklet-level legibility gate: mark tracklet as clearly -1 if it has too little evidence of legible frames
TRACKLET_MIN_GOOD_FRAMES = 3         # require at least this many "good" frames
TRACKLET_MIN_GOOD_RATIO  = 0.05      # or at least this fraction of frames "good"
TRACKLET_TOPK_FRAC       = 0.20      # evaluate top-K where K = frac * N
TRACKLET_TOPK_MIN_K      = 3
TRACKLET_TOPK_MAX_K      = 20
TRACKLET_TOPK_MEAN_THRESH = 0.60     # if even top-K mean is below this, it's clearly illegible

# 3) Digit model confidence gate (optional):
# if None, no extra gate beyond decoding logic.
DIGIT_CONF_THRESH: Optional[float] = None  # e.g. 0.50

# How much to weight digit confidence by legibility prob during aggregation
LEGIBILITY_WEIGHT_ALPHA = 1.5  # conf *= (p_legible ** alpha)

# Tracklet decision: if -1 score is close to best score, output -1
NEG1_DOMINATE_RATIO = 0.90

# Batch sizes for speed
LEGIBILITY_BS = 256
DIGIT_BS = 256
# ============================================================


# ============================================================
# LABEL CONVENTIONS (must match your digit-wise training)
# ============================================================
ONES_UNK = 10            # ones: 0..9 + UNK(10)  => 11 classes
TENS_BLANK = 10          # tens: 0..9 + BLANK(10) + UNK(11) => 12 classes
TENS_UNK = 11
NUM_UNK = 0              # num_digits: UNK(0), 1, 2 => 3 classes
# ============================================================


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Utilities
# ----------------------------
def list_tracklets(img_root: str) -> List[str]:
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"TEST_IMG_ROOT not found: {img_root}")
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


# ----------------------------
# 1) Sharpness (Variance of Laplacian)
# ----------------------------
def _laplacian_var(gray_u8: np.ndarray) -> float:
    """
    gray_u8: HxW uint8
    Returns variance of Laplacian response (higher = sharper).
    Pure numpy implementation (no cv2).
    """
    g = gray_u8.astype(np.float32)

    center = g[1:-1, 1:-1]
    up     = g[:-2,  1:-1]
    down   = g[2:,   1:-1]
    left   = g[1:-1, :-2]
    right  = g[1:-1, 2:]

    lap = (up + down + left + right) - 4.0 * center
    return float(np.var(lap))


def sharpness_score(img_rgb: Image.Image, resize_to: int = 224) -> float:
    # compute sharpness on resized grayscale for consistent scale
    im = img_rgb.resize((resize_to, resize_to))
    gray = np.array(im.convert("L"), dtype=np.uint8)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
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
    """
    Returns P(legible) for each item in batch.
    """
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[:, 1]
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
    def __init__(self, dropout: float = 0.2):
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

    model = DigitWiseHeadNet(dropout=0.0)  # dropout not needed at eval
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
    Decode per-frame logits -> jersey_number in {-1, 0..99}.
    Returns:
      jersey_pred: [B] long
      conf: [B] float (confidence proxy)
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

    # NOTE: keeping your original convention: for 1-digit, tens must be BLANK
    one_ok = (num_pred == 1) & (~unk) & (tens_pred == TENS_BLANK) & (ones_pred <= 9)
    jersey[one_ok] = ones_pred[one_ok]

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
    """
    Confidence-weighted voting. Returns best jersey or -1.
    """
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


def _topk_mean(vals: List[float], k: int) -> float:
    if not vals:
        return 0.0
    k = max(1, min(k, len(vals)))
    s = sorted(vals, reverse=True)[:k]
    return float(sum(s) / len(s))


def _select_frames_by_legibility(paths: List[str], probs: List[float]) -> Tuple[List[str], List[float]]:
    """
    Choose frames for digit model:
      - include all frames with prob >= LEGIBILITY_PROB_THRESH
      - plus ensure we have at least TOPK frames by adding top-K
    Returns selected_paths, selected_probs aligned.
    """
    assert len(paths) == len(probs)

    # First, take all above threshold
    selected = [(p, pr) for p, pr in zip(paths, probs) if pr >= LEGIBILITY_PROB_THRESH]

    # Ensure at least top-K are included (so we don't end up with 0 frames on hard tracklets)
    N = len(probs)
    k = int(TRACKLET_TOPK_FRAC * N)
    k = max(TRACKLET_TOPK_MIN_K, min(TRACKLET_TOPK_MAX_K, k))
    order = sorted(range(N), key=lambda i: probs[i], reverse=True)
    topk = [(paths[i], probs[i]) for i in order[:k]]

    # Merge (dedupe by path)
    seen = set()
    merged: List[Tuple[str, float]] = []
    for p, pr in selected + topk:
        if p not in seen:
            seen.add(p)
            merged.append((p, pr))

    merged.sort(key=lambda x: x[1], reverse=True)  # optional: high prob first
    sel_paths = [p for p, _ in merged]
    sel_probs = [float(pr) for _, pr in merged]
    return sel_paths, sel_probs


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

    # Stage A: Sharpness filter (CPU, per-image; cheap)
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

    # Stage B: Legibility model -> compute P(legible) for ALL kept frames
    all_paths: List[str] = []
    all_probs: List[float] = []

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
            all_paths.append(p)
            all_probs.append(float(pr))

    if not all_paths:
        return -1

    # Stage B2: Tracklet-level legibility gate (filter out clearly -1 tracklets)
    N = len(all_probs)
    good = [pr for pr in all_probs if pr >= LEGIBILITY_PROB_THRESH]
    num_good = len(good)
    good_ratio = num_good / max(1, N)

    k_for_gate = int(TRACKLET_TOPK_FRAC * N)
    k_for_gate = max(TRACKLET_TOPK_MIN_K, min(TRACKLET_TOPK_MAX_K, k_for_gate))
    topk_mean = _topk_mean(all_probs, k_for_gate)

    # Clearly illegible => return -1
    # (Need either enough good frames, or strong top-K evidence)
    if (num_good < TRACKLET_MIN_GOOD_FRAMES) and (good_ratio < TRACKLET_MIN_GOOD_RATIO) and (topk_mean < TRACKLET_TOPK_MEAN_THRESH):
        return -1

    # Stage B3: choose frames for digit model
    legible_paths, legible_probs = _select_frames_by_legibility(all_paths, all_probs)
    if not legible_paths:
        return -1

    # Stage C: Digit model on selected frames (weight confidence by legibility prob)
    frame_preds: List[int] = []
    frame_confs: List[float] = []

    # We keep legible_probs aligned with legible_paths
    for i in range(0, len(legible_paths), DIGIT_BS):
        batch_paths = legible_paths[i:i + DIGIT_BS]
        batch_probs = legible_probs[i:i + DIGIT_BS]

        xs = []
        ok_probs: List[float] = []
        for p, pr in zip(batch_paths, batch_probs):
            im = safe_open_rgb(p)
            if im is None:
                continue
            xs.append(TF_EVAL(im))
            ok_probs.append(float(pr))

        if not xs:
            continue

        x = torch.stack(xs, dim=0).to(DEVICE)
        num_logits, tens_logits, ones_logits = digit_model(x)

        jersey_pred, conf = decode_frame(num_logits, tens_logits, ones_logits, conf_thresh=DIGIT_CONF_THRESH)
        jersey_pred = jersey_pred.detach().cpu().numpy().tolist()
        conf = conf.detach().cpu().numpy().tolist()

        # Weight digit confidence by legibility (helps suppress false positives on GT=-1 tracklets)
        for j, c, pr in zip(jersey_pred, conf, ok_probs):
            w = float(c) * (float(pr) ** LEGIBILITY_WEIGHT_ALPHA)
            frame_preds.append(int(j))
            frame_confs.append(w)

    # If digit stage produced nothing (all corrupted etc.)
    if not frame_preds:
        return -1

    # Stage D: Aggregate
    return aggregate_scores(frame_preds, frame_confs)


def main():
    print("[INFO] device:", DEVICE, flush=True)
    print("[INFO] test root:", TEST_IMG_ROOT, flush=True)
    print("[INFO] leg ckpt:", LEGIBILITY_CKPT, flush=True)
    print("[INFO] digit ckpt:", DIGIT_CKPT, flush=True)

    leg_model = load_legibility_model(LEGIBILITY_CKPT, DEVICE)
    digit_model = load_digit_model(DIGIT_CKPT, DEVICE)

    tids = list_tracklets(TEST_IMG_ROOT)
    out: Dict[str, int] = {}

    for tid in tqdm(tids, desc="final pipeline (tracklet-gated)"):
        tdir = os.path.join(TEST_IMG_ROOT, tid)
        pred = process_tracklet(tdir, leg_model, digit_model)
        out[str(tid)] = int(pred)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"[OK] wrote -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
