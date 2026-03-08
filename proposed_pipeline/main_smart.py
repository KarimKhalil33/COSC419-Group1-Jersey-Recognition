import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# ============================================================
# PATHS (KEEP YOUR PATHS)
# ============================================================
TEST_IMG_ROOT = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/test/images"
OUT_JSON = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/test_smart_pred_final.txt"

# NOTE: Use the checkpoints you trained
LEGIBILITY_CKPT = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline/runs_legibility/best.pt"
DIGIT_CKPT = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline/runs_finetune_digitwise/best.pt"

# ============================================================
# SETTINGS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
LEG_BS = 64
DIGIT_BS = 64

# Sharpness filtering
SHARPNESS_MIN = 20.0

# Legibility stage
# train_legibility.py mapping: (-1 -> class 0), (0 -> class 1) => LEGIBLE index = 1
LEGIBLE_CLASS_INDEX = 1
LEGIBILITY_PROB_THRESH = 0.10

# Tracklet-level legibility decision (kept from your old code)
TRACKLET_MIN_LEGIBLE_FRAC = 0.05   # if < 5% of sharp frames are legible -> output -1
TRACKLET_MIN_LEGIBLE_COUNT = 3     # require at least this many legible frames

# ============================================================
# NEW: FAST ACC BOOST SETTINGS
# ============================================================

# Keep only the best frames per tracklet (quality = leg_prob * sharp_weight)
TOP_M_FRAMES_PER_TRACKLET = 12  # 5~15 is typical. Start with 12.

# Per-frame digit distribution filter (drop very uncertain frames)
DIGIT_FRAME_MAXPROB_THRESH = 0.20  # keep frame only if max prob over 0..99 >= this

# Tracklet-level decision based on aggregated probability distribution
TRACKLET_MIN_VALID_FRAMES = 3
TRACKLET_CONF_THRESH = 0.05        # raise if you want fewer wrong guesses (more -1)
TRACKLET_MARGIN_THRESH = 0.02      # (top1 - top2)/sum >= margin

# Digit-wise encoding constants (MUST match finetune_cnn.py)
# num_digits: UNK=0, 1-digit=1, 2-digit=2
NUM_UNK = 0
TENS_BLANK = 10
TENS_UNK = 11
ONES_UNK = 10

# Debug (prints a small summary; safe on HPC)
DEBUG_FIRST_K_TRACKLETS = 0  # set to 20 to debug; keep 0 for normal run

# ============================================================
# TRANSFORMS
# ============================================================
TF_EVAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# MODELS
# ============================================================
class LegibilityResNet50(nn.Module):
    """
    Matches train_legibility.py: self.net = resnet50(...); self.net.fc -> out_dim (usually 2)
    """
    def __init__(self, out_dim: int = 2):
        super().__init__()
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
        self.net = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DigitWiseHeadNet(nn.Module):
    """
    Matches finetune_cnn.py: backbone + 3 heads
      - num_digits: 3 classes (UNK=0, 1, 2)
      - tens: 12 classes (0-9, BLANK=10, UNK=11)
      - ones: 11 classes (0-9, UNK=10)
    """
    def __init__(self, name: str = "resnet50", dropout: float = 0.2):
        super().__init__()
        name = name.lower()

        if name == "resnet50":
            m = models.resnet50(weights=None)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        elif name == "efficientnet_b2":
            m = models.efficientnet_b2(weights=None)
            feat_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            self.backbone = m
        else:
            raise ValueError("MODEL_NAME must be 'resnet50' or 'efficientnet_b2'")

        self.drop = nn.Dropout(dropout)
        self.head_num = nn.Linear(feat_dim, 3)
        self.head_tens = nn.Linear(feat_dim, 12)
        self.head_ones = nn.Linear(feat_dim, 11)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.drop(feat)
        return self.head_num(feat), self.head_tens(feat), self.head_ones(feat)

# ============================================================
# CHECKPOINT LOADING (ROBUST)
# ============================================================
def _extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    """
    Best-effort extraction of a PyTorch state_dict from many checkpoint formats.

    Supports:
      - OrderedDict / Mapping of {name: tensor}
      - dict with nested Mapping under keys like:
          'state_dict', 'model_state_dict', 'model', 'net', 'weights', 'params',
          'model_state'   <--- IMPORTANT for your legibility trainer
      - dict with an nn.Module under 'model'
      - nn.Module directly
      - tuple/list where first item contains a state dict
    """
    from collections.abc import Mapping

    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict()

    if isinstance(ckpt_obj, Mapping):
        for k in ("state_dict", "model_state_dict", "model_state", "model", "net", "weights", "params"):
            if k in ckpt_obj:
                inner = ckpt_obj[k]
                if isinstance(inner, nn.Module):
                    return inner.state_dict()
                if isinstance(inner, Mapping):
                    if any(isinstance(v, torch.Tensor) for v in inner.values()):
                        return dict(inner)

        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return dict(ckpt_obj)

    if isinstance(ckpt_obj, (tuple, list)) and len(ckpt_obj) > 0:
        first = ckpt_obj[0]
        if isinstance(first, nn.Module):
            return first.state_dict()
        if isinstance(first, dict) or hasattr(first, "keys"):
            return _extract_state_dict(first)

    raise RuntimeError("Unrecognized checkpoint format: cannot find a state_dict inside the checkpoint object.")


def _normalize_legibility_keys_to_net(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize any checkpoint keys to match LegibilityResNet50 which expects 'net.<resnet keys>'.
    Guarantees:
      - all keys start with 'net.'
      - never 'net.net.'
    """
    out = {}
    for k, v in sd.items():
        nk = k
        for p in ("module.", "model.", "state_dict."):
            if nk.startswith(p):
                nk = nk[len(p):]
        if nk.startswith("net."):
            nk2 = nk
        else:
            nk2 = "net." + nk
        while nk2.startswith("net.net."):
            nk2 = "net." + nk2[len("net.net."):]
        out[nk2] = v
    return out


def _infer_legibility_out_dim(sd: Dict[str, torch.Tensor]) -> int:
    for key in ("net.fc.weight", "fc.weight"):
        if key in sd and isinstance(sd[key], torch.Tensor):
            return int(sd[key].shape[0])
    raise RuntimeError("Could not infer legibility out_dim (missing fc.weight or net.fc.weight).")


def load_legibility_model(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd_raw = _extract_state_dict(ckpt)

    out_dim = _infer_legibility_out_dim(sd_raw)
    model = LegibilityResNet50(out_dim=out_dim).to(device).eval()

    sd = _normalize_legibility_keys_to_net(sd_raw)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    model_keys = set(model.state_dict().keys())
    loaded_keys = model_keys.intersection(sd.keys())
    coverage = len(loaded_keys) / max(1, len(model_keys))
    print(f"[INFO] Legibility out_dim={out_dim} load coverage: {len(loaded_keys)}/{len(model_keys)} ({coverage:.1%})", flush=True)

    if unexpected:
        print(f"[WARN] Legibility unexpected keys (showing 10): {unexpected[:10]}", flush=True)
    if missing:
        print(f"[WARN] Legibility missing keys (showing 10): {missing[:10]}", flush=True)

    return model


def _normalize_digit_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in ("module.", "model.", "state_dict."):
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def load_digit_model(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd_raw = _extract_state_dict(ckpt)

    model = DigitWiseHeadNet(name="resnet50", dropout=0.2).to(device).eval()
    sd = _normalize_digit_keys(sd_raw)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    model_keys = set(model.state_dict().keys())
    loaded_keys = model_keys.intersection(sd.keys())
    coverage = len(loaded_keys) / max(1, len(model_keys))
    print(f"[INFO] Digit model load coverage: {len(loaded_keys)}/{len(model_keys)} ({coverage:.1%})", flush=True)

    if unexpected:
        print(f"[WARN] Digit unexpected keys (showing 10): {unexpected[:10]}", flush=True)
    if missing:
        print(f"[WARN] Digit missing keys (showing 10): {missing[:10]}", flush=True)

    return model

# ============================================================
# UTILS
# ============================================================
def safe_open_rgb(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def image_sharpness_laplacian(im: Image.Image) -> float:
    """
    Laplacian-variance sharpness on grayscale.
    """
    arr = np.array(im.convert("L"), dtype=np.float32)
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    h, w = arr.shape
    if h < 3 or w < 3:
        return 0.0
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for i in range(h - 2):
        for j in range(w - 2):
            patch = arr[i:i+3, j:j+3]
            out[i, j] = float(np.sum(patch * k))
    return float(out.var())


def list_tracklets(root: str) -> List[str]:
    tids = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    try:
        tids.sort(key=lambda x: int(x))
    except Exception:
        tids.sort()
    return tids


def list_images(tracklet_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    fs = [f for f in os.listdir(tracklet_dir) if f.lower().endswith(exts)]
    fs.sort()
    return [os.path.join(tracklet_dir, f) for f in fs]


@torch.no_grad()
def legibility_prob_legible(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Returns P(legible) for each item in batch.
    Supports:
      - 2-logit head (softmax) => uses LEGIBLE_CLASS_INDEX
      - 1-logit head (sigmoid) => assumes sigmoid output is P(legible)
    """
    logits = model(x)
    if logits.ndim != 2:
        raise RuntimeError(f"Legibility output must be [B,C], got {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, LEGIBLE_CLASS_INDEX]
    raise RuntimeError(f"Unsupported legibility channels: {logits.shape[1]}")


@torch.no_grad()
def digit_logits_to_number_probs(num_logits: torch.Tensor,
                                 tens_logits: torch.Tensor,
                                 ones_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert digit-wise head logits into P(number=k) for k in [0..99].
    Output: probs [B, 100], each row sums to <= 1 (mass can be lost to UNK states).
    Encoding:
      num_digits: UNK=0, 1-digit=1, 2-digit=2
      tens: 0..9, BLANK=10 (for 1-digit), UNK=11
      ones: 0..9, UNK=10
    """
    B = num_logits.shape[0]
    num_p = torch.softmax(num_logits, dim=1)   # [B,3]
    tens_p = torch.softmax(tens_logits, dim=1) # [B,12]
    ones_p = torch.softmax(ones_logits, dim=1) # [B,11]

    p_num1 = num_p[:, 1]  # 1-digit
    p_num2 = num_p[:, 2]  # 2-digit

    p_tens_blank = tens_p[:, TENS_BLANK]  # [B]
    p_tens_0_9 = tens_p[:, 0:10]          # [B,10]
    p_ones_0_9 = ones_p[:, 0:10]          # [B,10]

    out = torch.zeros((B, 100), device=num_logits.device, dtype=torch.float32)

    # one-digit: 0..9 => P(num=1)*P(tens=BLANK)*P(ones=d)
    one_digit = (p_num1 * p_tens_blank).unsqueeze(1) * p_ones_0_9  # [B,10]
    out[:, 0:10] = one_digit

    # two-digit: 10..99 => P(num=2)*P(tens=t)*P(ones=o)
    # outer product for each batch: [B,10,10]
    two_outer = (p_num2.unsqueeze(1) * p_tens_0_9).unsqueeze(2) * p_ones_0_9.unsqueeze(1)  # [B,10,10]
    # map (t,o) -> 10*t + o
    for t in range(10):
        out[:, t*10:(t+1)*10] = two_outer[:, t, :]

    # If you prefer to forbid leading-zero two-digit numbers (00..09 as "1-digit"),
    # uncomment the next line to zero them out:
    # out[:, 0:10] = one_digit

    return out


def _sharpness_weight(sharp: float) -> float:
    """
    Map sharpness -> [0,1] weight (simple, robust).
    - below SHARPNESS_MIN => 0
    - around 2*SHARPNESS_MIN => ~0.5
    - higher => closer to 1
    """
    if sharp <= SHARPNESS_MIN:
        return 0.0
    # smooth saturating curve
    x = (sharp - SHARPNESS_MIN) / max(1e-6, SHARPNESS_MIN)
    w = x / (1.0 + x)  # in (0,1)
    return float(w)


def _topk_select(items: List[Tuple[str, float, float]], k: int) -> List[Tuple[str, float, float]]:
    """
    items: [(path, leg_prob, sharp), ...]
    Select top-k by quality = leg_prob * sharp_weight(sharp).
    """
    if not items:
        return []
    scored = []
    for p, legp, sh in items:
        q = float(legp) * _sharpness_weight(float(sh))
        scored.append((q, p, float(legp), float(sh)))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = scored[:max(1, k)]
    return [(p, legp, sh) for (_, p, legp, sh) in keep]


def _tracklet_decision_from_scores(score_vec: np.ndarray,
                                   valid_frames: int,
                                   debug: bool = False) -> int:
    """
    score_vec: [100] aggregated unnormalized scores.
    Returns predicted number or -1.
    """
    total = float(score_vec.sum())
    if valid_frames < TRACKLET_MIN_VALID_FRAMES or total <= 0.0:
        return -1

    # top1/top2
    top_idx = int(score_vec.argmax())
    top1 = float(score_vec[top_idx])
    score_vec2 = score_vec.copy()
    score_vec2[top_idx] = -1.0
    top2 = float(score_vec2.max())

    conf = top1 / total
    margin = (top1 - top2) / total if total > 0 else 0.0

    if debug:
        print(f"[DBG] agg total={total:.4f} top={top_idx} conf={conf:.3f} margin={margin:.3f} valid_frames={valid_frames}", flush=True)

    if conf < TRACKLET_CONF_THRESH:
        return -1
    if margin < TRACKLET_MARGIN_THRESH:
        return -1

    return top_idx


def process_tracklet(tracklet_dir: str,
                     leg_model: nn.Module,
                     digit_model: nn.Module,
                     debug: bool = False) -> int:
    """
    Updated pipeline per tracklet:
      A) sharpness filter (keep sharp frames + record sharpness)
      B) legibility model per frame (keep legible frames + leg prob)
      C) tracklet gate by legibility count/fraction (same as before)
      D) select top-M frames by quality = leg_prob * sharp_weight(sharp)
      E) digit model -> per-frame P(number 0..99)
      F) aggregate probabilities (weighted) -> tracklet decision by confidence+margin
    """
    paths = list_images(tracklet_dir)
    if not paths:
        return -1

    # Stage A: sharpness
    sharp_keep: List[Tuple[str, float]] = []  # (path, sharpness)
    for p in paths:
        im = safe_open_rgb(p)
        if im is None:
            continue
        sharp = image_sharpness_laplacian(im)
        if sharp >= SHARPNESS_MIN:
            sharp_keep.append((p, float(sharp)))

    if not sharp_keep:
        return -1

    keep_paths = [p for p, _ in sharp_keep]
    sharp_map = {p: sh for p, sh in sharp_keep}

    # Stage B: legibility per frame
    legible_items: List[Tuple[str, float, float]] = []  # (path, leg_prob, sharpness)
    all_leg_probs: List[float] = []

    for i in range(0, len(keep_paths), LEG_BS):
        batch_paths = keep_paths[i:i + LEG_BS]
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
            prf = float(pr)
            all_leg_probs.append(prf)
            if prf >= LEGIBILITY_PROB_THRESH:
                legible_items.append((p, prf, float(sharp_map.get(p, SHARPNESS_MIN))))

    if debug:
        if all_leg_probs:
            mn = min(all_leg_probs)
            mx = max(all_leg_probs)
            mean = sum(all_leg_probs) / len(all_leg_probs)
            print(f"[DBG] {os.path.basename(tracklet_dir)} leg_prob min/mean/max = {mn:.3f}/{mean:.3f}/{mx:.3f} | sharp={len(keep_paths)} leg={len(legible_items)}", flush=True)
        else:
            print(f"[DBG] {os.path.basename(tracklet_dir)} no readable frames after open/transform", flush=True)

    if not legible_items:
        return -1

    # Stage C: tracklet-level legibility decision (same as before)
    leg_cnt = len(legible_items)
    keep_cnt = len(keep_paths)
    if (leg_cnt < TRACKLET_MIN_LEGIBLE_COUNT) or (leg_cnt / max(1, keep_cnt) < TRACKLET_MIN_LEGIBLE_FRAC):
        return -1

    # Stage D: select top-M frames by quality
    selected = _topk_select(legible_items, TOP_M_FRAMES_PER_TRACKLET)
    if not selected:
        return -1

    # Stage E/F: digit -> per-frame distribution -> aggregate
    agg = np.zeros((100,), dtype=np.float64)
    valid_frames = 0

    for i in range(0, len(selected), DIGIT_BS):
        batch = selected[i:i + DIGIT_BS]
        batch_paths = [p for (p, _, _) in batch]
        leg_probs = [lp for (_, lp, _) in batch]
        sharps = [sh for (_, _, sh) in batch]

        xs = []
        ok_meta = []  # (leg_prob, sharp)
        for p, lp, sh in zip(batch_paths, leg_probs, sharps):
            im = safe_open_rgb(p)
            if im is None:
                continue
            xs.append(TF_EVAL(im))
            ok_meta.append((float(lp), float(sh)))

        if not xs:
            continue

        x = torch.stack(xs, dim=0).to(DEVICE)
        num_logits, tens_logits, ones_logits = digit_model(x)

        probs_0_99 = digit_logits_to_number_probs(num_logits, tens_logits, ones_logits)  # [B,100]
        probs_np = probs_0_99.detach().cpu().numpy().astype(np.float64)

        # For each frame, optionally drop it if it's too uncertain
        for row, (lp, sh) in zip(probs_np, ok_meta):
            mprob = float(row.max())
            if mprob < DIGIT_FRAME_MAXPROB_THRESH:
                continue

            w = float(lp) * _sharpness_weight(float(sh))
            if w <= 0.0:
                continue

            agg += w * row
            valid_frames += 1

    # Stage G: decide using confidence + margin
    pred = _tracklet_decision_from_scores(agg, valid_frames=valid_frames, debug=debug)
    return int(pred)


def main():
    print("[INFO] device:", DEVICE, flush=True)
    print("[INFO] test root:", TEST_IMG_ROOT, flush=True)
    print("[INFO] leg ckpt:", LEGIBILITY_CKPT, flush=True)
    print("[INFO] digit ckpt:", DIGIT_CKPT, flush=True)

    print("[INFO] TOP_M_FRAMES_PER_TRACKLET:", TOP_M_FRAMES_PER_TRACKLET, flush=True)
    print("[INFO] DIGIT_FRAME_MAXPROB_THRESH:", DIGIT_FRAME_MAXPROB_THRESH, flush=True)
    print("[INFO] TRACKLET_MIN_VALID_FRAMES:", TRACKLET_MIN_VALID_FRAMES, flush=True)
    print("[INFO] TRACKLET_CONF_THRESH:", TRACKLET_CONF_THRESH, flush=True)
    print("[INFO] TRACKLET_MARGIN_THRESH:", TRACKLET_MARGIN_THRESH, flush=True)

    leg_model = load_legibility_model(LEGIBILITY_CKPT, DEVICE)
    digit_model = load_digit_model(DIGIT_CKPT, DEVICE)

    tids = list_tracklets(TEST_IMG_ROOT)
    out: Dict[str, int] = {}

    for idx, tid in enumerate(tqdm(tids, desc="final pipeline")):
        tdir = os.path.join(TEST_IMG_ROOT, tid)
        debug = (DEBUG_FIRST_K_TRACKLETS > 0 and idx < DEBUG_FIRST_K_TRACKLETS)
        pred = process_tracklet(tdir, leg_model, digit_model, debug=debug)
        out[str(tid)] = int(pred)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"[OK] wrote -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()