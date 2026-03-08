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
OUT_JSON = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/test_pred_final.txt"

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
LEGIBILITY_PROB_THRESH = 0.50

# Tracklet-level legibility decision
TRACKLET_MIN_LEGIBLE_FRAC = 0.25   # if < 25% of sharp frames are legible -> output -1
TRACKLET_MIN_LEGIBLE_COUNT = 3     # require at least this many legible frames

# Digit stage
DIGIT_CONF_THRESH = 0.35

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

    # 1) If it's an nn.Module, just return its state_dict
    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict()

    # 2) If it's a Mapping, try common nesting patterns
    if isinstance(ckpt_obj, Mapping):
        # NOTE: include 'model_state' because train_legibility.py saves that key
        for k in ("state_dict", "model_state_dict", "model_state", "model", "net", "weights", "params"):
            if k in ckpt_obj:
                inner = ckpt_obj[k]
                if isinstance(inner, nn.Module):
                    return inner.state_dict()
                if isinstance(inner, Mapping):
                    if any(isinstance(v, torch.Tensor) for v in inner.values()):
                        return dict(inner)

        # might already be a flat state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return dict(ckpt_obj)

    # 3) Some checkpoints are saved as a tuple/list like (state_dict, ...)
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
        # strip wrappers, but do NOT strip 'net.' here (handled below)
        for p in ("module.", "model.", "state_dict."):
            if nk.startswith(p):
                nk = nk[len(p):]
        # if it already starts with net., keep; else prefix
        if nk.startswith("net."):
            nk2 = nk
        else:
            nk2 = "net." + nk
        # guard against double prefix
        while nk2.startswith("net.net."):
            nk2 = "net." + nk2[len("net.net."):]
        out[nk2] = v
    return out


def _infer_legibility_out_dim(sd: Dict[str, torch.Tensor]) -> int:
    # works before or after normalization
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

    # Coverage sanity check (prevents silent "loads nothing")
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
    """
    DigitWiseHeadNet expects keys like 'backbone.*', 'head_num.*', etc.
    Remove common wrappers.
    """
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

    # defaults match finetune_cnn.py
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
    (Kept simple; OK for your current pipeline.)
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


def decode_frame(num_logits: torch.Tensor,
                 tens_logits: torch.Tensor,
                 ones_logits: torch.Tensor,
                 conf_thresh: Optional[float] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert per-frame head logits to jersey number prediction.
    Encoding (matches finetune_cnn.py):
      - num_digits: UNK=0, 1-digit=1, 2-digit=2
      - tens: 0..9, BLANK=10 (for 1-digit), UNK=11
      - ones: 0..9, UNK=10
    """
    num_prob = torch.softmax(num_logits, dim=1)
    tens_prob = torch.softmax(tens_logits, dim=1)
    ones_prob = torch.softmax(ones_logits, dim=1)

    num_pred = torch.argmax(num_prob, dim=1)
    tens_pred = torch.argmax(tens_prob, dim=1)
    ones_pred = torch.argmax(ones_prob, dim=1)

    conf = (
        num_prob.gather(1, num_pred[:, None]).squeeze(1) *
        tens_prob.gather(1, tens_pred[:, None]).squeeze(1) *
        ones_prob.gather(1, ones_pred[:, None]).squeeze(1)
    )

    if conf_thresh is not None:
        low = conf < conf_thresh
        if low.any():
            num_pred = num_pred.clone()
            tens_pred = tens_pred.clone()
            ones_pred = ones_pred.clone()
            num_pred[low] = NUM_UNK
            tens_pred[low] = TENS_UNK
            ones_pred[low] = ONES_UNK

    B = num_pred.shape[0]
    jersey = torch.full((B,), -1, dtype=torch.long, device=num_pred.device)

    unk = (num_pred == NUM_UNK) | (tens_pred == TENS_UNK) | (ones_pred == ONES_UNK)

    one_ok = (num_pred == 1) & (tens_pred == TENS_BLANK) & (ones_pred <= 9)
    two_ok = (num_pred == 2) & (tens_pred <= 9) & (ones_pred <= 9)

    jersey[one_ok] = ones_pred[one_ok]
    jersey[two_ok] = tens_pred[two_ok] * 10 + ones_pred[two_ok]
    jersey[unk] = -1

    return jersey, conf


def aggregate_scores(frame_preds: List[int], frame_confs: List[float]) -> int:
    """
    Weighted vote over frames. Ignores -1 unless it's the only thing.
    """
    if not frame_preds:
        return -1
    scores = defaultdict(float)
    for p, c in zip(frame_preds, frame_confs):
        scores[int(p)] += float(c)

    if len(scores) == 1:
        return int(next(iter(scores.keys())))

    if -1 in scores:
        del scores[-1]

    if not scores:
        return -1

    return int(max(scores.items(), key=lambda kv: kv[1])[0])


def process_tracklet(tracklet_dir: str,
                     leg_model: nn.Module,
                     digit_model: nn.Module,
                     debug: bool = False) -> int:
    """
    Pipeline per tracklet:
      A) sharpness filter
      B) legibility model per frame
      C) tracklet gate by legibility count/fraction
      D) digit model on legible frames
      E) aggregate
    """
    paths = list_images(tracklet_dir)
    if not paths:
        return -1

    # Stage A: sharpness
    keep_paths: List[str] = []
    for p in paths:
        im = safe_open_rgb(p)
        if im is None:
            continue
        sharp = image_sharpness_laplacian(im)
        if sharp >= SHARPNESS_MIN:
            keep_paths.append(p)

    if not keep_paths:
        return -1

    # Stage B: legibility per frame
    legible_paths: List[str] = []
    legible_probs: List[float] = []
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
                legible_paths.append(p)
                legible_probs.append(prf)

    if debug:
        if all_leg_probs:
            mn = min(all_leg_probs)
            mx = max(all_leg_probs)
            mean = sum(all_leg_probs) / len(all_leg_probs)
            print(f"[DBG] {os.path.basename(tracklet_dir)} leg_prob min/mean/max = {mn:.3f}/{mean:.3f}/{mx:.3f} | sharp={len(keep_paths)} leg={len(legible_paths)}", flush=True)
        else:
            print(f"[DBG] {os.path.basename(tracklet_dir)} no readable frames after open/transform", flush=True)

    if not legible_paths:
        return -1

    # Stage C: tracklet-level decision
    leg_cnt = len(legible_paths)
    keep_cnt = len(keep_paths)
    if (leg_cnt < TRACKLET_MIN_LEGIBLE_COUNT) or (leg_cnt / max(1, keep_cnt) < TRACKLET_MIN_LEGIBLE_FRAC):
        return -1

    leg_prob_map = {p: pr for p, pr in zip(legible_paths, legible_probs)}

    # Stage D: digit model
    frame_preds: List[int] = []
    frame_confs: List[float] = []

    for i in range(0, len(legible_paths), DIGIT_BS):
        batch_paths = legible_paths[i:i + DIGIT_BS]
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
        num_logits, tens_logits, ones_logits = digit_model(x)

        jersey_pred, conf = decode_frame(num_logits, tens_logits, ones_logits, conf_thresh=DIGIT_CONF_THRESH)
        jersey_pred = jersey_pred.detach().cpu().numpy().tolist()
        conf = conf.detach().cpu().numpy().tolist()

        for pth, j, c in zip(ok_paths, jersey_pred, conf):
            c2 = float(c) * float(leg_prob_map.get(pth, 1.0))
            frame_preds.append(int(j))
            frame_confs.append(float(c2))

    # Stage E: aggregate
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