#!/usr/bin/env python3
"""
train_legibility_resnet50.py

Legibility model: classify whether a crop is illegible (-1) or legible (0).

- Reads:
  /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline/splits_legibility_balanced
    - train_legibility.csv
    - val_legibility.csv

- Loads ResNet50 ImageNet weights LOCALLY (no internet):
  /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline/torch_weights/resnet50_imagenet.pth

- Saves:
  /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline/runs_legibility/
    - best.pt
    - last.pt
    - loss_curve.png
    - val_accuracy_curve.png
    - val_precision_recall_f1.png

Label mapping for training:
  raw -1  -> class 0 (illegible)
  raw  0  -> class 1 (legible)
"""

import os
import csv
import time
import random
from typing import Any, Tuple, List, Dict

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

# === plotting (non-interactive; saves PNGs) ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# HARD-CODED PATHS (MATCH YOUR SETUP)
# ============================================================
BASE_DIR = "/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline"

TRAIN_CSV = os.path.join(BASE_DIR, "splits_legibility_balanced", "train_legibility_balanced.csv")
VAL_CSV   = os.path.join(BASE_DIR, "splits_legibility", "val_legibility.csv")

WEIGHT_PATH = os.path.join(BASE_DIR, "torch_weights", "resnet50_imagenet.pth")

RUN_DIR = os.path.join(BASE_DIR, "runs_legibility")
BEST_CKPT = os.path.join(RUN_DIR, "best.pt")
LAST_CKPT = os.path.join(RUN_DIR, "last.pt")

# (Optional but recommended on Sockeye) avoid matplotlib cache warnings if you ever import it
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
# ============================================================


# ============================================================
# TRAINING HYPERPARAMS (EDIT HERE IF YOU WANT)
# ============================================================
SEED = 1337
IMGSZ = 224
EPOCHS = 12
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
WORKERS = 4
GRAD_CLIP = 5.0
MAX_RETRY = 30
USE_CLASS_WEIGHTS = True   # helps if -1 vs 0 is imbalanced
USE_AMP = True             # mixed precision on GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================


PATH_COLS  = ["image_path", "path", "img_path", "filepath", "file_path", "image"]
LABEL_COLS = ["label", "jersey", "jersey_number", "jerseynumber", "number", "target", "y"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def detect_columns(fieldnames: List[str]) -> Tuple[str, str]:
    lower = [f.lower() for f in fieldnames]
    path_col = next((fieldnames[lower.index(c)] for c in PATH_COLS if c in lower), None)
    label_col = next((fieldnames[lower.index(c)] for c in LABEL_COLS if c in lower), None)
    if path_col is None or label_col is None:
        raise RuntimeError(f"Cannot detect columns. Found: {fieldnames}")
    return path_col, label_col


def parse_label(x: Any) -> int:
    return int(float(str(x).strip()))


def map_legibility_label(y_raw: torch.Tensor) -> torch.Tensor:
    """
    Input y_raw: -1 or 0
    Output class index:
      -1 -> 0 (illegible)
       0 -> 1 (legible)
    """
    return torch.where(y_raw == -1, torch.zeros_like(y_raw), torch.ones_like(y_raw)).long()


class SafeLegibilityCSV(Dataset):
    def __init__(self, csv_path: str, transform=None, max_retry: int = 20):
        self.csv_path = csv_path
        self.transform = transform
        self.max_retry = max_retry

        self.samples: List[Tuple[str, int]] = []
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                raise RuntimeError(f"No header in {csv_path}")
            path_col, label_col = detect_columns(r.fieldnames)
            for row in r:
                p = str(row[path_col]).strip()
                y = parse_label(row[label_col])
                # these CSVs should already be only -1/0, but keep it robust
                if y not in (-1, 0):
                    continue
                self.samples.append((p, y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        for _ in range(self.max_retry):
            path, y = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, torch.tensor(y, dtype=torch.long)
            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
                idx = random.randint(0, len(self.samples) - 1)

        raise RuntimeError(f"Too many failed image loads (>{self.max_retry}). Last path: {self.samples[idx][0]}")


class LegibilityResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet50(weights=None)  # NO INTERNET. We'll load weights manually.
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, 2)  # 2-class

    def forward(self, x):
        return self.net(x)


def load_resnet50_imagenet_backbone(model: LegibilityResNet50, weight_path: str) -> None:
    """
    Loads local ImageNet weights into model.net, skipping fc.* if shapes mismatch.
    Works with common torchvision weight dict formats.
    """
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Local weight file not found: {weight_path}")

    ckpt = torch.load(weight_path, map_location="cpu")

    # torchvision sometimes stores raw state_dict, sometimes wrapped
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("Unknown checkpoint format for weight file.")

    # Remove possible prefixes (e.g., 'module.' or 'net.')
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("net."):
            nk = nk[len("net."):]
        cleaned[nk] = v

    # Drop fc weights if present (ImageNet has 1000 classes)
    cleaned.pop("fc.weight", None)
    cleaned.pop("fc.bias", None)

    missing, unexpected = model.net.load_state_dict(cleaned, strict=False)

    # Expect missing to include fc.weight/fc.bias (because we replaced it with 2-class)
    print("[OK] Loaded local ResNet50 ImageNet weights.")
    print(f"     weight_path: {weight_path}")
    if unexpected:
        print(f"[WARN] Unexpected keys (showing up to 10): {unexpected[:10]}")
    # missing should at least include fc.*
    if missing:
        print(f"     Missing keys (expected fc.*). Count={len(missing)}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    tp = fp = tn = fn = 0  # positive = legible (class 1)

    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        y = map_legibility_label(y_raw.to(device, non_blocking=True))

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    return {"acc": acc, "precision_legible": prec, "recall_legible": rec, "f1_legible": f1}


def save_curves(run_dir: str,
                train_losses: List[float],
                val_accs: List[float],
                val_precs: List[float],
                val_recs: List[float],
                val_f1s: List[float]) -> None:
    epochs = list(range(1, len(train_losses) + 1))

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    # Validation accuracy
    plt.figure()
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_accuracy_curve.png"))
    plt.close()

    # Precision / Recall / F1
    plt.figure()
    plt.plot(epochs, val_precs, label="Precision (legible)")
    plt.plot(epochs, val_recs, label="Recall (legible)")
    plt.plot(epochs, val_f1s, label="F1 (legible)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Precision / Recall / F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_precision_recall_f1.png"))
    plt.close()

    print("[SAVE] Curves saved to", run_dir)


def main():
    seed_everything(SEED)
    ensure_dir(RUN_DIR)

    print("[INFO] Using device:", DEVICE)
    print("[INFO] Train CSV:", TRAIN_CSV)
    print("[INFO] Val CSV  :", VAL_CSV)

    # Transforms: keep simple + realistic
    train_tf = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)], p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = SafeLegibilityCSV(TRAIN_CSV, transform=train_tf, max_retry=MAX_RETRY)
    val_ds = SafeLegibilityCSV(VAL_CSV, transform=val_tf, max_retry=MAX_RETRY)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True
    )

    # Build model + load local weights
    model = LegibilityResNet50().to(DEVICE)
    load_resnet50_imagenet_backbone(model, WEIGHT_PATH)

    # Class weights (optional but usually helps)
    class_weights = None
    if USE_CLASS_WEIGHTS:
        ys = [y for _, y in train_ds.samples]  # raw labels: -1 or 0
        n_illeg = sum(1 for y in ys if y == -1)   # class 0
        n_leg = len(ys) - n_illeg                 # class 1
        # inverse frequency (balanced)
        w0 = len(ys) / max(1, 2 * n_illeg)
        w1 = len(ys) / max(1, 2 * n_leg)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)
        print(f"[INFO] Class weights: illegible(w0)={w0:.4f} legible(w1)={w1:.4f} | n_illeg={n_illeg} n_leg={n_leg}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.startswith("cuda")))

    # === metric buffers for plotting ===
    train_losses: List[float] = []
    val_accs: List[float] = []
    val_precs: List[float] = []
    val_recs: List[float] = []
    val_f1s: List[float] = []

    best_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        running_loss = 0.0
        seen = 0

        for x, y_raw in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = map_legibility_label(y_raw.to(DEVICE, non_blocking=True))

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            running_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=running_loss / max(1, seen), lr=optimizer.param_groups[0]["lr"])

        scheduler.step()

        metrics = evaluate(model, val_loader, DEVICE)
        epoch_train_loss = running_loss / max(1, seen)

        # === record metrics for plots ===
        train_losses.append(epoch_train_loss)
        val_accs.append(metrics["acc"])
        val_precs.append(metrics["precision_legible"])
        val_recs.append(metrics["recall_legible"])
        val_f1s.append(metrics["f1_legible"])

        print(f"[VAL] epoch={epoch} "
              f"acc={metrics['acc']:.4f} "
              f"prec_leg={metrics['precision_legible']:.4f} "
              f"rec_leg={metrics['recall_legible']:.4f} "
              f"f1_leg={metrics['f1_legible']:.4f}")

        # Save best
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            torch.save({
                "epoch": epoch,
                "best_acc": best_acc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": {
                    "IMGSZ": IMGSZ, "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY, "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS
                },
                "paths": {
                    "TRAIN_CSV": TRAIN_CSV, "VAL_CSV": VAL_CSV, "WEIGHT_PATH": WEIGHT_PATH
                }
            }, BEST_CKPT)
            print(f"[SAVE] best -> {BEST_CKPT} (acc={best_acc:.4f})")

    # === save plots at end ===
    save_curves(RUN_DIR, train_losses, val_accs, val_precs, val_recs, val_f1s)

    # Save last
    torch.save({
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
        "cfg": {
            "IMGSZ": IMGSZ, "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY, "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS
        },
        "paths": {
            "TRAIN_CSV": TRAIN_CSV, "VAL_CSV": VAL_CSV, "WEIGHT_PATH": WEIGHT_PATH
        }
    }, LAST_CKPT)
    print(f"[SAVE] last -> {LAST_CKPT}")


if __name__ == "__main__":
    main()
