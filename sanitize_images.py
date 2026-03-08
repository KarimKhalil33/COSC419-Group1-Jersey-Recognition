#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from PIL import Image, ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = False  # keep strict; we want to catch bad files

def is_image_ok_pil(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # verifies header + integrity
        return True
    except Exception:
        return False

def is_image_ok_cv2(path: Path) -> bool:
    try:
        img = cv2.imread(str(path))
        return img is not None
    except Exception:
        return False

def main():
    roots = [
        Path("data/SoccerNet/jersey-2023/train/images")
    ]
    quarantine_root = Path("data/SoccerNet/jersey-2023/_quarantine_bad_images")
    quarantine_root.mkdir(parents=True, exist_ok=True)

    bad_list_path = quarantine_root / "bad_images.txt"
    moved = 0
    checked = 0

    with open(bad_list_path, "w") as out:
        for root in roots:
            if not root.exists():
                print(f"[WARN] Missing root: {root}")
                continue

            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                # (optional) only check common image extensions
                if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    continue

                checked += 1
                ok = is_image_ok_pil(p) and is_image_ok_cv2(p)
                if ok:
                    continue

                # move to quarantine, preserving relative path under split
                rel = p.relative_to(root)
                dst = quarantine_root / root.parent.name / root.name / rel  # keeps train/images/... structure
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dst))
                out.write(str(p) + "\n")
                moved += 1

                if moved % 100 == 0:
                    print(f"[INFO] moved {moved} bad images so far...")

    print(f"[DONE] checked={checked}, moved_bad={moved}")
    print(f"[DONE] bad list at: {bad_list_path}")

if __name__ == "__main__":
    main()