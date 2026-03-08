from pathlib import Path
import sys
import os
import argparse

ROOT = './reid/centroids-reid/'
sys.path.append(str(ROOT))  # add ROOT to PATH

import numpy as np
import torch
from tqdm import tqdm
import cv2
from PIL import Image

from config import cfg
from train_ctl_model import CTLModel
from datasets.transforms import ReidTransforms


# Based on this repo: https://github.com/mikwieczorek/centroids-reid
# Trained model from here: https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK
CONFIG_FILE = str(ROOT + '/configs/256_resnet50.yml')
MODEL_FILE = str(ROOT + '/models/resnet50-19c8e357.pth')

# dict used to get model config and weights using model version
ver_to_specs = {}
ver_to_specs["res50_market"] = (
    ROOT + '/configs/256_resnet50.yml',
    ROOT + '/models/market1501_resnet50_256_128_epoch_120.ckpt'
)
ver_to_specs["res50_duke"] = (
    ROOT + '/configs/256_resnet50.yml',
    ROOT + '/models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt'
)


def get_specs_from_version(model_version):
    conf, weights = ver_to_specs[model_version]
    conf, weights = str(conf), str(weights)
    return conf, weights


def batched(items, batch_size):
    """Yield successive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def _load_and_transform_image(full_img_path, val_transforms):
    """
    Load one image and apply the same transform behavior as before.
    Returns a tensor or None if the image cannot be read.
    """
    img = cv2.imread(full_img_path)
    if img is None:
        return None

    # Keep behavior consistent with the original script:
    # original code used Image.fromarray(cv2.imread(...)) directly.
    input_img = Image.fromarray(img)
    input_tensor = val_transforms(input_img)
    return input_tensor


def generate_features(input_folder, output_folder, model_version='res50_market', batch_size=64):
    # load model
    config_file, model_file = get_specs_from_version(model_version)
    cfg.merge_from_file(config_file)
    opts = [
        "MODEL.PRETRAIN_PATH", model_file,
        "MODEL.PRETRAINED", True,
        "TEST.ONLY_TEST", True,
        "MODEL.RESUME_TRAINING", False
    ]
    cfg.merge_from_list(opts)

    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)
    model.to(device)
    model.eval()

    if use_cuda:
        print("using GPU")
    else:
        print("using CPU")

    tracks = sorted(os.listdir(input_folder))
    transforms_base = ReidTransforms(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False)

    for track in tqdm(tracks):
        track_path = os.path.join(input_folder, track)
        if not os.path.isdir(track_path):
            continue

        images = sorted([x for x in os.listdir(track_path) if _is_image_file(x)])
        output_file = os.path.join(output_folder, f"{track}_features.npy")

        # Collect per-batch arrays here, then concatenate once at the end.
        feature_chunks = []

        with torch.no_grad():
            for image_batch in batched(images, batch_size):
                batch_tensors = []

                for img_name in image_batch:
                    full_img_path = os.path.join(track_path, img_name)
                    input_tensor = _load_and_transform_image(full_img_path, val_transforms)
                    if input_tensor is not None:
                        batch_tensors.append(input_tensor)

                if len(batch_tensors) == 0:
                    continue

                batch_tensor = torch.stack(batch_tensors, dim=0).to(device, non_blocking=use_cuda)

                _, global_feat = model.backbone(batch_tensor)
                global_feat = model.bn(global_feat)

                feature_chunks.append(global_feat.detach().cpu().numpy())

        if len(feature_chunks) == 0:
            # Preserve empty output case in a safe way.
            np_feat = np.empty((0,), dtype=np.float32)
        else:
            np_feat = np.concatenate(feature_chunks, axis=0)

        with open(output_file, 'wb') as f:
            np.save(f, np_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for ReID feature extraction")
    parser.add_argument('--model_version', type=str, default='res50_market',
                        choices=list(ver_to_specs.keys()),
                        help="Which pretrained ReID model spec to use")
    args = parser.parse_args()

    # create if does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    generate_features(
        args.tracklets_folder,
        args.output_folder,
        model_version=args.model_version,
        batch_size=args.batch_size
    )