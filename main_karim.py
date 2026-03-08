print("[BOOT] main_karim.py starting", flush=True)
import argparse
import os
import legibility_classifier as lc
import numpy as np
import cv2
import json
import shutil
import helpers
from tqdm import tqdm
import configuration as config
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Karim Khalil – COSC 419B  |  Team 1
#
# KEY MODIFICATIONS over main_original.py:
#
#   1. Multi-metric frame quality scoring
#         Combines three independent image-quality signals into one score:
#           a) Sharpness  – variance of Laplacian (high-frequency energy)
#           b) Contrast   – RMS contrast (standard deviation of pixel intensities)
#           c) Edge density – fraction of Canny edge pixels (structural detail)
#         Each signal is min-max normalised within the tracklet before being
#         blended with fixed weights [0.50, 0.30, 0.20].
#         Unlike Yijun's approach (pure Laplacian), the composite score is more
#         robust to cases where an image is sharp but low-contrast (e.g. white
#         jersey on bright background).
#
#   2. Temporally-diverse window sampling (instead of global top-K)
#         The tracklet frames are divided into W equal-width time windows.
#         The single highest-scoring frame is selected from each window,
#         guaranteeing temporal diversity.  This avoids picking K near-identical
#         consecutive frames that happen to be sharp, and ensures coverage across
#         the whole player trajectory.
#         The number of windows W is capped between MIN_WINDOWS and max_windows
#         (default 10) and adapts to tracklet length.
#
#   3. CLAHE preprocessing on crops before STR inference
#         Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied
#         per colour channel to every crop before the STR model sees it.  This
#         normalises local lighting variation – a common issue with cropped
#         soccer player images under mixed stadium lighting – and effectively
#         makes digits more visible to the recognition model without changing
#         geometry or introducing blur.
#
#   4. Adaptive legibility threshold (per-tracklet median)
#         Instead of the fixed global threshold (0.5) used in the original
#         pipeline, each tracklet uses the median of its own legibility scores
#         as the threshold when that median is higher than 0.5.  This adapts to
#         easy tracklets (high median → keep only clearly visible frames) and
#         hard ones (low median → fall back to global 0.5) automatically.
# ─────────────────────────────────────────────────────────────────────────────


# ── Constants ─────────────────────────────────────────────────────────────────
QUALITY_W_SHARPNESS   = 0.50   # weight for Laplacian-variance component
QUALITY_W_CONTRAST    = 0.30   # weight for RMS-contrast component
QUALITY_W_EDGE        = 0.20   # weight for Canny edge-density component
MIN_WINDOWS           = 2      # minimum number of time windows per tracklet
CLAHE_CLIP            = 2.0    # CLAHE clip limit (higher → more aggressive)
CLAHE_GRID            = 8      # CLAHE tile grid size (each axis)
ADAPTIVE_LEG_FLOOR    = 0.50   # never go below the original threshold


# ── Image-quality helpers ─────────────────────────────────────────────────────

def _sharpness(img_bgr: np.ndarray) -> float:
    """Variance of Laplacian on grayscale – higher is sharper."""
    if img_bgr is None:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _rms_contrast(img_bgr: np.ndarray) -> float:
    """Root-mean-square contrast (std of pixel intensities on grayscale)."""
    if img_bgr is None:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(np.std(gray.astype(np.float32)))


def _edge_density(img_bgr: np.ndarray) -> float:
    """Fraction of pixels flagged as edges by Canny – structural detail proxy."""
    if img_bgr is None:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return float(np.count_nonzero(edges)) / max(edges.size, 1)


def _multi_metric_score(img_bgr: np.ndarray) -> tuple:
    """Return (sharpness, contrast, edge_density) raw values for one image."""
    return _sharpness(img_bgr), _rms_contrast(img_bgr), _edge_density(img_bgr)


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns uniform array if all equal."""
    rng = arr.max() - arr.min()
    if rng < 1e-9:
        return np.ones_like(arr, dtype=float) / len(arr)
    return (arr - arr.min()) / rng


def _composite_scores(raw_metrics: list) -> np.ndarray:
    """
    Convert a list of (sharpness, contrast, edge_density) tuples into a
    single composite quality score per frame using weighted blending after
    per-metric min-max normalisation within the set.
    """
    arr = np.array(raw_metrics, dtype=float)  # shape (N, 3)
    s_norm = _normalise(arr[:, 0])
    c_norm = _normalise(arr[:, 1])
    e_norm = _normalise(arr[:, 2])
    return (QUALITY_W_SHARPNESS * s_norm
            + QUALITY_W_CONTRAST  * c_norm
            + QUALITY_W_EDGE      * e_norm)


# ── Temporally-diverse window sampling ────────────────────────────────────────

def select_diverse_topk_per_tracklet(
    crops_imgs_dir: str,
    max_windows: int = 10,
    min_keep: int = 2,
    verbose: bool = True,
) -> None:
    """
    In-place pruning of crop images using temporally-diverse window sampling.

    Algorithm
    ---------
    1. Sort frames within each tracklet by their filename (= temporal order).
    2. Divide the sorted list into W equal windows  (W = min(max_windows, N)).
    3. Score every frame with the multi-metric composite quality score.
    4. Keep exactly one frame per window – the one with the highest composite
       score.  Delete all others.

    This guarantees both quality (best frame per window) and diversity (one
    frame from each part of the tracklet timeline), addressing the pitfall of
    picking K consecutive near-duplicate sharp frames.
    """
    if max_windows <= 0:
        return
    if not os.path.isdir(crops_imgs_dir):
        if verbose:
            print(f"[Karim/WindSample] crops dir not found: {crops_imgs_dir}")
        return

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(crops_imgs_dir) if f.lower().endswith(exts)]
    if not files:
        return

    # Group by tracklet id (prefix before first '_')
    by_track: dict[str, list] = {}
    for fn in files:
        track = fn.split('_')[0]
        by_track.setdefault(track, []).append(fn)

    removed = kept = 0
    for track, fns in by_track.items():
        fns_sorted = sorted(fns)   # lexicographic ≈ temporal when zero-padded
        n = len(fns_sorted)
        w = max(MIN_WINDOWS, min(max_windows, n))

        if n <= w:
            kept += n
            continue  # nothing to prune

        # Multi-metric scoring
        raw_metrics = []
        for fn in fns_sorted:
            img = cv2.imread(os.path.join(crops_imgs_dir, fn))
            raw_metrics.append(_multi_metric_score(img))
        scores = _composite_scores(raw_metrics)

        # Window-based best-frame selection
        windows = np.array_split(np.arange(n), w)
        keep_indices = set()
        for win in windows:
            if len(win) == 0:
                continue
            best_in_win = win[np.argmax(scores[win])]
            keep_indices.add(best_in_win)

        # Enforce minimum keep count (add highest-scoring frames if needed)
        while len(keep_indices) < min(min_keep, n):
            remaining = [i for i in range(n) if i not in keep_indices]
            if not remaining:
                break
            best_remaining = remaining[np.argmax(scores[remaining])]
            keep_indices.add(best_remaining)

        # Delete everything not in keep_indices
        for i, fn in enumerate(fns_sorted):
            if i not in keep_indices:
                try:
                    os.remove(os.path.join(crops_imgs_dir, fn))
                    removed += 1
                except Exception:
                    pass
            else:
                kept += 1

    if verbose:
        print(
            f"[Karim/WindSample] kept {kept} crops, removed {removed} crops "
            f"(max_windows={max_windows}) from {crops_imgs_dir}"
        )


# ── CLAHE preprocessing ───────────────────────────────────────────────────────

def apply_clahe_to_crops(crops_imgs_dir: str, verbose: bool = True) -> None:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in-place to
    every image inside crops_imgs_dir.

    CLAHE operates channel-by-channel on the LAB colour space so that only the
    Luminance channel is equalised, leaving colour information intact.  This
    corrects local lighting variation – common in stadium crops – without
    introducing colour artefacts.
    """
    if not os.path.isdir(crops_imgs_dir):
        if verbose:
            print(f"[Karim/CLAHE] crops dir not found: {crops_imgs_dir}")
        return

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(crops_imgs_dir) if f.lower().endswith(exts)]
    if not files:
        return

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                             tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    processed = 0
    for fn in files:
        path = os.path.join(crops_imgs_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        # Convert BGR → LAB, equalise L, convert back
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        cv2.imwrite(path, img_eq)
        processed += 1

    if verbose:
        print(f"[Karim/CLAHE] applied CLAHE to {processed} crops in {crops_imgs_dir}")


# ── Adaptive legibility classification ────────────────────────────────────────

def get_soccer_net_legibility_results_adaptive(args, exclude_balls: bool = True):
    """
    Legibility classification with a per-tracklet adaptive threshold.

    For each tracklet the raw legibility scores are collected.  The threshold
    used to decide legible/illegible is:
        max(ADAPTIVE_LEG_FLOOR, median(scores))

    This means:
      - Easy tracklets (high median) → stricter threshold → only clearly
        visible frames are forwarded, reducing noise in the STR stage.
      - Hard tracklets (low median) → falls back to ADAPTIVE_LEG_FLOOR (0.5),
        matching the original behaviour so we don't throw away all frames.
    """
    root_dir  = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = os.listdir(path_to_images)

    # Load Gaussian-filtered frame lists (same as original)
    path_to_filter = os.path.join(
        config.dataset['SoccerNet']['working_dir'],
        config.dataset['SoccerNet'][args.part]['gauss_filtered'],
    )
    with open(path_to_filter, 'r') as f:
        filtered = json.load(f)

    # Optionally exclude soccer balls
    if exclude_balls:
        soccer_ball_list = os.path.join(
            config.dataset['SoccerNet']['working_dir'],
            config.dataset['SoccerNet'][args.part]['soccer_ball_list'],
        )
        with open(soccer_ball_list, 'r') as f:
            ball_list = json.load(f)['ball_tracks']
        tracklets = [t for t in tracklets if t not in ball_list]

    legible_tracklets   = {}
    illegible_tracklets = []

    for directory in tqdm(tracklets, desc="[Karim/AdaptiveLeg]"):
        track_dir          = os.path.join(path_to_images, directory)
        images             = filtered[directory]
        images_full_path   = [os.path.join(track_dir, x) for x in images]

        # Raw scores (threshold=-1 → return all probabilities)
        raw_scores = lc.run(
            images_full_path,
            config.dataset['SoccerNet']['legibility_model'],
            threshold=-1,
            arch=config.dataset['SoccerNet']['legibility_model_arch'],
        )
        raw_scores = np.asarray(raw_scores, dtype=float)

        # Adaptive per-tracklet threshold
        if len(raw_scores) > 0:
            adaptive_threshold = max(ADAPTIVE_LEG_FLOOR, float(np.median(raw_scores)))
        else:
            adaptive_threshold = ADAPTIVE_LEG_FLOOR

        legible_indices = np.where(raw_scores >= adaptive_threshold)[0]

        if len(legible_indices) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_tracklets[directory] = [images_full_path[i] for i in legible_indices]

    # Persist results (same paths as original so downstream stages work)
    wd   = config.dataset['SoccerNet']['working_dir']
    part = config.dataset['SoccerNet'][args.part]

    leg_path   = os.path.join(wd, part['legible_result'])
    illeg_path = os.path.join(wd, part['illegible_result'])

    with open(leg_path, 'w') as f:
        json.dump(legible_tracklets, f, indent=4)
    with open(illeg_path, 'w') as f:
        json.dump({'illegible': illegible_tracklets}, f, indent=4)

    return legible_tracklets, illegible_tracklets


# ── Utility: clean previous run artifacts ─────────────────────────────────────

def clean_soccer_net_artifacts(part: str, clean_crops: bool = True, verbose: bool = True) -> None:
    """Remove outputs from previous runs so a re-run reflects full pipeline time."""
    try:
        wd = config.dataset['SoccerNet']['working_dir']
        d  = config.dataset['SoccerNet'][part]
        paths_files = [
            os.path.join(wd, d.get('illegible', '')),
            os.path.join(wd, d.get('full_legible', '')),
            os.path.join(wd, d.get('pose_input_json', '')),
            os.path.join(wd, d.get('pose_output_json', '')),
            os.path.join(wd, d.get('jersey_id_result', '')),
            os.path.join(wd, d.get('final_result', '')),
        ]
        for fp in paths_files:
            if fp and os.path.isfile(fp):
                try:
                    os.remove(fp)
                    if verbose:
                        print(f"[Clean] removed file: {fp}")
                except Exception:
                    pass
        for k in ['sim_filtered', 'gauss_filtered']:
            if k in d:
                fp = os.path.join(wd, d[k])
                if fp and os.path.isfile(fp):
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
        if clean_crops and 'crops_folder' in d:
            crops_dir = os.path.join(wd, d['crops_folder'])
            if os.path.isdir(crops_dir):
                try:
                    shutil.rmtree(crops_dir)
                    if verbose:
                        print(f"[Clean] removed dir: {crops_dir}")
                except Exception:
                    pass
    except Exception as e:
        if verbose:
            print(f"[Clean] warning: {e}")


# ── Shared helpers (unchanged from original) ──────────────────────────────────

def get_soccer_net_raw_legibility_results(args, use_filtered=True, filter='gauss', exclude_balls=True):
    root_dir      = config.dataset['SoccerNet']['root_dir']
    image_dir     = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets     = os.listdir(path_to_images)
    results_dict  = {x: [] for x in tracklets}

    if use_filtered:
        key = 'sim_filtered' if filter == 'sim' else 'gauss_filtered'
        fpath = os.path.join(config.dataset['SoccerNet']['working_dir'],
                             config.dataset['SoccerNet'][args.part][key])
        with open(fpath, 'r') as f:
            filtered = json.load(f)

    if exclude_balls:
        ball_list_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(ball_list_path, 'r') as f:
            ball_list = json.load(f)['ball_tracks']
        tracklets = [t for t in tracklets if t not in ball_list]

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        images    = filtered[directory] if use_filtered else os.listdir(track_dir)
        images_fp = [os.path.join(track_dir, x) for x in images]
        results_dict[directory] = lc.run(
            images_fp,
            config.dataset['SoccerNet']['legibility_model'],
            threshold=-1,
            arch=config.dataset['SoccerNet']['legibility_model_arch'],
        )

    out_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                            config.dataset['SoccerNet'][args.part]['raw_legible_result'])
    with open(out_path, 'w') as f:
        json.dump(results_dict, f)
    return results_dict


def generate_json_for_pose_estimator(args, legible=None):
    all_files = []
    if legible is not None:
        for key in legible.keys():
            all_files.extend(os.path.join(os.getcwd(), e) for e in legible[key])
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset['SoccerNet']['root_dir'])
        image_dir = config.dataset['SoccerNet'][args.part]['images']
        path_to_images = os.path.join(root_dir, image_dir)
        for tr in os.listdir(path_to_images):
            track_dir = os.path.join(path_to_images, tr)
            for img in os.listdir(track_dir):
                all_files.append(os.path.join(track_dir, img))
    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet'][args.part]['pose_input_json'])
    helpers.generate_json(all_files, output_json)


def consolidated_results(image_dir, dict, illegible_path, soccer_ball_list=None):
    if soccer_ball_list is not None:
        with open(soccer_ball_list, 'r') as sf:
            for entry in json.load(sf)['ball_tracks']:
                dict[str(entry)] = 1
    with open(illegible_path, 'r') as f:
        all_illegible = json.load(f)['illegible']
    for entry in all_illegible:
        if str(entry) not in dict:
            dict[str(entry)] = -1
    for t in os.listdir(image_dir):
        if t not in dict:
            dict[t] = -1
        else:
            dict[t] = int(dict[t])
    return dict


def train_parseq(args):
    parseq_dir  = config.str_home
    current_dir = os.getcwd()
    os.chdir(parseq_dir)
    if args.dataset == 'Hockey':
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'],
                                 config.dataset['Hockey']['numbers_data'])
    else:
        data_root = os.path.join(current_dir, config.dataset['SoccerNet']['root_dir'],
                                 config.dataset['SoccerNet']['numbers_data'])
    command = (
        f"conda run -n {config.str_env} python3 train.py +experiment=parseq dataset=real "
        f"data.root_dir={data_root} trainer.max_epochs=25 pretrained=parseq "
        f"trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 "
        f"data.max_label_length=2"
    )
    os.system(command)
    os.chdir(current_dir)
    print("Done training")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def soccer_net_pipeline(args):
    print("[PIPE] entered soccer_net_pipeline (main_karim.py)", flush=True)
    legible_dict      = None
    legible_results   = None
    consolidated_dict = None

    Path(config.dataset['SoccerNet']['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True

    image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'],
                             config.dataset['SoccerNet'][args.part]['images'])
    soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                    config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
    features_dir       = config.dataset['SoccerNet'][args.part]['feature_output_folder']
    full_legible_path  = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['legible_result'])
    illegible_path     = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['illegible_result'])
    gt_path            = os.path.join(config.dataset['SoccerNet']['root_dir'],
                                      config.dataset['SoccerNet'][args.part]['gt'])
    input_json         = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['pose_input_json'])
    output_json        = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['pose_output_json'])

    # ── Stage 1: Soccer-ball filter ───────────────────────────────────────────
    if args.pipeline['soccer_ball_filter']:
        print("Determine soccer ball")
        success = helpers.identify_soccer_balls(image_dir, soccer_ball_list)
        print("Done determine soccer ball")

    # ── Stage 2: Feature extraction (ReID) ────────────────────────────────────
    if args.pipeline['feat']:
        print("Generate features")
        command = (f"conda run -n {config.reid_env} python3 {config.reid_script} "
                   f"--tracklets_folder {image_dir} --output_folder {features_dir}")
        success = os.system(command) == 0
        print("Done generating features")

    # ── Stage 3: Gaussian outlier removal ─────────────────────────────────────
    if args.pipeline['filter'] and success:
        print("Identify and remove outliers")
        command = (f"python3 gaussian_outliers.py "
                   f"--tracklets_folder {image_dir} --output_folder {features_dir}")
        success = os.system(command) == 0
        print("Done removing outliers")

    # ── Stage 4: Legibility classification (ADAPTIVE threshold) ───────────────
    if args.pipeline['legible'] and success:
        print("Classifying Legibility (adaptive threshold):")
        try:
            legible_dict, _ = get_soccer_net_legibility_results_adaptive(args, exclude_balls=True)
        except Exception as error:
            print(f"Failed to run legibility classifier: {error}")
            success = False
        print("Done classifying legibility")

    # ── Stage 4.5: Legibility evaluation ──────────────────────────────────────
    if args.pipeline['legible_eval'] and success:
        print("Evaluate Legibility results:")
        try:
            if legible_dict is None:
                with open(full_legible_path, 'r') as f:
                    legible_dict = json.load(f)
            helpers.evaluate_legibility(gt_path, illegible_path, legible_dict,
                                        soccer_ball_list=soccer_ball_list)
        except Exception as e:
            print(e)
            success = False
        print("Done evaluating legibility")

    # ── Stage 5: Pose estimation ───────────────────────────────────────────────
    if args.pipeline['pose'] and success:
        print("Generating json for pose")
        try:
            if legible_dict is None:
                with open(full_legible_path, 'r') as f:
                    legible_dict = json.load(f)
            generate_json_for_pose_estimator(args, legible=legible_dict)
        except Exception as e:
            print(e)
            success = False
        print("Done generating json for pose")

        if success:
            print("Detecting pose")
            command = (
                f"conda run -n {config.pose_env} python3 pose.py "
                f"{config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/"
                f"ViTPose_huge_coco_256x192.py "
                f"{config.pose_home}/checkpoints/vitpose-h.pth "
                f"--img-root / --json-file {input_json} --out-json {output_json}"
            )
            success = os.system(command) == 0
            print("Done detecting pose")

    # ── Stage 6: Crop generation → window sampling → CLAHE ────────────────────
    if args.pipeline['crops'] and success:
        print("Generate crops")
        try:
            crops_destination_dir = os.path.join(
                config.dataset['SoccerNet']['working_dir'],
                config.dataset['SoccerNet'][args.part]['crops_folder'],
                'imgs',
            )
            Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)

            if legible_results is None:
                with open(full_legible_path, 'r') as f:
                    legible_results = json.load(f)

            helpers.generate_crops(output_json, crops_destination_dir, legible_results)

            # ── Modification 2: temporally-diverse window sampling ─────────────
            if args.max_windows > 0:
                select_diverse_topk_per_tracklet(
                    crops_destination_dir,
                    max_windows=args.max_windows,
                    min_keep=2,
                    verbose=True,
                )

            # ── Modification 3: CLAHE preprocessing ───────────────────────────
            if args.use_clahe:
                apply_clahe_to_crops(crops_destination_dir, verbose=True)

        except Exception as e:
            print(e)
            success = False
        print("Done generating crops")

    str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                   config.dataset['SoccerNet'][args.part]['jersey_id_result'])

    # ── Stage 7: STR inference ─────────────────────────────────────────────────
    if args.pipeline['str'] and success:
        print("Predict numbers")
        crops_dir = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                 config.dataset['SoccerNet'][args.part]['crops_folder'])
        crops_imgs_dir = os.path.join(crops_dir, 'imgs')

        # Allow window sampling even when crops stage was skipped (e.g. reuse)
        if args.max_windows > 0:
            select_diverse_topk_per_tracklet(
                crops_imgs_dir,
                max_windows=args.max_windows,
                min_keep=2,
                verbose=True,
            )

        # Allow CLAHE even when crops stage was skipped
        if args.use_clahe:
            apply_clahe_to_crops(crops_imgs_dir, verbose=True)

        command = (
            f"conda run -n {config.str_env} python3 str.py "
            f"{config.dataset['SoccerNet']['str_model']} "
            f"--data_root={crops_dir} "
            f"--batch_size={args.str_batch_size} "
            f"--inference --result_file {str_result_file}"
        )
        success = os.system(command) == 0
        print("Done predict numbers")

    # ── Stage 8: Combine / aggregate predictions ───────────────────────────────
    if args.pipeline['combine'] and success:
        analysis_results = None
        # Try Bayesian (with TS) → Bayesian (without TS) → simple weighted vote
        try:
            results_dict, analysis_results = helpers.process_jersey_id_predictions_bayesian(
                str_result_file, useTS=True, useBias=True, useTh=True
            )
        except Exception as e1:
            print(f"[Combine] Bayesian+TS failed: {e1}")
            try:
                results_dict, analysis_results = helpers.process_jersey_id_predictions_bayesian(
                    str_result_file, useTS=False, useBias=True, useTh=True
                )
            except Exception as e2:
                print(f"[Combine] Bayesian(raw) failed: {e2}")
                print("[Combine] Falling back to simple combiner.")
                results_dict, analysis_results = helpers.process_jersey_id_predictions(
                    str_result_file, useBias=True
                )

        consolidated_dict = consolidated_results(
            crops_dir if 'crops_dir' in dir() else image_dir,
            results_dict,
            illegible_path,
            soccer_ball_list=soccer_ball_list,
        )

        final_results_path = os.path.join(
            config.dataset['SoccerNet']['working_dir'],
            config.dataset['SoccerNet'][args.part]['final_result'],
        )
        with open(final_results_path, 'w') as f:
            json.dump(consolidated_dict, f)

    # ── Stage 9: Evaluate ──────────────────────────────────────────────────────
    if args.pipeline['eval'] and success:
        if consolidated_dict is None:
            with open(final_results_path, 'r') as f:
                consolidated_dict = json.load(f)
        with open(gt_path, 'r') as gf:
            gt_dict = json.load(gf)
        print(len(consolidated_dict.keys()), len(gt_dict.keys()))
        helpers.evaluate_results(consolidated_dict, gt_dict, full_results=analysis_results)


def hockey_pipeline(args):
    success = True
    if args.pipeline['legible']:
        root_dir = os.path.join(config.dataset['Hockey']['root_dir'],
                                config.dataset['Hockey']['legibility_data'])
        command = (f"python3 legibility_classifier.py --data {root_dir} "
                   f"--arch resnet34 --trained_model {config.dataset['Hockey']['legibility_model']}")
        success = os.system(command) == 0

    if success and args.pipeline['str']:
        current_dir = os.getcwd()
        data_root   = os.path.join(current_dir, config.dataset['Hockey']['root_dir'],
                                   config.dataset['Hockey']['numbers_data'])
        command = (f"conda run -n {config.str_env} python3 str.py "
                   f"{config.dataset['Hockey']['str_model']} --data_root={data_root}")
        os.system(command)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Jersey Number Recognition – Karim Khalil (Team 1)'
    )
    parser.add_argument('dataset', help="Options: 'SoccerNet', 'Hockey'")
    parser.add_argument('part',    help="Options: 'test', 'val', 'train', 'challenge'")

    # ── Speed / quality knobs ─────────────────────────────────────────────────
    parser.add_argument(
        '--str_batch_size', type=int, default=64,
        help='Batch size for STR inference (higher = faster; default 64).',
    )
    parser.add_argument(
        '--max_windows', type=int, default=10,
        help=(
            'If >0, divide each tracklet into this many time windows and keep '
            'the single best-quality frame from each window (temporally-diverse '
            'sampling).  Set 0 to disable.  Default: 10.'
        ),
    )
    parser.add_argument(
        '--use_clahe', action='store_true', default=False,
        help='Apply CLAHE contrast normalisation to crops before STR inference.',
    )

    # ── Pipeline control ──────────────────────────────────────────────────────
    parser.add_argument(
        '--full_pipeline', action='store_true', default=False,
        help='Run every stage (feat+filter+legible+pose+crops+str+combine+eval).',
    )
    parser.add_argument(
        '--clean', action='store_true', default=False,
        help='Delete previous run artefacts before running.',
    )
    parser.add_argument(
        '--keep_crops', action='store_true', default=False,
        help='When --clean is set, keep the existing crops folder.',
    )
    parser.add_argument(
        '--train_str', action='store_true', default=False,
        help='Run PARSeq training instead of inference.',
    )
    args = parser.parse_args()

    if not args.train_str:
        if args.dataset == 'SoccerNet':
            if args.full_pipeline:
                actions = {
                    'soccer_ball_filter': False,
                    'feat':         True,
                    'filter':       True,
                    'legible':      True,
                    'legible_eval': False,
                    'pose':         True,
                    'crops':        True,
                    'str':          True,
                    'combine':      True,
                    'eval':         True,
                }
            else:
                # Default: only STR → combine → eval  (crops assumed pre-generated)
                actions = {
                    'soccer_ball_filter': False,
                    'feat':         False,
                    'filter':       False,
                    'legible':      False,
                    'legible_eval': False,
                    'pose':         False,
                    'crops':        False,
                    'str':          True,
                    'combine':      True,
                    'eval':         True,
                }
            args.pipeline = actions

            if args.clean:
                clean_soccer_net_artifacts(args.part,
                                           clean_crops=(not args.keep_crops),
                                           verbose=True)

            print("[BOOT] calling soccer_net_pipeline()", flush=True)
            soccer_net_pipeline(args)

        elif args.dataset == 'Hockey':
            args.pipeline = {'legible': True, 'str': True}
            hockey_pipeline(args)

        else:
            print("Unknown dataset")
    else:
        train_parseq(args)
