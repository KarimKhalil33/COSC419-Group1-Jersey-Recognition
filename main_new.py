print("[BOOT] main_new.py starting", flush=True)
import argparse
print("[BOOT] importing argparse", flush=True)
import os
print("[BOOT] importing os", flush=True)
import legibility_classifier as lc
print("[BOOT] importing legibility_classifier", flush=True)
import numpy as np
print("[BOOT] importing numpy", flush=True)
import json
print("[BOOT] importing json", flush=True)
import shutil
print("[BOOT] importing shutil", flush=True)
import helpers
print("[BOOT] importing helpers", flush=True)
from tqdm import tqdm
import configuration as config
print("[BOOT] importing configuration", flush=True)
from pathlib import Path
import random
print("[BOOT] imports finished", flush=True)
print("[BOOT] __file__ =", os.path.abspath(__file__), flush=True)
print("[BOOT] cwd =", os.getcwd(), flush=True)


QUALITY_W_SHARPNESS = 0.50   # weight for Laplacian sharpness in composite score
QUALITY_W_CONTRAST  = 0.30   # weight for RMS contrast
QUALITY_W_EDGE      = 0.20   # weight for Canny edge density
MIN_WINDOWS         = 2      # minimum number of time windows in diverse sampling
CLAHE_CLIP          = 2.0    # CLAHE clip limit
CLAHE_GRID          = 8      # CLAHE tile grid size (8×8)
CLAHE_MIN_DIM       = 32     # skip CLAHE for crops smaller than this on either axis
MIN_TRACKLET_FRAMES = 2      # fewer frames than this after filtering → illegible without CNN
ADAPTIVE_LEG_FLOOR  = 0.50   # lower bound for per-tracklet adaptive threshold




def sample_images(images,
                  keep_ratio_range=(0.8, 0.9),
                  min_keep=5,
                  seed=42,
                  track_id=None):
    """
    Uniformly sample images without modifying files.
    Returns
    -------
    list
        Sampled image filenames.
    """

    if len(images) == 0:
        return images

    # deterministic per-track seed
    local_seed = hash((seed, track_id)) % (2**32)
    rng = random.Random(local_seed)

    keep_ratio = rng.uniform(*keep_ratio_range)

    keep_n = max(min_keep, int(round(len(images) * keep_ratio)))
    keep_n = min(keep_n, len(images))

    return rng.sample(images, keep_n)


def _is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def clean_soccer_net_artifacts(part, clean_crops=True, verbose=True):
    """Remove outputs from previous runs so a rerun reflects full pipeline time."""
    try:
        wd = config.dataset['SoccerNet']['working_dir']
        d = config.dataset['SoccerNet'][part]
        paths_files = [
            os.path.join(wd, d.get('illegible_result', '')),
            os.path.join(wd, d.get('legible_result', '')),
            os.path.join(wd, d.get('raw_legible_result', '')),
            os.path.join(wd, d.get('pose_input_json', '')),
            os.path.join(wd, d.get('pose_output_json', '')),
            os.path.join(wd, d.get('jersey_id_result', '')),
            os.path.join(wd, d.get('final_result', '')),
            os.path.join(wd, f"{part}_crop_legibility_results.json"),
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
                        if verbose:
                            print(f"[Clean] removed file: {fp}")
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


def get_soccer_net_raw_legibility_results(args, use_filtered=True, filter='gauss', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = os.listdir(path_to_images)
    results_dict = {x: [] for x in tracklets}

    filtered = {}
    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if track not in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path,
                               config.dataset['SoccerNet']['legibility_model'],
                               threshold=-1,
                               arch=config.dataset['SoccerNet']['legibility_model_arch'],
                               batch_size=args.legible_batch_size)
        results_dict[directory] = track_results

    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['raw_legible_result'])
    with open(full_legibile_path, "w") as outfile:
        json.dump(results_dict, outfile)

    return results_dict


def get_soccer_net_legibility_results(args, use_filtered=False, filter='sim', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = os.listdir(path_to_images)

    filtered = {}
    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    legible_tracklets = {}
    illegible_tracklets = []

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if track not in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]

        if len(images_full_path) < MIN_TRACKLET_FRAMES:
            illegible_tracklets.append(directory)
            continue


        raw_scores = lc.run(
            images_full_path,
            config.dataset['SoccerNet']['legibility_model'],
            arch=config.dataset['SoccerNet']['legibility_model_arch'],
            threshold=-1,
            batch_size=args.legible_batch_size,
        )
        raw_scores = np.asarray(raw_scores, dtype=float)
        legible = list(np.where(raw_scores >= ADAPTIVE_LEG_FLOOR)[0])


        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            legible_tracklets[directory] = legible_images

    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['legible_result'])
    with open(full_legibile_path, "w") as outfile:
        outfile.write(json.dumps(legible_tracklets, indent=4))

    full_illegibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['illegible_result'])
    with open(full_illegibile_path, "w") as outfile:
        outfile.write(json.dumps({'illegible': illegible_tracklets}, indent=4))

    return legible_tracklets, illegible_tracklets


def generate_json_for_pose_estimator(args, legible=None):
    all_files = []
    if legible is not None:
        for key in legible.keys():
            for entry in legible[key]:
                all_files.append(os.path.join(os.getcwd(), entry))
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset['SoccerNet']['root_dir'])
        image_dir = config.dataset['SoccerNet'][args.part]['images']
        path_to_images = os.path.join(root_dir, image_dir)
        tracks = os.listdir(path_to_images)
        for tr in tracks:
            track_dir = os.path.join(path_to_images, tr)
            imgs = os.listdir(track_dir)
            for img in imgs:
                all_files.append(os.path.join(track_dir, img))

    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet'][args.part]['pose_input_json'])
    helpers.generate_json(all_files, output_json)


def consolidated_results(image_dir, pred_dict, illegible_path, soccer_ball_list=None):
    if soccer_ball_list is not None:
        with open(soccer_ball_list, 'r') as sf:
            balls_json = json.load(sf)
        balls_list = balls_json['ball_tracks']
        for entry in balls_list:
            pred_dict[str(entry)] = 1

    with open(illegible_path, 'r') as f:
        illegile_dict = json.load(f)
    all_illegible = illegile_dict['illegible']
    for entry in all_illegible:
        if str(entry) not in pred_dict.keys():
            pred_dict[str(entry)] = -1

    all_tracks = os.listdir(image_dir)
    for t in all_tracks:
        if t not in pred_dict.keys():
            pred_dict[t] = -1
        else:
            pred_dict[t] = int(pred_dict[t])
    return pred_dict


def _score_to_keep_mask(scores, threshold=0.5):
    arr = np.asarray(scores).reshape(-1)
    if arr.size == 0:
        return []
    try:
        arr = arr.astype(float)
    except Exception:
        arr = np.array([float(x) for x in arr], dtype=float)
    keep = [bool(x >= threshold) for x in arr]
    if not any(keep):
        keep[int(np.argmax(arr))] = True
    return keep


def run_crop_legibility_classifier(crops_imgs_dir, model_path, output_path, threshold=0.5,
                                   prune_in_place=False, verbose=True):
    """
    Run the extra crop-legibility stage from main_new, but make it usable in main_fast.
    Supports either:
      1) imgs/<tracklet>/<crop files>
      2) imgs/<flat crop files named like trackid_xxx.jpg>

    Returns a dict:
      {
        tracklet_id: {
          "images": [...],
          "scores": [...],
          "keep_mask": [...]
        },
        ...
      }
    """
    if not os.path.isdir(crops_imgs_dir):
        raise FileNotFoundError(f"Crops directory not found: {crops_imgs_dir}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Crop legibility model not found: {model_path}")

    if verbose:
        print(f"[CropLegible] loading model from: {model_path}")


    crop_legible_results = {}
    nested_dirs = [d for d in os.listdir(crops_imgs_dir) if os.path.isdir(os.path.join(crops_imgs_dir, d))]

    if nested_dirs:
        iterator = tqdm(sorted(nested_dirs), desc="Crop legibility")
        for directory in iterator:
            track_dir = os.path.join(crops_imgs_dir, directory)
            images = sorted([x for x in os.listdir(track_dir) if _is_image_file(x)])
            if len(images) == 0:
                crop_legible_results[directory] = {"images": [], "scores": [], "keep_mask": []}
                continue

            images_full_path = [os.path.join(track_dir, x) for x in images]
            scores = lc.run(
                images_full_path,
                model_path,
                threshold=-1,
                arch='resnet34',   # or pass this in as an argument
                batch_size=128
            )
            keep_mask = _score_to_keep_mask(scores, threshold=threshold)

            if prune_in_place:
                for img_name, keep in zip(images, keep_mask):
                    if not keep:
                        try:
                            os.remove(os.path.join(track_dir, img_name))
                        except Exception:
                            pass

            crop_legible_results[directory] = {
                "images": images,
                "scores": np.asarray(scores).reshape(-1).astype(float).tolist(),
                "keep_mask": keep_mask,
            }
    else:
        files = sorted([f for f in os.listdir(crops_imgs_dir) if _is_image_file(f)])
        by_track = {}
        for fn in files:
            track = fn.split('_')[0]
            by_track.setdefault(track, []).append(fn)

        iterator = tqdm(sorted(by_track.keys()), desc="Crop legibility")
        for track in iterator:
            images = sorted(by_track[track])
            images_full_path = [os.path.join(crops_imgs_dir, x) for x in images]
            scores = lc.run(
                images_full_path,
                model_path,
                threshold=-1,
                arch='resnet34',   # or pass this in as an argument
                batch_size=128
            )
            keep_mask = _score_to_keep_mask(scores, threshold=threshold)

            if prune_in_place:
                for img_name, keep in zip(images, keep_mask):
                    if not keep:
                        try:
                            os.remove(os.path.join(crops_imgs_dir, img_name))
                        except Exception:
                            pass

            crop_legible_results[track] = {
                "images": images,
                "scores": np.asarray(scores).reshape(-1).astype(float).tolist(),
                "keep_mask": keep_mask,
            }

    with open(output_path, "w") as outfile:
        json.dump(crop_legible_results, outfile, indent=2)

    if verbose:
        total_tracks = len(crop_legible_results)
        total_imgs = sum(len(v.get("images", [])) for v in crop_legible_results.values())
        total_kept = sum(sum(1 for k in v.get("keep_mask", []) if k) for v in crop_legible_results.values())
        print(f"[CropLegible] wrote: {output_path}")
        print(f"[CropLegible] tracks={total_tracks}, images={total_imgs}, kept={total_kept}, removed={total_imgs - total_kept}")

    return crop_legible_results


def train_parseq(args):
    if args.dataset == 'Hockey':
        print("Train PARSeq for Hockey")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'], config.dataset['Hockey']['numbers_data'])
        command = (
            f"conda run -n {config.str_env} python3 train.py +experiment=parseq "
            f"dataset=real data.root_dir={data_root} trainer.max_epochs=25 pretrained=parseq "
            f"trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
        )
        os.system(command)
        os.chdir(current_dir)
        print("Done training")
    else:
        print("Train PARSeq for Soccer")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet']['numbers_data'])
        command = (
            f"conda run -n {config.str_env} python3 train.py +experiment=parseq "
            f"dataset=real data.root_dir={data_root} trainer.max_epochs=25 pretrained=parseq "
            f"trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
        )
        os.system(command)
        os.chdir(current_dir)
        print("Done training")


def hockey_pipeline(args):
    success = True
    if args.pipeline['legible']:
        root_dir = os.path.join(config.dataset["Hockey"]["root_dir"], config.dataset["Hockey"]["legibility_data"])
        print("Test legibility classifier")
        command = f"python3 legibility_classifier.py --data {root_dir} --arch resnet34 --trained_model {config.dataset['Hockey']['legibility_model']}"
        success = os.system(command) == 0
        print("Done legibility classifier")

    if success and args.pipeline['str']:
        print("Predict numbers")
        current_dir = os.getcwd()
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'], config.dataset['Hockey']['numbers_data'])
        command = f"conda run -n {config.str_env} python3 str.py {config.dataset['Hockey']['str_model']} --data_root={data_root}"
        os.system(command)
        print("Done predict numbers")


def soccer_net_pipeline(args):
    print("[PIPE] entered soccer_net_pipeline", flush=True)
    legible_dict = None
    legible_results = None
    consolidated_dict = None
    analysis_results = None

    Path(config.dataset['SoccerNet']['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True

    image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['images'])
    soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                    config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
    features_dir = config.dataset['SoccerNet'][args.part]['feature_output_folder']
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['legible_result'])
    illegible_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                  config.dataset['SoccerNet'][args.part]['illegible_result'])
    gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'],
                           config.dataset['SoccerNet'][args.part]['gt'])

    input_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                              config.dataset['SoccerNet'][args.part]['pose_input_json'])
    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet'][args.part]['pose_output_json'])
    crops_destination_dir = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                         config.dataset['SoccerNet'][args.part]['crops_folder'], 'imgs')
    crop_legible_output_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                            f"{args.part}_crop_legibility_results.json")
    str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                   config.dataset['SoccerNet'][args.part]['jersey_id_result'])
    final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['final_result'])

    if args.pipeline['soccer_ball_filter']:
        print("Determine soccer ball")
        success = helpers.identify_soccer_balls(image_dir, soccer_ball_list)
        print("Done determine soccer ball")

    if args.pipeline['feat'] and success:
        print("Generate features")
        command = f"conda run -n {config.reid_env} python3 -u {config.reid_script} --tracklets_folder {image_dir} --output_folder {features_dir} --batch_size 2048"
        success = os.system(command) == 0
        print("Done generating features")

    if args.pipeline['filter'] and success:
        print("Identify and remove outliers")
        command = f"python3 -u gaussian_outliers.py --tracklets_folder {image_dir} --output_folder {features_dir}"
        success = os.system(command) == 0
        print("Done removing outliers")

    if args.pipeline['legible'] and success:
        print("Classifying Legibility:")
        try:
            legible_dict, _ = get_soccer_net_legibility_results(args, use_filtered=True, filter='gauss', exclude_balls=True)
        except Exception as error:
            print(f"Failed to run legibility classifier: {error}")
            success = False
        print("Done classifying legibility")

    if args.pipeline['legible_eval'] and success:
        print("Evaluate Legibility results:")
        try:
            if legible_dict is None:
                with open(full_legibile_path, 'r') as openfile:
                    legible_dict = json.load(openfile)
            helpers.evaluate_legibility(gt_path, illegible_path, legible_dict, soccer_ball_list=soccer_ball_list)
        except Exception as e:
            print(e)
            success = False
        print("Done evaluating legibility")

    if args.pipeline['pose'] and success:
        print("Generating json for pose")
        try:
            if legible_dict is None:
                with open(full_legibile_path, 'r') as openfile:
                    legible_dict = json.load(openfile)
            generate_json_for_pose_estimator(args, legible=legible_dict)
        except Exception as e:
            print(e)
            success = False
        print("Done generating json for pose")

        if success:
            print("Detecting pose")
            command = (
                f"conda run -n {config.pose_env} python3 -u pose.py "
                f"{config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py "
                f"{config.pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} --out-json {output_json}"
            )
            success = os.system(command) == 0
            print("Done detecting pose")

    if args.pipeline['crops'] and success:
        print("Generate crops")
        try:
            Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
            if legible_results is None:
                with open(full_legibile_path, "r") as outfile:
                    legible_results = json.load(outfile)
            helpers.generate_crops(output_json, crops_destination_dir, legible_results, topk=args.topk_crops,use_clahe=args.use_clahe,max_windows=args.max_windows)
        except Exception as e:
            print(e)
            success = False
        print("Done generating crops")

    if args.pipeline['str'] and success:
        print("Predict numbers")
        image_dir_for_str = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                         config.dataset['SoccerNet'][args.part]['crops_folder'])
        command = (
            f"conda run -n {config.str_env} python3 -u str.py {config.dataset['SoccerNet']['str_model']} "
            f"--data_root={image_dir_for_str} --batch_size={args.str_batch_size} --inference --result_file {str_result_file}"
        )
        success = os.system(command) == 0
        print("Done predict numbers")

    if args.pipeline['combine'] and success:
        print("Combine results")
        analysis_results = None

        # read predicted results, stack unique predictions, sum confidence scores for each, choose argmax
        results_dict, analysis_results = helpers.process_jersey_id_predictions(
            str_result_file, useBias=True
        )

        consolidated_dict = consolidated_results(
            image_dir, results_dict, illegible_path, soccer_ball_list=soccer_ball_list
        )

        with open(final_results_path, 'w') as f:
            json.dump(consolidated_dict, f)

        print("Done combine results")

    if args.pipeline['eval'] and success:
        if consolidated_dict is None:
            with open(final_results_path, 'r') as f:
                consolidated_dict = json.load(f)
        with open(gt_path, 'r') as gf:
            gt_dict = json.load(gf)
        print(len(consolidated_dict.keys()), len(gt_dict.keys()))
        helpers.evaluate_results(consolidated_dict, gt_dict, full_results=analysis_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Options: 'SoccerNet', 'Hockey'")
    parser.add_argument('part', help="Options: 'test', 'val', 'train', 'challenge'")
    parser.add_argument('--str_batch_size', type=int, default=1,
                        help='Batch size for STR inference (higher = faster; adjust to GPU memory).')
    parser.add_argument('--legible_batch_size', type=int, default=4,
                        help='Batch size for legibility inference.')
    parser.add_argument('--topk_crops', type=int, default=0,
                        help='If >0, keep only top-K sharpest crops per tracklet before STR (speed + often accuracy).')
    parser.add_argument('--full_pipeline', action='store_true', default=False,
                        help='Run the entire pipeline. Default is only str+combine+eval.')
    parser.add_argument('--clean', action='store_true', default=False,
                        help='Delete previous run artifacts before running.')
    parser.add_argument('--keep_crops', action='store_true', default=False,
                        help='When used with --clean, keep existing crops folder.')
    parser.add_argument('--crop_legible_model', type=str,
                        default='./experiments/legibility_resnet34_20260304-203231.pth',
                        help='Path to the crop-legibility model checkpoint.')
    parser.add_argument('--crop_legible_threshold', type=float, default=0.5,
                        help='Threshold used to decide whether a crop is legible.')
    parser.add_argument('--crop_legible_prune', action='store_true', default=False,
                        help='If set, delete crop images predicted as illegible before STR.')
    parser.add_argument('--train_str', action='store_true', default=False,
                        help='Run training of jersey number recognition')
    parser.add_argument('--use_clahe', action='store_true', default=False,
                        help='Apply CLAHE (LAB L-channel) to crops in the fused preprocessing pass.')
    parser.add_argument('--max_windows', type=int, default=0,
                        help=(
                            'If >0, divide each tracklet into this many time windows and keep '
                            'the best-quality (multi-metric) frame per window. '
                            'Runs as part of the fused select+CLAHE pass. '
                            'Default 0 = disabled.'
                        ))
    args = parser.parse_args()

    if not args.train_str:
        if args.dataset == 'SoccerNet':
            if args.full_pipeline:
                actions = {
                    "soccer_ball_filter": True,
                    "feat": True,
                    "filter": True,
                    "legible": True,
                    "legible_eval": False,
                    "pose": True,
                    "crops": True,
                    "str": True,
                    "combine": True,
                    "eval": True,
                }
            else:
                actions = {
                    "soccer_ball_filter": False,
                    "feat": False,
                    "filter": False,
                    "legible": False,
                    "legible_eval": False,
                    "pose": False,
                    "crops": False,
                    "str": True,
                    "combine": True,
                    "eval": True,
                }
            args.pipeline = actions

            if args.clean:
                clean_soccer_net_artifacts(args.part, clean_crops=(not args.keep_crops), verbose=True)
            print("[BOOT] calling soccer_net_pipeline()", flush=True)
            soccer_net_pipeline(args)
        elif args.dataset == 'Hockey':
            actions = {"legible": True, "str": True}
            args.pipeline = actions
            hockey_pipeline(args)
        else:
            print("Unknown dataset")
    else:
        train_parseq(args)
