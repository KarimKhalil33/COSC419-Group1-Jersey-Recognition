#!/bin/bash
#SBATCH --job-name=generate_crops
#SBATCH --account=st-li1210-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/crops_%j_out.txt
#SBATCH --error=logs/crops_%j_err.txt
#SBATCH --mail-user=yijun127@student.ubc.ca
#SBATCH --mail-type=ALL

module load gcc
module load cuda


export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
export XDG_CACHE_HOME=$SLURM_TMPDIR/xdg_cache
mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME

python generate_srt_crops.py \
  --images_root /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/train/images \
  --gt_json /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/train/train_gt.json \
  --out_dir /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline/out_srt \
  --pose_weights yolov8s-pose.pt \
  --target_total 50000 \
  --frame_stride 8 \
  --sharpness_thresh 80 \
  --max_per_tracklet 40
