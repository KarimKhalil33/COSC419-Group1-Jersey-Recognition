#!/bin/bash
#SBATCH --job-name=benchmark_legibility
#SBATCH --account=st-li1210-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH --output=./logs/benchmark——legibility_output.txt
#SBATCH --error=./logs/benchmark——legibility_error.txt
#SBATCH --mail-user=yijun127@student.ubc.ca
#SBATCH --mail-type=ALL

module load gcc
module load cuda
set -x
date
hostname
which python
python --version

export TORCH_HOME=/scratch/st-li1210-1/yijun/.cache/torch
export XDG_CACHE_HOME=/scratch/st-li1210-1/yijun/.cache
export TORCH_HOME=/scratch/st-li1210-1/yijun/.cache/torch
export LD_LIBRARY_PATH=/arc/home/yijun127/miniforge3/envs/parseq2/lib:$LD_LIBRARY_PATH

mkdir -p "$TORCH_HOME"
mkdir -p "$XDG_CACHE_HOME"

python3 benchmark.py \
  --run_legibility \
  --leg_script /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/legibility_classifier.py \
  --leg_image_dir /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/out/SoccerNetResults/test/crops \
  --leg_model /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/models/legibility_resnet34_soccer_20240215.pth \
  --leg_env centroids \
  --leg_batch_sizes 16 64 512 1024\
  --repeats 3 \
  --warmup \
  --output_dir /scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/batch_benchmark