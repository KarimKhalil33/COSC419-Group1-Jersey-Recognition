#!/bin/bash
#SBATCH --job-name=two_stage_experiment
#SBATCH --account=st-li1210-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=./logs/two_stage_experiment_output.txt
#SBATCH --error=./logs/two_stage_experiment_error.txt
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

python -u main.py SoccerNet test --full_pipeline --clean --max_windows 10 --str_batch_size 512 --legible_batch_size 4 --use_clahe 