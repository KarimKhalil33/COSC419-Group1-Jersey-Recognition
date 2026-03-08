#!/bin/bash
#SBATCH --job-name=produce_interim_data
#SBATCH --account=st-li1210-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/jersey_run_on_train_%j.out
#SBATCH --error=logs/jersey_run_on_train_%j.err
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
export LD_LIBRARY_PATH=/arc/home/yijun127/miniforge3/envs/parseq2/lib:$LD_LIBRARY_PATH

mkdir -p "$TORCH_HOME"
mkdir -p "$XDG_CACHE_HOME"

python -u main.py SoccerNet train

date
