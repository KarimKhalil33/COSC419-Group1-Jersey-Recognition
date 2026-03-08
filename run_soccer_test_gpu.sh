#!/bin/bash
#SBATCH --job-name=jersey_test
#SBATCH --account=st-li1210-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/jersey_%j.out
#SBATCH --error=logs/jersey_%j.err
#SBATCH --mail-user=yijun127@student.ubc.ca
#SBATCH --mail-type=ALL

module load gcc
module load cuda
export TORCH_HOME=/scratch/st-li1210-1/yijun/.cache/torch
export XDG_CACHE_HOME=/scratch/st-li1210-1/yijun/.cache
export TORCH_HOME=/scratch/st-li1210-1/yijun/.cache/torch
export LD_LIBRARY_PATH=/arc/home/yijun127/miniforge3/envs/parseq2/lib:$LD_LIBRARY_PATH

mkdir -p "$TORCH_HOME"
mkdir -p "$XDG_CACHE_HOME"
python main.py SoccerNet test

