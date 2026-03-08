#!/bin/bash
#SBATCH --job-name=legibility_finetune
#SBATCH --account=st-li1210-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --output=./logs/finetune_legibility_output.txt
#SBATCH --error=./logs/finetune_legibility_error.txt
#SBATCH --mail-user=yijun127@student.ubc.ca
#SBATCH --mail-type=ALL




module load gcc
module load cuda


export HOME=/scratch/st-li1210-1/yijun
export XDG_CONFIG_HOME=$HOME/.config
export MPLCONFIGDIR=$HOME/.config/matplotlib
export WANDB_MODE=offline

mkdir -p $XDG_CONFIG_HOME/Ultralytics
mkdir -p $MPLCONFIGDIR

python legibility_classifier.py \
  --finetune \
  --arch resnet34 \
  --data data/SoccerNet/jersey-2023/legibility_finetune_subset \
  --trained_model_path models/legibility_resnet34_soccer_20240215.pth