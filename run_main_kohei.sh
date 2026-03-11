#!/bin/bash
#SBATCH --job-name=Kohei_experiment
#SBATCH --account=st-li1210-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=./logs/Kohei_experiment_output.txt
#SBATCH --error=./logs/Kohei_experiment_error.txt
#SBATCH --mail-user=yijun127@student.ubc.ca
#SBATCH --mail-type=ALL




export HOME=/scratch/st-li1210-1/yijun
export XDG_CONFIG_HOME=$HOME/.config
export MPLCONFIGDIR=$HOME/.config/matplotlib
export WANDB_MODE=offline

mkdir -p $XDG_CONFIG_HOME/Ultralytics
mkdir -p $MPLCONFIGDIR

python main_kohei.py SoccerNet test --full_pipeline --max_windows 10 --use_clahe --str_batch_size 64
