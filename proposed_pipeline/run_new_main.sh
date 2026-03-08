#!/bin/bash
#SBATCH --job-name=test_pipeline
#SBATCH --account=st-li1210-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/test_pipeline_%j_out.txt
#SBATCH --error=logs/test_pipeline_%j_err.txt
#SBATCH --mail-user=yijun127@student.ubc.ca
#SBATCH --mail-type=ALL

module load gcc
module load cuda


export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
export XDG_CACHE_HOME=$SLURM_TMPDIR/xdg_cache
mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME

python new_main.py
