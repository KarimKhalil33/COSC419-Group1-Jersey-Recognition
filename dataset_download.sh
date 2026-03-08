#!/bin/bash

#SBATCH --time=24:00:00

#SBATCH --account=st-li1210-1

#SBATCH --job-name=jersey_data_download

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --mem=8G

#SBATCH --output=download_array_%A_%a.out

#SBATCH --error=download_array_%A_%a.err

#SBATCH --mail-user=yijun127@student.ubc.ca

#SBATCH --mail-type=ALL

#SBATCH --array=1


python download_dataset.py
