#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --account=st-li1210-1
#SBATCH --job-name=data_processing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --output=soccernet_legibility_data.out
#SBATCH --error=soccernet_legibility_data.err
#SBATCH --mail-user=yijun127@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1

python make_legibility_csv.py