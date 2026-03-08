#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --account=st-li1210-1
#SBATCH --job-name=make_legibility_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/make_legibility_dataset_%A_%a.out
#SBATCH --error=logs/make_legibility_dataset_%A_%a.err
#SBATCH --mail-user=yijun127@student.ubc.ca
#SBATCH --mail-type=ALL


python make_legibility_dataset.py \
  --str-json out/SoccerNetResults/jersey_id_results.json \
  --crops-dir out/SoccerNetResults/crops/imgs \
  --gt-json data/SoccerNet/jersey-2023/train/train_gt.json \
  --out-dir data/SoccerNet/jersey-2023/legibility_finetune_subset \
  --balance \
  --val-frac 0.2 \
  --pos-quantile 0.3 \
  --neg-quantile 0.2 \
  --max-pos-per-tracklet 30 \
  --max-neg-per-tracklet 30
