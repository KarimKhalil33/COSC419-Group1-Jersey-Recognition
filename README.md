# COSC419 Group 1 — Jersey Number Recognition
Reproduction and improvement of an automated jersey number recognition pipeline for soccer broadcast footage, based on the SoccerNet Jersey Number Recognition benchmark.

## Pipeline Overview
<img width="856" height="275" alt="image" src="https://github.com/user-attachments/assets/8a5ab63a-a91b-44be-85c3-6253d5d366c5" />

- **Main Subject Filtering** uses a Re-ID model to remove distractor images, such as other players, from the frame
- **Legibility Classification** filters out frames where the jersey number isn't visible
- **Pose Estimation** uses ViTPose to locate keypoints on the shoulders and hips and crops the torso region
- **Scene Text Recognition** runs PARSeq on each crop to predict the jersey number
- **Prediction Aggregation** combines per-frame predictions across a tracklet using Bayesian inference

# Our Improvements
- **Batch processing** across all three models (Re-ID, legibility classifier, PARSeq) improves the speed of the model
- **Top-K sharpness filtering** scores each crop using Laplacian variance and keeps only the sharpest frames per tracklet. This reduces noise before scene text recognition 
- **Multi-metric frame scoring** composite quality score combining sharpness (50%), contrast (30%), and edge density (20%) for more robust frame selection
- **Temporal diversity sampling** divides each tracklet into time windows and selects the best frame per window, ensuring selected frames are spread across the full tracklet rather than clustered
- **CLAHE contrast normalization** applies adaptive histogram equalization to each crop to correct for uneven stadium lighting before recognition
- **Adaptive legibility threshold** sets a per-tracklet threshold based on the median legibility score rather than a fixed global value, reducing noise on easy tracklets while staying robust on hard ones
  
# Results
Tested on the SoccerNet Jersey-2023 test set:

**Accuracy:** 85.3%
**Runtime:** 9hr 36min on UBC Sockeye (1x GPU, 64GB RAM)

# Reproducibility
The full pipeline requires a GPU server. We tested our model on the UBC Sockeye with 1x GPU, 64GB RAM.

# Dataset
SoccerNet Jersey-2023 dataset. Requires registration at SoccerNet to download.
