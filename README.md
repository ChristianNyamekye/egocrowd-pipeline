# Data Collection Pipeline

## Overview
Consumer hardware → physics-enriched manipulation data → robot training format (LeRobot/RLDS)

Aligned with EgoScale's (NVIDIA, Feb 2026) two-stage framework:
- **Stage 1**: Large-scale egocentric human pretraining data (crowdsourced)
- **Stage 2**: Aligned human-robot mid-training data (lab-collected)

## Hardware Kit
- iPhone (chest/head mount) → RGB video + LiDAR SLAM
- Apple Watch → 6-DoF wrist IMU at 100Hz
- UDCAP Data Glove → 21 joint angles at 120Hz

## Pipeline Stages
1. **Capture** — synchronized multi-sensor recording (iPhone app)
2. **Ingest** — upload raw recordings to cloud storage
3. **Process** — GPU pipeline: hand pose estimation, SLAM, sensor fusion
4. **Retarget** — map human hand joints → robot hand joint space
5. **Package** — output in EgoScale action representation / LeRobot/RLDS format

## Data Format
See `schema/` for the output format specification.

## Processing (GPU - RunPod)
See `processing/` for the GPU pipeline code.
