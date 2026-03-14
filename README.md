# Flexa Pipeline

**Phone capture -> robot training data.** Record a task with your iPhone, get a LeRobot/RLDS-compatible dataset out.

## Quick Start

```bash
# Install
pip install mujoco numpy pillow opencv-python liblzfse
# For GPU stages (HaMeR, GroundingDINO):
pip install torch torchvision transformers

# Synthetic mode (no hardware needed):
python run_pipeline.py --synthetic --robot g1 --task stack

# Real R3D file:
python run_pipeline.py --r3d path/to/recording.r3d --robot g1 --task stack

# With manual object positions (skip GPU detection):
python run_pipeline.py --r3d recording.r3d --robot g1 --task stack \
    --objects '[[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]'
```

## Architecture

```
                    run_pipeline.py (canonical entry point)
                            |
    +-----------+-----------+-----------+-----------+
    |           |           |           |           |
 1.INGEST   2.HANDS     3.OBJECTS   4.WRIST3D   5.CALIBRATE -> 6.SIMULATE
 r3d_ingest  HaMeR/MP   GDino/     reconstruct  calibrate      mujoco_*
             hand_tracker detect_obj  _wrist_3d   _workspace
```

### Stage 1: Ingest (`r3d_ingest.py`)
- Input: `.r3d` file from Record3D app (iPhone LiDAR)
- Extracts: RGB frames, LZ-FSE compressed depth maps, camera intrinsics
- Output: `r3d_output/<session>/` with frames + depth + metadata

### Stage 2: Hand Tracking
- **HaMeR** (GPU, via Modal): 3D hand mesh recovery — preferred
- **MediaPipe** (CPU): fallback when no GPU available
- Output: wrist pixel positions + grasp state per frame

### Stage 3: Object Detection
- **GroundingDINO** (GPU): zero-shot object detection with depth-based 3D positioning
- Requires `--objects` flag if GPU detection is unavailable
- Output: `object_detections/<session>_objects_clean.json`

### Stage 4-5: 3D Reconstruction + Calibration
- Projects 2D hand detections into 3D using depth + camera intrinsics
- Aligns phone coordinate frame to robot workspace
- Output: `wrist_trajectories/<session>_calibrated.json`

### Stage 6: Simulation (MuJoCo)
- **G1 humanoid** (`mujoco_g1_v10.py`): Kinematic arm + smooth-blend block attachment
- **Franka Panda** (`mujoco_franka_v9.py`): Same kinematic pattern, parallel gripper
- **H1 + Shadow Hand** (`mujoco_h1_shadow_v1.py`): WIP
- Uses [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) robot models
- Output: `sim_renders/<session>_<robot>.mp4`

## Key Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | End-to-end pipeline entry point |
| `r3d_ingest.py` | Parse .r3d -> frames + depth + intrinsics |
| `hand_tracker_v2.py` | MediaPipe hand tracking (CPU) |
| `egocrowd/hand_pose.py` | HaMeR hand mesh recovery (GPU) |
| `detect_objects.py` | GroundingDINO object detection |
| `reconstruct_wrist_3d.py` | 2D->3D wrist trajectory via depth |
| `calibrate_workspace.py` | Phone->robot coordinate alignment |
| `synthetic_data.py` | Generate test trajectories |
| `mujoco_g1_v10.py` | G1 humanoid simulation |
| `mujoco_franka_v9.py` | Franka Panda simulation |
| `pipeline_config.py` | Centralized path configuration |

## Data Flow

```
iPhone (.r3d)
  +-- RGB frames (30fps, 1920x1440)
  +-- Depth maps (LiDAR, 256x192, LZ-FSE compressed)
  +-- Camera intrinsics + poses
        |
        v
  HaMeR/MediaPipe -> wrist position per frame
  GroundingDINO -> object 3D positions
        |
        v
  Depth reprojection -> wrist XYZ trajectory
  Workspace calibration -> sim-space coordinates
        |
        v
  MuJoCo simulation -> robot replays manipulation task
  LeRobot HDF5 export -> training data
```

## Status

- Working: R3D ingest, hand tracking, GroundingDINO detection, 3D reconstruction, calibration
- Working: Franka sim (clean kinematic attach), G1 sim (kinematic attach)
- WIP: H1 + Shadow Hand sim, multi-episode batch processing
