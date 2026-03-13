# Flexa Pipeline

**Phone capture → robot training data.** Record a task with your iPhone, get a LeRobot/RLDS-compatible dataset out.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. CAPTURE  │────▶│  2. INGEST   │────▶│ 3. RETARGET  │────▶│  4. EXPORT   │────▶│  5. SIM/VIZ  │
│  iPhone R3D  │     │ Extract + Det│     │ Human→Robot  │     │ LeRobot/RLDS │     │  MuJoCo Val  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Stage 1: Capture (`r3d_ingest.py`)
- Input: `.r3d` file from Record3D app (iPhone LiDAR)
- Extracts: RGB frames, depth maps, camera intrinsics, confidence maps
- Output: `r3d_output/<session>/` with frames + metadata

### Stage 2: Hand Tracking + Object Detection
- **Hand tracking** (`hand_tracker_v2.py` or `modal_hamer.py`):
  - Runs HaMeR (Hand Mesh Recovery) on extracted frames
  - Produces 3D hand mesh + wrist position per frame
  - Can run locally or on Modal (GPU cloud) via `run_modal_pipeline.py`
- **Object detection** (`detect_objects.py`):
  - Runs GroundingDINO for object bounding boxes
  - Detects task-relevant objects (blocks, mugs, etc.)
  - Output: `object_detections/<session>.json`

### Stage 3: Spatial Calibration + Retargeting
- **3D reconstruction** (`reconstruct_wrist_3d.py`):
  - Projects 2D hand detections into 3D using depth + camera intrinsics
  - Output: `wrist_trajectories/<task>_wrist3d.json`
- **Workspace calibration** (`calibrate_workspace.py`):
  - Aligns phone coordinate frame → robot workspace
  - Maps detected object positions to sim-space
  - Output: `wrist_trajectories/<task>_calibrated.json`
- **Robot retargeting** (`validate_and_retarget.py`):
  - Maps human wrist trajectory → robot joint positions via IK
  - Handles grasp detection (open/close from hand mesh)
  - Currently targets: Franka Panda (7-DOF arm + parallel gripper)

### Stage 4: Dataset Export
- **LeRobot format** (`egodex_to_lerobot.py`):
  - Exports to LeRobot-compatible HDF5 + JSON metadata
  - Fields: `qpos_arm[7]`, `qvel_arm[7]`, `ee_pos[3]`, `gripper_state[1]`, `target_qpos[7]`, `target_gripper[1]`
  - Output: `lerobot_dataset_v5/`
- **RLDS/Open X-Embodiment format** (`egodex_to_oxe.py`):
  - Exports to TFRecord format compatible with RT-X ecosystem

### Stage 5: Simulation Validation (MuJoCo)
- **Franka sim** (`mujoco_franka_v9.py`): Replays retargeted trajectory on Franka arm — this one works cleanly
- **G1 humanoid sim** (`mujoco_g1_v10.py`): Replays on Unitree G1 — WIP, adhesion-based grasping
- **H1 + Shadow Hand** (`mujoco_h1_shadow_v1.py`): H1 body + dexterous hand — WIP
- Uses [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) robot models

## Quick Start

### Prerequisites
```bash
pip install mujoco numpy pillow opencv-python
pip install record3d  # for R3D file parsing
# For GPU stages (HaMeR):
pip install torch torchvision
# For object detection:
pip install groundingdino-py
```

### Run the pipeline end-to-end

```bash
# 1. Ingest R3D file
python r3d_ingest.py path/to/recording.r3d

# 2. Run hand tracking (local or cloud)
python hand_tracker_v2.py r3d_output/<session>/
# OR for cloud GPU:
python run_modal_pipeline.py <session>

# 3. Detect objects
python detect_objects.py r3d_output/<session>/

# 4. Reconstruct 3D wrist trajectory
python reconstruct_wrist_3d.py <session> --task stack2

# 5. Calibrate to robot workspace
python calibrate_workspace.py wrist_trajectories/stack2_wrist3d.json

# 6. Export to LeRobot format
python egodex_to_lerobot.py

# 7. Validate in simulation (optional)
python mujoco_franka_v9.py stack2
```

### View the R3D capture
Open `r3d_viewer/index.html` in a browser, or visit: https://r3dviewer.vercel.app

## Key Files

| File | Purpose |
|------|---------|
| `r3d_ingest.py` | Parse .r3d → frames + depth + intrinsics |
| `hand_tracker_v2.py` | HaMeR hand mesh recovery (local) |
| `modal_hamer.py` | HaMeR on Modal cloud GPU |
| `detect_objects.py` | GroundingDINO object detection |
| `reconstruct_wrist_3d.py` | 2D→3D wrist trajectory via depth |
| `calibrate_workspace.py` | Phone→robot coordinate alignment |
| `validate_and_retarget.py` | Human→robot IK retargeting |
| `egodex_to_lerobot.py` | Export LeRobot HDF5 dataset |
| `egodex_to_oxe.py` | Export RLDS/OXE TFRecord dataset |
| `mujoco_franka_v9.py` | Franka sim validation (working) |
| `mujoco_g1_v10.py` | G1 humanoid sim (WIP) |
| `schema/episode.py` | Data schema definitions |

## Data Flow

```
iPhone (.r3d)
  ├── RGB frames (30fps, 1920x1440)
  ├── Depth maps (LiDAR, 256x192)
  └── Camera intrinsics + poses
        │
        ▼
  HaMeR → 3D hand mesh per frame
  GroundingDINO → object bounding boxes
        │
        ▼
  Depth reprojection → wrist XYZ trajectory
  Workspace calibration → sim-space coordinates
        │
        ▼
  IK retargeting → robot joint angles
  Grasp detection → gripper open/close
        │
        ▼
  LeRobot HDF5 dataset
  ├── observations: qpos[7], ee_pos[3], gripper[1]
  ├── actions: target_qpos[7], target_gripper[1]
  └── metadata: task, fps, episode phases
```

## Status

- ✅ R3D ingest + frame extraction
- ✅ HaMeR hand tracking (local + cloud)
- ✅ Object detection (GroundingDINO)
- ✅ 3D wrist trajectory reconstruction
- ✅ Workspace calibration
- ✅ Franka IK retargeting
- ✅ LeRobot dataset export
- ✅ Franka MuJoCo sim replay (clean)
- 🔧 G1 humanoid sim (grasping needs work — adhesion-based, not physically realistic yet)
- 🔧 H1 + Shadow Hand sim (WIP)
- 📋 Multi-episode batch processing
- 📋 Automated quality validation
