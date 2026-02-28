# EgoCrowd

**Crowdsourced egocentric manipulation data pipeline for robot learning.**

Turn iPhone recordings into robot training data. Zero additional hardware required.

[![Paper](https://img.shields.io/badge/arXiv-EgoCrowd-b31b1b)](https://arxiv.org/abs/TODO)
[![Dataset](https://img.shields.io/badge/HuggingFace-egocrowd-yellow)](https://huggingface.co/datasets/egocrowd/pick-mug-v5)
[![Demo](https://img.shields.io/badge/Demo-Site-green)](https://demo-iota-six-98.vercel.app)

## Install

```bash
pip install egocrowd
```

**With GPU support** (for local object detection + hand pose):
```bash
pip install egocrowd[gpu]
```

**With simulation** (for MuJoCo replay):
```bash
pip install egocrowd[sim]
```

## Quick Start

### Download and explore the dataset

```python
from egocrowd import download_dataset
import h5py

# Download from HuggingFace
path = download_dataset("egocrowd/pick-mug-v5")

with h5py.File(path, "r") as f:
    ep = f["episode_0"]
    qpos = ep["observations/qpos_arm"][:]       # (270, 7) joint positions
    ee = ep["observations/ee_pos"][:]            # (270, 3) end-effector XYZ
    actions = ep["actions/target_qpos"][:]       # (270, 7) target joints
    print(f"Mug lift: {ep.attrs['mug_lift_cm']:.1f}cm")
# -> Mug lift: 17.7cm
```

### Process a new recording

```bash
# Parse .r3d file from Record3D
egocrowd process recording.r3d --object mug --output ./my_dataset

# With cloud GPU processing (GroundingDINO + HaMeR)
egocrowd process recording.r3d --cloud --output ./my_dataset
```

### Use as a library

```python
from egocrowd import parse_r3d, spatial_trajectory
from egocrowd.export import to_lerobot_hdf5, to_rlds_json

# Parse iPhone recording
data = parse_r3d("recording.r3d", output_dir="parsed/")

# Generate robot trajectory (after hand pose extraction)
traj = spatial_trajectory(
    hamer_results="parsed/hamer_results.json",
    object_poses="parsed/object_poses_3d.json",
)

# Export to LeRobot format
to_lerobot_hdf5(traj, qpos_data, "output/data.hdf5")
```

## Pipeline Architecture

```
iPhone (.r3d)
    |
    v
[1. Parse] ──> RGB frames + LiDAR depth + camera poses
    |
    v
[2. Detect] ──> GroundingDINO: open-vocab object detection
    |
    v
[3. Hand Pose] ──> HaMeR: 3D hand mesh reconstruction (93.7% coverage)
    |
    v
[4. Retarget] ──> Spatial trajectory: hand motion -> robot EE targets
    |
    v
[5. Export] ──> LeRobot HDF5 | RLDS JSON | Raw JSON
```

## Key Results

- **18.1cm clean mug lift** in MuJoCo simulation from a single iPhone recording
- **93.7% hand pose coverage** via HaMeR (no wearable sensors needed)
- **$0 contributor hardware cost** (core tier: iPhone with LiDAR only)
- **9-55x cheaper** than teleoperation-based data collection

## Supported Formats

| Format | File | Use Case |
|--------|------|----------|
| LeRobot HDF5 | `data.hdf5` | HuggingFace ecosystem, policy training |
| RLDS JSON | `episode.json` | RT-X, Octo, Open X-Embodiment |
| Raw JSON | `raw.json` | Custom pipelines, analysis |

## Citation

```bibtex
@article{nyamekye2026egocrowd,
  title={EgoCrowd: Crowdsourced Egocentric Manipulation Data at Consumer Cost},
  author={Nyamekye, Christian},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```

## License

CC-BY-4.0
