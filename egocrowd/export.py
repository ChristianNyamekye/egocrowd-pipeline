"""Export trajectories to LeRobot HDF5, RLDS JSON, and Raw JSON formats."""

import json
import numpy as np
from pathlib import Path
from typing import Optional


def to_lerobot_hdf5(
    trajectory: dict,
    qpos_data: np.ndarray,
    output_path: str,
    episode_id: int = 0,
) -> str:
    """Export to LeRobot-compatible HDF5.
    
    Args:
        trajectory: Output from spatial_trajectory()
        qpos_data: Joint positions array (N, 7)
        output_path: Path to output .hdf5 file
        episode_id: Episode index
        
    Returns:
        Path to saved HDF5 file
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n = len(trajectory["ee_targets"])
    ee = np.array(trajectory["ee_targets"])
    grip = np.array(trajectory["gripper_cmds"]).reshape(-1, 1)
    mug = np.tile(trajectory["mug_sim"], (n, 1))
    act_grip = grip.copy()
    
    with h5py.File(output_path, 'w') as f:
        ep = f.create_group(f"episode_{episode_id}")
        
        # Observations
        obs = ep.create_group("observations")
        obs.create_dataset("qpos_arm", data=qpos_data)
        obs.create_dataset("ee_pos", data=ee)
        obs.create_dataset("mug_pos", data=mug)
        obs.create_dataset("gripper_state", data=grip)
        
        # Actions
        act = ep.create_group("actions")
        act.create_dataset("target_qpos", data=qpos_data)
        act.create_dataset("target_gripper", data=act_grip)
        
        # Metadata
        ep.attrs["n_steps"] = n
        ep.attrs["fps"] = 30
        ep.attrs["robot"] = "franka_panda"
        
        # Compute mug lift
        mug_z = np.array(trajectory["mug_sim"])[2]
        final_z = ee[-1, 2]
        ep.attrs["mug_lift_cm"] = (final_z - mug_z) * 100
    
    print(f"Saved LeRobot HDF5: {output_path} ({n} steps)")
    return str(output_path)


def to_rlds_json(trajectory: dict, qpos_data: np.ndarray, output_path: str, task: str = "pick up the mug") -> str:
    """Export to RLDS-compatible JSON."""
    n = len(trajectory["ee_targets"])
    ee = np.array(trajectory["ee_targets"])
    grip = np.array(trajectory["gripper_cmds"])
    mug = trajectory["mug_sim"]
    
    steps = []
    for i in range(n):
        steps.append({
            "observation": {
                "qpos_arm": qpos_data[i].tolist(),
                "ee_pos": ee[i].tolist(),
                "mug_pos": mug,
                "gripper_state": float(grip[i]),
            },
            "action": qpos_data[i].tolist() + [float(grip[i])],
            "reward": 1.0 if i == n - 1 else 0.0,
            "is_first": i == 0,
            "is_last": i == n - 1,
            "is_terminal": i == n - 1,
            "language_instruction": task,
        })
    
    episode = {
        "format": "rlds",
        "dataset_name": "egocrowd",
        "num_steps": n,
        "steps": steps,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(episode, f, indent=2)
    
    print(f"Saved RLDS JSON: {output_path} ({n} steps)")
    return output_path


def to_raw_json(trajectory: dict, qpos_data: np.ndarray, output_path: str) -> str:
    """Export to raw JSON with full state per timestep."""
    n = len(trajectory["ee_targets"])
    ee = np.array(trajectory["ee_targets"])
    grip = np.array(trajectory["gripper_cmds"])
    mug = trajectory["mug_sim"]
    
    timesteps = []
    for i in range(n):
        timesteps.append({
            "t": i,
            "qpos_arm": qpos_data[i].tolist(),
            "ee_pos": ee[i].tolist(),
            "mug_pos": mug,
            "gripper_state": float(grip[i]),
            "action_qpos": qpos_data[i].tolist(),
            "action_gripper": float(grip[i]),
        })
    
    mug_z = mug[2]
    final_z = ee[-1, 2]
    
    episode = {
        "episode_id": "episode_0",
        "num_steps": n,
        "mug_lift_cm": float((final_z - mug_z) * 100),
        "source": "egocrowd",
        "timesteps": timesteps,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(episode, f, indent=2)
    
    print(f"Saved Raw JSON: {output_path} ({n} steps)")
    return output_path


def download_dataset(repo_id: str = "egocrowd/pick-mug-v5", filename: str = "data/data.hdf5") -> str:
    """Download an EgoCrowd dataset from HuggingFace.
    
    Args:
        repo_id: HuggingFace dataset repo ID
        filename: File to download
        
    Returns:
        Local path to downloaded file
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    print(f"Downloaded {repo_id}/{filename} -> {path}")
    return path
