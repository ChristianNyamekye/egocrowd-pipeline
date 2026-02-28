"""Spatial trajectory retargeting: hand motion -> robot end-effector trajectory.

Converts vision-extracted hand trajectories (relative to manipulation target)
into robot joint-space trajectories via IK solving.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def spatial_trajectory(
    hamer_results: str,
    object_poses: str,
    n_frames: int = 270,
    pre_z: float = 0.415,
    grasp_z: float = 0.315,
    lift_z: float = 0.515,
    spatial_frac: float = 0.26,
    converge_frac: float = 0.35,
    mug_sim: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Generate a hybrid spatial trajectory from hand pose data.
    
    Architecture:
      Phase 1 (spatial): EE follows hand trajectory relative to object
      Phase 2 (converge): XY blends toward object center, Z holds at pre_z
      Phase 3 (expert): pregrasp -> cosine descend -> close -> lift
    
    Args:
        hamer_results: Path to HaMeR results JSON
        object_poses: Path to object poses JSON  
        n_frames: Number of output frames
        pre_z: Pre-grasp height above table
        grasp_z: Grasp height (object top)
        lift_z: Lift target height
        spatial_frac: Fraction of frames for spatial approach
        converge_frac: Fraction for XY convergence
        mug_sim: Override object position in sim (x, y, z)
        
    Returns:
        dict with trajectory data (ee_targets, phases, gripper_cmds)
    """
    # Load hand trajectory
    with open(hamer_results) as f:
        hamer = json.load(f)
    with open(object_poses) as f:
        obj = json.load(f)
    
    # Extract hand world positions
    frames = sorted(hamer.keys(), key=lambda x: int(x))
    hand_world = []
    for fr in frames:
        t = hamer[fr].get("translation", [0, 0, 0])
        hand_world.append(t)
    hand_world = np.array(hand_world)
    
    # Object position (use first detection)
    obj_pos = np.array(obj[0]["position_3d"]) if obj else np.array([0.5, 0, 0.3])
    
    if mug_sim is None:
        mug_sim = np.array([0.5, 0.0, 0.295])
    else:
        mug_sim = np.array(mug_sim)
    
    # Compute relative offsets (hand - object in world)
    offsets = hand_world - obj_pos
    
    # Normalize and scale to sim workspace
    scale = 0.3
    offsets_norm = offsets / (np.max(np.abs(offsets)) + 1e-8) * scale
    
    # Phase boundaries
    sp_end = int(n_frames * spatial_frac)
    conv_end = int(n_frames * converge_frac)
    
    # Phase definitions
    phases = {
        "spatial": [0, sp_end],
        "converge": [sp_end, conv_end],
        "pre": [conv_end, conv_end + 15],
        "descend": [conv_end + 15, conv_end + 70],
        "close": [conv_end + 70, conv_end + 95],
        "lift": [conv_end + 95, conv_end + 145],
        "hold": [conv_end + 145, n_frames],
    }
    
    # Generate EE targets
    ee_targets = np.zeros((n_frames, 3))
    gripper_cmds = np.ones(n_frames) * 0.04  # open
    
    home_ee = np.array([0.677, 0.0, 0.378])
    
    for t in range(n_frames):
        if t < sp_end:
            # Spatial: follow hand direction
            frac = t / max(sp_end - 1, 1)
            idx = min(int(frac * (len(offsets_norm) - 1)), len(offsets_norm) - 1)
            ee_targets[t] = home_ee + offsets_norm[idx]
            ee_targets[t, 2] = max(ee_targets[t, 2], pre_z)
            
        elif t < conv_end:
            # Converge XY to mug center
            frac = (t - sp_end) / max(conv_end - sp_end - 1, 1)
            blend = 0.5 * (1 - np.cos(np.pi * frac))
            prev = ee_targets[sp_end - 1].copy()
            target_xy = mug_sim[:2]
            ee_targets[t, 0] = prev[0] + blend * (target_xy[0] - prev[0])
            ee_targets[t, 1] = prev[1] + blend * (target_xy[1] - prev[1])
            ee_targets[t, 2] = pre_z
            
        elif t < phases["descend"][0]:
            # Pre-grasp: hold above mug
            ee_targets[t] = [mug_sim[0], mug_sim[1], pre_z]
            
        elif t < phases["close"][0]:
            # Descend with cosine profile
            frac = (t - phases["descend"][0]) / max(phases["close"][0] - phases["descend"][0] - 1, 1)
            blend = 0.5 * (1 - np.cos(np.pi * frac))
            ee_targets[t] = [mug_sim[0], mug_sim[1], pre_z + blend * (grasp_z - pre_z)]
            
        elif t < phases["lift"][0]:
            # Close gripper
            ee_targets[t] = [mug_sim[0], mug_sim[1], grasp_z]
            gripper_cmds[t] = 0.0
            
        elif t < phases["hold"][0]:
            # Lift
            frac = (t - phases["lift"][0]) / max(phases["hold"][0] - phases["lift"][0] - 1, 1)
            blend = 0.5 * (1 - np.cos(np.pi * frac))
            ee_targets[t] = [mug_sim[0], mug_sim[1], grasp_z + blend * (lift_z - grasp_z)]
            gripper_cmds[t] = 0.0
            
        else:
            # Hold
            ee_targets[t] = [mug_sim[0], mug_sim[1], lift_z]
            gripper_cmds[t] = 0.0
    
    result = {
        "n_frames": n_frames,
        "ee_targets": ee_targets.tolist(),
        "gripper_cmds": gripper_cmds.tolist(),
        "phases": phases,
        "mug_sim": mug_sim.tolist(),
        "pre_z": pre_z,
        "grasp_z": grasp_z,
        "lift_z": lift_z,
    }
    
    return result


def save_trajectory(traj: dict, output_path: str):
    """Save trajectory to JSON."""
    with open(output_path, 'w') as f:
        json.dump(traj, f, indent=2)
    print(f"Saved trajectory to {output_path}")
