"""Generate synthetic calibrated JSON for testing simulation without R3D files.

Creates a known-good stacking trajectory: hover → descend → grasp → lift → carry → stack → release.

Usage:
    python synthetic_data.py --robot g1 --task stack
    python synthetic_data.py --robot franka --task stack
"""
import argparse
import json
import numpy as np
from pathlib import Path

from pipeline_config import CALIB_DIR

# Robot-specific workspace parameters
ROBOT_PARAMS = {
    "g1": {
        "table_z": 0.78,
        "block_half": 0.03,
        "pick_xy": [0.38, -0.03],
        "support_xy": [0.38, -0.31],
        "hover_z_offset": 0.12,
        "lift_z_offset": 0.18,
    },
    "franka": {
        "table_z": 0.40,
        "block_half": 0.025,
        "pick_xy": [0.45, 0.08],
        "support_xy": [0.55, -0.08],
        "hover_z_offset": 0.15,
        "lift_z_offset": 0.20,
    },
    "h1": {
        "table_z": 1.24,
        "block_half": 0.04,
        "pick_xy": [0.30, -0.05],
        "support_xy": [0.42, 0.03],
        "hover_z_offset": 0.10,
        "lift_z_offset": 0.12,
    },
}

# Total frames for synthetic trajectory
N_FRAMES = 200


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def generate_stack_trajectory(robot="g1"):
    """Generate a 7-phase stacking trajectory."""
    params = ROBOT_PARAMS[robot]
    table_z = params["table_z"]
    bh = params["block_half"]
    pick = np.array(params["pick_xy"])
    support = np.array(params["support_xy"])
    hover_off = params["hover_z_offset"]
    lift_off = params["lift_z_offset"]

    z_block = table_z + bh
    z_hover = z_block + hover_off
    z_grasp = z_block - 0.005
    z_lift = z_block + lift_off
    z_stack = table_z + 3 * bh + 0.002  # on top of support

    wrist = np.zeros((N_FRAMES, 3))
    grasping = np.zeros(N_FRAMES, dtype=int)

    # Phase boundaries (as fractions of N_FRAMES)
    phases = [
        (0.00, 0.15, "hover"),      # hover over pick
        (0.15, 0.28, "descend"),     # descend to grasp
        (0.28, 0.42, "grasp"),       # dwell + grasp
        (0.42, 0.55, "lift"),        # lift block
        (0.55, 0.72, "carry"),       # carry to support
        (0.72, 0.85, "stack"),       # descend to stack
        (0.85, 1.00, "release"),     # release + retreat
    ]

    for i in range(N_FRAMES):
        p = i / (N_FRAMES - 1)

        if p < 0.15:
            # Hover over pick position
            wrist[i] = [pick[0], pick[1], z_hover]
        elif p < 0.28:
            # Descend to grasp height
            t = smoothstep((p - 0.15) / 0.13)
            wrist[i] = [pick[0], pick[1], z_hover + (z_grasp - z_hover) * t]
        elif p < 0.42:
            # Dwell at grasp height (fingers close)
            wrist[i] = [pick[0], pick[1], z_grasp]
            grasping[i] = 1
        elif p < 0.55:
            # Lift
            t = smoothstep((p - 0.42) / 0.13)
            wrist[i] = [pick[0], pick[1], z_grasp + (z_lift - z_grasp) * t]
            grasping[i] = 1
        elif p < 0.72:
            # Carry to support (arc path)
            t = smoothstep((p - 0.55) / 0.17)
            mid = (pick + support) / 2
            if t < 0.5:
                t2 = t * 2
                cx = pick[0] + (mid[0] - pick[0]) * t2
                cy = pick[1] + (mid[1] - pick[1]) * t2
            else:
                t2 = (t - 0.5) * 2
                cx = mid[0] + (support[0] - mid[0]) * t2
                cy = mid[1] + (support[1] - mid[1]) * t2
            wrist[i] = [cx, cy, z_lift]
            grasping[i] = 1
        elif p < 0.85:
            # Descend to stack
            t = smoothstep((p - 0.72) / 0.13)
            wrist[i] = [support[0], support[1], z_lift + (z_stack - z_lift) * t]
            grasping[i] = 1
        else:
            # Release + retreat
            t = smoothstep((p - 0.85) / 0.15)
            wrist[i] = [support[0], support[1], z_stack + 0.02 + (z_hover - z_stack) * t]

    # Object positions
    objects_sim = [
        [pick[0], pick[1], z_block],
        [support[0], support[1], z_block],
    ]

    return {
        "session": f"synthetic_stack",
        "wrist_sim": wrist.tolist(),
        "grasping": grasping.tolist(),
        "objects_sim": objects_sim,
        "r3d_to_sim": {
            "scale": 1.0,
            "obj_centroid_r3d": [0, 0, 0],
            "obj_centroid_sim": [0, 0, 0],
        },
    }


def generate_synthetic(robot="g1", task="stack"):
    """Generate and save synthetic calibrated data."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    if task == "stack":
        calib = generate_stack_trajectory(robot)
    else:
        # Default to stack for other tasks
        calib = generate_stack_trajectory(robot)
        calib["session"] = f"synthetic_{task}"

    session_name = f"synthetic_{task}"
    out_path = CALIB_DIR / f"{session_name}_calibrated.json"
    with open(out_path, "w") as f:
        json.dump(calib, f)
    print(f"  Generated synthetic data: {out_path}")
    print(f"  Robot: {robot}, Task: {task}, Frames: {N_FRAMES}")
    return session_name, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic calibrated data")
    parser.add_argument("--robot", default="g1", choices=["g1", "franka", "h1"])
    parser.add_argument("--task", default="stack", choices=["stack", "pick_place", "sort"])
    args = parser.parse_args()
    generate_synthetic(args.robot, args.task)
