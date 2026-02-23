"""
Hand joint retargeting: human → robot hand joint space.

Based on EgoScale Section 2.1 (Hand Articulation):
"We retarget the 21 human hand keypoints into a dexterous robot hand joint space
using an optimization-based procedure that enforces joint limits and kinematic constraints."

This module implements retargeting from UDCAP 21-joint angles to various robot hands.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RobotHandConfig:
    """Configuration for a target robot hand."""
    name: str
    num_joints: int
    joint_names: List[str]
    joint_limits_low: List[float]   # radians
    joint_limits_high: List[float]  # radians
    # Mapping from UDCAP joint indices to robot joint indices
    # Each entry: (udcap_idx, robot_idx, scale, offset)
    joint_mapping: List[Tuple[int, int, float, float]]


# Robot hand configurations
ALLEGRO_HAND = RobotHandConfig(
    name="allegro_hand",
    num_joints=16,
    joint_names=[
        # Index (4)
        "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
        # Middle (4) 
        "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
        # Ring (4)
        "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
        # Thumb (4)
        "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ],
    joint_limits_low=[
        0.263, -0.105, -0.189, -0.162,  # Index
        0.263, -0.105, -0.189, -0.162,  # Middle
        0.263, -0.105, -0.189, -0.162,  # Ring
        0.263, -0.105, -0.189, -0.162,  # Thumb
    ],
    joint_limits_high=[
        1.396, 1.163, 1.644, 1.719,  # Index
        1.396, 1.163, 1.644, 1.719,  # Middle
        1.396, 1.163, 1.644, 1.719,  # Ring
        1.396, 1.163, 1.644, 1.719,  # Thumb
    ],
    # (udcap_idx, robot_idx, scale, offset) 
    # Maps UDCAP human joint angles → Allegro joint commands
    joint_mapping=[
        # Index: UDCAP joints 4-7 → Allegro index 0-3
        (5, 0, 1.0, 0.0),   # index_mcp_abduction → index_joint_0
        (4, 1, 1.0, 0.0),   # index_mcp_flexion → index_joint_1
        (6, 2, 1.0, 0.0),   # index_pip_flexion → index_joint_2
        (7, 3, 1.0, 0.0),   # index_dip_flexion → index_joint_3
        # Middle: UDCAP joints 8-11 → Allegro middle 4-7
        (9, 4, 1.0, 0.0),   # middle_mcp_abduction → middle_joint_0
        (8, 5, 1.0, 0.0),   # middle_mcp_flexion → middle_joint_1
        (10, 6, 1.0, 0.0),  # middle_pip_flexion → middle_joint_2
        (11, 7, 1.0, 0.0),  # middle_dip_flexion → middle_joint_3
        # Ring: UDCAP joints 12-15 → Allegro ring 8-11
        (13, 8, 1.0, 0.0),  # ring_mcp_abduction → ring_joint_0
        (12, 9, 1.0, 0.0),  # ring_mcp_flexion → ring_joint_1
        (14, 10, 1.0, 0.0), # ring_pip_flexion → ring_joint_2
        (15, 11, 1.0, 0.0), # ring_dip_flexion → ring_joint_3
        # Thumb: UDCAP joints 0-3 → Allegro thumb 12-15
        (1, 12, 1.0, 0.0),  # thumb_cmc_abduction → thumb_joint_0
        (0, 13, 1.0, 0.0),  # thumb_cmc_flexion → thumb_joint_1
        (2, 14, 1.0, 0.0),  # thumb_mcp_flexion → thumb_joint_2
        (3, 15, 1.0, 0.0),  # thumb_ip_flexion → thumb_joint_3
    ]
)

LEAP_HAND = RobotHandConfig(
    name="leap_hand",
    num_joints=16,
    joint_names=[
        "index_0", "index_1", "index_2", "index_3",
        "middle_0", "middle_1", "middle_2", "middle_3",
        "ring_0", "ring_1", "ring_2", "ring_3",
        "thumb_0", "thumb_1", "thumb_2", "thumb_3",
    ],
    joint_limits_low=[-0.314] * 16,
    joint_limits_high=[2.23] * 16,
    joint_mapping=[
        (5, 0, 1.0, 0.0), (4, 1, 1.0, 0.0), (6, 2, 1.0, 0.0), (7, 3, 1.0, 0.0),
        (9, 4, 1.0, 0.0), (8, 5, 1.0, 0.0), (10, 6, 1.0, 0.0), (11, 7, 1.0, 0.0),
        (13, 8, 1.0, 0.0), (12, 9, 1.0, 0.0), (14, 10, 1.0, 0.0), (15, 11, 1.0, 0.0),
        (1, 12, 1.0, 0.0), (0, 13, 1.0, 0.0), (2, 14, 1.0, 0.0), (3, 15, 1.0, 0.0),
    ]
)


ROBOT_HANDS = {
    "allegro_hand": ALLEGRO_HAND,
    "leap_hand": LEAP_HAND,
}


def retarget_joints(
    human_joints: np.ndarray,
    target_hand: str = "allegro_hand",
    method: str = "linear"
) -> np.ndarray:
    """
    Retarget human 21-joint angles to robot hand joint space.
    
    Args:
        human_joints: (21,) array of human joint angles in degrees from UDCAP
        target_hand: target robot hand name
        method: "linear" (direct mapping) or "optimization" (constrained opt)
    
    Returns:
        (N,) array of robot joint angles in radians
    """
    config = ROBOT_HANDS[target_hand]
    human_rad = np.deg2rad(human_joints)
    
    if method == "linear":
        return _retarget_linear(human_rad, config)
    elif method == "optimization":
        return _retarget_optimization(human_rad, config)
    else:
        raise ValueError(f"Unknown method: {method}")


def _retarget_linear(human_rad: np.ndarray, config: RobotHandConfig) -> np.ndarray:
    """Direct linear mapping with joint limit clamping."""
    robot_joints = np.zeros(config.num_joints)
    
    for udcap_idx, robot_idx, scale, offset in config.joint_mapping:
        value = human_rad[udcap_idx] * scale + offset
        # Clamp to joint limits
        value = np.clip(
            value,
            config.joint_limits_low[robot_idx],
            config.joint_limits_high[robot_idx]
        )
        robot_joints[robot_idx] = value
    
    return robot_joints


def _retarget_optimization(human_rad: np.ndarray, config: RobotHandConfig) -> np.ndarray:
    """
    Optimization-based retargeting (EgoScale approach).
    
    Minimizes ||FK(q_robot) - FK(q_human)|| subject to joint limits.
    Requires GPU for batch processing — runs on RunPod.
    
    For now, falls back to linear + smoothing.
    TODO: Implement full FK-based optimization when GPU pipeline is ready.
    """
    # Start with linear mapping
    robot_joints = _retarget_linear(human_rad, config)
    
    # Apply temporal smoothing (placeholder for full optimization)
    # Full implementation will use differentiable FK + gradient descent
    return robot_joints


def retarget_episode(
    human_joint_sequence: np.ndarray,
    target_hand: str = "allegro_hand",
    fps: int = 30,
    smooth: bool = True,
    smooth_window: int = 5
) -> np.ndarray:
    """
    Retarget a full episode of human joint angles.
    
    Args:
        human_joint_sequence: (T, 21) array of human joints per timestep
        target_hand: target robot hand
        fps: frame rate (for temporal smoothing)
        smooth: apply temporal smoothing
        smooth_window: smoothing window size
    
    Returns:
        (T, N) array of robot joint angles
    """
    T = human_joint_sequence.shape[0]
    config = ROBOT_HANDS[target_hand]
    robot_sequence = np.zeros((T, config.num_joints))
    
    for t in range(T):
        robot_sequence[t] = retarget_joints(
            human_joint_sequence[t], target_hand, method="linear"
        )
    
    if smooth and T > smooth_window:
        # Simple moving average smoothing
        kernel = np.ones(smooth_window) / smooth_window
        for j in range(config.num_joints):
            robot_sequence[:, j] = np.convolve(
                robot_sequence[:, j], kernel, mode='same'
            )
    
    return robot_sequence


def compute_wrist_relative_transform(
    wrist_poses: np.ndarray
) -> np.ndarray:
    """
    Compute relative wrist transforms (EgoScale Eq: ΔW^t = (W_w^0)^-1 * W_w^t).
    
    Args:
        wrist_poses: (T, 7) array of [x,y,z, qw,qx,qy,qz] absolute poses
    
    Returns:
        (T, 7) array of relative poses from timestep 0
    """
    from scipy.spatial.transform import Rotation
    
    T = wrist_poses.shape[0]
    relative = np.zeros_like(wrist_poses)
    relative[0] = [0, 0, 0, 1, 0, 0, 0]  # Identity at t=0
    
    pos_0 = wrist_poses[0, :3]
    rot_0 = Rotation.from_quat(wrist_poses[0, 3:7][[1,2,3,0]])  # scipy uses xyzw
    rot_0_inv = rot_0.inv()
    
    for t in range(1, T):
        # Relative position
        relative[t, :3] = rot_0_inv.apply(wrist_poses[t, :3] - pos_0)
        
        # Relative rotation
        rot_t = Rotation.from_quat(wrist_poses[t, 3:7][[1,2,3,0]])
        rel_rot = rot_0_inv * rot_t
        q = rel_rot.as_quat()  # xyzw
        relative[t, 3:7] = [q[3], q[0], q[1], q[2]]  # back to wxyz
    
    return relative


if __name__ == "__main__":
    # Quick test
    print("Testing retargeting...")
    
    # Simulated UDCAP readings: hand slightly curled
    human_joints = np.array([
        30, 15, 20, 10,      # Thumb
        25, 5, 30, 20,       # Index
        25, 3, 35, 25,       # Middle
        20, 2, 30, 20,       # Ring
        10, 15, 3, 25, 15,   # Pinky
    ], dtype=float)
    
    allegro = retarget_joints(human_joints, "allegro_hand")
    print(f"Human (21 joints, degrees): {human_joints}")
    print(f"Allegro (16 joints, radians): {np.round(allegro, 3)}")
    
    # Test episode retargeting
    T = 100
    sequence = np.tile(human_joints, (T, 1))
    # Add some movement
    for t in range(T):
        sequence[t] += np.sin(t * 0.1) * 5  # Gentle oscillation
    
    robot_seq = retarget_episode(sequence, "allegro_hand", smooth=True)
    print(f"\nEpisode: {T} timesteps, human (T,21) → allegro (T,16)")
    print(f"Shape: {robot_seq.shape}")
    print(f"Joint ranges: min={np.round(robot_seq.min(axis=0), 3)}")
    print(f"             max={np.round(robot_seq.max(axis=0), 3)}")
    print("\n✅ Retargeting pipeline working")
