#!/usr/bin/env python3
"""
EgoCrowd Retarget + Export Pipeline
====================================
Takes HaMeR 3D hand results + object detection + camera poses from an iPhone .r3d recording
and converts them into a LeRobot-compatible HDF5 dataset for robot training.

Pipeline: HaMeR cam_t (camera space) → world space → robot workspace → Panda EE targets → LeRobot HDF5

Usage:
    python tools/retarget_export.py
    python tools/retarget_export.py --output pipeline/lerobot_dataset
    python tools/retarget_export.py --viz  # also produce visualization
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
R3D_OUTPUT = "pipeline/r3d_output"
DEFAULT_OUTPUT = "pipeline/lerobot_dataset"

# Franka Panda workspace limits (meters)
PANDA_WORKSPACE = {
    "x": (0.25, 0.75),   # forward from base
    "y": (-0.4, 0.4),    # left-right
    "z": (0.01, 0.60),   # height above table
}
PANDA_HOME = np.array([0.5, 0.0, 0.35])  # home EE position
TABLE_HEIGHT = 0.01  # minimum z (table surface)

# Gripper heuristic: distance between thumb tip and index tip
GRIPPER_CLOSE_THRESH = 0.04  # meters — below this = closed
GRIPPER_OPEN_THRESH = 0.07   # meters — above this = open


def load_data(r3d_dir):
    """Load all pipeline outputs."""
    r3d = Path(r3d_dir)
    
    hamer = json.load(open(r3d / "hamer_results.json"))
    meta = json.load(open(r3d / "metadata.json"))
    obj_data = json.load(open(r3d / "object_poses_3d.json"))
    
    camera_K = np.array(meta["camera_K"])
    frames = meta["frames"]
    
    return hamer, frames, camera_K, obj_data


def get_camera_pose(frame):
    """Extract 4x4 camera-to-world transform from frame pose (quat + translation)."""
    pose = frame["pose"]
    # pose format: [qx, qy, qz, qw, tx, ty, tz] (Record3D format)
    qx, qy, qz, qw = pose[0], pose[1], pose[2], pose[3]
    tx, ty, tz = pose[4], pose[5], pose[6]
    
    # Quaternion to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def cam_t_to_world(cam_t, camera_pose):
    """Convert HaMeR camera-space position to world coordinates."""
    pos_cam = np.array([cam_t[0], cam_t[1], cam_t[2], 1.0])
    pos_world = camera_pose @ pos_cam
    return pos_world[:3]


def world_to_robot(world_pos, world_positions):
    """
    Map world coordinates to robot workspace.
    Uses the range of observed positions to normalize into Panda workspace.
    """
    wp = np.array(world_positions)
    
    # Normalize each axis to [0, 1] based on observed range
    mins = wp.min(axis=0)
    maxs = wp.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 0.001] = 1.0  # avoid division by zero
    
    normalized = (world_pos - mins) / ranges
    
    # Map to Panda workspace
    robot_pos = np.array([
        PANDA_WORKSPACE["x"][0] + normalized[0] * (PANDA_WORKSPACE["x"][1] - PANDA_WORKSPACE["x"][0]),
        PANDA_WORKSPACE["y"][0] + normalized[1] * (PANDA_WORKSPACE["y"][1] - PANDA_WORKSPACE["y"][0]),
        PANDA_WORKSPACE["z"][0] + normalized[2] * (PANDA_WORKSPACE["z"][1] - PANDA_WORKSPACE["z"][0]),
    ])
    
    # Clamp to workspace
    robot_pos[0] = np.clip(robot_pos[0], *PANDA_WORKSPACE["x"])
    robot_pos[1] = np.clip(robot_pos[1], *PANDA_WORKSPACE["y"])
    robot_pos[2] = np.clip(robot_pos[2], *PANDA_WORKSPACE["z"])
    
    return robot_pos


def compute_gripper_state(hand_data):
    """
    Compute gripper open/close from fingertip distances.
    Returns value in [0, 1] where 0=closed, 1=open.
    """
    ft = hand_data.get("fingertips", {})
    if not ft or "thumb" not in ft or "index" not in ft:
        return 0.5  # unknown
    
    thumb = np.array(ft["thumb"])
    index = np.array(ft["index"])
    dist = np.linalg.norm(thumb - index)
    
    # Linear interpolation between thresholds
    gripper = np.clip((dist - GRIPPER_CLOSE_THRESH) / (GRIPPER_OPEN_THRESH - GRIPPER_CLOSE_THRESH), 0, 1)
    return float(gripper)


def smooth_trajectory(positions, window=5):
    """Apply moving average smoothing to trajectory."""
    if len(positions) < window:
        return positions
    
    smoothed = np.copy(positions)
    half = window // 2
    for i in range(half, len(positions) - half):
        smoothed[i] = positions[i-half:i+half+1].mean(axis=0)
    return smoothed


def detect_episodes(gripper_states, min_episode_len=15):
    """
    Detect pick-and-place episodes from gripper state transitions.
    An episode = open → close → open (or close → open → close).
    If no clear episodes, treat the whole thing as one episode.
    """
    # Find gripper close events
    gs = np.array(gripper_states)
    closed = gs < 0.3
    
    # Find transitions
    episodes = []
    in_grasp = False
    ep_start = 0
    
    for i in range(1, len(gs)):
        if not in_grasp and closed[i] and not closed[i-1]:
            # Grasp started — episode started a bit before
            ep_start = max(0, i - 10)
            in_grasp = True
        elif in_grasp and not closed[i] and closed[i-1]:
            # Release — episode ends a bit after
            ep_end = min(len(gs) - 1, i + 10)
            if ep_end - ep_start >= min_episode_len:
                episodes.append((ep_start, ep_end))
            in_grasp = False
    
    # If still grasping at end
    if in_grasp and len(gs) - ep_start >= min_episode_len:
        episodes.append((ep_start, len(gs) - 1))
    
    # If no episodes detected, treat whole sequence as one
    if not episodes:
        episodes = [(0, len(gs) - 1)]
    
    return episodes


def export_lerobot_hdf5(output_dir, episodes_data, fps=30):
    """Export to LeRobot HDF5 format."""
    try:
        import h5py
    except ImportError:
        print("h5py not installed. Installing...")
        os.system(f"{sys.executable} -m pip install h5py -q")
        import h5py
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    h5_path = out / "data.hdf5"
    
    with h5py.File(h5_path, "w") as f:
        # Dataset metadata
        f.attrs["fps"] = fps
        f.attrs["robot_type"] = "franka_panda"
        f.attrs["source"] = "egocrowd_iphone_r3d"
        f.attrs["num_episodes"] = len(episodes_data)
        
        for ep_idx, ep in enumerate(episodes_data):
            grp = f.create_group(f"episode_{ep_idx}")
            
            # Observations: EE position (3) + gripper state (1) + object position (3)
            obs = np.column_stack([
                ep["ee_positions"],           # (T, 3)
                ep["gripper_states"][:, None], # (T, 1)
                ep["object_positions"],        # (T, 3)
            ])
            
            # Actions: delta EE position (3) + gripper command (1)
            ee = ep["ee_positions"]
            deltas = np.zeros_like(ee)
            deltas[1:] = ee[1:] - ee[:-1]
            actions = np.column_stack([
                deltas,                        # (T, 3)
                ep["gripper_states"][:, None],  # (T, 1)
            ])
            
            grp.create_dataset("observation", data=obs.astype(np.float32))
            grp.create_dataset("action", data=actions.astype(np.float32))
            grp.create_dataset("ee_pos", data=ee.astype(np.float32))
            grp.create_dataset("gripper", data=ep["gripper_states"].astype(np.float32))
            grp.create_dataset("object_pos", data=ep["object_positions"].astype(np.float32))
            
            grp.attrs["num_steps"] = len(ee)
            grp.attrs["episode_idx"] = ep_idx
            grp.attrs["fps"] = fps
            
            print(f"  Episode {ep_idx}: {len(ee)} steps, "
                  f"EE range x=[{ee[:,0].min():.3f},{ee[:,0].max():.3f}] "
                  f"z=[{ee[:,2].min():.3f},{ee[:,2].max():.3f}], "
                  f"gripper min={ep['gripper_states'].min():.2f} max={ep['gripper_states'].max():.2f}")
    
    print(f"\n✅ Saved HDF5: {h5_path} ({h5_path.stat().st_size / 1024:.1f} KB)")
    return h5_path


def export_lerobot_json(output_dir, episodes_data, fps=30):
    """Export to LeRobot JSON format (backup/compatibility)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "fps": fps,
        "robot_type": "franka_panda",
        "source": "egocrowd_iphone_r3d",
        "episodes": []
    }
    
    for ep_idx, ep in enumerate(episodes_data):
        ee = ep["ee_positions"]
        deltas = np.zeros_like(ee)
        deltas[1:] = ee[1:] - ee[:-1]
        
        episode = {
            "episode_idx": ep_idx,
            "num_steps": len(ee),
            "steps": []
        }
        
        for t in range(len(ee)):
            step = {
                "observation": {
                    "ee_pos": ee[t].tolist(),
                    "gripper_state": float(ep["gripper_states"][t]),
                    "object_pos": ep["object_positions"][t].tolist(),
                },
                "action": {
                    "delta_ee": deltas[t].tolist(),
                    "gripper_cmd": float(ep["gripper_states"][t]),
                }
            }
            episode["steps"].append(step)
        
        dataset["episodes"].append(episode)
    
    json_path = out / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Saved JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    return json_path


def visualize_trajectory(episodes_data, output_path):
    """Create a 3D trajectory visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available, skipping viz")
        return
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(121, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes_data)))
    
    for ep_idx, ep in enumerate(episodes_data):
        ee = ep["ee_positions"]
        ax1.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=colors[ep_idx], 
                label=f"Ep {ep_idx}", linewidth=2)
        ax1.scatter(*ee[0], color=colors[ep_idx], marker="o", s=50)  # start
        ax1.scatter(*ee[-1], color=colors[ep_idx], marker="x", s=50)  # end
        
        # Object position
        obj = ep["object_positions"]
        ax1.scatter(obj[0, 0], obj[0, 1], obj[0, 2], color="red", marker="^", s=100, label="Object" if ep_idx == 0 else "")
    
    ax1.set_xlabel("X (forward)")
    ax1.set_ylabel("Y (lateral)")
    ax1.set_zlabel("Z (height)")
    ax1.set_title("Robot EE Trajectory")
    ax1.legend()
    
    # Gripper state over time
    ax2 = fig.add_subplot(122)
    for ep_idx, ep in enumerate(episodes_data):
        t = np.arange(len(ep["gripper_states"]))
        ax2.plot(t, ep["gripper_states"], color=colors[ep_idx], label=f"Ep {ep_idx}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Gripper (0=closed, 1=open)")
    ax2.set_title("Gripper State")
    ax2.legend()
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved viz: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EgoCrowd Retarget + Export")
    parser.add_argument("--input", default=R3D_OUTPUT, help="r3d_output directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--viz", action="store_true", help="Generate visualization")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS")
    args = parser.parse_args()
    
    print("=" * 60)
    print("EgoCrowd Retarget + Export Pipeline")
    print("=" * 60)
    
    # 1. Load data
    print("\n📂 Loading pipeline data...")
    hamer, frames, camera_K, obj_data = load_data(args.input)
    print(f"   HaMeR frames: {len(hamer)}")
    print(f"   Camera frames: {len(frames)}")
    print(f"   Object detections: {len(obj_data['poses'])}")
    
    # 2. Convert cam_t to world coordinates
    print("\n🌍 Converting camera-space → world coordinates...")
    world_positions = []
    gripper_states = []
    frame_indices = []
    
    for fid in sorted(hamer.keys(), key=int):
        hands = hamer[fid]
        if not hands:
            continue
        
        hand = hands[0]  # take first (usually right) hand
        fidx = int(fid)
        
        if fidx >= len(frames):
            continue
        
        # Get camera pose for this frame
        cam_pose = get_camera_pose(frames[fidx])
        
        # Convert cam_t to world
        world_pos = cam_t_to_world(hand["cam_t"], cam_pose)
        world_positions.append(world_pos)
        
        # Compute gripper state from fingertip distances
        grip = compute_gripper_state(hand)
        gripper_states.append(grip)
        frame_indices.append(fidx)
    
    world_positions = np.array(world_positions)
    gripper_states = np.array(gripper_states)
    
    print(f"   World positions: {len(world_positions)} frames")
    print(f"   World range: x=[{world_positions[:,0].min():.3f},{world_positions[:,0].max():.3f}] "
          f"y=[{world_positions[:,1].min():.3f},{world_positions[:,1].max():.3f}] "
          f"z=[{world_positions[:,2].min():.3f},{world_positions[:,2].max():.3f}]")
    
    # 3. Map to robot workspace
    print("\n🤖 Mapping to Panda robot workspace...")
    robot_positions = np.array([
        world_to_robot(wp, world_positions) for wp in world_positions
    ])
    
    # Smooth trajectory
    robot_positions = smooth_trajectory(robot_positions, window=5)
    
    print(f"   Robot EE range: x=[{robot_positions[:,0].min():.3f},{robot_positions[:,0].max():.3f}] "
          f"y=[{robot_positions[:,1].min():.3f},{robot_positions[:,1].max():.3f}] "
          f"z=[{robot_positions[:,2].min():.3f},{robot_positions[:,2].max():.3f}]")
    
    # 4. Get object positions in robot frame
    print("\n📦 Mapping object positions to robot frame...")
    # Use the detected 3D object positions
    obj_poses = obj_data["poses"]
    
    # Build per-frame object position (interpolate for missing frames)
    obj_frame_map = {}
    for op in obj_poses:
        if op["detected"]:
            obj_frame_map[op["frame"]] = np.array(op["pose_3d"])
    
    # For each hand frame, find nearest object detection
    object_positions = []
    default_obj = np.array([0.5, 0.0, 0.02])  # default: center of table
    
    if obj_frame_map:
        obj_frames = sorted(obj_frame_map.keys())
        for fidx in frame_indices:
            # Find nearest detected object frame
            nearest = min(obj_frames, key=lambda x: abs(x - fidx))
            obj_world = obj_frame_map[nearest]
            # Map object to robot frame using same transform
            obj_robot = world_to_robot(obj_world, world_positions)
            obj_robot[2] = TABLE_HEIGHT  # object sits on table
            object_positions.append(obj_robot)
    else:
        object_positions = [default_obj] * len(frame_indices)
    
    object_positions = np.array(object_positions)
    
    # 5. Detect episodes
    print("\n🎬 Detecting pick-and-place episodes...")
    episodes = detect_episodes(gripper_states)
    print(f"   Found {len(episodes)} episode(s)")
    
    # 6. Build episode data
    episodes_data = []
    for ep_idx, (start, end) in enumerate(episodes):
        ep = {
            "ee_positions": robot_positions[start:end+1],
            "gripper_states": gripper_states[start:end+1],
            "object_positions": object_positions[start:end+1],
        }
        episodes_data.append(ep)
        print(f"   Episode {ep_idx}: frames {start}-{end} ({end-start+1} steps)")
    
    # 7. Export
    print("\n💾 Exporting to LeRobot format...")
    h5_path = export_lerobot_hdf5(args.output, episodes_data, fps=args.fps)
    json_path = export_lerobot_json(args.output, episodes_data, fps=args.fps)
    
    # 8. Validation
    print("\n✅ Validation checks:")
    for ep_idx, ep in enumerate(episodes_data):
        ee = ep["ee_positions"]
        gs = ep["gripper_states"]
        
        # Check trajectory is within workspace
        in_ws = (
            (ee[:, 0] >= PANDA_WORKSPACE["x"][0]).all() and
            (ee[:, 0] <= PANDA_WORKSPACE["x"][1]).all() and
            (ee[:, 1] >= PANDA_WORKSPACE["y"][0]).all() and
            (ee[:, 1] <= PANDA_WORKSPACE["y"][1]).all() and
            (ee[:, 2] >= PANDA_WORKSPACE["z"][0]).all() and
            (ee[:, 2] <= PANDA_WORKSPACE["z"][1]).all()
        )
        print(f"   Ep {ep_idx}: workspace bounds ✅" if in_ws else f"   Ep {ep_idx}: workspace bounds ❌")
        
        # Check smoothness (max step size)
        if len(ee) > 1:
            steps = np.linalg.norm(np.diff(ee, axis=0), axis=1)
            print(f"   Ep {ep_idx}: max step={steps.max():.4f}m, mean={steps.mean():.4f}m")
        
        # Check gripper has variation
        grip_range = gs.max() - gs.min()
        print(f"   Ep {ep_idx}: gripper range={grip_range:.3f} {'✅' if grip_range > 0.1 else '⚠️ low variation'}")
    
    # 9. Visualization
    if args.viz:
        print("\n📊 Generating visualization...")
        viz_path = Path(args.output) / "trajectory_viz.png"
        visualize_trajectory(episodes_data, viz_path)
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE")
    print(f"   HDF5:  {h5_path}")
    print(f"   JSON:  {json_path}")
    print(f"   Episodes: {len(episodes_data)}")
    total_steps = sum(len(ep['ee_positions']) for ep in episodes_data)
    print(f"   Total steps: {total_steps}")
    print("=" * 60)


if __name__ == "__main__":
    main()
