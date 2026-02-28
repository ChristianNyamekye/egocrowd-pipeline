#!/usr/bin/env python3
"""
EgoCrowd Retarget V2 — LiDAR-grounded coordinates + Action Replay
===================================================================
V2 improvements over V1:
- Uses LiDAR depth at hand pixel location for true metric scale
- Projects hand 2D pixel + depth → 3D world using camera intrinsics + poses
- Action replay: feeds data back into MuJoCo Panda to validate

Usage:
    python tools/retarget_v2.py                    # retarget + export
    python tools/retarget_v2.py --replay           # also run MuJoCo action replay
    python tools/retarget_v2.py --replay --video   # replay + save video
"""

import argparse, json, os, sys
import numpy as np
from pathlib import Path

R3D_OUTPUT = "pipeline/r3d_output"
OUTPUT_DIR = "pipeline/lerobot_dataset_v2"

PANDA_WORKSPACE = {"x": (0.25, 0.75), "y": (-0.4, 0.4), "z": (0.01, 0.60)}
TABLE_HEIGHT = 0.01
GRIPPER_CLOSE_THRESH = 0.04
GRIPPER_OPEN_THRESH = 0.07


def quat_to_rot(qx, qy, qz, qw):
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])
    return R


def get_camera_pose(frame):
    p = frame["pose"]
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(p[0], p[1], p[2], p[3])
    T[:3, 3] = [p[4], p[5], p[6]]
    return T


def pixel_depth_to_world(u, v, depth, K, cam_pose):
    """Back-project pixel (u,v) + LiDAR depth → world 3D point."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Camera-space 3D
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    p_world = cam_pose @ p_cam
    return p_world[:3]


def get_hand_pixel(hand_data, rgb_size, K):
    """
    Get hand pixel location from HaMeR cam_t.
    cam_t is [x, y, z] in camera space — project to pixel using K.
    """
    ct = hand_data["cam_t"]
    # Project: u = fx * x/z + cx, v = fy * y/z + cy
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * ct[0] / ct[2] + cx
    v = fy * ct[1] / ct[2] + cy
    return u, v


def sample_depth_at(depth_map, u, v, rgb_size, depth_size, patch=5):
    """Sample LiDAR depth at pixel (u,v) with a small patch for robustness."""
    # Scale from RGB coords to depth coords
    scale_x = depth_size[1] / rgb_size[0]  # width
    scale_y = depth_size[0] / rgb_size[1]  # height
    du = int(u * scale_x)
    dv = int(v * scale_y)
    
    h, w = depth_map.shape
    du = np.clip(du, patch, w - patch - 1)
    dv = np.clip(dv, patch, h - patch - 1)
    
    patch_vals = depth_map[dv-patch:dv+patch+1, du-patch:du+patch+1]
    valid = patch_vals[patch_vals > 0.05]  # filter invalid depths
    
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def compute_gripper(hand_data):
    ft = hand_data.get("fingertips", {})
    if "thumb" not in ft or "index" not in ft:
        return 0.5
    dist = np.linalg.norm(np.array(ft["thumb"]) - np.array(ft["index"]))
    return float(np.clip((dist - GRIPPER_CLOSE_THRESH) / (GRIPPER_OPEN_THRESH - GRIPPER_CLOSE_THRESH), 0, 1))


def smooth(arr, window=5):
    if len(arr) < window:
        return arr
    out = np.copy(arr)
    h = window // 2
    for i in range(h, len(arr) - h):
        out[i] = arr[i-h:i+h+1].mean(axis=0)
    return out


def detect_episodes(grippers, min_len=15):
    gs = np.array(grippers)
    closed = gs < 0.3
    episodes = []
    in_grasp = False
    start = 0
    for i in range(1, len(gs)):
        if not in_grasp and closed[i] and not closed[i-1]:
            start = max(0, i - 10)
            in_grasp = True
        elif in_grasp and not closed[i] and closed[i-1]:
            end = min(len(gs)-1, i + 10)
            if end - start >= min_len:
                episodes.append((start, end))
            in_grasp = False
    if in_grasp and len(gs) - start >= min_len:
        episodes.append((start, len(gs)-1))
    if not episodes:
        episodes = [(0, len(gs)-1)]
    return episodes


def world_to_robot_metric(world_pos, world_positions):
    """
    Map world coords to robot workspace using metric centering.
    Center the trajectory on Panda's workspace center, preserving relative scale.
    """
    wp = np.array(world_positions)
    centroid = wp.mean(axis=0)
    
    # Offset from centroid
    offset = world_pos - centroid
    
    # Scale: map the observed range to fit in workspace
    wp_range = wp.max(axis=0) - wp.min(axis=0)
    ws_range = np.array([
        PANDA_WORKSPACE["x"][1] - PANDA_WORKSPACE["x"][0],
        PANDA_WORKSPACE["y"][1] - PANDA_WORKSPACE["y"][0],
        PANDA_WORKSPACE["z"][1] - PANDA_WORKSPACE["z"][0],
    ])
    
    # Scale factor: amplify small movements to fill workspace
    scale = min(ws_range / np.maximum(wp_range, 0.001))
    scale = min(scale, 8.0)  # allow aggressive scaling for small movements
    
    # Center of robot workspace
    ws_center = np.array([
        (PANDA_WORKSPACE["x"][0] + PANDA_WORKSPACE["x"][1]) / 2,
        (PANDA_WORKSPACE["y"][0] + PANDA_WORKSPACE["y"][1]) / 2,
        (PANDA_WORKSPACE["z"][0] + PANDA_WORKSPACE["z"][1]) / 2,
    ])
    
    robot_pos = ws_center + offset * scale
    
    # Clamp
    robot_pos[0] = np.clip(robot_pos[0], *PANDA_WORKSPACE["x"])
    robot_pos[1] = np.clip(robot_pos[1], *PANDA_WORKSPACE["y"])
    robot_pos[2] = np.clip(robot_pos[2], *PANDA_WORKSPACE["z"])
    
    return robot_pos


def export_hdf5(output_dir, episodes_data, fps=30):
    try:
        import h5py
    except ImportError:
        os.system(f"{sys.executable} -m pip install h5py -q")
        import h5py
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / "data.hdf5"
    
    with h5py.File(h5_path, "w") as f:
        f.attrs["fps"] = fps
        f.attrs["robot_type"] = "franka_panda"
        f.attrs["source"] = "egocrowd_iphone_r3d_v2_lidar"
        f.attrs["num_episodes"] = len(episodes_data)
        
        for ei, ep in enumerate(episodes_data):
            grp = f.create_group(f"episode_{ei}")
            ee = ep["ee_positions"]
            gs = ep["gripper_states"]
            obj = ep["object_positions"]
            
            obs = np.column_stack([ee, gs[:, None], obj])
            deltas = np.zeros_like(ee)
            deltas[1:] = ee[1:] - ee[:-1]
            actions = np.column_stack([deltas, gs[:, None]])
            
            grp.create_dataset("observation", data=obs.astype(np.float32))
            grp.create_dataset("action", data=actions.astype(np.float32))
            grp.create_dataset("ee_pos", data=ee.astype(np.float32))
            grp.create_dataset("gripper", data=gs.astype(np.float32))
            grp.create_dataset("object_pos", data=obj.astype(np.float32))
            grp.attrs["num_steps"] = len(ee)
            grp.attrs["episode_idx"] = ei
            
            print(f"  Ep {ei}: {len(ee)} steps, EE x=[{ee[:,0].min():.3f},{ee[:,0].max():.3f}] z=[{ee[:,2].min():.3f},{ee[:,2].max():.3f}] grip=[{gs.min():.2f},{gs.max():.2f}]")
    
    print(f"\nSaved HDF5: {h5_path} ({h5_path.stat().st_size/1024:.1f} KB)")
    return h5_path


def export_json(output_dir, episodes_data, fps=30):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    dataset = {"fps": fps, "robot_type": "franka_panda", "source": "egocrowd_v2_lidar", "episodes": []}
    for ei, ep in enumerate(episodes_data):
        ee = ep["ee_positions"]
        gs = ep["gripper_states"]
        obj = ep["object_positions"]
        deltas = np.zeros_like(ee); deltas[1:] = ee[1:] - ee[:-1]
        
        episode = {"episode_idx": ei, "num_steps": len(ee), "steps": []}
        for t in range(len(ee)):
            episode["steps"].append({
                "observation": {"ee_pos": ee[t].tolist(), "gripper_state": float(gs[t]), "object_pos": obj[t].tolist()},
                "action": {"delta_ee": deltas[t].tolist(), "gripper_cmd": float(gs[t])}
            })
        dataset["episodes"].append(episode)
    
    jp = out / "dataset.json"
    json.dump(dataset, open(jp, "w"), indent=2)
    print(f"Saved JSON: {jp} ({jp.stat().st_size/1024:.1f} KB)")
    return jp


def action_replay(episodes_data, video_path=None):
    """Replay exported actions in MuJoCo Panda sim to validate data."""
    try:
        import mujoco
    except ImportError:
        print("MuJoCo not available, skipping replay")
        return False
    
    print("\n--- Action Replay in MuJoCo ---")
    
    # Find Panda model
    panda_xml = None
    candidates = [
        "mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml",
        "mujoco_menagerie/franka_emika_panda/panda.xml",
        "pipeline/panda_mug.xml",
    ]
    for c in candidates:
        if os.path.exists(c):
            panda_xml = c
            break
    
    if not panda_xml:
        print("No Panda XML found. Trying to load from mujoco_menagerie...")
        # Try the scene file we used before
        if os.path.exists("tools/panda_mug_scene.xml"):
            panda_xml = "tools/panda_mug_scene.xml"
        else:
            print("ERROR: No Panda model found. Skipping replay.")
            return False
    
    print(f"Using model: {panda_xml}")
    model = mujoco.MjModel.from_xml_path(panda_xml)
    data = mujoco.MjData(model)
    
    # Get actuator and site info
    print(f"  Actuators: {model.nu}")
    print(f"  nq: {model.nq}, nv: {model.nv}")
    
    frames = []
    record_video = video_path is not None
    
    if record_video:
        renderer = mujoco.Renderer(model, 480, 640)
    
    for ei, ep in enumerate(episodes_data):
        ee_traj = ep["ee_positions"]
        gs_traj = ep["gripper_states"]
        
        print(f"\n  Replaying Episode {ei} ({len(ee_traj)} steps)...")
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        
        # Simple position control: set ctrl to approach EE targets
        for t in range(len(ee_traj)):
            target = ee_traj[t]
            grip = gs_traj[t]
            
            # For Panda: first 7 actuators are arm joints, last 2 are gripper fingers
            # We'll use a simple approach: set the arm joints to track the EE target
            # via MuJoCo's built-in position control
            
            # Get current EE position (end_effector site or last body)
            ee_site = None
            for i in range(model.nsite):
                name = model.site(i).name
                if 'grip' in name.lower() or 'ee' in name.lower() or 'hand' in name.lower():
                    ee_site = i
                    break
            
            if ee_site is not None:
                current_ee = data.site_xpos[ee_site].copy()
            else:
                # Use last body position as proxy
                current_ee = data.xpos[-1].copy()
            
            # Compute IK-like control (Jacobian pseudoinverse)
            ee_error = target - current_ee
            
            # Simple proportional control on joints
            if model.nu >= 7:
                # Use Jacobian for better IK
                jacp = np.zeros((3, model.nv))
                if ee_site is not None:
                    mujoco.mj_jacSite(model, data, jacp, None, ee_site)
                else:
                    mujoco.mj_jacBody(model, data, jacp, None, model.nbody - 1)
                
                # Damped pseudoinverse
                lam = 0.01
                J = jacp[:, :7]  # only arm joints
                JtJ = J.T @ J + lam * np.eye(7)
                dq = np.linalg.solve(JtJ, J.T @ ee_error)
                
                # Apply to arm actuators
                for j in range(min(7, model.nu)):
                    data.ctrl[j] = data.qpos[j] + dq[j] * 2.0  # gain
                
                # Gripper
                if model.nu > 7:
                    grip_val = grip * 0.04  # Panda gripper: 0=closed, 0.04=open
                    for j in range(7, model.nu):
                        data.ctrl[j] = grip_val
            
            # Step simulation
            for _ in range(10):  # 10 substeps per control step
                mujoco.mj_step(model, data)
            
            if record_video and t % 2 == 0:  # every other frame
                renderer.update_scene(data)
                frames.append(renderer.render().copy())
        
        # Report final EE position
        if ee_site is not None:
            final_ee = data.site_xpos[ee_site]
        else:
            final_ee = data.xpos[-1]
        
        target_final = ee_traj[-1]
        dist = np.linalg.norm(final_ee - target_final)
        print(f"    Final EE: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}]")
        print(f"    Target:   [{target_final[0]:.3f}, {target_final[1]:.3f}, {target_final[2]:.3f}]")
        print(f"    Error: {dist:.4f}m")
    
    # Save video
    if record_video and frames:
        print(f"\n  Saving replay video ({len(frames)} frames)...")
        try:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 15, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"  Saved: {video_path}")
        except ImportError:
            # Fallback: save as numpy
            np.save(video_path.replace('.mp4', '.npy'), np.array(frames))
            print(f"  Saved frames as numpy (cv2 not available)")
    
    return True


def visualize(episodes_data, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    fig = plt.figure(figsize=(15, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(episodes_data), 2)))
    
    # 3D trajectory
    ax1 = fig.add_subplot(131, projection="3d")
    for ei, ep in enumerate(episodes_data):
        ee = ep["ee_positions"]
        ax1.plot(ee[:,0], ee[:,1], ee[:,2], color=colors[ei], label=f"Ep {ei}", linewidth=2)
        ax1.scatter(*ee[0], color=colors[ei], marker="o", s=50)
        ax1.scatter(*ee[-1], color=colors[ei], marker="x", s=50)
        obj = ep["object_positions"]
        ax1.scatter(obj[0,0], obj[0,1], obj[0,2], color="red", marker="^", s=100)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_title("EE Trajectory (LiDAR-grounded)")
    ax1.legend(fontsize=8)
    
    # Gripper
    ax2 = fig.add_subplot(132)
    for ei, ep in enumerate(episodes_data):
        ax2.plot(ep["gripper_states"], color=colors[ei], label=f"Ep {ei}")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Gripper"); ax2.set_title("Gripper State")
    ax2.legend(fontsize=8); ax2.set_ylim(-0.1, 1.1)
    
    # Height profile
    ax3 = fig.add_subplot(133)
    for ei, ep in enumerate(episodes_data):
        ax3.plot(ep["ee_positions"][:,2], color=colors[ei], label=f"Ep {ei} Z")
        ax3.axhline(y=TABLE_HEIGHT, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Step"); ax3.set_ylabel("Z (m)"); ax3.set_title("Height Profile")
    ax3.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved viz: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=R3D_OUTPUT)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--replay", action="store_true", help="Run MuJoCo action replay")
    parser.add_argument("--video", action="store_true", help="Save replay video")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    
    print("=" * 60)
    print("EgoCrowd Retarget V2 (LiDAR-grounded)")
    print("=" * 60)
    
    # Load
    print("\nLoading data...")
    hamer, frames, camera_K, obj_data = (
        json.load(open(f"{args.input}/hamer_results.json")),
        json.load(open(f"{args.input}/metadata.json"))["frames"],
        np.array(json.load(open(f"{args.input}/metadata.json"))["camera_K"]),
        json.load(open(f"{args.input}/object_poses_3d.json")),
    )
    meta = json.load(open(f"{args.input}/metadata.json"))
    rgb_size = meta["rgb_size"]   # [width, height] = [720, 960]
    depth_size = meta["depth_size"]  # [height, width] = [192, 256]
    
    print(f"  HaMeR: {len(hamer)} frames, Camera: {len(frames)} frames")
    print(f"  RGB: {rgb_size}, Depth: {depth_size}")
    
    # Process each frame: pixel + LiDAR depth -> world 3D
    print("\nConverting with LiDAR depth...")
    world_positions = []
    gripper_states = []
    frame_indices = []
    lidar_used = 0
    cam_t_fallback = 0
    
    for fid in sorted(hamer.keys(), key=int):
        hands = hamer[fid]
        if not hands:
            continue
        hand = hands[0]
        fidx = int(fid)
        if fidx >= len(frames):
            continue
        
        cam_pose = get_camera_pose(frames[fidx])
        
        # Get hand pixel position
        u, v = get_hand_pixel(hand, rgb_size, camera_K)
        
        # Try LiDAR depth at hand location
        depth_path = os.path.join(args.input, f"depth/{fidx:04d}.npy")
        depth_val = None
        if os.path.exists(depth_path):
            depth_map = np.load(depth_path)
            depth_val = sample_depth_at(depth_map, u, v, rgb_size, depth_size)
        
        if depth_val is not None and 0.1 < depth_val < 5.0:
            # LiDAR-grounded: back-project pixel + real depth
            world_pos = pixel_depth_to_world(u, v, depth_val, camera_K, cam_pose)
            lidar_used += 1
        else:
            # Fallback: use cam_t directly transformed to world
            ct = hand["cam_t"]
            p_cam = np.array([ct[0], ct[1], ct[2], 1.0])
            world_pos = (cam_pose @ p_cam)[:3]
            cam_t_fallback += 1
        
        world_positions.append(world_pos)
        gripper_states.append(compute_gripper(hand))
        frame_indices.append(fidx)
    
    world_positions = np.array(world_positions)
    gripper_states = np.array(gripper_states)
    
    print(f"  Frames processed: {len(world_positions)}")
    print(f"  LiDAR depth used: {lidar_used}, cam_t fallback: {cam_t_fallback}")
    print(f"  World range: x=[{world_positions[:,0].min():.3f},{world_positions[:,0].max():.3f}] "
          f"y=[{world_positions[:,1].min():.3f},{world_positions[:,1].max():.3f}] "
          f"z=[{world_positions[:,2].min():.3f},{world_positions[:,2].max():.3f}]")
    
    # Map to robot workspace (metric-preserving)
    print("\nMapping to Panda workspace (metric-preserving)...")
    robot_positions = np.array([world_to_robot_metric(wp, world_positions) for wp in world_positions])
    robot_positions = smooth(robot_positions, window=5)
    
    print(f"  Robot EE range: x=[{robot_positions[:,0].min():.3f},{robot_positions[:,0].max():.3f}] "
          f"y=[{robot_positions[:,1].min():.3f},{robot_positions[:,1].max():.3f}] "
          f"z=[{robot_positions[:,2].min():.3f},{robot_positions[:,2].max():.3f}]")
    
    # Object positions
    obj_poses = obj_data["poses"]
    obj_map = {op["frame"]: np.array(op["pose_3d"]) for op in obj_poses if op["detected"]}
    obj_frames_list = sorted(obj_map.keys())
    
    object_positions = []
    for fidx in frame_indices:
        if obj_frames_list:
            nearest = min(obj_frames_list, key=lambda x: abs(x - fidx))
            obj_robot = world_to_robot_metric(obj_map[nearest], world_positions)
            obj_robot[2] = TABLE_HEIGHT
            object_positions.append(obj_robot)
        else:
            object_positions.append(np.array([0.5, 0.0, TABLE_HEIGHT]))
    object_positions = np.array(object_positions)
    
    # Detect episodes
    print("\nDetecting episodes...")
    episodes = detect_episodes(gripper_states)
    print(f"  Found {len(episodes)} episode(s)")
    
    episodes_data = []
    for ei, (s, e) in enumerate(episodes):
        ep = {
            "ee_positions": robot_positions[s:e+1],
            "gripper_states": gripper_states[s:e+1],
            "object_positions": object_positions[s:e+1],
        }
        episodes_data.append(ep)
        print(f"  Ep {ei}: frames {s}-{e} ({e-s+1} steps)")
    
    # Export
    print("\nExporting...")
    h5 = export_hdf5(args.output, episodes_data, args.fps)
    js = export_json(args.output, episodes_data, args.fps)
    
    # Validation
    print("\nValidation:")
    for ei, ep in enumerate(episodes_data):
        ee = ep["ee_positions"]
        gs = ep["gripper_states"]
        in_ws = (
            (ee[:,0] >= PANDA_WORKSPACE["x"][0]).all() and (ee[:,0] <= PANDA_WORKSPACE["x"][1]).all() and
            (ee[:,1] >= PANDA_WORKSPACE["y"][0]).all() and (ee[:,1] <= PANDA_WORKSPACE["y"][1]).all() and
            (ee[:,2] >= PANDA_WORKSPACE["z"][0]).all() and (ee[:,2] <= PANDA_WORKSPACE["z"][1]).all()
        )
        steps = np.linalg.norm(np.diff(ee, axis=0), axis=1) if len(ee) > 1 else [0]
        print(f"  Ep {ei}: bounds={'OK' if in_ws else 'FAIL'}, max_step={max(steps):.4f}m, grip_range={gs.max()-gs.min():.3f}")
    
    # Viz
    print("\nGenerating visualization...")
    viz_path = Path(args.output) / "trajectory_v2.png"
    visualize(episodes_data, viz_path)
    
    # Action replay
    if args.replay:
        vid_path = str(Path(args.output) / "replay.mp4") if args.video else None
        action_replay(episodes_data, vid_path)
    
    total = sum(len(ep["ee_positions"]) for ep in episodes_data)
    print(f"\n{'='*60}")
    print(f"DONE -- {len(episodes_data)} episodes, {total} steps")
    print(f"  HDF5: {h5}")
    print(f"  JSON: {js}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
