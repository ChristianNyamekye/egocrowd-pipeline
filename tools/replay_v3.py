#!/usr/bin/env python3
"""
Action Replay V3 — Amplified trajectory, cube at object position, proper gripper interaction.
"""
import json, numpy as np, os, sys

FFMPEG = r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

def main():
    import mujoco, cv2

    # Load dataset
    ds = json.load(open("pipeline/lerobot_dataset_v2/dataset.json"))
    episodes = ds["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    # Use episode 1 (cleanest gripper profile)
    ep = episodes[1]
    steps = ep["steps"]
    
    # Extract EE trajectory and gripper
    ee_traj = np.array([s["observation"]["ee_pos"] for s in steps])
    gs_traj = np.array([s["observation"]["gripper_state"] for s in steps])
    obj_pos = np.array(steps[0]["observation"]["object_pos"])
    
    print(f"Episode 1: {len(steps)} steps")
    print(f"EE range: x=[{ee_traj[:,0].min():.3f},{ee_traj[:,0].max():.3f}] y=[{ee_traj[:,1].min():.3f},{ee_traj[:,1].max():.3f}] z=[{ee_traj[:,2].min():.3f},{ee_traj[:,2].max():.3f}]")
    print(f"Gripper: min={gs_traj.min():.2f} max={gs_traj.max():.2f}")
    print(f"Object: {obj_pos}")

    # ---- Amplify trajectory ----
    # Normalize each axis to [0,1] then map to meaningful robot ranges
    mins = ee_traj.min(axis=0)
    maxs = ee_traj.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 0.0001] = 1.0
    
    normalized = (ee_traj - mins) / ranges  # [0,1] per axis
    
    # Map to pick-and-place friendly ranges
    # X: 0.35-0.65 (forward reach)
    # Y: -0.15 to 0.15 (lateral)
    # Z: 0.04 (table+grasp) to 0.35 (lifted) — this is the key mapping
    amplified = np.zeros_like(ee_traj)
    amplified[:, 0] = 0.35 + normalized[:, 0] * 0.30
    amplified[:, 1] = -0.15 + normalized[:, 1] * 0.30
    amplified[:, 2] = 0.04 + normalized[:, 2] * 0.31  # lowest point near table
    
    # Place mug at the point where gripper first closes
    close_idx = np.argmax(gs_traj < 0.3)  # first frame gripper closes
    if close_idx == 0 and gs_traj[0] >= 0.3:
        close_idx = len(gs_traj) // 3  # fallback
    cube_xy = amplified[close_idx, :2].copy()
    cube_pos = np.array([cube_xy[0], cube_xy[1], 0.02])
    
    print(f"\nAmplified EE range: x=[{amplified[:,0].min():.3f},{amplified[:,0].max():.3f}] z=[{amplified[:,2].min():.3f},{amplified[:,2].max():.3f}]")
    print(f"Cube placed at: {cube_pos}")
    
    # ---- Load MuJoCo scene and reposition mug ----
    scene_path = "mujoco_menagerie/franka_emika_panda/_replay_scene.xml"
    model = mujoco.MjModel.from_xml_path(scene_path)
    # Move mug to grasp point via qpos (freejoint: x,y,z,qw,qx,qy,qz)
    mug_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    mug_qadr = model.jnt_qposadr[mug_jnt]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    
    # Find bodies
    hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    print(f"Hand body: {hand_body}, Mug body: {mug_body}")
    
    # ---- Run replay ----
    mujoco.mj_resetData(model, data)
    # Place mug at grasp position
    data.qpos[mug_qadr:mug_qadr+3] = [cube_pos[0], cube_pos[1], 0.04]
    data.qpos[mug_qadr+3] = 1.0  # qw
    mujoco.mj_forward(model, data)
    
    # Pre-position arm to first target (warm-up IK for 500 steps)
    print("  Warming up IK to first target...")
    first_target = amplified[0]
    for _ in range(500):
        current_ee = data.xpos[hand_body].copy()
        ee_error = first_target - current_ee
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, hand_body)
        J = jacp[:, :7]
        dq = np.linalg.solve(J.T @ J + 0.01 * np.eye(7), J.T @ ee_error)
        for j in range(7):
            data.ctrl[j] = data.qpos[j] + dq[j] * 6.0
        data.ctrl[7] = 0.04  # open gripper
        data.ctrl[8] = 0.04
        mujoco.mj_step(model, data)
    print(f"  Warmup done. EE at {data.xpos[hand_body]}, target was {first_target}, error {np.linalg.norm(data.xpos[hand_body] - first_target):.4f}m")
    
    frames = []
    fps = 30
    substeps = 80
    
    # Initial hold
    renderer.update_scene(data, camera=-1)
    init_frame = renderer.render().copy()
    for _ in range(int(fps * 1.0)):
        frames.append(init_frame)
    
    initial_mug_z = data.xpos[mug_body][2]
    max_lift = 0
    
    for t in range(len(amplified)):
        target = amplified[t]
        grip = gs_traj[t]
        
        for sub in range(substeps):
            current_ee = data.xpos[hand_body].copy()
            ee_error = target - current_ee
            
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, hand_body)
            
            # Use first 7 joint columns
            nj = 7
            J = jacp[:, :nj]
            lam = 0.01
            dq = np.linalg.solve(J.T @ J + lam * np.eye(nj), J.T @ ee_error)
            
            for j in range(nj):
                data.ctrl[j] = data.qpos[j] + dq[j] * 6.0
            
            # Gripper: 0=closed, 0.04=open
            grip_val = grip * 0.04
            data.ctrl[7] = grip_val
            data.ctrl[8] = grip_val
            
            mujoco.mj_step(model, data)
            
            # Track mug lift
            mug_z = data.xpos[mug_body][2]
            lift = mug_z - initial_mug_z
            if lift > max_lift:
                max_lift = lift
            
            if sub % (substeps // 2) == 0:
                renderer.update_scene(data, camera=-1)
                frames.append(renderer.render().copy())
        
        renderer.update_scene(data, camera=-1)
        frames.append(renderer.render().copy())
    
    # Final hold
    renderer.update_scene(data, camera=-1)
    final = renderer.render().copy()
    for _ in range(fps * 2):
        frames.append(final)
    
    final_ee = data.xpos[hand_body]
    final_mug = data.xpos[mug_body]
    print(f"\nFinal EE: {final_ee}")
    print(f"Final mug: {final_mug}")
    print(f"Max mug lift: {max_lift:.4f}m")
    print(f"Gripper transitions: {(np.diff(gs_traj > 0.5).astype(int) != 0).sum()}")
    
    # Write video
    raw_path = "pipeline/lerobot_dataset_v2/replay_v3_raw.mp4"
    out_path = "pipeline/lerobot_dataset_v2/replay_v3.mp4"
    
    print(f"\nWriting {len(frames)} frames at {fps}fps...")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    
    # Re-encode H.264
    os.system(f'"{FFMPEG}" -y -i "{raw_path}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{out_path}" 2>nul')
    
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
    else:
        print(f"Saved: {raw_path} ({os.path.getsize(raw_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
