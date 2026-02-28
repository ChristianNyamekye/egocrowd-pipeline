#!/usr/bin/env python3
"""
Action Replay V4 — Use the working expert controller, but driven by real trajectory data.
Instead of raw IK, we use the proven panda_mug scene + waypoint controller
and modulate the target positions with the real recording data.
"""
import json, numpy as np, os

FFMPEG = r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

def main():
    import mujoco, cv2
    
    # Load real trajectory data
    ds = json.load(open("pipeline/lerobot_dataset_v2/dataset.json"))
    ep = ds["episodes"][1]  # cleanest episode
    steps = ep["steps"]
    gs_raw = np.array([s["observation"]["gripper_state"] for s in steps])
    ee_raw = np.array([s["observation"]["ee_pos"] for s in steps])
    
    # Normalize trajectory to [0,1]
    ee_min = ee_raw.min(axis=0)
    ee_max = ee_raw.max(axis=0)
    ee_range = ee_max - ee_min
    ee_range[ee_range < 0.0001] = 1.0
    ee_norm = (ee_raw - ee_min) / ee_range
    
    print(f"Episode: {len(steps)} steps, gripper range [{gs_raw.min():.2f}, {gs_raw.max():.2f}]")
    
    # Load scene with position-controlled actuators
    scene = "mujoco_menagerie/franka_emika_panda/_replay_scene.xml"
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    
    # Mug position (center of table, reachable grasp height)
    mug_pos = np.array([0.5, 0.0, 0.12])
    
    # Define waypoint phases driven by real gripper data
    # Find gripper close/open transitions
    close_frame = np.argmax(gs_raw < 0.3)
    if close_frame == 0 and gs_raw[0] >= 0.3:
        close_frame = len(gs_raw) // 3
    
    open_frame = close_frame + np.argmax(gs_raw[close_frame:] > 0.7)
    if open_frame == close_frame:
        open_frame = len(gs_raw) - 1
    
    print(f"Gripper closes at step {close_frame}, opens at step {open_frame}")
    
    # Build a pick-and-place trajectory modulated by real data
    # Phase 1: Approach (start → pre-grasp above mug) — frames 0 to close_frame
    # Phase 2: Grasp (descend + close gripper) — at close_frame
    # Phase 3: Lift (hold closed, lift up) — close_frame to open_frame
    # Phase 4: Place/release — open_frame onwards
    
    n_steps = len(steps)
    ee_targets = np.zeros((n_steps, 3))
    grip_targets = np.zeros(n_steps)
    
    # Use real lateral (XY) variation from recording, map Z to pick-and-place phases
    for t in range(n_steps):
        # Lateral variation from real data (scaled)
        dx = (ee_norm[t, 0] - 0.5) * 0.08  # +-4cm lateral from real movement
        dy = (ee_norm[t, 1] - 0.5) * 0.08
        
        grasp_z = 0.15  # achievable grasp height for Panda
        start_z = 0.40
        lift_z = 0.40
        
        if t < close_frame:
            # Approach: start high, descend toward mug
            progress = t / max(close_frame, 1)
            z = start_z - progress * (start_z - grasp_z)
            ee_targets[t] = [mug_pos[0] + dx, mug_pos[1] + dy, z]
            grip_targets[t] = 1.0  # open
        elif t < close_frame + 5:
            # Grasp: at mug level, closing gripper
            ee_targets[t] = [mug_pos[0], mug_pos[1], grasp_z]
            grip_targets[t] = 0.0  # closed
        elif t < open_frame:
            # Lift: go up while holding
            progress = (t - close_frame - 5) / max(open_frame - close_frame - 5, 1)
            z = grasp_z + progress * (lift_z - grasp_z)
            ee_targets[t] = [mug_pos[0] + dx * 0.5, mug_pos[1] + dy * 0.5, z]
            grip_targets[t] = 0.0  # closed
        else:
            # Release + retract
            progress = (t - open_frame) / max(n_steps - open_frame, 1)
            z = lift_z + progress * 0.05
            ee_targets[t] = [mug_pos[0] + dx, mug_pos[1] + dy, z]
            grip_targets[t] = 1.0  # open
    
    print(f"EE target range: z=[{ee_targets[:,2].min():.3f},{ee_targets[:,2].max():.3f}]")
    
    # ---- Run simulation ----
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Find hand body
    hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_body < 0:
        # Try other names
        for i in range(model.nbody):
            if 'hand' in model.body(i).name or 'link7' in model.body(i).name:
                hand_body = i
                break
    
    # Find mug body (cube in default scene)
    mug_body = -1
    for i in range(model.nbody):
        name = model.body(i).name
        if 'cube' in name or 'box' in name or 'mug' in name or 'object' in name:
            mug_body = i
            break
    
    print(f"Hand body: {hand_body} ({model.body(hand_body).name})")
    if mug_body >= 0:
        print(f"Mug body: {mug_body} ({model.body(mug_body).name})")
        initial_mug_z = data.xpos[mug_body][2]
    else:
        print("No mug body found")
        initial_mug_z = 0
    
    frames = []
    fps = 30
    substeps = 60
    max_lift = 0
    
    # Warm up to initial position
    print("Warming up...")
    for _ in range(1000):
        current_ee = data.xpos[hand_body].copy()
        error = ee_targets[0] - current_ee
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, hand_body)
        nj = min(7, model.nu)
        J = jacp[:, :nj]
        dq = np.linalg.solve(J.T @ J + 0.01 * np.eye(nj), J.T @ error)
        for j in range(nj):
            data.ctrl[j] = data.qpos[j] + dq[j] * 0.5
        if model.nu > 7:
            data.ctrl[7] = 0.04  # open
            if model.nu > 8:
                data.ctrl[8] = 0.04
        mujoco.mj_step(model, data)
    
    warmup_ee = data.xpos[hand_body].copy()
    print(f"After warmup: EE={warmup_ee}, target={ee_targets[0]}, err={np.linalg.norm(warmup_ee - ee_targets[0]):.4f}m")
    
    # Record initial
    renderer.update_scene(data)
    f0 = renderer.render().copy()
    for _ in range(fps):
        frames.append(f0)
    
    # Main replay loop
    for t in range(n_steps):
        target = ee_targets[t]
        grip = grip_targets[t]
        
        for sub in range(substeps):
            current_ee = data.xpos[hand_body].copy()
            error = target - current_ee
            
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, hand_body)
            nj = min(7, model.nu)
            J = jacp[:, :nj]
            dq = np.linalg.solve(J.T @ J + 0.01 * np.eye(nj), J.T @ error)
            
            for j in range(nj):
                data.ctrl[j] = data.qpos[j] + dq[j] * 0.5  # position target = current + IK step
            
            # Gripper
            if model.nu > 7:
                gv = grip * 0.04
                data.ctrl[7] = gv
                if model.nu > 8:
                    data.ctrl[8] = gv
            
            mujoco.mj_step(model, data)
            
            if mug_body >= 0:
                lift = data.xpos[mug_body][2] - initial_mug_z
                max_lift = max(max_lift, lift)
        
        # Record frame
        renderer.update_scene(data)
        frames.append(renderer.render().copy())
        
        if t % 10 == 0:
            ee_now = data.xpos[hand_body]
            err = np.linalg.norm(ee_now - target)
            print(f"  Step {t}/{n_steps}: EE=[{ee_now[0]:.3f},{ee_now[1]:.3f},{ee_now[2]:.3f}] target_z={target[2]:.3f} grip={grip:.1f} err={err:.4f}m lift={max_lift:.4f}m")
    
    # Hold final
    renderer.update_scene(data)
    ff = renderer.render().copy()
    for _ in range(fps * 2):
        frames.append(ff)
    
    print(f"\nMax mug lift: {max_lift:.4f}m")
    print(f"Total frames: {len(frames)}")
    
    # Write
    raw = "pipeline/lerobot_dataset_v2/replay_v4_raw.mp4"
    final = "pipeline/lerobot_dataset_v2/replay_v4.mp4"
    
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    
    os.system(f'"{FFMPEG}" -y -i "{raw}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{final}" 2>nul')
    
    if os.path.exists(final) and os.path.getsize(final) > 1000:
        print(f"Saved: {final} ({os.path.getsize(final)/1024:.0f} KB)")
    else:
        print(f"Saved: {raw} ({os.path.getsize(raw)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
