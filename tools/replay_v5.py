#!/usr/bin/env python3
"""
Action Replay V5 — Uses the mjx_single_cube scene (which works with our ACT eval)
and the proven IK controller from eval_v3_act.py, driven by real trajectory data.
"""
import json, numpy as np, os

FFMPEG = r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

def main():
    import mujoco, cv2
    
    # Load real data
    ds = json.load(open("pipeline/lerobot_dataset_v2/dataset.json"))
    ep = ds["episodes"][1]
    steps = ep["steps"]
    gs_raw = np.array([s["observation"]["gripper_state"] for s in steps])
    
    # Find gripper transitions
    close_frame = int(np.argmax(gs_raw < 0.3))
    if close_frame == 0 and gs_raw[0] >= 0.3:
        close_frame = len(gs_raw) // 3
    open_frame = close_frame + int(np.argmax(gs_raw[close_frame:] > 0.7))
    if open_frame <= close_frame:
        open_frame = len(gs_raw) - 1
    
    print(f"Episode: {len(steps)} steps")
    print(f"Gripper: close@{close_frame}, open@{open_frame}")
    
    # Load scene
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Bodies
    hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    
    # Box starts at [0.5, 0, 0.03]
    box_start_z = data.xpos[box_body][2]
    print(f"Hand: body {hand_body} at {data.xpos[hand_body]}")
    print(f"Box: body {box_body} at {data.xpos[box_body]}")
    
    # Build pick-and-place waypoints driven by real gripper timing
    # Total sim time: 10 seconds at 200Hz = 2000 steps
    dt = model.opt.timestep
    total_sim_steps = 2000
    fps_render = 30
    render_every = int(1.0 / fps_render / dt)
    
    # Phase durations from real data (proportional)
    n = len(steps)
    approach_frac = close_frame / n
    hold_frac = 5 / n
    lift_frac = (open_frame - close_frame - 5) / n
    
    approach_steps = int(approach_frac * total_sim_steps)
    hold_steps = int(hold_frac * total_sim_steps)
    lift_steps = int(lift_frac * total_sim_steps)
    release_steps = total_sim_steps - approach_steps - hold_steps - lift_steps
    
    # Mug target position
    mug_xy = data.xpos[box_body][:2].copy()  # [0.5, 0]
    grasp_z = 0.12  # hand body z where fingertips align with box top (fingers are ~6cm below hand center)
    pre_grasp_z = 0.20
    lift_z = 0.25
    
    frames = []
    max_lift = 0
    
    # Render initial
    renderer.update_scene(data)
    f0 = renderer.render().copy()
    for _ in range(fps_render):
        frames.append(f0)
    
    step = 0
    phase = "approach"
    
    for sim_step in range(total_sim_steps):
        # Determine target
        if sim_step < approach_steps:
            progress = sim_step / max(approach_steps, 1)
            target = np.array([
                mug_xy[0],
                mug_xy[1],
                pre_grasp_z + (grasp_z - pre_grasp_z) * progress
            ])
            grip_open = True
        elif sim_step < approach_steps + hold_steps:
            target = np.array([mug_xy[0], mug_xy[1], grasp_z])
            grip_open = False
        elif sim_step < approach_steps + hold_steps + lift_steps:
            progress = (sim_step - approach_steps - hold_steps) / max(lift_steps, 1)
            target = np.array([
                mug_xy[0],
                mug_xy[1],
                grasp_z + (lift_z - grasp_z) * progress
            ])
            grip_open = False
        else:
            target = np.array([mug_xy[0], mug_xy[1], lift_z])
            grip_open = True
        
        # IK control (same approach as eval_v3_act.py)
        current_ee = data.xpos[hand_body].copy()
        ee_error = target - current_ee
        
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, hand_body)
        
        J = jacp[:, :7]
        lam = 0.01
        dq = np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ ee_error)
        
        # Apply joint velocity commands (this scene uses general actuators)
        gain = 2.5
        for j in range(min(7, model.nu)):
            data.ctrl[j] = data.qpos[j] + dq[j] * gain
            # Clamp to actuator limits
            lo, hi = model.actuator_ctrlrange[j]
            data.ctrl[j] = np.clip(data.ctrl[j], lo, hi)
        
        # Gripper (actuator 7)
        if model.nu > 7:
            data.ctrl[7] = 0.04 if grip_open else 0.0
        
        mujoco.mj_step(model, data)
        
        # Track lift
        lift = data.xpos[box_body][2] - box_start_z
        max_lift = max(max_lift, lift)
        
        # Render
        if sim_step % render_every == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())
        
        if sim_step % 200 == 0:
            ee = data.xpos[hand_body]
            err = np.linalg.norm(ee - target)
            print(f"  sim={sim_step}: EE=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}] tgt_z={target[2]:.3f} grip={'open' if grip_open else 'CLOSED'} err={err:.3f}m lift={max_lift:.4f}m")
    
    # Hold final
    renderer.update_scene(data)
    ff = renderer.render().copy()
    for _ in range(fps_render * 2):
        frames.append(ff)
    
    print(f"\nMax box lift: {max_lift:.4f}m")
    print(f"Frames: {len(frames)}")
    
    # Write
    raw = "pipeline/lerobot_dataset_v2/replay_v5_raw.mp4"
    final = "pipeline/lerobot_dataset_v2/replay_v5.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps_render, (w, h))
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
