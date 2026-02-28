#!/usr/bin/env python3
"""
Action Replay V6 — Proper orientation control. Uses 6-DOF IK (position + orientation).
The gripper must point downward to grasp the box.
"""
import json, numpy as np, os

FFMPEG = r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

def main():
    import mujoco, cv2
    
    # Real data for gripper timing
    ds = json.load(open("pipeline/lerobot_dataset_v2/dataset.json"))
    ep = ds["episodes"][1]
    gs_raw = np.array([s["observation"]["gripper_state"] for s in ep["steps"]])
    close_frame = int(np.argmax(gs_raw < 0.3))
    if close_frame == 0 and gs_raw[0] >= 0.3: close_frame = len(gs_raw)//3
    open_frame = close_frame + int(np.argmax(gs_raw[close_frame:] > 0.7))
    if open_frame <= close_frame: open_frame = len(gs_raw)-1
    n = len(gs_raw)
    
    print(f"Real data: {n} steps, close@{close_frame}, open@{open_frame}")
    
    # Scene
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    hand_body = 9
    box_body = 12
    box_z0 = data.xpos[box_body][2]
    box_xy = data.xpos[box_body][:2].copy()
    
    # Desired orientation: gripper pointing straight down
    # In home config, hand rotation matrix
    R_home = data.xmat[hand_body].reshape(3,3).copy()
    print(f"Home hand R:\n{R_home}")
    # We want Z-axis of hand to point down (0,0,-1)
    # and the fingers to open along Y-axis
    R_down = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=float)
    
    # Phase timing
    total_steps = 3000
    approach_end = int(close_frame / n * total_steps)
    hold_end = approach_end + int(5/n * total_steps)
    lift_end = hold_end + int((open_frame - close_frame - 5)/n * total_steps)
    
    grasp_z = 0.06   # fingertips at box top when hand oriented downward
    pre_z = 0.25
    lift_z = 0.30
    fps = 30
    dt = model.opt.timestep
    render_every = max(1, int(1.0/fps/dt))
    
    frames = []
    max_lift = 0.0
    
    # Initial frame
    renderer.update_scene(data)
    for _ in range(fps): frames.append(renderer.render().copy())
    
    for step in range(total_steps):
        # Target position
        if step < approach_end:
            p = step / max(approach_end, 1)
            tgt = np.array([box_xy[0], box_xy[1], pre_z + (grasp_z - pre_z)*p])
            grip_open = True
        elif step < hold_end:
            tgt = np.array([box_xy[0], box_xy[1], grasp_z])
            grip_open = False
        elif step < lift_end:
            p = (step - hold_end) / max(lift_end - hold_end, 1)
            tgt = np.array([box_xy[0], box_xy[1], grasp_z + (lift_z - grasp_z)*p])
            grip_open = False
        else:
            tgt = np.array([box_xy[0], box_xy[1], lift_z])
            grip_open = True
        
        # 6-DOF IK: position + orientation
        ee_pos = data.xpos[hand_body].copy()
        ee_mat = data.xmat[hand_body].reshape(3,3)
        
        # Position error
        pos_err = tgt - ee_pos
        
        # Orientation error (rotation error as axis*angle)
        R_err = R_down @ ee_mat.T
        # Extract axis-angle from rotation matrix
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        if angle < 1e-6:
            ori_err = np.zeros(3)
        else:
            ori_err = angle / (2 * np.sin(angle)) * np.array([
                R_err[2,1] - R_err[1,2],
                R_err[0,2] - R_err[2,0],
                R_err[1,0] - R_err[0,1]
            ])
        
        # Combined 6-DOF Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_body)
        
        J_pos = jacp[:, :7]
        J_rot = jacr[:, :7]
        
        # Stack: [pos_err; ori_err] = [J_pos; J_rot] * dq
        err_6d = np.concatenate([pos_err, ori_err * 0.3])  # weight orientation less
        J_6d = np.vstack([J_pos, J_rot[:, :7]])
        
        lam = 0.01
        dq = np.linalg.solve(J_6d.T @ J_6d + lam * np.eye(7), J_6d.T @ err_6d)
        
        gain = 2.0
        for j in range(7):
            data.ctrl[j] = data.qpos[j] + dq[j] * gain
            data.ctrl[j] = np.clip(data.ctrl[j], model.actuator_ctrlrange[j][0], model.actuator_ctrlrange[j][1])
        
        data.ctrl[7] = 0.04 if grip_open else 0.0
        
        mujoco.mj_step(model, data)
        
        lift = data.xpos[box_body][2] - box_z0
        max_lift = max(max_lift, lift)
        
        if step % render_every == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())
        
        if step % 300 == 0:
            ee = data.xpos[hand_body]
            lf = data.xpos[10]  # left finger
            print(f"  s={step}: hand=[{ee[0]:.3f},{ee[2]:.3f}] finger_z={lf[2]:.3f} tgt_z={tgt[2]:.3f} grip={'O' if grip_open else 'C'} lift={max_lift:.4f}m")
    
    # Hold
    renderer.update_scene(data)
    for _ in range(fps*2): frames.append(renderer.render().copy())
    
    print(f"\nMax lift: {max_lift:.4f}m")
    
    # Write
    raw = "pipeline/lerobot_dataset_v2/replay_v6_raw.mp4"
    final = "pipeline/lerobot_dataset_v2/replay_v6.mp4"
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames: wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    os.system(f'"{FFMPEG}" -y -i "{raw}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{final}" 2>nul')
    sz = os.path.getsize(final) if os.path.exists(final) else os.path.getsize(raw)
    print(f"Saved: {final if os.path.exists(final) else raw} ({sz/1024:.0f} KB)")

if __name__ == "__main__":
    main()
