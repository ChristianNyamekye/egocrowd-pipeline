#!/usr/bin/env python3
"""
Action Replay Final — Uses the exact same scene + IK as collect_expert_v3.py.
Drives the pick-and-place timing from real recording data.
"""
import os, json, sys, numpy as np, mujoco

PANDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mujoco_menagerie", "franka_emika_panda")
FFMPEG = r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

SCENE_XML = """
<mujoco model="panda_replay">
  <include file="{panda_dir}/mjx_panda.xml"/>
  <statistic center="0.3 0 0.4" extent="1"/>
  <option timestep="0.002" iterations="50" ls_iterations="20" integrator="implicitfast" gravity="0 0 -9.81">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody>
    <light pos="0.5 -0.5 1.5" dir="0 0.3 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" contype="1" conaffinity="1"/>
    <body name="table" pos="0.5 0 0.125">
      <geom type="box" size="0.25 0.3 0.125" rgba="0.55 0.35 0.18 1" contype="1" conaffinity="1"/>
    </body>
    <body name="mug" pos="0.5 0 0.295">
      <freejoint name="mug_joint"/>
      <geom name="mug_body" type="cylinder" size="0.03 0.04" mass="0.25"
        rgba="0.85 0.15 0.15 1" contype="2" conaffinity="1"
        friction="1.5 0.05 0.01" condim="4" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home"
      qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04   0.5 0 0.295 1 0 0 0"
      ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"/>
  </keyframe>
</mujoco>
""".replace("{panda_dir}", PANDA_DIR)


def cosine(t):
    return 0.5 * (1 - np.cos(np.pi * np.clip(t, 0, 1)))


def solve_ik(model, target, site_id, start_q=None):
    d2 = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d2, 0)
    if start_q is not None:
        d2.qpos[:7] = start_q
    q = d2.qpos[:7].copy()
    for _ in range(2000):
        d2.qpos[:7] = q
        mujoco.mj_forward(model, d2)
        err = target - d2.site_xpos[site_id].copy()
        if np.linalg.norm(err) < 0.003:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, None, site_id)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(3), err)
        q += dq * min(0.5, 0.1 / (np.linalg.norm(dq) + 1e-8))
        for j in range(7):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{j+1}')
            q[j] = np.clip(q[j], *model.jnt_range[jid])
    return q


def main():
    import cv2
    
    # Load real data for timing
    ds = json.load(open("pipeline/lerobot_dataset_v2/dataset.json"))
    ep = ds["episodes"][1]
    gs_raw = np.array([s["observation"]["gripper_state"] for s in ep["steps"]])
    n = len(gs_raw)
    
    close_frame = int(np.argmax(gs_raw < 0.3))
    if close_frame == 0 and gs_raw[0] >= 0.3: close_frame = n // 3
    open_frame = close_frame + int(np.argmax(gs_raw[close_frame:] > 0.7))
    if open_frame <= close_frame: open_frame = n - 1
    
    # Convert frame indices to time fractions
    close_t = close_frame / n  # when gripper closes
    open_t = open_frame / n    # when gripper opens
    
    print(f"Real data: {n} steps, close@{close_frame} ({close_t:.2f}), open@{open_frame} ({open_t:.2f})")
    
    # Build scene
    xml_path = os.path.join(PANDA_DIR, '_replay_final.xml')
    with open(xml_path, 'w') as f:
        f.write(SCENE_XML)
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    
    gs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripper')
    mb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'mug')
    
    # Reset to home keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    mug0 = data.xpos[mb].copy()
    print(f"Mug at: {mug0}")
    print(f"Gripper at: {data.site_xpos[gs]}")
    
    # Compute IK targets
    grasp_t = mug0.copy(); grasp_t[2] += 0.02
    pre_t = grasp_t.copy(); pre_t[2] += 0.12
    lift_t = grasp_t.copy(); lift_t[2] += 0.20
    
    grasp_q = solve_ik(model, grasp_t, gs)
    pre_q = solve_ik(model, pre_t, gs, grasp_q)
    lift_q = solve_ik(model, lift_t, gs, grasp_q)
    home_q = np.array([0, 0.3, 0, -1.57079, 0, 2.0, -0.7853])
    
    # Build phases using real gripper timing
    # Total duration 10s, map real timing
    DURATION = 10.0
    FPS = 30
    EP_LEN = int(DURATION * FPS)
    spf = int(1.0 / (FPS * model.opt.timestep))
    
    O, C = 0.04, 0.0
    
    # Map real timing to phase times
    t_approach_start = 0.0
    t_pre_grasp = DURATION * close_t * 0.6   # arrive at pre-grasp
    t_grasp = DURATION * close_t              # at mug, close gripper
    t_hold = t_grasp + 0.5                    # hold grip
    t_lift_start = t_hold
    t_lift_end = DURATION * open_t            # reach lift height
    t_release = t_lift_end + 0.3
    
    phases = [
        (0.0,           t_pre_grasp,  home_q,  pre_q,   O, O),   # home → pre-grasp
        (t_pre_grasp,   t_grasp,      pre_q,   grasp_q, O, O),   # descend to mug
        (t_grasp,       t_grasp+0.3,  grasp_q, grasp_q, O, C),   # close gripper
        (t_grasp+0.3,   t_hold,       grasp_q, grasp_q, C, C),   # hold
        (t_hold,        t_lift_end,   grasp_q, lift_q,  C, C),   # lift
        (t_lift_end,    t_release,    lift_q,  lift_q,  C, C),   # hold at top
        (t_release,     DURATION,     lift_q,  lift_q,  C, O),   # release
    ]
    
    print(f"Phase timing: pre-grasp@{t_pre_grasp:.1f}s, grasp@{t_grasp:.1f}s, lift@{t_hold:.1f}-{t_lift_end:.1f}s, release@{t_release:.1f}s")
    
    # Reset and run
    mujoco.mj_resetDataKeyframe(model, data, 0)
    frames = []
    
    for fi in range(EP_LEN):
        t = fi / FPS
        
        arm_q = phases[-1][3].copy()
        grip_tgt = phases[-1][5]
        for (t0, t1, q0, q1, g0, g1) in phases:
            if t0 <= t < t1:
                a = cosine((t - t0) / (t1 - t0))
                arm_q = q0 + a * (q1 - q0)
                grip_tgt = g0 + a * (g1 - g0)
                break
        
        data.ctrl[:7] = arm_q
        data.ctrl[7] = grip_tgt
        
        for _ in range(spf):
            mujoco.mj_step(model, data)
        
        # Render
        renderer.update_scene(data)
        frames.append(renderer.render().copy())
        
        if fi % 30 == 0:
            ee = data.site_xpos[gs]
            mugz = data.xpos[mb][2]
            lift = (mugz - mug0[2]) * 100
            print(f"  t={t:.1f}s: EE=[{ee[0]:.3f},{ee[2]:.3f}] grip={grip_tgt:.2f} mug_lift={lift:+.1f}cm")
    
    final_lift = (data.xpos[mb][2] - mug0[2]) * 100
    print(f"\nFinal mug lift: {final_lift:+.1f}cm")
    
    # Write video
    raw = "pipeline/lerobot_dataset_v2/replay_final_raw.mp4"
    final_path = "pipeline/lerobot_dataset_v2/replay_final.mp4"
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    for f in frames:
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    
    os.system(f'"{FFMPEG}" -y -i "{raw}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{final_path}" 2>nul')
    
    out = final_path if os.path.exists(final_path) and os.path.getsize(final_path) > 1000 else raw
    print(f"Saved: {out} ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
