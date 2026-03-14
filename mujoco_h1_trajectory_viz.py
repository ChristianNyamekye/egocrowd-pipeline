#!/usr/bin/env python3
"""H1 Trajectory Visualization: humanoid arm follows retargeted wrist trajectory.

Shows H1 standing at table, right arm tracking the pick-place-stack motion
from stack2_calibrated.json. No grasping physics — pure trajectory replay
to demonstrate data transfer from phone capture to humanoid kinematics.

Output: pipeline/sim_renders/h1_trajectory_viz.mp4
"""
import mujoco, numpy as np, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil, subprocess

ASSETS = Path(__file__).parent / "humanoidbench_assets"
CALIB = Path(__file__).parent / "wrist_trajectories"
OUT = Path(__file__).parent / "sim_renders"
OUT.mkdir(exist_ok=True)

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

def render_task(task_name="stack2"):
    # Load trajectory
    calib = json.loads((CALIB / f"{task_name}_calibrated.json").read_text())
    wrist = np.array(calib["wrist_sim"], dtype=float)
    grasping = np.array(calib["grasping"], dtype=float)
    n = len(wrist)
    print(f"Trajectory: {n} frames")

    # Load H1 cube scene (has table-height blocks)
    m = mujoco.MjModel.from_xml_path(str(ASSETS / "envs" / "h1hand_pos_cube.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"Model: nq={m.nq} nv={m.nv} nu={m.nu}")

    # Add table via spec rebuild
    spec = mujoco.MjSpec.from_file(str(ASSETS / "envs" / "h1hand_pos_cube.xml"))
    
    # Add table
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = np.array([0.45, 0.0, 0.475])
    tg = table.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = np.array([0.30, 0.40, 0.475])
    tg.rgba = np.array([0.55, 0.35, 0.18, 1.0])
    tg.mass = 100.0
    tg.contype = 1
    tg.conaffinity = 1

    # Recompile
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # Right arm joints for IK
    arm_names = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_yaw"]
    arm_jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_names]
    arm_qa = [m.jnt_qposadr[j] for j in arm_jids]
    arm_da = [m.jnt_dofadr[j] for j in arm_jids]
    arm_lo = np.array([m.jnt_range[j][0] for j in arm_jids])
    arm_hi = np.array([m.jnt_range[j][1] for j in arm_jids])

    # Right hand site
    rh_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_hand")
    rh_init = d.site_xpos[rh_sid].copy()
    print(f"Right hand site: {rh_init.round(3)}")

    # Table surface at z=0.95 (body at 0.475, geom half-height 0.475)
    table_top = 0.95
    block_z = table_top + 0.025 + 0.001

    # Blocks - place near right hand reach
    pick_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free_right_cube_to_rotate")
    sup_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free_left_cube_to_rotate")
    pick_qa = m.jnt_qposadr[pick_jnt]
    sup_qa = m.jnt_qposadr[sup_jnt]

    # Place blocks on table in right hand workspace
    # Freeze all joints except right arm
    standing = d.qpos.copy()
    free_set = set(arm_names)
    for j in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if jn.startswith("rh_") or "cube" in jn:
            free_set.add(jn)

    freeze_q, freeze_v = [], []
    for j in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"_j{j}"
        if jn in free_set: continue
        qa, da = m.jnt_qposadr[j], m.jnt_dofadr[j]
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            freeze_q.append((qa, 7)); freeze_v.append((da, 6))
        else:
            freeze_q.append((qa, 1)); freeze_v.append((da, 1))

    def freeze():
        for qa, cnt in freeze_q:
            d.qpos[qa:qa+cnt] = standing[qa:qa+cnt]
        for da, cnt in freeze_v:
            d.qvel[da:da+cnt] = 0

    # Settle
    for _ in range(300):
        freeze()
        mujoco.mj_step(m, d)

    # Place blocks on table AFTER settle (keyframe puts them at default positions)
    d.qpos[pick_qa:pick_qa+3] = [0.35, -0.22, block_z]
    d.qpos[pick_qa+3:pick_qa+7] = [1, 0, 0, 0]
    d.qpos[sup_qa:sup_qa+3] = [0.35, -0.10, block_z]
    d.qpos[sup_qa+3:sup_qa+7] = [1, 0, 0, 0]
    d.qvel[m.jnt_dofadr[pick_jnt]:m.jnt_dofadr[pick_jnt]+6] = 0
    d.qvel[m.jnt_dofadr[sup_jnt]:m.jnt_dofadr[sup_jnt]+6] = 0
    mujoco.mj_forward(m, d)
    print(f"Blocks placed: pick={d.qpos[pick_qa:pick_qa+3].round(3)} sup={d.qpos[sup_qa:sup_qa+3].round(3)}")

    # Map trajectory to workspace
    # Normalize wrist trajectory to right hand's reachable space
    ws = wrist.copy()
    ws_min, ws_max = ws.min(axis=0), ws.max(axis=0)
    ws_range = ws_max - ws_min
    ws_range[ws_range < 0.01] = 1.0

    # Map trajectory to full arm reach range
    # Default site at [0.328, -0.214, 1.077]
    # Use wide range so motion is clearly visible
    target_min = np.array([0.05, -0.35, 0.85])
    target_max = np.array([0.45, -0.05, 1.15])

    def traj_to_target(idx):
        t = (ws[idx] - ws_min) / ws_range
        return target_min + t * (target_max - target_min)

    # IK solver
    jac = np.zeros((3, m.nv))
    jac_r = np.zeros((3, m.nv))

    def ik_right_hand(target, max_iter=100):
        for _ in range(max_iter):
            mujoco.mj_forward(m, d)
            pos = d.site_xpos[rh_sid].copy()
            err = target - pos
            if np.linalg.norm(err) < 0.003:
                break
            mujoco.mj_jacSite(m, d, jac, jac_r, rh_sid)
            J = jac[:, arm_da]
            JJT = J @ J.T + 0.005 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            for i, qa in enumerate(arm_qa):
                d.qpos[qa] = np.clip(d.qpos[qa] + dq[i] * 0.5, arm_lo[i], arm_hi[i])

    # Finger actuators
    rh_acts = {}
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if an.startswith("rh_A_"):
            rh_acts[an] = ai

    closed = {
        "rh_A_THJ5": 1.0, "rh_A_THJ4": 1.2, "rh_A_THJ3": 0.2, "rh_A_THJ2": 0.7, "rh_A_THJ1": 1.5,
        "rh_A_FFJ4": 0.0, "rh_A_FFJ3": 1.5, "rh_A_FFJ0": 3.0,
        "rh_A_MFJ4": 0.0, "rh_A_MFJ3": 1.5, "rh_A_MFJ0": 3.0,
        "rh_A_RFJ4": 0.0, "rh_A_RFJ3": 1.5, "rh_A_RFJ0": 3.0,
        "rh_A_LFJ5": 0.7, "rh_A_LFJ4": 0.0, "rh_A_LFJ3": 1.5, "rh_A_LFJ0": 3.0,
    }

    # Arm actuators
    arm_acts = {}
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if an in arm_names:
            arm_acts[an] = ai

    # Render
    renderer = mujoco.Renderer(m, 480, 640)
    fd = OUT / f"_{task_name}_h1viz_frames"
    if fd.exists(): shutil.rmtree(fd)
    fd.mkdir()

    trail = []  # trajectory trail points

    for i in range(n):
        target = traj_to_target(i)
        want_grip = grasping[i] > 0

        # IK
        ik_right_hand(target)

        # Set arm actuators
        for an, ai in arm_acts.items():
            idx = arm_names.index(an)
            d.ctrl[ai] = d.qpos[arm_qa[idx]]

        # Fingers
        for an, ai in rh_acts.items():
            d.ctrl[ai] = closed.get(an, 0.0) if want_grip else 0.0

        # Step
        for _ in range(25):
            freeze()
            mujoco.mj_step(m, d)

        mujoco.mj_forward(m, d)

        # Render
        # cam_inhand is close and shows the arm well
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_inhand")
        renderer.update_scene(d, camera=cid if cid >= 0 else -1)
        frame = renderer.render()
        
        # Add text overlay
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        phase = "APPROACH" if not want_grip and i < 153 else "GRASP" if want_grip else "RELEASE"
        color = (50, 200, 50) if want_grip else (200, 200, 200)
        draw.text((10, 10), f"Frame {i}/{n}  |  {phase}", fill=color)
        draw.text((10, 25), f"Source: iPhone R3D → retargeted", fill=(180, 180, 180))
        
        img.save(fd / f"frame_{i:04d}.png")

        if i % 20 == 0:
            rh = d.site_xpos[rh_sid]
            pz = d.qpos[2]
            print(f"F{i:03d} phase={phase:8s} pelvis_z={pz:.3f} rh={rh.round(3)} target={target.round(3)}")

    renderer.close()

    # Encode
    ff = shutil.which("ffmpeg") or r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    out = OUT / f"{task_name}_h1_trajectory.mp4"
    subprocess.check_call([ff, "-y", "-framerate", "10", "-i", str(fd/"frame_%04d.png"),
                           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast", str(out)])
    print(f"OK {out}")
    return out

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    render_task(task)
