#!/usr/bin/env python3
"""Render H1 + Shadow Hand with dex-retargeted joint angles from stack2 capture.

Uses:
- HumanoidBench H1+Shadow Hand model (h1hand_pos_cube.xml)
- Retargeted Shadow Hand joints from retargeted/stack2_shadow_hand.json
- Wrist trajectory from wrist_trajectories/stack2_calibrated.json for arm IK

Output: pipeline/sim_renders/stack2_h1_retargeted.mp4
"""
import mujoco, numpy as np, json, sys
from pathlib import Path
from PIL import Image, ImageDraw
import shutil, subprocess

ASSETS = Path(__file__).parent / "humanoidbench_assets"
CALIB = Path(__file__).parent / "wrist_trajectories"
RETARGET = Path(__file__).parent / "retargeted"
OUT = Path(__file__).parent / "sim_renders"
OUT.mkdir(exist_ok=True)

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

def render_task(task_name="stack2"):
    # Load retargeted joint angles
    ret = json.loads((RETARGET / f"{task_name}_shadow_hand.json").read_text())
    joint_names = ret["joint_names"]  # 30 joints
    results = ret["results"]  # per frame
    n = len(results)
    print(f"Retargeted data: {n} frames, {ret['valid_frames']} valid")
    print(f"Joint names: {joint_names}")

    # Load wrist trajectory for arm IK
    calib = json.loads((CALIB / f"{task_name}_calibrated.json").read_text())
    wrist = np.array(calib["wrist_sim"], dtype=float)
    grasping = np.array(calib["grasping"], dtype=float)
    n_traj = len(wrist)
    print(f"Wrist trajectory: {n_traj} frames")

    # Build scene
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

    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"Model: nq={m.nq} nv={m.nv} nu={m.nu}")

    # Build Shadow Hand actuator mapping (retargeted joint name → actuator index)
    rh_act_map = {}
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if an.startswith("rh_A_"):
            # rh_A_FFJ3 → FFJ3
            short = an[5:]  # remove "rh_A_"
            rh_act_map[short] = ai

    print(f"Shadow Hand actuators available: {list(rh_act_map.keys())}")
    print(f"Retargeted joint names: {joint_names}")

    # Map retargeted joints to actuators
    joint_to_act = {}
    for jn in joint_names:
        if jn.startswith("dummy_"):
            continue
        if jn in rh_act_map:
            joint_to_act[jn] = rh_act_map[jn]
    print(f"Matched joints: {len(joint_to_act)} of {len(joint_names)}")

    # Right arm joints for IK
    arm_names = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_yaw"]
    arm_jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_names]
    arm_qa = [m.jnt_qposadr[j] for j in arm_jids]
    arm_da = [m.jnt_dofadr[j] for j in arm_jids]
    arm_lo = np.array([m.jnt_range[j][0] for j in arm_jids])
    arm_hi = np.array([m.jnt_range[j][1] for j in arm_jids])
    arm_acts = {}
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if an in arm_names:
            arm_acts[an] = ai

    rh_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_hand")

    # Freeze body
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
    for _ in range(200):
        freeze()
        mujoco.mj_step(m, d)

    # Place blocks on table after settle
    pick_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free_right_cube_to_rotate")
    sup_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free_left_cube_to_rotate")
    pick_qa_bl = m.jnt_qposadr[pick_jnt]
    sup_qa_bl = m.jnt_qposadr[sup_jnt]
    table_top_z = 0.95
    block_z = table_top_z + 0.025 + 0.001
    d.qpos[pick_qa_bl:pick_qa_bl+3] = [0.35, -0.22, block_z]
    d.qpos[pick_qa_bl+3:pick_qa_bl+7] = [1, 0, 0, 0]
    d.qpos[sup_qa_bl:sup_qa_bl+3] = [0.35, -0.10, block_z]
    d.qpos[sup_qa_bl+3:sup_qa_bl+7] = [1, 0, 0, 0]
    d.qvel[m.jnt_dofadr[pick_jnt]:m.jnt_dofadr[pick_jnt]+6] = 0
    d.qvel[m.jnt_dofadr[sup_jnt]:m.jnt_dofadr[sup_jnt]+6] = 0
    mujoco.mj_forward(m, d)

    # Wrist trajectory mapping
    ws = wrist.copy()
    ws_min, ws_max = ws.min(axis=0), ws.max(axis=0)
    ws_range = ws_max - ws_min
    ws_range[ws_range < 0.01] = 1.0
    target_min = np.array([0.05, -0.35, 0.85])
    target_max = np.array([0.45, -0.05, 1.15])

    def traj_to_target(idx):
        t = (ws[idx % n_traj] - ws_min) / ws_range
        return target_min + t * (target_max - target_min)

    # IK
    jac = np.zeros((3, m.nv))
    jac_r = np.zeros((3, m.nv))

    def ik_right_hand(target, max_iter=80):
        for _ in range(max_iter):
            mujoco.mj_forward(m, d)
            pos = d.site_xpos[rh_sid].copy()
            err = target - pos
            if np.linalg.norm(err) < 0.005:
                break
            mujoco.mj_jacSite(m, d, jac, jac_r, rh_sid)
            J = jac[:, arm_da]
            JJT = J @ J.T + 0.005 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            for i, qa in enumerate(arm_qa):
                d.qpos[qa] = np.clip(d.qpos[qa] + dq[i] * 0.5, arm_lo[i], arm_hi[i])

    # Render
    renderer = mujoco.Renderer(m, 480, 640)
    fd = OUT / f"_{task_name}_h1ret_frames"
    if fd.exists(): shutil.rmtree(fd)
    fd.mkdir()

    # Use frame mapping: n retargeted frames may not equal n_traj
    frame_count = max(n, n_traj)
    last_joints = None

    for i in range(frame_count):
        traj_idx = min(i, n_traj - 1)
        ret_idx = min(i, n - 1)
        
        target = traj_to_target(traj_idx)
        ret_frame = results[ret_idx]
        want_grip = grasping[traj_idx] > 0

        # IK for arm
        ik_right_hand(target)
        for an, ai in arm_acts.items():
            idx = arm_names.index(an)
            d.ctrl[ai] = d.qpos[arm_qa[idx]]

        # Apply retargeted finger joints
        if ret_frame["joints"] is not None:
            joints = ret_frame["joints"]
            for ji, jn in enumerate(joint_names):
                if jn.startswith("dummy_"):
                    continue
                if jn in joint_to_act:
                    ai = joint_to_act[jn]
                    val = np.clip(joints[ji], m.actuator_ctrlrange[ai][0], m.actuator_ctrlrange[ai][1])
                    d.ctrl[ai] = val
            last_joints = joints
        elif last_joints is not None:
            # Repeat last known pose
            for ji, jn in enumerate(joint_names):
                if jn.startswith("dummy_"):
                    continue
                if jn in joint_to_act:
                    ai = joint_to_act[jn]
                    val = np.clip(last_joints[ji], m.actuator_ctrlrange[ai][0], m.actuator_ctrlrange[ai][1])
                    d.ctrl[ai] = val

        # Step
        for _ in range(25):
            freeze()
            mujoco.mj_step(m, d)

        mujoco.mj_forward(m, d)
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_inhand")
        renderer.update_scene(d, camera=cam_id if cam_id >= 0 else -1)
        frame = renderer.render()

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        has_ret = "✓" if ret_frame["joints"] is not None else "○"
        phase = "GRASP" if want_grip else "FREE"
        color = (50, 200, 50) if want_grip else (200, 200, 200)
        draw.text((5, 5), f"F{i}/{frame_count} {has_ret} {phase}", fill=color)
        draw.text((5, 20), "dex-retargeting → H1 Shadow Hand", fill=(180, 180, 180))
        img.save(fd / f"frame_{i:04d}.png")

        if i % 30 == 0:
            rh = d.site_xpos[rh_sid]
            print(f"F{i:03d} grip={want_grip} rh={rh.round(3)} ret={'OK' if ret_frame['joints'] else '--'}")

    renderer.close()

    ff = shutil.which("ffmpeg") or r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    out = OUT / f"{task_name}_h1_retargeted.mp4"
    subprocess.check_call([ff, "-y", "-framerate", "10", "-i", str(fd/"frame_%04d.png"),
                           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast", str(out)])
    print(f"OK {out}")
    return out

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    render_task(task)
