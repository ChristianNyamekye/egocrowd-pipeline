#!/usr/bin/env python3
"""H1 + Shadow Hand: block stacking with real physics grasping.

Uses H1 humanoid body + Shadow Hand (20+ DOF dexterous fingers with proper
collision geometry). Adhesion actuators on fingertips for physics-based grip.
No welds, no teleporting.

Output: pipeline/sim_renders/stack2_h1_shadow.mp4
"""

import mujoco
import numpy as np
import json, sys, tempfile, subprocess, shutil, os
from pathlib import Path
from PIL import Image
from compose_h1_shadow import build_model, find_body, BLOCK_HALF

from pipeline_config import CALIB_DIR, OUT_DIR, H1_DIR, SHADOW_DIR

RENDER_W, RENDER_H = 640, 480
FPS = 10
TIMESTEP = 0.002
SUBSTEPS = max(10, min(50, int(1.0 / (FPS * TIMESTEP))))

TABLE_HEIGHT = 1.24  # H1 palm at ~1.34 in neutral, table surface just below palm

# H1 right arm joints (4 DOF)
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow",
]
# Shadow Hand wrist joints (2 DOF) - for reaching
WRIST_JOINTS = ["rh_WRJ2", "rh_WRJ1"]

ARM_SEED = np.array([0.0, 0.0, 0.0, 0.0])  # neutral arm
WRIST_SEED = np.array([0.0, 0.0])

# Shadow Hand finger joints for grasping
FINGER_CLOSE_MAP = {
    # joint_name: closed_value
    "rh_FFJ3": 1.2, "rh_FFJ2": 1.0, "rh_FFJ1": 0.5,
    "rh_MFJ3": 1.2, "rh_MFJ2": 1.0, "rh_MFJ1": 0.5,
    "rh_RFJ3": 1.2, "rh_RFJ2": 1.0, "rh_RFJ1": 0.5,
    "rh_LFJ3": 1.2, "rh_LFJ2": 1.0, "rh_LFJ1": 0.5,
    "rh_THJ4": 1.2, "rh_THJ3": 0.5, "rh_THJ2": 0.3, "rh_THJ1": 0.8,
}

# Adhesion bodies (fingertips + thumb tip)
ADHESION_BODIES = [
    "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal",
]
ADHESION_GAIN = 120.0

# Palm center bodies for IK targeting
PALM_BODIES = ["rh_ffdistal", "rh_mfdistal", "rh_thdistal"]

HAND_COLLIDER_SCALE = 1.8  # Shadow Hand already has decent geoms, just need mild upscale
HAND_FRICTION = np.array([5.0, 0.05, 0.001])


def jid(m, name, t):
    return mujoco.mj_name2id(m, t, name)


def palm_center(model, data):
    bid = jid(model, "rh_palm", mujoco.mjtObj.mjOBJ_BODY)
    return data.xpos[bid].copy()


def ik_solve(model, data, target, arm_qa, arm_da, max_iter=150):
    """IK targeting rh_palm body to target position."""
    palm_bid = jid(model, "rh_palm", mujoco.mjtObj.mjOBJ_BODY)
    jac = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        pc = data.xpos[palm_bid].copy()
        err = target - pc
        if np.linalg.norm(err) < 0.005:
            break
        mujoco.mj_jacBody(model, data, jac, jac_r, palm_bid)
        J = jac[:, arm_da]
        JJT = J @ J.T + 0.01 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)
        # Clamp step to avoid divergence
        step = min(0.5, 0.2 + np.linalg.norm(err) * 0.5)
        for i, a in enumerate(arm_qa):
            data.qpos[a] = np.clip(data.qpos[a] + dq[i] * step,
                                    model.jnt_range[0][0] if model.jnt_limited[0] else -10,
                                    model.jnt_range[0][1] if model.jnt_limited[0] else 10)
    
    mujoco.mj_forward(model, data)
    return np.array([data.qpos[a] for a in arm_qa])


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def render_task(task_name: str):
    calib = json.loads((CALIB_DIR / f"{task_name}_calibrated.json").read_text())
    wrist = np.array(calib["wrist_sim"], dtype=float)
    grasping = np.array(calib["grasping"], dtype=float)
    objects_raw = calib["objects_sim"]

    if isinstance(objects_raw, dict):
        obj_names = list(objects_raw.keys())
    else:
        obj_names = ["block_a", "block_b"] if len(objects_raw) >= 2 else [f"obj_{chr(97+i)}" for i in range(len(objects_raw))]

    # Build model with adhesion actuators via MjSpec
    from compose_h1_shadow import build_model as _bm, find_body as _fb
    import mujoco as mj

    h1 = mj.MjSpec.from_file(str(H1_DIR / "h1.xml"))
    shadow = mj.MjSpec.from_file(str(SHADOW_DIR / "right_hand.xml"))
    
    # Fix naming conflicts
    for m in shadow.materials:
        if m.name == 'black':
            m.name = 'sh_black'
        elif m.name == 'white':
            m.name = 'sh_white'
    
    # Attach shadow hand to H1 right elbow
    elbow = find_body(h1.worldbody, 'right_elbow_link')
    frame = elbow.add_frame()
    frame.pos = np.array([0.28, 0.0, -0.015])
    frame.quat = np.array([0.707107, 0.0, -0.707107, 0.0])
    sh_forearm = find_body(shadow.worldbody, 'rh_forearm')
    frame.attach_body(sh_forearm)
    
    # Options
    h1.option.timestep = TIMESTEP
    h1.option.gravity = np.array([0, 0, -9.81])
    h1.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
    h1.option.impratio = 10.0
    h1.visual.global_.offwidth = RENDER_W
    h1.visual.global_.offheight = RENDER_H
    
    # Textures
    tex = h1.add_texture()
    tex.name = "skybox"
    tex.type = mj.mjtTexture.mjTEXTURE_SKYBOX
    tex.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
    tex.rgb1 = np.array([0.3, 0.5, 0.7])
    tex.rgb2 = np.array([0.0, 0.0, 0.0])
    tex.width = 512
    tex.height = 3072
    
    tex2 = h1.add_texture()
    tex2.name = "groundplane"
    tex2.type = mj.mjtTexture.mjTEXTURE_2D
    tex2.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER
    tex2.rgb1 = np.array([0.15, 0.2, 0.25])
    tex2.rgb2 = np.array([0.1, 0.15, 0.2])
    tex2.width = 300
    tex2.height = 300
    
    mat = h1.add_material()
    mat.name = "groundplane"
    mat.texrepeat = np.array([5.0, 5.0])
    mat.reflectance = 0.2
    
    # Floor
    floor = h1.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mj.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05])
    floor.contype = 1
    floor.conaffinity = 1
    floor.rgba = np.array([0.2, 0.25, 0.3, 1.0])
    
    # Light
    light = h1.worldbody.add_light()
    light.pos = np.array([0.0, 0.0, 2.5])
    light.dir = np.array([0.0, 0.0, -1.0])
    
    light2 = h1.worldbody.add_light()
    light2.pos = np.array([1.5, -0.5, 2.0])
    light2.dir = np.array([-0.5, 0.3, -1.0])
    light2.diffuse = np.array([0.4, 0.4, 0.4])
    
    # Table
    table = h1.worldbody.add_body()
    table.name = "table"
    table.pos = np.array([0.30, 0.0, TABLE_HEIGHT / 2])
    tg = table.add_geom()
    tg.type = mj.mjtGeom.mjGEOM_BOX
    tg.size = np.array([0.4, 0.6, TABLE_HEIGHT / 2])
    tg.rgba = np.array([0.55, 0.35, 0.18, 1.0])
    tg.mass = 50.0
    tg.friction = np.array([1.0, 0.005, 0.0001])
    tg.contype = 1
    tg.conaffinity = 1
    
    # Blocks
    colors = {"block_a": [0.95, 0.15, 0.15, 1.0], "block_b": [0.15, 0.25, 0.95, 1.0]}
    for nm in obj_names:
        block = h1.worldbody.add_body()
        block.name = nm
        bj = block.add_freejoint()
        bj.name = f"{nm}_jnt"
        bg = block.add_geom()
        bg.type = mj.mjtGeom.mjGEOM_BOX
        bg.size = np.array([BLOCK_HALF, BLOCK_HALF, BLOCK_HALF])
        bg.rgba = np.array(colors.get(nm, [0.2, 0.8, 0.3, 1.0]))
        bg.mass = 0.05
        bg.friction = np.array([2.0, 0.01, 0.001])
        bg.contype = 1
        bg.conaffinity = 1
    
    # Camera
    cam = h1.worldbody.add_camera()
    cam.name = "front"
    cam.pos = np.array([1.6, 0.9, 1.5])
    cam.fovy = 50.0
    
    # Adhesion actuators
    for bname in ADHESION_BODIES:
        body = find_body(h1.worldbody, bname)
        if body:
            a = h1.add_actuator()
            a.name = f"adh_{bname}"
            a.trntype = mj.mjtTrn.mjTRN_BODY
            a.dyntype = mj.mjtDyn.mjDYN_NONE
            a.gaintype = mj.mjtGain.mjGAIN_FIXED
            a.biastype = mj.mjtBias.mjBIAS_NONE
            a.gainprm = np.zeros(10)
            a.gainprm[0] = ADHESION_GAIN
            a.ctrlrange = np.array([0.0, 1.0])
            a.target = bname
    
    model = h1.compile()
    data = mj.MjData(model)
    
    print(f"Model: {model.nq}q {model.nv}v {model.nu}u {model.nbody}b {model.ngeom}g")
    
    # Scale up Shadow Hand collision geoms
    hand_geom_ids = set()
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith("rh_") and (model.geom_contype[gi] > 0 or model.geom_conaffinity[gi] > 0):
            # Scale primitive geoms (capsule, cylinder, sphere, box)
            if model.geom_type[gi] != mj.mjtGeom.mjGEOM_MESH:
                model.geom_size[gi] *= HAND_COLLIDER_SCALE
            model.geom_friction[gi] = HAND_FRICTION
            hand_geom_ids.add(gi)
    print(f"  {len(hand_geom_ids)} hand collision geoms scaled {HAND_COLLIDER_SCALE}x")
    
    # Finger actuator gains
    for ai in range(model.nu):
        aname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if aname.startswith("rh_A_"):
            model.actuator_gainprm[ai, 0] *= 5.0
            model.actuator_biasprm[ai, 1] *= 5.0
    
    # Find adhesion actuator indices
    adhesion_act_ids = []
    for ai in range(model.nu):
        aname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if aname.startswith("adh_"):
            adhesion_act_ids.append(ai)
    print(f"  {len(adhesion_act_ids)} adhesion actuators: {adhesion_act_ids}")
    
    # H1 standing pose (no keyframe available)
    # Pelvis freejoint is joint 0, qpos[0:7] = [x, y, z, qw, qx, qy, qz]
    data.qpos[0:3] = [0, 0, 1.0]
    data.qpos[3:7] = [1, 0, 0, 0]
    # All other joints default to 0 (legs straight = standing)
    print(f"  H1 standing pose: pelvis at z=1.0")
    
    # Arm + wrist joint indices
    arm_qa, arm_da = [], []
    for jnm in RIGHT_ARM_JOINTS + WRIST_JOINTS:
        j = jid(model, jnm, mj.mjtObj.mjOBJ_JOINT)
        if j >= 0:
            arm_qa.append(model.jnt_qposadr[j])
            arm_da.append(model.jnt_dofadr[j])
    print(f"  IK joints: {len(arm_qa)} (arm={len(RIGHT_ARM_JOINTS)} + wrist={len(WRIST_JOINTS)})")
    
    # Object setup
    obj_jids = [jid(model, f"{nm}_jnt", mj.mjtObj.mjOBJ_JOINT) for nm in obj_names]
    obj_qadr = [model.jnt_qposadr[j] for j in obj_jids]
    
    block_geom_ids = set()
    for nm in obj_names:
        bid = jid(model, nm, mj.mjtObj.mjOBJ_BODY)
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] == bid:
                block_geom_ids.add(gi)
    
    # Place blocks in reachable workspace
    mj.mj_forward(model, data)
    pc = palm_center(model, data)
    print(f"  Initial palm center: {pc.round(4)}")
    
    # Pick block directly below palm, support block offset
    pick_xy = np.array([pc[0], pc[1]])
    support_xy = pick_xy + np.array([0.12, 0.08])
    
    block_z = TABLE_HEIGHT + BLOCK_HALF + 0.01
    for nm, qa in zip(obj_names, obj_qadr):
        if nm == "block_b":  # pick target
            data.qpos[qa:qa+3] = [pick_xy[0], pick_xy[1], block_z]
        else:  # support
            data.qpos[qa:qa+3] = [support_xy[0], support_xy[1], block_z]
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
    
    # Set arm seed
    arm_seed = np.concatenate([ARM_SEED, WRIST_SEED])
    for qa, v in zip(arm_qa, arm_seed):
        data.qpos[qa] = v
    
    # Freeze setup
    arm_joint_set = set(RIGHT_ARM_JOINTS + WRIST_JOINTS)
    obj_joint_set = set([f"{nm}_jnt" for nm in obj_names])
    # Also keep finger joints free
    finger_joint_set = set()
    for i in range(model.njnt):
        n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if n and n.startswith("rh_") and n not in arm_joint_set:
            finger_joint_set.add(n)
    
    mj.mj_forward(model, data)
    standing_qpos = data.qpos.copy()
    
    freeze_q_slices = []
    freeze_v_slices = []
    for jn in range(model.njnt):
        jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jn) or f"_unnamed_{jn}"
        if jname in arm_joint_set or jname in obj_joint_set or jname in finger_joint_set:
            continue
        qa = model.jnt_qposadr[jn]
        da = model.jnt_dofadr[jn]
        if model.jnt_type[jn] == mj.mjtJoint.mjJNT_FREE:
            freeze_q_slices.append((qa, 7))
            freeze_v_slices.append((da, 6))
        else:
            freeze_q_slices.append((qa, 1))
            freeze_v_slices.append((da, 1))
    
    def freeze_robot():
        for qa, cnt in freeze_q_slices:
            data.qpos[qa:qa+cnt] = standing_qpos[qa:qa+cnt]
        for da, cnt in freeze_v_slices:
            data.qvel[da:da+cnt] = 0
    
    # Settle
    for _ in range(300):
        freeze_robot()
        mj.mj_step(model, data)
    
    for nm, qa in zip(obj_names, obj_qadr):
        bz = data.qpos[qa+2]
        print(f"  {nm} settled at z={bz:.4f} (expected ~{TABLE_HEIGHT + BLOCK_HALF:.3f})")
    
    # Grip window from calibrated trajectory
    n = len(wrist)
    grip_idx = np.where(grasping > 0)[0]
    late = grip_idx[grip_idx >= int(0.6*n)]
    if len(late) < 3:
        late = grip_idx[-max(1, len(grip_idx)//2):] if len(grip_idx) else np.array([n-10])
    ls, le = int(late[0]), int(min(late[-1]+5, n-1))
    win = max(1, le-ls)
    
    pick_nm = "block_b"
    support_nm = "block_a"
    pick_xy0 = pick_xy.copy()
    
    z_grasp = TABLE_HEIGHT + BLOCK_HALF - 0.005
    z_hover = z_grasp + 0.10
    z_lift = z_grasp + 0.12
    
    print(f"\nTrajectory: {n} frames, grip [{ls},{le}], z_grasp={z_grasp:.3f}")
    print(f"  pick_xy={pick_xy0.round(3)} support_xy={support_xy.round(3)}")
    
    # Finger actuator mapping
    finger_act_map = {}
    for ai in range(model.nu):
        aname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if aname.startswith("rh_A_"):
            jname = "rh_" + aname[5:]  # rh_A_FFJ3 -> rh_FFJ3
            finger_act_map[jname] = ai
    
    renderer = mj.Renderer(model, RENDER_H, RENDER_W)
    fd = OUT_DIR / f"_{task_name}_h1shadow_frames"
    if fd.exists():
        shutil.rmtree(fd)
    fd.mkdir()
    
    last_arm_q = arm_seed.copy()
    
    for i in range(n):
        p = 0.0 if (i < ls) else (1.0 if i > le else (i-ls)/win)
        
        # Support position (for stacking target)
        sidx = obj_names.index(support_nm)
        support_pos = data.qpos[obj_qadr[sidx]:obj_qadr[sidx]+3].copy()
        place_xy = support_pos[:2]
        place_z = support_pos[2] + 2*BLOCK_HALF + 0.002
        
        # Trajectory phases
        if i < ls:
            target = np.array([pick_xy0[0], pick_xy0[1], z_hover])
            want_grip = False
        elif i > le:
            target = np.array([place_xy[0], place_xy[1], z_hover])
            want_grip = False
        else:
            if p < 0.12:
                target = np.array([pick_xy0[0], pick_xy0[1], z_hover])
                want_grip = False
            elif p < 0.25:
                t = smoothstep((p-0.12)/0.13)
                target = np.array([pick_xy0[0], pick_xy0[1], z_hover + (z_grasp - z_hover)*t])
                want_grip = t > 0.5
            elif p < 0.42:
                target = np.array([pick_xy0[0], pick_xy0[1], z_grasp])
                want_grip = True
            elif p < 0.55:
                t = smoothstep((p-0.42)/0.13)
                target = np.array([pick_xy0[0], pick_xy0[1], z_grasp + (z_lift - z_grasp)*t])
                want_grip = True
            elif p < 0.75:
                t = smoothstep((p-0.55)/0.20)
                cx = pick_xy0[0] + (place_xy[0]-pick_xy0[0])*t
                cy = pick_xy0[1] + (place_xy[1]-pick_xy0[1])*t
                target = np.array([cx, cy, z_lift])
                want_grip = True
            elif p < 0.88:
                t = smoothstep((p-0.75)/0.13)
                target = np.array([place_xy[0], place_xy[1], z_lift + (place_z-z_lift)*t])
                want_grip = True
            elif p < 0.94:
                target = np.array([place_xy[0], place_xy[1], place_z + 0.02])
                want_grip = False
            else:
                t = smoothstep((p-0.94)/0.06)
                target = np.array([place_xy[0], place_xy[1], place_z + 0.02 + (z_hover-place_z)*t])
                want_grip = False
        
        # IK
        for qa, v in zip(arm_qa, last_arm_q):
            data.qpos[qa] = v
        mj.mj_forward(model, data)
        arm_q = ik_solve(model, data, target, arm_qa, arm_da)
        last_arm_q = arm_q.copy()
        
        # Finger control
        for jname, close_val in FINGER_CLOSE_MAP.items():
            if jname in finger_act_map:
                ai = finger_act_map[jname]
                data.ctrl[ai] = close_val if want_grip else 0.0
        
        # Adhesion
        adhesion_val = 1.0 if want_grip else 0.0
        for aid in adhesion_act_ids:
            data.ctrl[aid] = adhesion_val
        
        # Simulate
        for _ in range(SUBSTEPS):
            freeze_robot()
            for qa, v in zip(arm_qa, arm_q):
                data.qpos[qa] = v
            for da in arm_da:
                data.qvel[da] = 0
            mj.mj_step(model, data)
        
        renderer.update_scene(data, camera="front")
        Image.fromarray(renderer.render()).save(fd / f"frame_{i:04d}.png")
        
        if i % 10 == 0:
            pc = palm_center(model, data)
            contacts = 0
            for c in range(data.ncon):
                g1, g2 = data.contact[c].geom1, data.contact[c].geom2
                if (g1 in hand_geom_ids and g2 in block_geom_ids) or \
                   (g1 in block_geom_ids and g2 in hand_geom_ids):
                    contacts += 1
            block_zs = {nm: data.qpos[obj_qadr[obj_names.index(nm)]+2] for nm in obj_names}
            print(f"F{i:03d} p={p:.2f} grip={want_grip} adh={adhesion_val:.0f} contacts={contacts} palm={pc.round(3)} block_z={{{', '.join(f'{k}:{v:.3f}' for k,v in block_zs.items())}}}")
    
    for nm, qa in zip(obj_names, obj_qadr):
        pos = data.qpos[qa:qa+3]
        print(f"FINAL {nm}: xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    za = data.qpos[obj_qadr[0]+2]
    zb = data.qpos[obj_qadr[1]+2] if len(obj_qadr) > 1 else 0
    top_z = max(za, zb)
    bot_z = min(za, zb)
    stacked = abs(top_z - bot_z - 2*BLOCK_HALF) < 0.02
    print(f"STACK CHECK: top_z={top_z:.3f} bot_z={bot_z:.3f} gap={top_z-bot_z:.3f} expected={2*BLOCK_HALF:.3f} STACKED={stacked}")
    
    renderer.close()
    
    out = OUT_DIR / f"{task_name}_h1_shadow.mp4"
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found in PATH")
    subprocess.check_call([ff, "-y", "-framerate", str(FPS), "-i", str(fd/"frame_%04d.png"),
                           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast", str(out)])
    print("OK", out)
    return out


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    render_task(task)
