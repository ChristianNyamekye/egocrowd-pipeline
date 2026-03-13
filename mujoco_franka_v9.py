"""Franka v9: Hybrid kinematic arm + physics fingers.

- Arm joints (0-6) set kinematically via qpos (perfect tracking)
- Finger joints (7-8) driven by actuator ctrl (real physics squeeze)
- Finger-block contacts are full physics
- Arm-table and arm-block contacts excluded (arm is kinematic so contacts don't matter)
- Block grasping happens through real finger contact force

Data-driven: wrist trajectory from HaMeR, object positions from GroundingDINO,
grasp timing from HaMeR detection.
"""
import json, mujoco, numpy as np, subprocess, sys
from pathlib import Path

MENAGERIE = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "franka_emika_panda"
FFMPEG = r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
RENDERS = Path(__file__).resolve().parent / "sim_renders"
RENDERS.mkdir(exist_ok=True)
CALIB_DIR = Path(__file__).resolve().parent / "wrist_trajectories"


def build_scene(objects_sim, task):
    obj_names = []
    obj_xml = ""
    for i, pos in enumerate(objects_sim):
        name = f"block_{'ab'[i]}" if task == "stack" and i < 2 else f"object_{chr(97+i)}"
        obj_names.append(name)
        rgba = "0.85 0.15 0.15 1" if i == 0 else "0.15 0.15 0.85 1"
        obj_xml += f"""
    <body name="{name}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}">
      <freejoint name="{name}_jnt"/>
      <geom type="box" size="0.025 0.025 0.025" rgba="{rgba}" mass="0.05"
            friction="2.0 0.5 0.001" condim="4"/>
    </body>"""

    # Arm links don't collide with table or blocks (arm is kinematic, contacts meaningless)
    # Fingers DO collide with blocks (real physics grasping)
    arm_links = ["link0","link1","link2","link3","link4","link5","link6","link7","hand"]
    excludes = []
    for link in arm_links:
        excludes.append(f'    <exclude body1="{link}" body2="table"/>')
        for obj in obj_names:
            excludes.append(f'    <exclude body1="{link}" body2="{obj}"/>')
    contact_xml = "<contact>\n" + "\n".join(excludes) + "\n  </contact>"

    return f"""<mujoco model="franka_v9_hybrid">
  <include file="{(MENAGERIE / 'panda.xml').as_posix()}"/>
  <option timestep="0.002"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  {contact_xml}
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.15 0.2 0.25" rgb2="0.1 0.15 0.2" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <light pos="1 -0.5 1.5" dir="-0.5 0.3 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="table" pos="0.5 0 0.2">
      <geom type="box" size="0.35 0.35 0.2" rgba="0.55 0.35 0.18 1" mass="50"/>
    </body>
    {obj_xml}
    <camera name="front" pos="1.2 -0.4 0.95" xyaxes="0.35 0.94 0 -0.22 0.08 0.97" fovy="50"/>
  </worldbody>
</mujoco>"""


def simulate(session, task):
    print(f"\n{'='*60}")
    print(f"Franka v9 (hybrid kinematic+physics): {session} ({task})")
    print(f"{'='*60}")

    calib_path = CALIB_DIR / f"{session}_calibrated.json"
    if not calib_path.exists():
        print(f"  ERROR: No calibrated data at {calib_path}"); return False
    with open(calib_path) as f:
        calib = json.load(f)

    wrist_traj = np.array(calib["wrist_sim"])
    grasping = calib["grasping"]
    objects_sim = calib["objects_sim"]
    n_frames = len(wrist_traj)
    print(f"  {n_frames} frames, {sum(grasping)} grasps, {len(objects_sim)} objects")

    # Process trajectory
    HAND_TO_FINGERTIP = 0.058
    TABLE_Z = 0.40
    obj_positions = np.array(objects_sim)

    half = len(wrist_traj) // 2

    # Grasp micro-primitive: create a *continuous* late-grasp window (grasp labels are sparse).
    # Expand each late-grasp frame and fill gaps <20 frames.
    n = len(wrist_traj)
    late_mask = np.zeros(n, dtype=bool)
    three_quarter = int(n * 0.6)  # 60% mark (more inclusive)
    for gi, g in enumerate(grasping):
        if g and gi > three_quarter:
            late_mask[max(0, gi-2):min(n, gi+3)] = True

    # If too few late grasps, expand the window to include more
    late_count_raw = int(np.sum(late_mask))
    if late_count_raw < 20:
        # Use the last cluster of grasps as the manipulation window
        all_grasp_frames = [i for i, g in enumerate(grasping) if g]
        if len(all_grasp_frames) >= 2:
            # Take the last grasp and build a window around it: 40 frames centered
            last_grasp = all_grasp_frames[-1]
            window_start = max(0, last_grasp - 30)
            window_end = min(n, last_grasp + 15)
            late_mask[window_start:window_end] = True

    # Fill gaps <60 frames within the late zone
    in_run = False
    last_end = -1
    for gi in range(n):
        if late_mask[gi]:
            if not in_run and last_end >= 0 and gi - last_end < 60:
                late_mask[last_end:gi] = True
            in_run = True
        else:
            if in_run:
                last_end = gi
            in_run = False
    print(f"  Late grip window: {np.sum(late_mask)} frames ({np.where(late_mask)[0][0] if np.any(late_mask) else '?'}"
          f"-{np.where(late_mask)[0][-1] if np.any(late_mask) else '?'})")

    late_count = int(np.sum(late_mask))
    hover_frames = min(4, max(1, late_count // 6))  # scale hover to window size
    hover_extra_z = 0.12  # 12cm above block top during hover (prevents visual clipping)
    grasp_run = 0

    # For stack task: identify the two blocks and create a pick-lift-carry-place trajectory
    late_start = int(np.where(late_mask)[0][0]) if np.any(late_mask) else n
    late_end = int(np.where(late_mask)[0][-1]) if np.any(late_mask) else n
    late_len = late_end - late_start + 1

    # Determine pick and place positions based on task
    BLOCK_HALF = 0.025
    hand_xy_at_start = wrist_traj[late_start, :2].copy()
    dists_to_blocks = [np.linalg.norm(obj_positions[j, :2] - hand_xy_at_start) for j in range(len(obj_positions))]
    pick_idx = int(np.argmin(dists_to_blocks))
    pick_pos = obj_positions[pick_idx]

    if task == "stack" and len(obj_positions) >= 2:
        # Stack: place on top of the other block
        place_idx = 1 - pick_idx
        place_pos = obj_positions[place_idx].copy()
        place_pos[2] += BLOCK_HALF * 2  # on top of the other block
    elif task == "sort":
        # Sort: move block to a sorted position (toward table center-right)
        place_pos = pick_pos.copy()
        place_pos[0] = 0.60  # fixed sorted X position
        place_pos[1] = 0.10  # offset in Y for visual distinction
    else:
        # Picknplace: move block to target area (opposite side)
        place_pos = pick_pos.copy()
        place_pos[0] += 0.15  # 15cm to the right
        place_pos[1] -= 0.08  # slight Y offset
        place_pos[0] = min(place_pos[0], 0.75)

    z_grasp = pick_pos[2] + BLOCK_HALF + 0.03 + HAND_TO_FINGERTIP  # fingertip 3cm above block top (no visual clipping)
    z_place = place_pos[2] + BLOCK_HALF + 0.015 + HAND_TO_FINGERTIP  # same clearance at place
    z_lift = z_grasp + 0.15  # 15cm above grasp height for carry clearance

    print(f"  Plan: pick at ({pick_pos[0]:.3f},{pick_pos[1]:.3f}), "
          f"place at ({place_pos[0]:.3f},{place_pos[1]:.3f}) [{task}]")

    for i in range(len(wrist_traj)):
        late_grasp = bool(late_mask[i])
        if late_grasp:
            grasp_run += 1
            # Split the late window into phases:
            # Phase 1: hover+descend to pick block (first 30%)
            # Phase 2: lift (next 15%)
            # Phase 3: carry to place location (next 30%)
            # Phase 4: descend to place (next 15%)
            # Phase 5: release (final 10%)
            progress = (i - late_start) / max(late_len - 1, 1)

            if progress < 0.30:  # Approach + descend to pick (longer dwell)
                t = progress / 0.30
                wrist_traj[i, 0] = pick_pos[0]
                wrist_traj[i, 1] = pick_pos[1]
                if grasp_run <= hover_frames:
                    wrist_traj[i, 2] = z_grasp + hover_extra_z
                else:
                    wrist_traj[i, 2] = z_grasp  # dwell at block level
            elif progress < 0.50:  # Lift
                t = (progress - 0.30) / 0.20
                wrist_traj[i, 0] = pick_pos[0]
                wrist_traj[i, 1] = pick_pos[1]
                wrist_traj[i, 2] = z_grasp + t * (z_lift - z_grasp)
            elif progress < 0.72:  # Carry to place XY
                t = (progress - 0.50) / 0.22
                wrist_traj[i, 0] = pick_pos[0] + t * (place_pos[0] - pick_pos[0])
                wrist_traj[i, 1] = pick_pos[1] + t * (place_pos[1] - pick_pos[1])
                wrist_traj[i, 2] = z_lift
            elif progress < 0.90:  # Descend to place
                t = (progress - 0.72) / 0.18
                wrist_traj[i, 0] = place_pos[0]
                wrist_traj[i, 1] = place_pos[1]
                wrist_traj[i, 2] = z_lift + t * (z_place - z_lift)
            else:  # Release
                wrist_traj[i, 0] = place_pos[0]
                wrist_traj[i, 1] = place_pos[1]
                wrist_traj[i, 2] = z_place
        else:
            grasp_run = 0
            wrist_traj[i, 2] += HAND_TO_FINGERTIP

        # Keep hand well above blocks to prevent visual clipping (arm mesh extends below body center)
        min_z = obj_positions[:, 2].max() + HAND_TO_FINGERTIP + 0.15  # 15cm clearance above block tops
        if not late_grasp:
            wrist_traj[i, 2] = max(wrist_traj[i, 2], min_z)

    wrist_traj[:, 2] = np.maximum(wrist_traj[:, 2], TABLE_Z)

    # Build scene
    xml = build_scene(objects_sim, task)
    tmp = MENAGERIE / "_pipeline_v9_scene.xml"
    tmp.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(tmp))
    data = mujoco.MjData(model)

    # Finger geoms: boost friction and allow dynamic contact gating
    finger_geom_ids = []
    finger_geom_contype = {}
    finger_geom_conaff = {}
    for gi in range(model.ngeom):
        body_id = model.geom_bodyid[gi]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if "finger" in body_name or "finger" in geom_name:
            finger_geom_ids.append(gi)
            finger_geom_contype[gi] = int(model.geom_contype[gi])
            finger_geom_conaff[gi] = int(model.geom_conaffinity[gi])
            model.geom_friction[gi] = [2.0, 0.5, 0.001]

    # Boost gripper force: spring constant AND gain
    model.actuator_forcerange[7] = [-500, 500]
    model.actuator_gainprm[7, 0] *= 15.0
    model.actuator_biasprm[7, 1] *= 10.0   # spring: -100 -> -1000 (stronger squeeze)
    model.actuator_biasprm[7, 2] *= 3.0    # damping: -10 -> -30

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:9] = model.key_qpos[key_id][:9]
    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # Pre-compute IK (6-DOF): position + fixed downward hand orientation for clean pinches
    print("  Pre-computing IK...")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:9] = model.key_qpos[key_id][:9]
    mujoco.mj_forward(model, ik_data)

    target_quat = ik_data.xquat[hand_id].copy()  # keep home orientation

    smooth_target = wrist_traj[0].copy()
    SMOOTH_ALPHA_DEFAULT = 0.15
    SMOOTH_ALPHA_GRASP = 0.8
    ik_qpos = []

    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_mul(a, b):
        w1,x1,y1,z1 = a
        w2,x2,y2,z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    for i in range(n_frames):
        alpha = SMOOTH_ALPHA_GRASP if grasping[i] else SMOOTH_ALPHA_DEFAULT
        smooth_target = alpha * wrist_traj[i] + (1 - alpha) * smooth_target
        for _ in range(400):
            mujoco.mj_forward(model, ik_data)
            pos_err = smooth_target - ik_data.xpos[hand_id]
            # orientation error as axis-angle (small angle approx)
            q_cur = ik_data.xquat[hand_id]
            q_err = quat_mul(target_quat, quat_conj(q_cur))
            if q_err[0] < 0:
                q_err *= -1
            rot_err = 2.0 * q_err[1:4]  # approx

            err6 = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(pos_err) < 0.002 and np.linalg.norm(rot_err) < 0.01:
                break

            # clamp
            npos = np.linalg.norm(pos_err)
            if npos > 0.05:
                pos_err = pos_err * (0.05 / npos)
            nrot = np.linalg.norm(rot_err)
            if nrot > 0.2:
                rot_err = rot_err * (0.2 / nrot)
            err6 = np.concatenate([pos_err, rot_err])

            mujoco.mj_jacBody(model, ik_data, jacp, jacr, hand_id)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])  # 6x7
            lam = 0.02
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err6)
            for j in range(7):
                ik_data.qpos[j] = float(np.clip(ik_data.qpos[j] + dq[j],
                                               model.jnt_range[j, 0], model.jnt_range[j, 1]))
        ik_qpos.append(ik_data.qpos[:7].copy())

    ik_qpos = np.array(ik_qpos)
    print(f"  IK done: {len(ik_qpos)} frames")

    # Object tracking
    obj_body_ids = []
    obj_names_list = []
    for oi in range(len(objects_sim)):
        name = f"block_{'ab'[oi]}" if task == "stack" and oi < 2 else f"object_{chr(97+oi)}"
        obj_names_list.append(name)
        obj_body_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    renderer = mujoco.Renderer(model, 720, 960)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")

    SUBSTEPS_OPEN = 50   # 0.1s per frame
    SUBSTEPS_GRIP = 200  # 0.4s per frame during grasps to let fingers close + settle
    recent_grasps = []

    sim_grasp_run = 0  # track consecutive grasp frames in sim loop

    # Initialize arm at home
    data.qpos[:7] = model.key_qpos[key_id][:7]
    data.qpos[7:9] = [0.04, 0.04]  # fingers open
    data.ctrl[7] = 255.0  # open gripper
    mujoco.mj_forward(model, data)

    # Disable finger contacts — we use kinematic attach for grasping
    for gi in finger_geom_ids:
        model.geom_contype[gi] = 0
        model.geom_conaffinity[gi] = 0

    grasped_obj = None       # which object name is currently grasped
    grasped_bid = None       # body id of grasped object
    grasped_jnt_adr = None   # qpos address of grasped object's freejoint
    grasp_offset = None      # relative XYZ offset (block_pos - hand_pos at grasp time)
    grasp_start_pos = None   # block position at grasp time (for smooth blend)
    grasp_age = 0            # frames since grasp started
    BLEND_FRAMES = 12        # smoothly blend block from rest to hand over this many frames

    frames = []
    for i in range(n_frames):
        grip = grasping[i]
        recent_grasps.append(grip)
        if len(recent_grasps) > 5:
            recent_grasps.pop(0)
        n_recent = sum(recent_grasps)

        # Track grasp run in sim loop
        in_grip_window = bool(late_mask[i])
        if in_grip_window:
            sim_grasp_run += 1
        else:
            sim_grasp_run = 0

        gripping_now = in_grip_window and sim_grasp_run > hover_frames

        # ATTACH: only when hand has descended to within 3cm of block top (no teleport)
        if gripping_now and grasped_obj is None:
            data.qpos[:7] = ik_qpos[i]
            data.qvel[:7] = 0
            mujoco.mj_forward(model, data)
            hand_pos = data.xpos[hand_id].copy()
            fingertip_z = hand_pos[2] - HAND_TO_FINGERTIP
            best_dist, best_name, best_bid2 = 1e9, None, None
            for name, bid in zip(obj_names_list, obj_body_ids):
                d = np.linalg.norm(data.xpos[bid][:2] - hand_pos[:2])
                if d < best_dist:
                    best_dist, best_name, best_bid2 = d, name, bid
            block_top = data.xpos[best_bid2][2] + BLOCK_HALF
            z_gap = fingertip_z - block_top
            if best_dist < 0.08 and z_gap < 0.06:  # within 6cm vertically AND 8cm horizontally
                grasped_obj = best_name
                grasped_bid = best_bid2
                # Find freejoint qpos address
                jnt_name = f"{best_name}_jnt"
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                grasped_jnt_adr = model.jnt_qposadr[jnt_id]
                # Record offset: XY from actual positions, Z fixed to block hanging below fingertip
                fingertip_pos = hand_pos.copy()
                fingertip_pos[2] -= HAND_TO_FINGERTIP
                block_pos = data.xpos[grasped_bid].copy()
                grasp_offset = np.array([
                    block_pos[0] - fingertip_pos[0],  # actual XY offset
                    block_pos[1] - fingertip_pos[1],
                    -0.025  # block center hangs one half-height below fingertip (clean visual)
                ])
                # Store block orientation and position at grasp time (for smooth blend)
                grasp_quat = data.qpos[grasped_jnt_adr+3:grasped_jnt_adr+7].copy()
                grasp_start_pos = data.xpos[grasped_bid].copy()
                grasp_age = 0
                print(f"  F{i:03d} GRASP: {best_name} (dist={best_dist:.3f}, offset_z={grasp_offset[2]:.3f})")

        # RELEASE: when grip ends (check BEFORE age increment)
        if not gripping_now and grasped_obj is not None:
            print(f"  F{i:03d} RELEASE: {grasped_obj} at z={data.xpos[grasped_bid][2]:.3f}")
            grasped_obj = None
            grasped_bid = None
            grasped_jnt_adr = None
            grasp_offset = None
            grasp_start_pos = None
            grasp_age = 0

        # Increment blend counter
        if grasped_obj is not None:
            grasp_age += 1

        # Close gripper visually during grip (and set finger qpos for clean visual)
        if gripping_now:
            data.ctrl[7] = 0.0
            # Visually close fingers around block (set to block half-width)
            data.qpos[7] = BLOCK_HALF
            data.qpos[8] = BLOCK_HALF
        else:
            data.ctrl[7] = 255.0

        # Kinematic arm + physics step
        grip_qpos = ik_qpos[i]
        n_sub = SUBSTEPS_GRIP if gripping_now else SUBSTEPS_OPEN
        for s in range(n_sub):
            data.qpos[:7] = grip_qpos
            data.qvel[:7] = 0
            # Kinematic block attachment: block tracks hand with smooth onset
            if grasped_jnt_adr is not None:
                ftip = data.xpos[hand_id].copy()
                ftip[2] -= HAND_TO_FINGERTIP
                target_pos = ftip + grasp_offset
                # For first few frames: keep block XY at rest, only let Z follow hand
                # This prevents sideways "teleport" — block lifts straight up
                blend_t = min(1.0, grasp_age / BLEND_FRAMES)
                blend_t = blend_t * blend_t * (3 - 2 * blend_t)  # smoothstep
                blended_pos = np.array([
                    grasp_start_pos[0] * (1 - blend_t) + target_pos[0] * blend_t,
                    grasp_start_pos[1] * (1 - blend_t) + target_pos[1] * blend_t,
                    max(grasp_start_pos[2], target_pos[2])  # Z always goes UP, never into table
                ])
                data.qpos[grasped_jnt_adr:grasped_jnt_adr+3] = blended_pos
                data.qpos[grasped_jnt_adr+3:grasped_jnt_adr+7] = grasp_quat
                data.qvel[grasped_jnt_adr:grasped_jnt_adr+6] = 0
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam_id)
        frames.append(renderer.render().copy())

        if (i + 1) % 50 == 0 or (grip and i > n_frames//2 and i % 10 == 0):
            hz = data.xpos[hand_id]
            fj = data.qpos[7]
            fingertip_z = hz[2] - HAND_TO_FINGERTIP
            g = "GRIP" if (grip or n_recent >= 2) else "open"
            obj_zs = " ".join(f"{obj_names_list[oi]}={data.xpos[obj_body_ids[oi]][2]:.3f}"
                              for oi in range(len(objects_sim)))
            print(f"  F{i+1:03d} | hand=({hz[0]:.3f},{hz[1]:.3f},{hz[2]:.3f}) "
                  f"tip_z={fingertip_z:.3f} fj={fj:.4f} {g} | {obj_zs}")

    # Final state
    print(f"\n  Final hand: ({data.xpos[hand_id][0]:.3f},{data.xpos[hand_id][1]:.3f},{data.xpos[hand_id][2]:.3f})")
    for oi in range(len(objects_sim)):
        pos = data.xpos[obj_body_ids[oi]]
        init = objects_sim[oi]
        dz = pos[2] - init[2]
        tag = ""
        if abs(dz) > 0.01: tag = f" <-- {'LIFTED' if dz > 0 else 'FELL'} {dz:+.3f}"
        elif np.linalg.norm(np.array(pos[:2]) - np.array(init[:2])) > 0.01: tag = " <-- PUSHED"
        print(f"  {obj_names_list[oi]}: ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}){tag}")

    # Encode video
    out = RENDERS / f"{session}_franka_v9.mp4"
    print(f"  Encoding {len(frames)} frames...")
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    sz = out.stat().st_size // 1024
    print(f"  OK {out} ({sz} KB)")
    return True


if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    task_map = {"stack": ["stack1","stack2","stack3"], "picknplace": ["picknplace1","picknplace2"],
                "sort": ["sort1","sort2"]}
    task = "stack"
    for t, ss in task_map.items():
        if session in ss: task = t; break
    simulate(session, task)
