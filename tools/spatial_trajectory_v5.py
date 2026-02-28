#!/usr/bin/env python3
"""Spatial trajectory V5 — hybrid: spatial approach + expert grasp funnel.

Architecture:
  Phase 1 (spatial): EE follows hand trajectory (relative to mug) for approach direction.
  Phase 2 (converge): XY blends toward mug center, Z holds at pre_z.
  Phase 3 (expert): pregrasp directly above mug → slow cosine descend → close → lift.

The key insight: vision-based XY has ~2-4cm error at contact scale,
so we MUST converge to mug_sim XY before descending.
"""

import os, json, argparse
import numpy as np

PANDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mujoco_menagerie", "franka_emika_panda")
R3D_DIR = "pipeline/r3d_output"
OUT_DIR = "pipeline/spatial_v5"
FFMPEG = r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

SCENE_XML = """
<mujoco model="panda_spatial_v5">
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


def cosine01(a):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(a, 0, 1))


def smooth_vecs(x, w=11):
    if len(x) < w:
        return x
    y = np.copy(x)
    h = w // 2
    for i in range(h, len(x) - h):
        y[i] = x[i - h:i + h + 1].mean(axis=0)
    return y


def quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])


def camera_pose(frame):
    p = frame["pose"]
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(p[0], p[1], p[2], p[3])
    T[:3, 3] = [p[4], p[5], p[6]]
    return T


def get_hand_pixel(hand, K):
    ct = hand["cam_t"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * ct[0] / ct[2] + cx
    v = fy * ct[1] / ct[2] + cy
    return float(u), float(v)


def sample_depth(depth_map, u, v, rgb_size, depth_size, patch=4):
    sx = depth_size[1] / rgb_size[0]
    sy = depth_size[0] / rgb_size[1]
    du = int(u * sx)
    dv = int(v * sy)
    h, w = depth_map.shape
    du = int(np.clip(du, patch, w - patch - 1))
    dv = int(np.clip(dv, patch, h - patch - 1))
    p = depth_map[dv - patch:dv + patch + 1, du - patch:du + patch + 1]
    valid = p[p > 0.05]
    return float(np.median(valid)) if len(valid) > 0 else None


def backproject(u, v, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) / fx * depth, (v - cy) / fy * depth, depth, 1.0])


def compute_gripper(hand):
    ft = hand.get("fingertips", {})
    if "thumb" not in ft or "index" not in ft:
        return 0.5
    dist = np.linalg.norm(np.array(ft["thumb"]) - np.array(ft["index"]))
    return float(np.clip((dist - 0.04) / 0.03, 0, 1))


def solve_ik(model, target, site_id, start_q=None):
    import mujoco
    d2 = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d2, 0)
    if start_q is not None:
        d2.qpos[:7] = start_q
    q = d2.qpos[:7].copy()
    for _ in range(1400):
        d2.qpos[:7] = q
        mujoco.mj_forward(model, d2)
        err = target - d2.site_xpos[site_id].copy()
        if np.linalg.norm(err) < 0.003:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, None, site_id)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(3), err)
        q += dq * min(0.35, 0.07 / (np.linalg.norm(dq) + 1e-8))
        for j in range(7):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{j+1}')
            q[j] = np.clip(q[j], *model.jnt_range[jid])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=200)
    # Phase durations
    ap.add_argument("--approach_frac", type=float, default=0.35, help="fraction of frames for spatial approach")
    ap.add_argument("--converge_frames", type=int, default=20, help="frames to blend XY toward mug center")
    ap.add_argument("--pre_frames", type=int, default=15, help="hold above mug")
    ap.add_argument("--desc_frames", type=int, default=50, help="slow cosine descent")
    ap.add_argument("--close_frames", type=int, default=20, help="close gripper")
    ap.add_argument("--lift_frames", type=int, default=50, help="lift up")
    ap.add_argument("--hold_frames", type=int, default=30, help="hold at top")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    meta = json.load(open(os.path.join(R3D_DIR, "metadata.json")))
    frames = meta["frames"]
    K = np.array(meta["camera_K"], dtype=float)
    rgb_size, depth_size = meta["rgb_size"], meta["depth_size"]

    hamer = json.load(open(os.path.join(R3D_DIR, "hamer_results.json")))
    obj = json.load(open(os.path.join(R3D_DIR, "object_poses_3d.json")))

    det = [p["pose_3d"] for p in obj["poses"] if p.get("detected")]
    mug_world = np.median(np.array(det, dtype=float), axis=0)

    # Extract hand world positions
    offsets, grips = [], []
    for fid in sorted(hamer.keys(), key=int):
        fidx = int(fid)
        if fidx >= len(frames) or len(offsets) >= args.max_frames:
            break
        hands = hamer[fid]
        if not hands:
            continue
        hand = hands[0]
        u, v = get_hand_pixel(hand, K)
        depth_path = os.path.join(R3D_DIR, "depth", f"{fidx:04d}.npy")
        if not os.path.exists(depth_path):
            continue
        dval = sample_depth(np.load(depth_path), u, v, rgb_size, depth_size)
        if dval is None:
            continue
        p_cam = backproject(u, v, dval, K)
        Twc = camera_pose(frames[fidx])
        hw = (Twc @ p_cam)[:3]
        offsets.append(hw - mug_world)
        grips.append(compute_gripper(hand))

    offsets = smooth_vecs(np.array(offsets), w=13)
    grips = np.array(grips)
    n_spatial = len(offsets)
    print(f"Spatial samples: {n_spatial}")

    import mujoco, cv2

    xml_path = os.path.join(PANDA_DIR, "_spatial_v5.xml")
    with open(xml_path, "w") as f:
        f.write(SCENE_XML)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    gs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    mb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    mug_sim = data.xpos[mb].copy()  # [0.5, 0, 0.295]
    home_ee = data.site_xpos[gs].copy()

    # Heights
    pre_z = mug_sim[2] + 0.12   # 0.415 — well above mug
    grasp_z = mug_sim[2] + 0.02  # 0.315 — gripper center at mug mid-height
    lift_z = mug_sim[2] + 0.22   # 0.515

    # Phase 1: Spatial approach
    # Use the first approach_frac of spatial frames, scaled and rebased
    n_approach = int(n_spatial * args.approach_frac)
    # Rebase: offset[0] is start, we want to map home_ee → ... → near mug
    off_start = offsets[0]
    spatial_ee = []
    for i in range(n_approach):
        rel = offsets[i] - off_start
        # Target: start from home_ee, move toward mug_sim + small offset
        alpha = cosine01(i / max(n_approach - 1, 1))
        # Blend from home_ee toward a point above the mug
        approach_target = home_ee + alpha * (np.array([mug_sim[0], mug_sim[1], pre_z]) - home_ee)
        # Add lateral spatial variation (scaled down to avoid overshooting)
        lateral = args.scale * rel[:2] * (1 - alpha)  # spatial influence fades as we approach
        approach_target[0] += lateral[0]
        approach_target[1] += lateral[1]
        spatial_ee.append(approach_target.copy())

    # Phase 2: Converge XY to mug center at pre_z
    last_spatial = spatial_ee[-1] if spatial_ee else np.array([mug_sim[0], mug_sim[1], pre_z])
    converge_ee = []
    for i in range(args.converge_frames):
        alpha = cosine01(i / max(args.converge_frames - 1, 1))
        xy = last_spatial[:2] + alpha * (mug_sim[:2] - last_spatial[:2])
        converge_ee.append(np.array([xy[0], xy[1], pre_z]))

    # Phase 3: Pre-hold directly above mug
    pre_ee = [np.array([mug_sim[0], mug_sim[1], pre_z])] * args.pre_frames

    # Phase 4: Slow cosine descent
    desc_ee = []
    for i in range(args.desc_frames):
        alpha = cosine01(i / max(args.desc_frames - 1, 1))
        z = pre_z + alpha * (grasp_z - pre_z)
        desc_ee.append(np.array([mug_sim[0], mug_sim[1], z]))

    # Phase 5: Close (hold at grasp_z)
    close_ee = [np.array([mug_sim[0], mug_sim[1], grasp_z])] * args.close_frames

    # Phase 6: Lift
    lift_ee = []
    for i in range(args.lift_frames):
        alpha = cosine01(i / max(args.lift_frames - 1, 1))
        z = grasp_z + alpha * (lift_z - grasp_z)
        lift_ee.append(np.array([mug_sim[0], mug_sim[1], z]))

    # Phase 7: Hold at top
    hold_ee = [np.array([mug_sim[0], mug_sim[1], lift_z])] * args.hold_frames

    # Concatenate all phases
    all_ee = spatial_ee + converge_ee + pre_ee + desc_ee + close_ee + lift_ee + hold_ee
    all_ee = np.array(all_ee)

    # Gripper: open for all except close + lift + hold
    n_total = len(all_ee)
    grip_vals = np.ones(n_total)
    close_start = len(spatial_ee) + len(converge_ee) + len(pre_ee) + len(desc_ee)
    close_end = close_start + args.close_frames
    # Gradual close during close phase
    for i in range(close_start, close_end):
        alpha = (i - close_start) / max(args.close_frames - 1, 1)
        grip_vals[i] = max(0.0, 1.0 - alpha)
    # Closed for lift + hold
    grip_vals[close_end:] = 0.0

    phase_info = {
        "spatial": [0, len(spatial_ee)],
        "converge": [len(spatial_ee), len(spatial_ee) + len(converge_ee)],
        "pre": [len(spatial_ee) + len(converge_ee), close_start - len(desc_ee) - args.close_frames + len(desc_ee)],
        "descend": [close_start - len(desc_ee), close_start],
        "close": [close_start, close_end],
        "lift": [close_end, close_end + len(lift_ee)],
        "hold": [close_end + len(lift_ee), n_total],
    }

    print(f"Total frames: {n_total}")
    print(f"Phases: spatial={len(spatial_ee)}, converge={len(converge_ee)}, "
          f"pre={args.pre_frames}, desc={args.desc_frames}, "
          f"close={args.close_frames}, lift={args.lift_frames}, hold={args.hold_frames}")
    print(f"Grasp z={grasp_z:.3f}, pre_z={pre_z:.3f}, lift_z={lift_z:.3f}")

    # IK solve all frames
    prev_q = None
    q_all = []
    for i in range(n_total):
        q = solve_ik(model, all_ee[i], gs, prev_q)
        prev_q = q
        q_all.append(q)
        if i % 40 == 0:
            print(f"IK {i}/{n_total}")

    # Simulate
    FPS = args.fps
    spf = int(1.0 / (FPS * model.opt.timestep))

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    frames_out = []
    # 1s pause at start
    renderer.update_scene(data)
    im0 = renderer.render().copy()
    for _ in range(FPS):
        frames_out.append(im0)

    for i in range(n_total):
        data.ctrl[:7] = q_all[i]
        data.ctrl[7] = 0.04 * float(np.clip(grip_vals[i], 0, 1))
        for _ in range(spf):
            mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frames_out.append(renderer.render().copy())

    # 1s pause at end
    renderer.update_scene(data)
    imf = renderer.render().copy()
    for _ in range(FPS):
        frames_out.append(imf)

    # Write video
    raw = os.path.join(OUT_DIR, "spatial_replay_raw.mp4")
    final = os.path.join(OUT_DIR, "spatial_replay.mp4")
    h, w = frames_out[0].shape[:2]
    wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for im in frames_out:
        wr.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    wr.release()

    os.system(f'"{FFMPEG}" -y -i "{raw}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{final}" 2>nul')
    out_path = final if os.path.exists(final) and os.path.getsize(final) > 1000 else raw

    # Save trajectory JSON for audit
    with open(os.path.join(OUT_DIR, "spatial_traj.json"), "w") as f:
        json.dump({
            "scale": args.scale,
            "fps": FPS,
            "n_total": n_total,
            "phases": {k: v for k, v in phase_info.items()},
            "ee_targets": all_ee.tolist(),
            "grippers": grip_vals.tolist(),
            "params": {"pre_z": pre_z, "grasp_z": grasp_z, "lift_z": lift_z},
        }, f)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
