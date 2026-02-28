#!/usr/bin/env python3
"""Spatial trajectory retarget V3 ("perfect" approach)

Goal: visually perfect, physically plausible grasp trajectory while still being driven by
real spatial signal.

Key changes vs V2:
- Use stabilized mug_world (median) + smoothed offsets + grasp rebase
- Add *grasp funnel* near contact:
    If EE target is within funnel_radius of mug_sim, override targets with:
      pregrasp (above mug) -> slow descend -> close -> lift
  while preserving lateral alignment from the spatial signal.
- Slow descent by limiting max_step_m AND applying a separate max_z_step_m.
- Slightly close gripper before full close (soft-close) to avoid "crush" look.

Outputs: pipeline/spatial_v3/spatial_replay.mp4 + spatial_traj.json

Usage:
  python tools/spatial_trajectory_v3.py --scale 1.5
"""

import os, json, argparse
import numpy as np

PANDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mujoco_menagerie", "franka_emika_panda")
R3D_DIR = "pipeline/r3d_output"
OUT_DIR = "pipeline/spatial_v3"
FFMPEG = r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

SCENE_XML = """
<mujoco model="panda_spatial">
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


def quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])


def camera_pose(frame):
    p = frame["pose"]
    T = np.eye(4)
    T[:3,:3] = quat_to_rot(p[0], p[1], p[2], p[3])
    T[:3,3] = [p[4], p[5], p[6]]
    return T


def get_hand_pixel(hand, K):
    ct = hand["cam_t"]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u = fx * ct[0] / ct[2] + cx
    v = fy * ct[1] / ct[2] + cy
    return float(u), float(v)


def sample_depth(depth_map, u, v, rgb_size, depth_size, patch=4):
    sx = depth_size[1] / rgb_size[0]
    sy = depth_size[0] / rgb_size[1]
    du = int(u * sx)
    dv = int(v * sy)
    h, w = depth_map.shape
    du = int(np.clip(du, patch, w-patch-1))
    dv = int(np.clip(dv, patch, h-patch-1))
    p = depth_map[dv-patch:dv+patch+1, du-patch:du+patch+1]
    valid = p[p > 0.05]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def backproject(u, v, depth, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    return np.array([x, y, z, 1.0])


def compute_gripper(hand):
    ft = hand.get("fingertips", {})
    if "thumb" not in ft or "index" not in ft:
        return 0.5
    dist = np.linalg.norm(np.array(ft["thumb"]) - np.array(ft["index"]))
    close, open_ = 0.04, 0.07
    return float(np.clip((dist - close)/(open_-close), 0, 1))


def smooth_vecs(x, w=11):
    if len(x) < w:
        return x
    y = np.copy(x)
    h = w//2
    for i in range(h, len(x)-h):
        y[i] = x[i-h:i+h+1].mean(axis=0)
    return y


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
        if np.linalg.norm(err) < 0.0035:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, None, site_id)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05*np.eye(3), err)
        q += dq * min(0.35, 0.07/(np.linalg.norm(dq)+1e-8))
        for j in range(7):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{j+1}')
            q[j] = np.clip(q[j], *model.jnt_range[jid])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--max_step_m", type=float, default=0.006)
    ap.add_argument("--max_z_step_m", type=float, default=0.003)
    ap.add_argument("--funnel_radius", type=float, default=0.08)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    meta = json.load(open(os.path.join(R3D_DIR, "metadata.json")))
    frames = meta["frames"]
    K = np.array(meta["camera_K"], dtype=float)
    rgb_size = meta["rgb_size"]
    depth_size = meta["depth_size"]

    hamer = json.load(open(os.path.join(R3D_DIR, "hamer_results.json")))
    obj = json.load(open(os.path.join(R3D_DIR, "object_poses_3d.json")))

    det = [p["pose_3d"] for p in obj["poses"] if p.get("detected")]
    if not det:
        print("No object detections")
        return
    mug_world = np.median(np.array(det, dtype=float), axis=0)

    import mujoco, cv2

    xml_path = os.path.join(PANDA_DIR, "_spatial_v3.xml")
    with open(xml_path, "w") as f:
        f.write(SCENE_XML)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    gs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    mb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    mug_sim = data.xpos[mb].copy()

    # Build raw offsets
    rec = []
    offsets = []
    grips = []
    for fid in sorted(hamer.keys(), key=int):
        fidx = int(fid)
        if fidx >= len(frames):
            continue
        if len(rec) >= args.max_frames:
            break
        hands = hamer[fid]
        if not hands:
            continue
        hand = hands[0]

        u, v = get_hand_pixel(hand, K)
        depth_path = os.path.join(R3D_DIR, "depth", f"{fidx:04d}.npy")
        if not os.path.exists(depth_path):
            continue
        depth_map = np.load(depth_path)
        dval = sample_depth(depth_map, u, v, rgb_size, depth_size)
        if dval is None:
            continue

        p_cam = backproject(u, v, dval, K)
        Twc = camera_pose(frames[fidx])
        hand_world = (Twc @ p_cam)[:3]

        off = hand_world - mug_world
        g = compute_gripper(hand)
        rec.append({"frame": fidx, "hand_world": hand_world.tolist(), "gripper": g})
        offsets.append(off)
        grips.append(g)

    offsets = np.array(offsets)
    grips = np.array(grips)

    offsets_s = smooth_vecs(offsets, w=11)

    close_idx = int(np.argmax(grips < 0.3))
    if close_idx == 0 and grips[0] >= 0.3:
        close_idx = len(grips)//3

    off0 = offsets_s[close_idx]

    # Base mapping (spatial)
    grasp_offset_sim = np.array([0.0, 0.0, -0.015])
    ee_base = []
    prev = None
    for i in range(len(offsets_s)):
        rel = offsets_s[i] - off0
        ee = mug_sim + grasp_offset_sim + args.scale * rel
        # workspace clamp
        ee[0] = float(np.clip(ee[0], 0.40, 0.72))
        ee[1] = float(np.clip(ee[1], -0.18, 0.18))
        ee[2] = float(np.clip(ee[2], 0.24, 0.55))
        # delta clamp
        if prev is not None:
            d = ee - prev
            n = float(np.linalg.norm(d))
            if n > args.max_step_m:
                ee = prev + d/n * args.max_step_m
            # extra z clamp
            dz = ee[2] - prev[2]
            if abs(dz) > args.max_z_step_m:
                ee[2] = prev[2] + np.sign(dz) * args.max_z_step_m
        prev = ee
        ee_base.append(ee)
    ee_base = np.array(ee_base)

    # Grasp funnel override
    # Determine funnel entry: when base target gets within radius
    dist_to_mug = np.linalg.norm(ee_base - mug_sim[None, :], axis=1)
    enter_idx = int(np.argmax(dist_to_mug < args.funnel_radius))
    if enter_idx == 0 and dist_to_mug[0] >= args.funnel_radius:
        enter_idx = len(ee_base) // 2

    # Lateral alignment (use base XY at enter)
    align_xy = ee_base[enter_idx, :2].copy()

    # Build funnel trajectory segments
    pre_z = 0.38
    grasp_z = 0.30
    lift_z = 0.48

    ee_targets = ee_base.copy()

    # Pregrasp window
    n_pre = 25
    n_desc = 35
    n_close = 12
    n_lift = 35

    start = enter_idx
    end = min(len(ee_targets), start + n_pre + n_desc + n_close + n_lift)

    k = 0
    # pregrasp hold
    for i in range(start, min(end, start+n_pre)):
        ee_targets[i, 0] = align_xy[0]
        ee_targets[i, 1] = align_xy[1]
        ee_targets[i, 2] = pre_z
        k += 1
    # slow descend
    for i in range(start+n_pre, min(end, start+n_pre+n_desc)):
        a = (i - (start+n_pre)) / max(n_desc-1, 1)
        ee_targets[i, 0] = align_xy[0]
        ee_targets[i, 1] = align_xy[1]
        ee_targets[i, 2] = pre_z + (grasp_z - pre_z) * (0.5 - 0.5*np.cos(np.pi*a))
        k += 1
    # soft close (stay at grasp)
    for i in range(start+n_pre+n_desc, min(end, start+n_pre+n_desc+n_close)):
        ee_targets[i, 0] = align_xy[0]
        ee_targets[i, 1] = align_xy[1]
        ee_targets[i, 2] = grasp_z
        k += 1
    # lift
    for i in range(start+n_pre+n_desc+n_close, end):
        a = (i - (start+n_pre+n_desc+n_close)) / max(n_lift-1, 1)
        ee_targets[i, 0] = align_xy[0]
        ee_targets[i, 1] = align_xy[1]
        ee_targets[i, 2] = grasp_z + (lift_z - grasp_z) * (0.5 - 0.5*np.cos(np.pi*a))
        k += 1

    print(f"Samples={len(ee_targets)} close_idx={close_idx} enter_idx={enter_idx} scale={args.scale}")
    print(f"EE range: x[{ee_targets[:,0].min():.3f},{ee_targets[:,0].max():.3f}] z[{ee_targets[:,2].min():.3f},{ee_targets[:,2].max():.3f}]")

    # Build gripper targets: use real, but enforce soft-close in funnel
    g_targets = grips.copy()
    # during soft close window, ramp to closed
    c0 = start + n_pre + n_desc
    c1 = min(end, c0 + n_close)
    for i in range(c0, c1):
        a = (i - c0) / max((c1-c0)-1, 1)
        # soft close from open->closed (0.04 -> 0)
        g_targets[i] = max(0.0, 1.0 - a*1.1)
    # hold closed during lift window
    for i in range(c1, end):
        g_targets[i] = 0.0

    # IK solve
    q_list = []
    prev_q = None
    for i in range(len(ee_targets)):
        q = solve_ik(model, ee_targets[i], gs, prev_q)
        prev_q = q
        q_list.append(q)
        if i % 25 == 0:
            print(f"IK {i}/{len(ee_targets)}")

    # Replay
    FPS = args.fps
    spf = int(1.0 / (FPS * model.opt.timestep))

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    frames_out = []
    renderer.update_scene(data)
    im0 = renderer.render().copy()
    for _ in range(FPS):
        frames_out.append(im0)

    for i in range(len(ee_targets)):
        grip_ctrl = 0.04 * float(np.clip(g_targets[i], 0, 1))
        data.ctrl[:7] = q_list[i]
        data.ctrl[7] = grip_ctrl
        for _ in range(spf):
            mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frames_out.append(renderer.render().copy())

    renderer.update_scene(data)
    imf = renderer.render().copy()
    for _ in range(FPS):
        frames_out.append(imf)

    raw = os.path.join(OUT_DIR, "spatial_replay_raw.mp4")
    final = os.path.join(OUT_DIR, "spatial_replay.mp4")

    h, w = frames_out[0].shape[:2]
    wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for im in frames_out:
        wr.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    wr.release()

    os.system(f'"{FFMPEG}" -y -i "{raw}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{final}" 2>nul')
    out_path = final if os.path.exists(final) and os.path.getsize(final) > 1000 else raw

    traj = {
        "scale": args.scale,
        "fps": FPS,
        "close_idx": close_idx,
        "enter_idx": enter_idx,
        "mug_world": mug_world.tolist(),
        "mug_sim": mug_sim.tolist(),
        "ee_targets": ee_targets.tolist(),
        "grippers": g_targets.tolist(),
        "qpos": [q.tolist() for q in q_list],
        "notes": {
            "funnel_radius": args.funnel_radius,
            "max_step_m": args.max_step_m,
            "max_z_step_m": args.max_z_step_m,
            "pre_z": pre_z,
            "grasp_z": grasp_z,
            "lift_z": lift_z,
        }
    }
    with open(os.path.join(OUT_DIR, "spatial_traj.json"), "w") as f:
        json.dump(traj, f)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
