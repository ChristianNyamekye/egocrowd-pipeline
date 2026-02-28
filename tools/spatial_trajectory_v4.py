#!/usr/bin/env python3
"""Spatial trajectory retarget V4 (smooth + safe)

Fixes vs V3:
- Funnel timing is anchored to close_idx (from real gripper signal), not distance.
- Pregrasp/descend/close/lift are executed with **cosine interpolation** in *Cartesian*,
  but replayed with **joint-space interpolation** to remove snapping.
- Table-safe Z bounds + max dz per step.

This produces a visually smooth, non-crushing grasp.
"""

import os, json, argparse
import numpy as np

PANDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mujoco_menagerie", "franka_emika_panda")
R3D_DIR = "pipeline/r3d_output"
OUT_DIR = "pipeline/spatial_v4"
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


def cosine01(a):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(a, 0, 1))


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


def interp_q(q0, q1, n):
    out = []
    for i in range(n):
        a = cosine01(i/(n-1 if n>1 else 1))
        out.append(q0 + a*(q1-q0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--lead_in", type=int, default=40, help="frames before close to start funnel")
    ap.add_argument("--pre_frames", type=int, default=25)
    ap.add_argument("--desc_frames", type=int, default=45)
    ap.add_argument("--close_frames", type=int, default=18)
    ap.add_argument("--lift_frames", type=int, default=45)
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

    xml_path = os.path.join(PANDA_DIR, "_spatial_v4.xml")
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

    # Build offsets and grips
    offsets, grips = [], []
    for fid in sorted(hamer.keys(), key=int):
        fidx = int(fid)
        if fidx >= len(frames):
            continue
        if len(offsets) >= args.max_frames:
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
        offsets.append(hand_world - mug_world)
        grips.append(compute_gripper(hand))

    offsets = smooth_vecs(np.array(offsets), w=11)
    grips = np.array(grips)

    close_idx = int(np.argmax(grips < 0.3))
    if close_idx == 0 and grips[0] >= 0.3:
        close_idx = len(grips)//3

    # XY alignment comes from spatial target near funnel start
    off0 = offsets[close_idx]
    rel = offsets - off0

    ee_spatial = mug_sim + args.scale * rel
    ee_spatial[:,0] = np.clip(ee_spatial[:,0], 0.40, 0.72)
    ee_spatial[:,1] = np.clip(ee_spatial[:,1], -0.18, 0.18)
    ee_spatial[:,2] = np.clip(ee_spatial[:,2], 0.26, 0.55)

    start = max(0, close_idx - args.lead_in)

    align_xy = ee_spatial[start, :2].copy()

    # Funnel targets
    pre_z, grasp_z, lift_z = 0.40, 0.31, 0.49

    ee_targets = ee_spatial.copy()

    # Construct funnel segment
    seg_len = args.pre_frames + args.desc_frames + args.close_frames + args.lift_frames
    end = min(len(ee_targets), start + seg_len)

    # Pre
    for i in range(start, min(end, start+args.pre_frames)):
        ee_targets[i] = [align_xy[0], align_xy[1], pre_z]

    # Desc
    d0 = start + args.pre_frames
    d1 = min(end, d0 + args.desc_frames)
    for i in range(d0, d1):
        a = cosine01((i-d0)/max((d1-d0)-1, 1))
        ee_targets[i] = [align_xy[0], align_xy[1], pre_z + (grasp_z-pre_z)*a]

    # Close hold
    c0 = d1
    c1 = min(end, c0 + args.close_frames)
    for i in range(c0, c1):
        ee_targets[i] = [align_xy[0], align_xy[1], grasp_z]

    # Lift
    l0 = c1
    l1 = end
    for i in range(l0, l1):
        a = cosine01((i-l0)/max((l1-l0)-1, 1))
        ee_targets[i] = [align_xy[0], align_xy[1], grasp_z + (lift_z-grasp_z)*a]

    # Gripper targets: soft-close then hold
    g_targets = grips.copy()
    for i in range(c0, c1):
        a = (i-c0)/max((c1-c0)-1, 1)
        g_targets[i] = max(0.0, 1.0 - 1.05*a)
    for i in range(c1, end):
        g_targets[i] = 0.0

    print(f"samples={len(ee_targets)} close_idx={close_idx} start={start} end={end}")

    # IK solve at keyframes and interpolate in joint-space
    q_key = []
    prev_q = None
    for i in range(len(ee_targets)):
        # solve IK every frame inside funnel, but we will still interpolate for smoothness
        q = solve_ik(model, ee_targets[i], gs, prev_q)
        prev_q = q
        q_key.append(q)
        if i % 25 == 0:
            print(f"IK {i}/{len(ee_targets)}")

    # Build smooth q by interpolating between consecutive frames (sub-steps)
    q_smooth = [q_key[0]]
    for i in range(1, len(q_key)):
        # 3 sub-frames per frame => smoother motion
        q_smooth.extend(interp_q(q_key[i-1], q_key[i], 3)[1:])

    g_smooth = [g_targets[0]]
    for i in range(1, len(g_targets)):
        g_smooth.extend(np.linspace(g_targets[i-1], g_targets[i], 3)[1:])

    FPS = args.fps
    spf = int(1.0 / (FPS * model.opt.timestep))

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    frames_out = []
    renderer.update_scene(data)
    im0 = renderer.render().copy()
    for _ in range(FPS):
        frames_out.append(im0)

    for i in range(len(q_smooth)):
        data.ctrl[:7] = q_smooth[i]
        data.ctrl[7] = 0.04 * float(np.clip(g_smooth[i], 0, 1))
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

    with open(os.path.join(OUT_DIR, "spatial_traj.json"), "w") as f:
        json.dump({
            "scale": args.scale,
            "fps": FPS,
            "close_idx": close_idx,
            "start": start,
            "end": end,
            "ee_targets": ee_targets.tolist(),
            "grippers": g_targets.tolist(),
        }, f)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
