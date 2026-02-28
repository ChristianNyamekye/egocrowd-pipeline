#!/usr/bin/env python3
"""
Spatial Trajectory Retarget V1
=============================
Goal: make the robot follow the *spatial* trajectory extracted from the iPhone recording.

Approach:
- Use LiDAR depth + camera intrinsics/poses to compute hand_world(t)
- Use object_poses_3d.json to get mug_world(t)
- Compute relative trajectory: offset(t) = hand_world(t) - mug_world(t)
- In MuJoCo scene (mjx_panda.xml + mug), set mug_sim_pos and map:
    ee_target(t) = mug_sim_pos + scale * offset(t)
- Solve IK per frame using the gripper site (position-only, but site encodes orientation)
- Interpolate joint angles in time and replay

Outputs:
- pipeline/spatial_v1/spatial_traj.json (targets + qpos + gripper)
- pipeline/spatial_v1/spatial_replay.mp4 (H.264)

Usage:
  python tools/spatial_trajectory_v1.py
  python tools/spatial_trajectory_v1.py --scale 3.0
"""

import os, json, argparse
import numpy as np

PANDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mujoco_menagerie", "franka_emika_panda")

R3D_DIR = "pipeline/r3d_output"
OUT_DIR = "pipeline/spatial_v1"
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
    # rgb_size=[w,h], depth_size=[h,w]
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


def cosine(t):
    return 0.5 * (1 - np.cos(np.pi * np.clip(t, 0, 1)))


def solve_ik(model, target, site_id, start_q=None):
    import mujoco
    d2 = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d2, 0)
    if start_q is not None:
        d2.qpos[:7] = start_q
    q = d2.qpos[:7].copy()
    for _ in range(1200):
        d2.qpos[:7] = q
        mujoco.mj_forward(model, d2)
        err = target - d2.site_xpos[site_id].copy()
        if np.linalg.norm(err) < 0.004:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, None, site_id)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05*np.eye(3), err)
        q += dq * min(0.4, 0.08/(np.linalg.norm(dq)+1e-8))
        # clamp to joint ranges
        for j in range(7):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{j+1}')
            q[j] = np.clip(q[j], *model.jnt_range[jid])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=220, help="cap frames for faster iteration")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load metadata
    meta = json.load(open(os.path.join(R3D_DIR, "metadata.json")))
    frames = meta["frames"]
    K = np.array(meta["camera_K"], dtype=float)
    rgb_size = meta["rgb_size"]
    depth_size = meta["depth_size"]

    hamer = json.load(open(os.path.join(R3D_DIR, "hamer_results.json")))
    obj = json.load(open(os.path.join(R3D_DIR, "object_poses_3d.json")))

    obj_map = {p["frame"]: np.array(p["pose_3d"], dtype=float) for p in obj["poses"] if p.get("detected")}
    obj_frames = sorted(obj_map.keys())

    import mujoco
    import cv2

    # Build scene
    xml_path = os.path.join(PANDA_DIR, "_spatial_v1.xml")
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

    # Build per-frame targets
    samples = []
    for fid in sorted(hamer.keys(), key=int):
        fidx = int(fid)
        if fidx >= len(frames):
            continue
        if len(samples) >= args.max_frames:
            break

        hands = hamer[fid]
        if not hands:
            continue
        hand = hands[0]

        # nearest object
        if obj_frames:
            nearest = min(obj_frames, key=lambda x: abs(x - fidx))
            mug_world = obj_map[nearest]
        else:
            continue

        # LiDAR hand_world
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

        offset = hand_world - mug_world

        # Map into sim
        ee_tgt = mug_sim + args.scale * offset

        # Clamp to reachable-ish region
        ee_tgt[0] = float(np.clip(ee_tgt[0], 0.35, 0.75))
        ee_tgt[1] = float(np.clip(ee_tgt[1], -0.25, 0.25))
        ee_tgt[2] = float(np.clip(ee_tgt[2], 0.18, 0.55))

        g = compute_gripper(hand)
        samples.append({
            "frame": fidx,
            "hand_world": hand_world.tolist(),
            "mug_world": mug_world.tolist(),
            "offset": offset.tolist(),
            "ee_target": ee_tgt.tolist(),
            "gripper": g,
        })

    if len(samples) < 30:
        print(f"Not enough samples: {len(samples)}")
        return

    # Solve IK per sample
    q_list = []
    prev_q = None
    for i, s in enumerate(samples):
        q = solve_ik(model, np.array(s["ee_target"], dtype=float), gs, prev_q)
        prev_q = q
        q_list.append(q)
        if i % 20 == 0:
            print(f"IK {i}/{len(samples)}")

    # Replay + render
    FPS = args.fps
    spf = int(1.0 / (FPS * model.opt.timestep))

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    frames_out = []
    # lead-in
    renderer.update_scene(data)
    im0 = renderer.render().copy()
    for _ in range(FPS):
        frames_out.append(im0)

    # Run
    for i in range(len(samples)):
        q = q_list[i]
        grip = samples[i]["gripper"]
        # map to mujoco gripper ctrl: 0.04=open, 0=closed
        grip_ctrl = 0.04 * grip

        data.ctrl[:7] = q
        data.ctrl[7] = grip_ctrl

        for _ in range(spf):
            mujoco.mj_step(model, data)

        renderer.update_scene(data)
        frames_out.append(renderer.render().copy())

    # tail
    renderer.update_scene(data)
    imf = renderer.render().copy()
    for _ in range(FPS):
        frames_out.append(imf)

    raw = os.path.join(OUT_DIR, "spatial_replay_raw.mp4")
    final = os.path.join(OUT_DIR, "spatial_replay.mp4")

    h, w = frames_out[0].shape[:2]
    import cv2
    wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for im in frames_out:
        wr.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    wr.release()

    # h264 encode
    os.system(f'"{FFMPEG}" -y -i "{raw}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{final}" 2>nul')

    out_path = final if os.path.exists(final) and os.path.getsize(final) > 1000 else raw

    traj = {
        "scale": args.scale,
        "fps": FPS,
        "num_samples": len(samples),
        "samples": samples,
        "qpos": [q.tolist() for q in q_list],
    }
    with open(os.path.join(OUT_DIR, "spatial_traj.json"), "w") as f:
        json.dump(traj, f)

    print(f"Saved: {out_path}")
    print(f"Saved: {os.path.join(OUT_DIR, 'spatial_traj.json')}")


if __name__ == "__main__":
    main()
