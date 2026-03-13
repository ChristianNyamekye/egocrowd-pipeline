"""Reconstruct 3D wrist trajectories from wrist_pixel + R3D depth maps.
For each frame with a wrist_pixel, look up depth, unproject to 3D, transform to world.
Then normalize all world trajectories to a common sim workspace.
"""
import json, sys
import numpy as np
from pathlib import Path
import zipfile, io

RAW = Path(__file__).resolve().parent / "raw_captures"
RETARGET = Path(__file__).resolve().parent / "gpu_retargeted"
OUTPUT = Path(__file__).resolve().parent / "wrist_trajectories"
OUTPUT.mkdir(exist_ok=True)


def load_r3d_depth(r3d_path, frame_idx):
    """Load depth map for a specific R3D frame."""
    import liblzfse
    with zipfile.ZipFile(r3d_path) as z:
        meta = json.load(z.open("metadata"))
        with z.open(f"rgbd/{frame_idx}.depth") as f:
            raw = liblzfse.decompress(f.read())
        depth = np.frombuffer(raw, dtype=np.float32).reshape(meta["dh"], meta["dw"])
    return meta, depth


def pixel_to_world(u, v, depth_map, meta, r3d_frame):
    """Unproject pixel to world coordinates."""
    dh, dw = meta["dh"], meta["dw"]
    h, w = meta["h"], meta["w"]

    # Intrinsics
    if meta.get("perFrameIntrinsicCoeffs") and r3d_frame < len(meta["perFrameIntrinsicCoeffs"]):
        c = meta["perFrameIntrinsicCoeffs"][r3d_frame]
        fx, fy, cx, cy = c[0], c[1], c[2], c[3]
    else:
        K = meta["K"]
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

    # Depth lookup (median of 5x5 patch)
    du = int(np.clip(u * dw / w, 0, dw-1))
    dv = int(np.clip(v * dh / h, 0, dh-1))
    r = 3
    patch = depth_map[max(0,dv-r):dv+r+1, max(0,du-r):du+r+1]
    valid = patch[(patch > 0.05) & (patch < 3.0)]
    if len(valid) == 0:
        return None
    Z = float(np.median(valid))

    # Unproject to camera coords
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    point_cam = np.array([X, Y, Z])

    # Camera → world
    pose_idx = min(r3d_frame, len(meta["poses"]) - 1)
    pose = meta["poses"][pose_idx]
    tx, ty, tz = pose[0], pose[1], pose[2]
    qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
    ])
    point_world = R @ point_cam + np.array([tx, ty, tz])
    return point_world


def smooth_trajectory(traj, alpha=0.15):
    """Bidirectional EMA smoothing (zero-lag)."""
    n = len(traj)
    if n < 3: return traj
    arr = np.array(traj)
    # Forward pass
    fwd = np.zeros_like(arr)
    fwd[0] = arr[0]
    for i in range(1, n):
        fwd[i] = alpha * arr[i] + (1 - alpha) * fwd[i-1]
    # Backward pass
    bwd = np.zeros_like(arr)
    bwd[-1] = arr[-1]
    for i in range(n-2, -1, -1):
        bwd[i] = alpha * arr[i] + (1 - alpha) * bwd[i+1]
    # Average
    return ((fwd + bwd) / 2).tolist()


def process_session(session):
    print(f"\n{'='*60}")
    print(f"Wrist 3D Reconstruction: {session}")
    print(f"{'='*60}")

    # Load retarget data
    ret_path = RETARGET / f"{session}_retargeted.json"
    with open(ret_path) as f:
        ret = json.load(f)
    ts = ret["timesteps"]

    # Find R3D file
    r3d_dir = RAW / session
    r3d_files = list(r3d_dir.glob("*.r3d"))
    if not r3d_files:
        print(f"  ERROR: No .r3d in {r3d_dir}"); return None
    r3d_path = r3d_files[0]

    # Load metadata once
    import liblzfse
    with zipfile.ZipFile(r3d_path) as z:
        meta = json.load(z.open("metadata"))
    print(f"  R3D: {meta['w']}x{meta['h']} @ {meta['fps']}fps, {len(meta['poses'])} poses")
    fps_ratio = meta["fps"] / 30.0  # R3D 60fps, our pipeline 30fps

    # Process each frame
    wrist_3d_world = []
    grasping = []
    failed = 0
    for i, t in enumerate(ts):
        wp = t.get("wrist_pixel")
        if wp is None:
            wrist_3d_world.append(None)
            grasping.append(t.get("grasping", False))
            failed += 1
            continue

        r3d_idx = min(int(i * fps_ratio), len(meta["poses"]) - 1)

        # Load depth for this frame
        try:
            _, depth = load_r3d_depth(r3d_path, r3d_idx)
        except (KeyError, Exception):
            wrist_3d_world.append(None)
            grasping.append(t.get("grasping", False))
            failed += 1
            continue

        point = pixel_to_world(wp[0], wp[1], depth, meta, r3d_idx)
        wrist_3d_world.append(point.tolist() if point is not None else None)
        grasping.append(t.get("grasping", False))
        if point is None:
            failed += 1

    valid = sum(1 for w in wrist_3d_world if w is not None)
    print(f"  Reconstructed: {valid}/{len(ts)} frames ({failed} failed)")

    # Fill gaps with linear interpolation
    arr = []
    for w in wrist_3d_world:
        arr.append(w if w is not None else [np.nan, np.nan, np.nan])
    arr = np.array(arr)

    # Interpolate NaN gaps
    for ax in range(3):
        col = arr[:, ax]
        nans = np.isnan(col)
        if nans.all(): continue
        good = ~nans
        indices = np.arange(len(col))
        col[nans] = np.interp(indices[nans], indices[good], col[good])
        arr[:, ax] = col

    # Smooth
    smoothed = smooth_trajectory(arr.tolist(), alpha=0.12)
    smoothed = np.array(smoothed)

    # Stats
    print(f"  World coords range:")
    for ax, name in enumerate("XYZ"):
        mn, mx = smoothed[:, ax].min(), smoothed[:, ax].max()
        print(f"    {name}: [{mn:.4f}, {mx:.4f}] (range={mx-mn:.4f})")

    # Frame-to-frame displacement
    disps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    print(f"  Displacement: mean={disps.mean():.4f} max={disps.max():.4f}")

    # Save
    result = {
        "session": session,
        "n_frames": len(ts),
        "n_valid": valid,
        "wrist_world_raw": [w if w is not None else None for w in wrist_3d_world],
        "wrist_world_smooth": smoothed.tolist(),
        "grasping": grasping,
        "world_range": {
            "x": [float(smoothed[:, 0].min()), float(smoothed[:, 0].max())],
            "y": [float(smoothed[:, 1].min()), float(smoothed[:, 1].max())],
            "z": [float(smoothed[:, 2].min()), float(smoothed[:, 2].max())],
        }
    }
    out_path = OUTPUT / f"{session}_wrist3d.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


if __name__ == "__main__":
    sessions = sys.argv[1:] or ["stack1", "stack2", "picknplace1", "picknplace2", "sort2"]
    for s in sessions:
        try:
            process_session(s)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
