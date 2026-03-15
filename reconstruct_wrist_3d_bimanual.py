"""Reconstruct 3D wrist trajectories for BOTH hands using LiDAR depth.

Takes per-hand wrist_pixel from detection data + R3D depth maps → 3D world coords.
Output: wrist_trajectories/stack2_bimanual_wrist3d.json
"""
import json, sys, zipfile, io
import numpy as np
from pathlib import Path

RAW = Path(__file__).parent / "raw_captures"
MODAL = Path(__file__).parent / "modal_results"
OUTPUT = Path(__file__).parent / "wrist_trajectories"
OUTPUT.mkdir(exist_ok=True)


def load_r3d_depth(r3d_path, frame_idx):
    import liblzfse
    with zipfile.ZipFile(r3d_path) as z:
        meta = json.load(z.open("metadata"))
        depth_file = f"rgbd/{frame_idx}.depth"
        if depth_file not in z.namelist():
            return meta, None
        with z.open(depth_file) as f:
            raw = liblzfse.decompress(f.read())
        depth = np.frombuffer(raw, dtype=np.float32).reshape(meta["dh"], meta["dw"])
    return meta, depth


def pixel_to_3d(u, v, depth_map, meta, frame_idx):
    """Unproject pixel (u,v) to 3D camera coordinates using depth map."""
    dh, dw = meta["dh"], meta["dw"]
    h, w = meta["h"], meta["w"]

    # Intrinsics
    if meta.get("perFrameIntrinsicCoeffs") and frame_idx < len(meta["perFrameIntrinsicCoeffs"]):
        c = meta["perFrameIntrinsicCoeffs"][frame_idx]
        fx, fy, cx, cy = c[0], c[1], c[2], c[3]
    else:
        K = meta["K"]
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

    # Depth lookup (median of 5x5 patch)
    du = int(np.clip(u * dw / w, 0, dw - 1))
    dv = int(np.clip(v * dh / h, 0, dh - 1))
    r = 3
    patch = depth_map[max(0, dv-r):dv+r+1, max(0, du-r):du+r+1]
    valid = patch[(patch > 0.05) & (patch < 3.0)]
    if len(valid) == 0:
        return None
    Z = float(np.median(valid))

    # Unproject
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def run(session="stack2"):
    r3d_path = RAW / session / f"{session}.r3d"
    det_path = MODAL / f"{session}_gpu_hands.json"

    if not r3d_path.exists():
        print(f"R3D not found: {r3d_path}")
        return
    if not det_path.exists():
        print(f"Detection data not found: {det_path}")
        return

    dets = json.loads(det_path.read_text())
    print(f"Detections: {len(dets['results'])} frames")

    # Load R3D metadata
    with zipfile.ZipFile(r3d_path) as z:
        meta = json.load(z.open("metadata"))
    print(f"R3D: {meta['w']}x{meta['h']}, depth {meta['dw']}x{meta['dh']}")

    left_3d, right_3d = [], []
    valid_left, valid_right = 0, 0

    for ri, r in enumerate(dets["results"]):
        frame_idx = r["frame_idx"]

        # Get wrist pixels for each hand
        left_wp, right_wp = None, None
        for h in r.get("hands", []):
            wp = h.get("wrist_pixel")
            if not wp:
                continue
            side = h.get("hand", "unknown")
            if side == "left":
                left_wp = wp
            elif side == "right":
                right_wp = wp
            elif wp[0] > meta["w"] / 2:
                left_wp = wp
            else:
                right_wp = wp

        # Load depth for this frame
        _, depth = load_r3d_depth(r3d_path, frame_idx)

        l3d, r3d_pt = None, None
        if depth is not None:
            if left_wp:
                l3d = pixel_to_3d(left_wp[0], left_wp[1], depth, meta, frame_idx)
                if l3d is not None:
                    valid_left += 1
            if right_wp:
                r3d_pt = pixel_to_3d(right_wp[0], right_wp[1], depth, meta, frame_idx)
                if r3d_pt is not None:
                    valid_right += 1

        left_3d.append(l3d.tolist() if l3d is not None else None)
        right_3d.append(r3d_pt.tolist() if r3d_pt is not None else None)

        if (ri + 1) % 50 == 0:
            print(f"  {ri+1}/{len(dets['results'])} frames, left={valid_left} right={valid_right}")

    print(f"Valid: left={valid_left}, right={valid_right} of {len(dets['results'])}")

    # Interpolate gaps
    for traj in [left_3d, right_3d]:
        last_valid = None
        for i in range(len(traj)):
            if traj[i] is not None:
                last_valid = traj[i]
            elif last_valid is not None:
                traj[i] = last_valid

    out = {
        "session": session,
        "total_frames": len(dets["results"]),
        "valid_left": valid_left,
        "valid_right": valid_right,
        "left_wrist_3d": left_3d,
        "right_wrist_3d": right_3d,
    }

    out_path = OUTPUT / f"{session}_bimanual_wrist3d.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")

    # Print trajectory stats
    for name, traj in [("left", left_3d), ("right", right_3d)]:
        valid = [np.array(t) for t in traj if t is not None]
        if valid:
            arr = np.array(valid)
            print(f"  {name}: X=[{arr[:,0].min():.3f},{arr[:,0].max():.3f}] Y=[{arr[:,1].min():.3f},{arr[:,1].max():.3f}] Z=[{arr[:,2].min():.3f},{arr[:,2].max():.3f}]")


if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    run(session)
