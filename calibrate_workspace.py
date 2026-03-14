"""Calibrate the mapping from R3D world coordinates to sim workspace.

The sim has:
- Robot base at origin (0,0,0)
- Table at varying heights per robot
- Objects on table
- Workspace roughly x=[0.3,0.7], y=[-0.3,0.3], z=[0.4,0.8]

The R3D world has:
- Some arbitrary origin based on ARKit/Record3D
- Objects and wrist in that frame

Strategy: use detected object positions as anchor points to compute
a transform (scale + offset) from R3D → sim coordinates.
"""
import json
import numpy as np
from pathlib import Path

from pipeline_config import CALIB_DIR, OBJECT_DET_DIR

WRIST_DIR = CALIB_DIR
DET_DIR = OBJECT_DET_DIR

# Sim workspace (where objects should be)
SIM_TABLE_Z = 0.43  # block center height on table
SIM_CENTER_X = 0.50  # center of workspace
SIM_CENTER_Y = 0.00  # center of workspace


def r3d_to_sim_axes(pts):
    """Convert R3D (X_right, Y_up, Z_toward) to sim (X_fwd, Y_left, Z_up)."""
    pts = np.array(pts)
    if pts.ndim == 1:
        return np.array([-pts[2], -pts[0], pts[1]])
    return np.column_stack([-pts[:, 2], -pts[:, 0], pts[:, 1]])


def calibrate_session(session_name, wrist_dir=None, det_dir=None):
    """Calibrate a single session from R3D to sim coordinates.

    Args:
        session_name: Name of the session (e.g. 'stack1')
        wrist_dir: Directory containing wrist3d JSONs (default: WRIST_DIR)
        det_dir: Directory containing object detection JSONs (default: DET_DIR)

    Returns:
        Path to the calibrated JSON file, or None on failure.
    """
    if wrist_dir is None:
        wrist_dir = WRIST_DIR
    if det_dir is None:
        det_dir = DET_DIR
    wrist_dir = Path(wrist_dir)
    det_dir = Path(det_dir)

    print(f"\n{'='*50}")
    print(f"Calibration: {session_name}")

    # Load detected objects
    det_path = det_dir / f"{session_name}_objects_clean.json"
    if not det_path.exists():
        print(f"  No detections at {det_path}")
        return None
    with open(det_path) as f:
        dets = json.load(f)["detections"]

    # Load wrist trajectory
    wrist_path = wrist_dir / f"{session_name}_wrist3d.json"
    if not wrist_path.exists():
        print(f"  No wrist data at {wrist_path}")
        return None
    with open(wrist_path) as f:
        wrist_data = json.load(f)

    wrist = np.array(wrist_data["wrist_world_smooth"])
    grasping = wrist_data["grasping"]

    # Object positions in R3D world
    obj_positions = []
    for det in dets:
        if "pos_world" in det:
            obj_positions.append(det["pos_world"])
    obj_positions = np.array(obj_positions) if obj_positions else None

    if obj_positions is not None:
        print(f"  Objects ({len(obj_positions)}):")
        for i, pos in enumerate(obj_positions):
            print(f"    Obj{i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # Wrist during grasp frames
    grasp_frames = [i for i, g in enumerate(grasping) if g]
    if grasp_frames:
        grasp_wrist = wrist[grasp_frames]
        print(f"  Wrist during grasps ({len(grasp_frames)} frames):")
        print(f"    Mean: ({grasp_wrist[:,0].mean():.3f}, {grasp_wrist[:,1].mean():.3f}, {grasp_wrist[:,2].mean():.3f})")

    if obj_positions is None or len(obj_positions) == 0:
        print("  No object positions available for calibration")
        return None

    # Object centroid in R3D
    obj_centroid_r3d = obj_positions.mean(axis=0)

    # Target centroid in sim
    obj_centroid_sim = np.array([SIM_CENTER_X, SIM_CENTER_Y, SIM_TABLE_Z])

    # Convert all coordinates to sim axes
    obj_sim_axes = r3d_to_sim_axes(obj_positions)
    wrist_sim_axes = r3d_to_sim_axes(wrist)

    # Object centroid in sim axes
    obj_centroid_sim_axes = obj_sim_axes.mean(axis=0)

    # Scale: match wrist range to reasonable sim range (~0.35m)
    wrist_range = np.array([
        wrist_sim_axes[:, ax].max() - wrist_sim_axes[:, ax].min()
        for ax in range(3)
    ])
    target_range = 0.35
    if wrist_range.max() > 0.01:
        scale = target_range / wrist_range.max()
    else:
        scale = 1.0

    # Offset: place object centroid at sim workspace center
    offset = obj_centroid_sim - obj_centroid_sim_axes * scale

    print(f"\n  Transform:")
    print(f"    Axis swap: R3D(X,Y,Z) → sim(-Z,-X,Y)")
    print(f"    Scale: {scale:.3f}")
    print(f"    Offset: ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")

    # Apply
    wrist_sim = wrist_sim_axes * scale + offset
    objs_sim = obj_sim_axes * scale + offset

    # CRITICAL: Force object Z to table height (0.425)
    TABLE_Z = 0.425
    z_correction = TABLE_Z - objs_sim[:, 2].mean()
    objs_sim[:, 2] = TABLE_Z
    wrist_sim[:, 2] += z_correction

    print(f"\n  Transformed objects:")
    for i, pos in enumerate(objs_sim):
        print(f"    Obj{i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print(f"  Transformed wrist range:")
    for ax, name in enumerate("XYZ"):
        mn, mx = wrist_sim[:, ax].min(), wrist_sim[:, ax].max()
        print(f"    {name}: [{mn:.3f}, {mx:.3f}]")

    # Save calibration
    calib = {
        "session": session_name,
        "r3d_to_sim": {
            "obj_centroid_r3d": obj_centroid_r3d.tolist(),
            "obj_centroid_sim": obj_centroid_sim.tolist(),
            "scale": float(scale),
        },
        "objects_sim": objs_sim.tolist(),
        "wrist_sim": wrist_sim.tolist(),
        "grasping": grasping,
    }
    out_path = wrist_dir / f"{session_name}_calibrated.json"
    with open(out_path, "w") as f:
        json.dump(calib, f)
    print(f"  Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Calibrate specific sessions from CLI args
        for sess in sys.argv[1:]:
            calibrate_session(sess)
    else:
        # Default: calibrate known sessions
        sessions = ["stack1", "stack2", "picknplace2", "sort2"]
        for sess in sessions:
            try:
                calibrate_session(sess)
            except Exception as e:
                print(f"  ERROR: {e}")
