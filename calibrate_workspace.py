"""Calibrate the mapping from R3D world coordinates to sim workspace.

The sim has:
- Robot base at origin (0,0,0)
- Table at varying heights per robot
- Objects on table
- Workspace roughly x=[0.3,0.7], y=[-0.3,0.3], z=[0.78,1.2]

The R3D world has:
- Some arbitrary origin based on ARKit/Record3D
- Objects and wrist in that frame

Strategy: use grasp centroid as anchor point to align the wrist trajectory
with detected/manual object positions. Scale=1.0 because R3D depth data
is physical meters from Apple LiDAR.
"""
import json
import numpy as np
from pathlib import Path

from pipeline_config import CALIB_DIR, OBJECT_DET_DIR
from trim_trajectory import clean_grasping_signal

WRIST_DIR = CALIB_DIR
DET_DIR = OBJECT_DET_DIR

# Sim workspace (where objects should be)
SIM_TABLE_Z = 0.81  # block center height on G1 table (0.78 + BLOCK_HALF 0.03)
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

    # Load detected objects (full dicts, not just positions)
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

    # --- Object positions: handle manual vs detected ---
    obj_sim_positions = []
    for det in dets:
        if "pos_world" not in det:
            continue
        pos = det["pos_world"]
        if det.get("source") == "manual":
            # Manual objects are already in sim coordinates — no axis swap
            obj_sim_positions.append(np.array(pos))
            print(f"  Object '{det.get('label', '?')}': manual (sim coords) "
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        else:
            # Detected objects (GroundingDINO) are in R3D world — axis swap
            sim_pos = r3d_to_sim_axes(np.array(pos))
            obj_sim_positions.append(sim_pos)
            print(f"  Object '{det.get('label', '?')}': detected (R3D→sim) "
                  f"({sim_pos[0]:.3f}, {sim_pos[1]:.3f}, {sim_pos[2]:.3f})")

    if not obj_sim_positions:
        print("  No object positions available for calibration")
        return None

    objs_sim = np.array(obj_sim_positions)
    obj_centroid_sim = objs_sim.mean(axis=0)
    print(f"  Object centroid (sim): ({obj_centroid_sim[0]:.3f}, "
          f"{obj_centroid_sim[1]:.3f}, {obj_centroid_sim[2]:.3f})")

    # --- Wrist axis swap (always R3D world coords) ---
    wrist_sim_axes = r3d_to_sim_axes(wrist)

    # --- Scale = 1.0 (R3D is physical meters from Apple LiDAR) ---
    scale = 1.0

    # --- Clean grasping signal: debounce + onset detection ---
    # Proximity gate not possible here (objects not aligned yet), but
    # debounce + onset removes early-trajectory false positives
    grasping_arr = np.array(grasping, dtype=float)
    grasping_arr = clean_grasping_signal(grasping_arr)
    grasping = grasping_arr.tolist()

    # --- Anchor via grasp centroid ---
    grasp_frames = [i for i, g in enumerate(grasping) if g]
    if not grasp_frames:
        print("  WARNING: No grasp frames found, using full trajectory centroid")
        grasp_frames = list(range(len(grasping)))

    grasp_centroid = wrist_sim_axes[grasp_frames].mean(axis=0)
    print(f"  Grasp centroid ({len(grasp_frames)} frames): "
          f"({grasp_centroid[0]:.3f}, {grasp_centroid[1]:.3f}, {grasp_centroid[2]:.3f})")

    # Offset: shift so grasp centroid aligns with object centroid
    offset = obj_centroid_sim - grasp_centroid
    wrist_sim = wrist_sim_axes + offset

    print(f"\n  Transform:")
    print(f"    Axis swap: R3D(X,Y,Z) → sim(-Z,-X,Y)")
    print(f"    Scale: {scale:.3f}")
    print(f"    Anchor: grasp_centroid")
    print(f"    Offset: ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")

    # --- Z correction: force objects to table height ---
    G1_TABLE_HEIGHT = 0.78
    TABLE_Z = G1_TABLE_HEIGHT + 0.03  # block center (BLOCK_HALF = 0.03)
    z_correction = TABLE_Z - objs_sim[:, 2].mean()
    objs_sim[:, 2] = TABLE_Z
    wrist_sim[:, 2] += z_correction

    print(f"\n  Z correction: {z_correction:+.3f}")
    print(f"  Transformed objects:")
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
            "obj_centroid_sim": obj_centroid_sim.tolist(),
            "grasp_centroid": grasp_centroid.tolist(),
            "scale": float(scale),
            "anchor": "grasp_centroid",
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
