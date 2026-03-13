"""Calibrate the mapping from R3D world coordinates to Franka sim workspace.

The Franka sim has:
- Robot base at origin (0,0,0)
- Table at (0.5, 0, 0.2), top at z=0.4
- Objects on table at z~0.43
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

WRIST_DIR = Path(__file__).resolve().parent / "wrist_trajectories"
DET_DIR = Path(__file__).resolve().parent / "object_detections"

# Franka sim workspace (where objects should be)
SIM_TABLE_Z = 0.43  # block center height on table
SIM_CENTER_X = 0.50  # center of workspace
SIM_CENTER_Y = 0.00  # center of workspace

sessions = ["stack1", "stack2", "picknplace2", "sort2"]  # skip picknplace1 (broken)

for sess in sessions:
    print(f"\n{'='*50}")
    print(f"Calibration: {sess}")

    # Load detected objects
    det_path = DET_DIR / f"{sess}_objects_clean.json"
    if not det_path.exists():
        print(f"  No detections"); continue
    with open(det_path) as f:
        dets = json.load(f)["detections"]

    # Load wrist trajectory
    wrist_path = WRIST_DIR / f"{sess}_wrist3d.json"
    if not wrist_path.exists():
        print(f"  No wrist data"); continue
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

    # Compute transform: R3D → sim
    # Simple approach: translate so objects are at sim workspace center, scale to reasonable range
    if obj_positions is not None and len(obj_positions) > 0:
        # Object centroid in R3D
        obj_centroid_r3d = obj_positions.mean(axis=0)

        # Target centroid in sim
        obj_centroid_sim = np.array([SIM_CENTER_X, SIM_CENTER_Y, SIM_TABLE_Z])

        # ARKit/R3D world frame: Y up, Z toward viewer (backward), X right
        # MuJoCo/Franka sim: Z up, X forward, Y left
        # Mapping: R3D_X → sim_Y (negated, right→left), R3D_Y → sim_Z, R3D_Z → sim_X (negated, toward→forward)
        def r3d_to_sim_axes(pts):
            """Convert R3D (X_right, Y_up, Z_toward) to sim (X_fwd, Y_left, Z_up)."""
            pts = np.array(pts)
            if pts.ndim == 1:
                return np.array([-pts[2], -pts[0], pts[1]])
            return np.column_stack([-pts[:, 2], -pts[:, 0], pts[:, 1]])

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
        # The R3D depth→Z mapping is unreliable, but we KNOW objects sit on the table
        TABLE_Z = 0.425
        z_correction = TABLE_Z - objs_sim[:, 2].mean()
        objs_sim[:, 2] = TABLE_Z
        wrist_sim[:, 2] += z_correction  # shift wrist Z by same amount

        print(f"\n  Transformed objects:")
        for i, pos in enumerate(objs_sim):
            print(f"    Obj{i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        print(f"  Transformed wrist range:")
        for ax, name in enumerate("XYZ"):
            mn, mx = wrist_sim[:, ax].min(), wrist_sim[:, ax].max()
            print(f"    {name}: [{mn:.3f}, {mx:.3f}]")

        # Save calibration
        calib = {
            "session": sess,
            "r3d_to_sim": {
                "obj_centroid_r3d": obj_centroid_r3d.tolist(),
                "obj_centroid_sim": obj_centroid_sim.tolist(),
                "scale": float(scale),
            },
            "objects_sim": objs_sim.tolist(),
            "wrist_sim": wrist_sim.tolist(),
            "grasping": grasping,
        }
        out_path = WRIST_DIR / f"{sess}_calibrated.json"
        with open(out_path, "w") as f:
            json.dump(calib, f)
        print(f"  Saved: {out_path}")
