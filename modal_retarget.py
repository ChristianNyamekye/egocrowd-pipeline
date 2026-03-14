"""Run dex-retargeting on Modal (Linux GPU) for HaMeR MANO → Shadow Hand.

Usage: python modal_retarget.py stack2
"""
import modal
import json
import sys
from pathlib import Path

app = modal.App("dex-retarget-shadow")

retarget_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("numpy")
    .run_commands("pip uninstall pinocchio -y || true")  # remove wrong pinocchio package
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("dex_retargeting")
    .run_commands(
        "cd /root && git clone --depth 1 https://github.com/dexsuite/dex-urdf.git",
        "cp -r /root/dex-urdf/robots/hands/shadow_hand /root/shadow_hand",
        "ls /root/shadow_hand/",
    )
    .run_commands("python -c 'import dex_retargeting; print(\"OK\")'")
)

@app.function(image=retarget_image, timeout=300)
def retarget_session(hamer_data: dict, session: str) -> dict:
    """Retarget MANO joints to Shadow Hand joint angles."""
    import numpy as np
    from dex_retargeting.retargeting_config import RetargetingConfig
    from dex_retargeting.seq_retarget import SeqRetargeting
    import dex_retargeting
    from pathlib import Path as P

    pkg = P(dex_retargeting.__file__).parent

    # Find Shadow Hand URDF — search multiple locations
    import subprocess, glob
    
    # Try to find in dex_retargeting package
    urdf_path = None
    search_patterns = [
        str(pkg / "**" / "shadow_hand_right.urdf"),
        "/usr/local/lib/python3.11/site-packages/**/shadow_hand_right.urdf",
    ]
    for pattern in search_patterns:
        found = glob.glob(pattern, recursive=True)
        if found:
            urdf_path = found[0]
            break
    
    # If not found, try dex-urdf
    if not urdf_path:
        try:
            import dex_urdf
            urdf_pkg = P(dex_urdf.__file__).parent
            found = list(urdf_pkg.rglob("shadow_hand_right.urdf"))
            if found:
                urdf_path = str(found[0])
        except ImportError:
            pass
    
    if not urdf_path:
        # Use the URDF we cloned from dex-urdf
        urdf_path = "/root/shadow_hand/shadow_hand_right.urdf"
    
    if not P(urdf_path).exists():
        return {"error": f"Shadow Hand URDF not found at {urdf_path}"}
    
    print(f"URDF: {urdf_path}")
    config = RetargetingConfig(
        type="position",
        urdf_path=urdf_path,
        target_link_names=[
            "thtip", "fftip", "mftip", "rftip", "lftip",
            "thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle",
        ],
        target_link_human_indices=np.array([4, 8, 12, 16, 20, 2, 6, 10, 14, 18]),
        add_dummy_free_joint=True,
    )
    retargeting = config.build()
    joint_names = retargeting.joint_names
    print(f"Joint names ({len(joint_names)}): {joint_names}")

    results = []
    valid = 0
    total = len(hamer_data["results"])

    for r in hamer_data["results"]:
        frame_idx = r["frame_idx"]
        hands = r.get("hands", [])

        if not hands:
            results.append({"frame": frame_idx, "joints": None, "grasping": False})
            continue

        h = hands[0]
        j21 = h.get("joints_21", [])
        grasping = h.get("grasping", False)

        # Use landmarks_3d (MediaPipe 3D) if available, else skip
        lm3d = h.get("landmarks_3d", [])
        if not lm3d or len(lm3d) != 21:
            results.append({"frame": frame_idx, "joints": None, "grasping": grasping})
            continue

        # Convert landmarks_3d to Nx3 array
        # x, y are in pixel space; z is relative (in some unit)
        # Normalize to [0,1] space with z scaled
        joints_np = np.array([[lm["x"], lm["y"], lm.get("z", 0)] for lm in lm3d], dtype=float)
        
        # Check for valid z values
        if np.max(np.abs(joints_np[:, 2])) < 0.001:
            results.append({"frame": frame_idx, "joints": None, "grasping": grasping})
            continue
        
        # Normalize: center at wrist (joint 0), scale by hand size
        wrist = joints_np[0].copy()
        joints_np -= wrist  # center at wrist
        
        # Scale: typical pixel distance between wrist and middle finger tip ~200px
        # dex-retargeting expects ~cm scale
        scale = np.max(np.abs(joints_np[:, :2])) / 0.20  # normalize to ~20cm hand
        if scale > 0:
            joints_np[:, :2] /= scale
        # Z is already in relative units from MediaPipe, scale similarly
        joints_np[:, 2] /= max(scale * 0.01, 0.01)

        try:
            robot_qpos = retargeting.retarget(joints_np)
            results.append({
                "frame": frame_idx,
                "joints": robot_qpos.tolist(),
                "grasping": grasping,
            })
            valid += 1
        except Exception as e:
            results.append({"frame": frame_idx, "joints": None, "grasping": grasping, "error": str(e)})

    print(f"Retargeted: {valid}/{total} frames")

    return {
        "session": session,
        "total_frames": total,
        "valid_frames": valid,
        "joint_names": joint_names,
        "results": results,
    }


@app.local_entrypoint()
def main(session: str = "stack2"):
    
    modal_results = Path(__file__).parent / "modal_results"
    hamer_file = modal_results / f"{session}_gpu_hands.json"
    
    if not hamer_file.exists():
        print(f"HaMeR data not found: {hamer_file}")
        return
    
    print(f"Loading {hamer_file}...")
    hamer_data = json.loads(hamer_file.read_text())
    print(f"  {len(hamer_data['results'])} frames")
    
    print("Running retargeting on Modal...")
    result = retarget_session.remote(hamer_data, session)
    
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    
    out_dir = Path(__file__).parent / "retargeted"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{session}_shadow_hand.json"
    out_file.write_text(json.dumps(result, indent=2))
    
    print(f"Done! {result['valid_frames']}/{result['total_frames']} frames retargeted")
    print(f"Saved to {out_file}")
