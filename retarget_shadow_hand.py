#!/usr/bin/env python3
"""Retarget HaMeR MANO hand joints to Shadow Hand using dex-retargeting.

Input: modal_results/stack2_gpu_hands.json (MANO 21-joint positions per frame)
Output: pipeline/retargeted/stack2_shadow_hand.json (Shadow Hand joint angles per frame)

Run with conda:
  conda run -n dexretarget python retarget_shadow_hand.py stack2
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\Users\chris\clawd\_dex_retarget\src")

from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

MODAL = Path(__file__).parent / "modal_results"
OUTPUT = Path(__file__).parent / "retargeted"
OUTPUT.mkdir(exist_ok=True)

CONFIG = Path(r"C:\Users\chris\clawd\_dex_retarget\src\dex_retargeting\configs\offline\shadow_hand_right.yml")


def run_retargeting(session: str):
    print(f"Loading HaMeR data for {session}...")
    d = json.loads((MODAL / f"{session}_gpu_hands.json").read_text())
    
    total = len(d["results"])
    print(f"  {total} frames, {d['frames_with_hands']} with hands, detection rate {d['detection_rate']:.1%}")
    
    # Load retargeting config
    config = RetargetingConfig.from_dict(
        {"retargeting": {
            "type": "position",
            "urdf_path": str(Path(r"C:\Users\chris\clawd\_dex_retarget\src\dex_retargeting\configs") / "shadow_hand" / "shadow_hand_right.urdf"),
            "target_link_names": ["thtip", "fftip", "mftip", "rftip", "lftip",
                                   "thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle"],
            "target_link_human_indices": [4, 8, 12, 16, 20, 2, 6, 10, 14, 18],
            "add_dummy_free_joint": True,
            "low_pass_alpha": 0.8,
        }}
    )
    retargeting = SeqRetargeting(config)
    print(f"Retargeting joint names: {retargeting.joint_names}")
    
    results = []
    skipped = 0
    
    for r in d["results"]:
        frame_idx = r["frame_idx"]
        hands = r.get("hands", [])
        
        if not hands:
            results.append({"frame": frame_idx, "joints": None, "grasping": False})
            skipped += 1
            continue
        
        # Use first hand detection
        h = hands[0]
        joints_21 = h.get("joints_21", [])
        
        # Check if we have valid 3D joint data
        if not joints_21 or all(j == 0 for j in joints_21):
            # No 3D data — use wrist pixel as fallback
            results.append({"frame": frame_idx, "joints": None, "grasping": h.get("grasping", False)})
            skipped += 1
            continue
        
        # joints_21: MANO format, 21 joints
        # Each joint is [x, y, z] in some coordinate system
        joints_np = np.array(joints_21, dtype=float)
        if joints_np.ndim == 1:
            # Flat array → reshape
            if len(joints_np) == 63:  # 21*3
                joints_np = joints_np.reshape(21, 3)
            else:
                results.append({"frame": frame_idx, "joints": None, "grasping": h.get("grasping", False)})
                skipped += 1
                continue
        
        # Retarget
        try:
            robot_qpos = retargeting.retarget(joints_np)
            results.append({
                "frame": frame_idx,
                "joints": robot_qpos.tolist(),
                "joint_names": retargeting.joint_names,
                "grasping": h.get("grasping", False),
            })
        except Exception as e:
            results.append({"frame": frame_idx, "joints": None, "grasping": h.get("grasping", False)})
            skipped += 1
    
    valid = sum(1 for r in results if r["joints"] is not None)
    print(f"  Retargeted: {valid}/{total} frames ({skipped} skipped)")
    
    out = OUTPUT / f"{session}_shadow_hand.json"
    out.write_text(json.dumps({
        "session": session,
        "total_frames": total,
        "valid_frames": valid,
        "joint_names": retargeting.joint_names,
        "results": results,
    }, indent=2))
    print(f"  Saved to {out}")
    return out


if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    run_retargeting(session)
