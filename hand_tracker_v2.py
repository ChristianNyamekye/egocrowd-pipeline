"""
Hand Tracker v2 — Uses MediaPipe Tasks API (0.10+).
Extracts 21 hand landmarks per frame from video, computes joint angles + grasp state.

Usage:
    python hand_tracker_v2.py input_video.mp4 -o output.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Model path
MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")

FINGER_LANDMARKS = {
    "thumb":  {"CMC": 1, "MCP": 2, "IP": 3, "TIP": 4},
    "index":  {"MCP": 5, "PIP": 6, "DIP": 7, "TIP": 8},
    "middle": {"MCP": 9, "PIP": 10, "DIP": 11, "TIP": 12},
    "ring":   {"MCP": 13, "PIP": 14, "DIP": 15, "TIP": 16},
    "pinky":  {"MCP": 17, "PIP": 18, "DIP": 19, "TIP": 20},
}
WRIST = 0
CURL_THRESHOLD = 2.0
GRASP_MIN_CURLED = 3


def _vec(a, b):
    return b - a

def _angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))

def landmarks_to_np(landmarks, w, h):
    """Convert MediaPipe NormalizedLandmark list to dict of numpy arrays."""
    pts = {}
    for i, lm in enumerate(landmarks):
        pts[i] = np.array([lm.x * w, lm.y * h, lm.z * w])
    return pts

def compute_joint_angles(pts):
    angles = {}
    for finger, ids in FINGER_LANDMARKS.items():
        if finger == "thumb":
            v1 = _vec(pts[ids["CMC"]], pts[WRIST])
            v2 = _vec(pts[ids["CMC"]], pts[ids["MCP"]])
            mcp_angle = _angle_between(v1, v2)
            v1 = _vec(pts[ids["MCP"]], pts[ids["CMC"]])
            v2 = _vec(pts[ids["MCP"]], pts[ids["IP"]])
            pip_angle = _angle_between(v1, v2)
            v1 = _vec(pts[ids["IP"]], pts[ids["MCP"]])
            v2 = _vec(pts[ids["IP"]], pts[ids["TIP"]])
            dip_angle = _angle_between(v1, v2)
        else:
            v1 = _vec(pts[ids["MCP"]], pts[WRIST])
            v2 = _vec(pts[ids["MCP"]], pts[ids["PIP"]])
            mcp_angle = _angle_between(v1, v2)
            v1 = _vec(pts[ids["PIP"]], pts[ids["MCP"]])
            v2 = _vec(pts[ids["PIP"]], pts[ids["DIP"]])
            pip_angle = _angle_between(v1, v2)
            v1 = _vec(pts[ids["DIP"]], pts[ids["PIP"]])
            v2 = _vec(pts[ids["DIP"]], pts[ids["TIP"]])
            dip_angle = _angle_between(v1, v2)
        angles[finger] = {
            "MCP": round(math.degrees(mcp_angle), 2),
            "PIP": round(math.degrees(pip_angle), 2),
            "DIP": round(math.degrees(dip_angle), 2),
        }
    return angles

def compute_finger_states(angles):
    states = {}
    for finger, a in angles.items():
        curled = a["PIP"] < math.degrees(CURL_THRESHOLD)
        states[finger] = "curled" if curled else "extended"
    return states

def detect_grasp(finger_states):
    return sum(1 for s in finger_states.values() if s == "curled") >= GRASP_MIN_CURLED

def compute_wrist_pose(pts):
    wrist = pts[WRIST]
    middle_mcp = pts[9]
    index_mcp = pts[5]
    v1 = _vec(wrist, middle_mcp)
    v2 = _vec(wrist, index_mcp)
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm > 1e-8:
        normal = normal / norm
    return {
        "position": {"x": round(float(wrist[0]), 2), "y": round(float(wrist[1]), 2), "z": round(float(wrist[2]), 2)},
        "palm_normal": {"x": round(float(normal[0]), 4), "y": round(float(normal[1]), 4), "z": round(float(normal[2]), 4)},
    }


def process_video(video_path: str, output_path: Optional[str] = None, max_hands: int = 2) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'", file=sys.stderr)
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path} | {w}x{h} @ {fps:.1f} fps | {total} frames")

    # Create hand landmarker
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    trajectory = {
        "metadata": {
            "source": "hand_tracker_v2",
            "video": str(Path(video_path).name),
            "fps": fps,
            "resolution": [w, h],
            "total_frames": total,
        },
        "frames": [],
    }

    frame_idx = 0
    hands_detected_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_data = {
                "frame": frame_idx,
                "timestamp": round(frame_idx / fps, 4),
                "hands": [],
            }

            if result.hand_landmarks:
                hands_detected_count += 1
                for hand_lms, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name.lower()
                    score = round(handedness[0].score, 3)
                    
                    pts = landmarks_to_np(hand_lms, w, h)
                    angles = compute_joint_angles(pts)
                    finger_states = compute_finger_states(angles)
                    grasping = detect_grasp(finger_states)
                    wrist = compute_wrist_pose(pts)

                    landmarks_3d = []
                    for lm in hand_lms:
                        landmarks_3d.append({
                            "x": round(lm.x * w, 2),
                            "y": round(lm.y * h, 2),
                            "z": round(lm.z * w, 2),
                        })

                    # Also produce flat 21-joint angle array for pipeline compatibility
                    joints_21 = []
                    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                        a = angles[finger]
                        joints_21.extend([a["MCP"], a["PIP"], a["DIP"]])
                    # Pad to 21 (we have 15 from 5 fingers x 3 joints, add 6 zeros for abduction etc)
                    joints_21.extend([0.0] * 6)

                    hand_data = {
                        "hand": label,
                        "confidence": score,
                        "wrist": wrist,
                        "joint_angles": angles,
                        "finger_states": finger_states,
                        "grasping": grasping,
                        "landmarks_3d": landmarks_3d,
                        "joints_21": joints_21,
                    }
                    frame_data["hands"].append(hand_data)

            trajectory["frames"].append(frame_data)
            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"  Processed {frame_idx}/{total} frames...")

    finally:
        cap.release()
        landmarker.close()

    detection_rate = hands_detected_count / max(frame_idx, 1) * 100
    print(f"Done. {frame_idx} frames, {hands_detected_count} with hands ({detection_rate:.0f}%)")

    if output_path is None:
        output_path = str(Path(video_path).with_suffix("")) + "_hand_trajectory.json"

    with open(output_path, "w") as f:
        json.dump(trajectory, f)
    size_mb = Path(output_path).stat().st_size / (1024*1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    return trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand tracking via MediaPipe Tasks API")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--max-hands", type=int, default=2)
    args = parser.parse_args()

    process_video(args.video, args.output, args.max_hands)
