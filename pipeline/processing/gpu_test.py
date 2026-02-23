"""GPU hand pose estimation test on real video."""
import cv2
import mediapipe as mp
import numpy as np
import json
import sys

def test_hand_detection(video_path="/tmp/test_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    
    frame_count = 0
    hand_detected = 0
    all_keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:  # 1fps sampling
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            sample = {"frame": frame_count, "hands": []}
            
            if results.multi_hand_landmarks:
                hand_detected += 1
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    kp = hand_landmarks.landmark
                    joints_21 = []
                    for i, lm in enumerate(kp):
                        joints_21.append({
                            "id": i,
                            "x": round(lm.x, 4),
                            "y": round(lm.y, 4),
                            "z": round(lm.z, 4)
                        })
                    sample["hands"].append({
                        "hand_index": hand_idx,
                        "wrist": {"x": kp[0].x, "y": kp[0].y, "z": kp[0].z},
                        "joints_21": joints_21
                    })
                    print(f"Frame {frame_count}: hand {hand_idx} detected, "
                          f"wrist=({kp[0].x:.3f},{kp[0].y:.3f},{kp[0].z:.3f})")
            else:
                print(f"Frame {frame_count}: no hands")
            
            all_keypoints.append(sample)
        frame_count += 1
    
    cap.release()
    hands.close()
    
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  Total frames: {frame_count}")
    print(f"  Sampled frames: {len(all_keypoints)}")
    print(f"  Hands detected: {hand_detected}/{len(all_keypoints)} "
          f"({100*hand_detected/max(len(all_keypoints),1):.0f}%)")
    
    # Save keypoints
    output_path = "/tmp/hand_keypoints.json"
    with open(output_path, "w") as f:
        json.dump(all_keypoints, f, indent=2)
    print(f"  Keypoints saved: {output_path}")
    print(f"{'='*50}")
    
    # Convert to our 21-joint angle format (simplified)
    if hand_detected > 0:
        print("\nSample joint data (first detection):")
        for kp in all_keypoints:
            if kp["hands"]:
                hand = kp["hands"][0]
                print(f"  Frame {kp['frame']}: 21 keypoints captured")
                print(f"  Wrist: ({hand['wrist']['x']:.4f}, {hand['wrist']['y']:.4f}, {hand['wrist']['z']:.4f})")
                # Print finger tips
                tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
                for tip_id in tips:
                    j = hand["joints_21"][tip_id]
                    names = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}
                    print(f"  {names[tip_id]}_tip: ({j['x']:.4f}, {j['y']:.4f}, {j['z']:.4f})")
                break
    
    return all_keypoints

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_video.mp4"
    test_hand_detection(video)
