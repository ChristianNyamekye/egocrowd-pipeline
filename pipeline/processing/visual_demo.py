"""Visual pipeline demo — processes video with hand detection overlays and saves output video + dashboard."""
import cv2
import mediapipe as mp
import numpy as np
import json
import os

def run_visual_pipeline(video_path="/tmp/test_video.mp4", output_dir="/tmp/pipeline_output"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/frames", exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames ({total_frames/fps:.1f}s)")
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    
    # Output video with overlays
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(f"{output_dir}/processed.mp4", fourcc, fps, (width, height))
    
    frame_count = 0
    hand_frames = 0
    all_data = []
    
    print(f"\nProcessing {total_frames} frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw overlays
        annotated = frame.copy()
        
        # Status bar at top
        cv2.rectangle(annotated, (0, 0), (width, 40), (20, 20, 20), -1)
        
        frame_data = {"frame": frame_count, "hands": []}
        
        if results.multi_hand_landmarks:
            hand_frames += 1
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                
                # Extract 21 keypoints
                kp = hand_landmarks.landmark
                joints = []
                for lm in kp:
                    joints.append({"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)})
                
                frame_data["hands"].append({
                    "hand_index": hand_idx,
                    "joints_21": joints,
                    "wrist": {"x": kp[0].x, "y": kp[0].y, "z": kp[0].z}
                })
                
                # Draw wrist position text
                wx, wy = int(kp[0].x * width), int(kp[0].y * height)
                cv2.putText(annotated, f"Hand {hand_idx}", (wx, wy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            status = f"Frame {frame_count}/{total_frames} | {len(results.multi_hand_landmarks)} hand(s) | TRACKING"
            color = (0, 255, 0)
        else:
            status = f"Frame {frame_count}/{total_frames} | No hands | SEARCHING"
            color = (0, 0, 255)
        
        cv2.putText(annotated, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Detection rate bar
        rate = hand_frames / max(frame_count + 1, 1)
        bar_width = int(rate * (width - 20))
        cv2.rectangle(annotated, (10, height - 20), (10 + bar_width, height - 10), (0, 255, 0), -1)
        cv2.rectangle(annotated, (10, height - 20), (width - 10, height - 10), (100, 100, 100), 1)
        cv2.putText(annotated, f"Detection: {rate*100:.0f}%", (10, height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out_video.write(annotated)
        all_data.append(frame_data)
        
        # Save sample frames (every 3 seconds)
        if frame_count % (fps * 3) == 0:
            cv2.imwrite(f"{output_dir}/frames/frame_{frame_count:06d}.jpg", annotated)
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} ({100*frame_count/total_frames:.0f}%) "
                  f"| Hands: {hand_frames}/{frame_count+1} ({100*hand_frames/(frame_count+1):.0f}%)")
        
        frame_count += 1
    
    cap.release()
    out_video.release()
    hands.close()
    
    # Save all keypoint data
    with open(f"{output_dir}/keypoints.json", "w") as f:
        json.dump(all_data, f)
    
    # Generate summary HTML
    sample_frames = sorted([f for f in os.listdir(f"{output_dir}/frames") if f.endswith('.jpg')])
    
    html = f"""<!DOCTYPE html>
<html><head><title>Pipeline Demo Results</title>
<style>
body {{ background: #1a1a2e; color: #eee; font-family: 'Courier New', monospace; padding: 20px; }}
h1 {{ color: #0ff; text-align: center; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
.stat {{ background: #16213e; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #0f3460; }}
.stat .value {{ font-size: 2em; color: #0ff; }}
.stat .label {{ color: #888; margin-top: 5px; }}
.frames {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 10px; margin-top: 20px; }}
.frames img {{ width: 100%; border-radius: 5px; border: 1px solid #333; }}
video {{ width: 100%; max-width: 800px; margin: 20px auto; display: block; border-radius: 10px; }}
</style></head>
<body>
<h1>PIPELINE DEMO RESULTS</h1>
<div class="stats">
  <div class="stat"><div class="value">{frame_count}</div><div class="label">Total Frames</div></div>
  <div class="stat"><div class="value">{hand_frames}</div><div class="label">Hands Detected</div></div>
  <div class="stat"><div class="value">{100*hand_frames/max(frame_count,1):.0f}%</div><div class="label">Detection Rate</div></div>
  <div class="stat"><div class="value">{frame_count/fps:.1f}s</div><div class="label">Duration</div></div>
</div>
<h2>Processed Video</h2>
<video controls><source src="processed.mp4" type="video/mp4"></video>
<h2>Sample Frames (with hand skeleton overlay)</h2>
<div class="frames">
{''.join(f'<img src="frames/{f}" alt="{f}">' for f in sample_frames)}
</div>
</body></html>"""
    
    with open(f"{output_dir}/index.html", "w") as f:
        f.write(html)
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Frames processed: {frame_count}")
    print(f"  Hands detected: {hand_frames} ({100*hand_frames/max(frame_count,1):.0f}%)")
    print(f"  Output video: {output_dir}/processed.mp4")
    print(f"  Keypoints JSON: {output_dir}/keypoints.json")
    print(f"  Dashboard: {output_dir}/index.html")
    print(f"  Sample frames: {len(sample_frames)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_visual_pipeline()
