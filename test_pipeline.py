"""
End-to-end pipeline test with synthetic sensor data.
Simulates a "pick up mug" task captured by our hardware kit.
"""

import json
import os
import numpy as np
from pathlib import Path

def generate_synthetic_session(output_dir: str):
    """Generate a fake capture session that mimics real sensor output."""
    os.makedirs(output_dir, exist_ok=True)
    
    duration_ms = 5000  # 5 second task
    
    # Metadata
    metadata = {
        "contributor_id": "test_user_001",
        "task": "Pick up the red mug from the table",
        "task_category": "pick_place",
        "environment": "kitchen",
        "environment_id": "env_test_001",
        "duration_ms": duration_ms,
        "kit_version": "v1",
        "capture_date": "2026-02-20T14:00:00",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Apple Watch IMU (100Hz = 500 samples for 5s)
    imu_samples = 500
    imu_data = []
    for i in range(imu_samples):
        t_ms = i * 10  # 100Hz = 10ms intervals
        # Simulate reaching forward, grasping, lifting
        phase = t_ms / duration_ms
        
        # Acceleration: gentle forward reach then lift
        ax = 0.5 * np.sin(phase * np.pi) + np.random.randn() * 0.1
        ay = 0.2 * np.sin(phase * np.pi * 2) + np.random.randn() * 0.1
        az = 9.81 + 0.3 * max(0, phase - 0.6) + np.random.randn() * 0.1
        
        # Gyroscope: wrist rotation during grasp
        wx = 0.5 * np.sin(phase * np.pi * 3) + np.random.randn() * 0.05
        wy = 0.3 * np.cos(phase * np.pi * 2) + np.random.randn() * 0.05
        wz = 0.1 * np.random.randn()
        
        imu_data.append({
            "timestamp": t_ms,
            "accel": {"x": ax, "y": ay, "z": az},
            "gyro": {"x": wx, "y": wy, "z": wz},
        })
    
    with open(os.path.join(output_dir, "watch_imu.json"), "w") as f:
        json.dump(imu_data, f)
    
    # UDCAP Glove joints (120Hz = 600 samples for 5s)
    glove_samples = 600
    glove_data = []
    for i in range(glove_samples):
        t_ms = i * (1000/120)  # 120Hz
        phase = t_ms / duration_ms
        
        # Simulate hand opening -> closing (grasp) -> holding
        if phase < 0.3:
            grasp = 0  # hand open
        elif phase < 0.5:
            grasp = (phase - 0.3) / 0.2  # closing
        else:
            grasp = 1.0  # closed grasp
        
        # 21 joint angles in degrees
        joints = [
            # Thumb (4): curl during grasp
            20 + 40 * grasp, 10 + 20 * grasp, 15 + 35 * grasp, 5 + 25 * grasp,
            # Index (4)
            10 + 60 * grasp, 3 + 5 * grasp, 15 + 55 * grasp, 10 + 40 * grasp,
            # Middle (4)
            10 + 65 * grasp, 2 + 4 * grasp, 15 + 60 * grasp, 10 + 45 * grasp,
            # Ring (4)
            8 + 55 * grasp, 2 + 3 * grasp, 12 + 50 * grasp, 8 + 40 * grasp,
            # Pinky (5)
            5 + 10 * grasp, 8 + 45 * grasp, 2 + 3 * grasp, 10 + 40 * grasp, 5 + 30 * grasp,
        ]
        # Add sensor noise
        joints = [j + np.random.randn() * 0.5 for j in joints]
        
        glove_data.append({
            "timestamp": t_ms,
            "joints_21": joints,
        })
    
    with open(os.path.join(output_dir, "glove_joints.json"), "w") as f:
        json.dump(glove_data, f)
    
    # ARKit camera poses (60Hz = 300 samples for 5s)
    arkit_samples = 300
    arkit_data = []
    for i in range(arkit_samples):
        t_ms = i * (1000/60)
        phase = t_ms / duration_ms
        
        # Camera moves forward (reaching) then up (lifting)
        x = 0.3 * phase
        y = 0.05 * np.sin(phase * np.pi)
        z = 0.15 * max(0, phase - 0.5)
        
        # Identity-ish rotation with slight changes
        transform = [
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1,
        ]
        
        arkit_data.append({
            "timestamp": t_ms,
            "transform_4x4": transform,
        })
    
    with open(os.path.join(output_dir, "arkit_poses.json"), "w") as f:
        json.dump(arkit_data, f)
    
    # Calibration
    calibration = {
        "neutral_pose_joints": [0] * 21,
        "camera_intrinsics": {
            "fx": 1500, "fy": 1500, "cx": 960, "cy": 540
        },
        "sensor_offsets": {
            "watch_to_wrist_cm": [0, -2, 0],
            "glove_calibrated": True,
        }
    }
    with open(os.path.join(output_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    
    print(f"Generated synthetic session: {output_dir}")
    print(f"  IMU: {imu_samples} samples (100Hz)")
    print(f"  Glove: {glove_samples} samples (120Hz)")
    print(f"  ARKit: {arkit_samples} samples (60Hz)")
    print(f"  Duration: {duration_ms}ms")


if __name__ == "__main__":
    # Generate synthetic data
    test_dir = str(Path(__file__).parent / "test_data" / "session_test_001")
    generate_synthetic_session(test_dir)
    
    # Run pipeline
    from processing.pipeline import process_session
    
    output_dir = str(Path(__file__).parent / "test_output")
    episode = process_session(
        test_dir,
        target_hand="allegro_hand",
        target_fps=30,
        output_dir=output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"PIPELINE TEST RESULTS")
    print(f"{'='*60}")
    print(f"Episode ID: {episode.episode_id}")
    print(f"Task: {episode.task_description}")
    print(f"Duration: {episode.duration_ms}ms")
    print(f"Timesteps: {episode.num_timesteps}")
    print(f"Target hand: {episode.target_embodiment}")
    print(f"Quality score: {episode.quality_score:.2f}")
    print(f"Has wrist data: {episode.timesteps[0].wrist_position != [0,0,0]}")
    print(f"Has human joints: {episode.timesteps[0].hand_joints_human is not None}")
    print(f"Has robot joints: {episode.timesteps[0].hand_joints_robot is not None}")
    
    # Check output files
    print(f"\nOutput files:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f}: {size:,} bytes")
    
    print(f"\nPIPELINE TEST PASSED")
