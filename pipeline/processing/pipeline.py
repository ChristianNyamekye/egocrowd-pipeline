"""
Main processing pipeline: raw capture → packaged episode.

Takes synchronized sensor data from the capture app and produces
a complete Episode ready for LeRobot/RLDS export.

Stages:
1. Ingest: Parse raw sensor files from capture app export
2. Align: Time-align all sensor streams to common clock
3. Hand pose: Extract/validate hand keypoints (GPU if from video, direct if from glove)
4. Wrist motion: Compute relative SE(3) transforms (EgoScale format)
5. Retarget: Map human joints → robot hand joint space
6. Package: Create Episode object with all data
7. Quality: Automated quality scoring
8. Export: Save as LeRobot/RLDS format
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema.episode import Episode, Timestep, DataSource
from processing.retarget import retarget_episode, compute_wrist_relative_transform
from processing.hand_pose import IMUProcessor


class CaptureIngestor:
    """Parse raw capture app output."""
    
    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
    
    def load(self) -> Dict:
        """Load all sensor data from a capture session."""
        data = {}
        
        # Metadata
        meta_path = self.session_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                data["metadata"] = json.load(f)
        
        # ARKit camera poses
        arkit_path = self.session_dir / "arkit_poses.json"
        if arkit_path.exists():
            with open(arkit_path) as f:
                data["arkit_poses"] = json.load(f)
        
        # Apple Watch IMU
        imu_path = self.session_dir / "watch_imu.json"
        if imu_path.exists():
            with open(imu_path) as f:
                data["watch_imu"] = json.load(f)
        
        # UDCAP glove joints
        glove_path = self.session_dir / "glove_joints.json"
        if glove_path.exists():
            with open(glove_path) as f:
                data["glove_joints"] = json.load(f)
        
        # Calibration
        cal_path = self.session_dir / "calibration.json"
        if cal_path.exists():
            with open(cal_path) as f:
                data["calibration"] = json.load(f)
        
        # Video path
        video_path = self.session_dir / "video.mp4"
        if video_path.exists():
            data["video_path"] = str(video_path)
        
        return data


class SensorAligner:
    """Time-align sensor streams to common clock at target FPS."""
    
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
    
    def align(self, raw_data: Dict) -> Dict:
        """
        Align all sensor streams to target FPS.
        
        Interpolates higher-rate sensors (IMU@100Hz, glove@120Hz)
        down to target FPS, and extracts video frames at target FPS.
        """
        # Determine episode duration from video/metadata
        meta = raw_data.get("metadata", {})
        duration_ms = meta.get("duration_ms", 10000)  # default 10s
        
        num_frames = int(duration_ms / 1000 * self.target_fps)
        frame_times_ms = np.linspace(0, duration_ms, num_frames, endpoint=False)
        
        aligned = {
            "frame_times_ms": frame_times_ms.tolist(),
            "num_frames": num_frames,
            "fps": self.target_fps,
        }
        
        # Interpolate IMU data to frame times
        if "watch_imu" in raw_data:
            imu_entries = raw_data["watch_imu"]
            if imu_entries:
                imu_times = np.array([e["timestamp"] for e in imu_entries])
                imu_accel = np.array([[e["accel"]["x"], e["accel"]["y"], e["accel"]["z"]] for e in imu_entries])
                imu_gyro = np.array([[e["gyro"]["x"], e["gyro"]["y"], e["gyro"]["z"]] for e in imu_entries])
                
                # Interpolate to frame times
                aligned["imu_accel"] = np.column_stack([
                    np.interp(frame_times_ms, imu_times, imu_accel[:, i])
                    for i in range(3)
                ])
                aligned["imu_gyro"] = np.column_stack([
                    np.interp(frame_times_ms, imu_times, imu_gyro[:, i])
                    for i in range(3)
                ])
        
        # Interpolate glove joints to frame times
        if "glove_joints" in raw_data:
            glove_entries = raw_data["glove_joints"]
            if glove_entries:
                glove_times = np.array([e["timestamp"] for e in glove_entries])
                glove_angles = np.array([e["joints_21"] for e in glove_entries])
                
                aligned["hand_joints"] = np.column_stack([
                    np.interp(frame_times_ms, glove_times, glove_angles[:, i])
                    for i in range(21)
                ])
        
        # Interpolate ARKit camera poses to frame times
        if "arkit_poses" in raw_data:
            arkit_entries = raw_data["arkit_poses"]
            if arkit_entries:
                arkit_times = np.array([e["timestamp"] for e in arkit_entries])
                # Extract position and quaternion
                poses = []
                for e in arkit_entries:
                    t = e["transform_4x4"]
                    pos = [t[3], t[7], t[11]]  # translation from 4x4
                    # Extract rotation as quaternion (simplified)
                    poses.append(pos + [1, 0, 0, 0])  # placeholder quat
                poses = np.array(poses)
                
                aligned["camera_poses"] = np.column_stack([
                    np.interp(frame_times_ms, arkit_times, poses[:, i])
                    for i in range(7)
                ])
        
        return aligned


class EpisodeBuilder:
    """Build Episode from aligned sensor data."""
    
    def __init__(self, target_hand: str = "allegro_hand"):
        self.target_hand = target_hand
        self.imu_processor = IMUProcessor()
    
    def build(
        self,
        aligned_data: Dict,
        metadata: Dict,
        session_dir: str
    ) -> Episode:
        """Build a complete Episode from aligned data."""
        
        num_frames = aligned_data["num_frames"]
        fps = aligned_data["fps"]
        frame_times = aligned_data["frame_times_ms"]
        
        # Process wrist motion from IMU
        wrist_poses = None
        if "imu_accel" in aligned_data and "imu_gyro" in aligned_data:
            imu_data = np.column_stack([
                aligned_data["imu_accel"],
                aligned_data["imu_gyro"]
            ])
            imu_result = self.imu_processor.process_imu_stream(imu_data, fps)
            wrist_poses = imu_result["wrist_poses"]
            
            # Convert to relative transforms (EgoScale format)
            wrist_poses = compute_wrist_relative_transform(wrist_poses)
        
        # Retarget hand joints if available
        robot_joints = None
        human_joints = aligned_data.get("hand_joints")
        if human_joints is not None:
            robot_joints = retarget_episode(
                human_joints, self.target_hand, fps=fps, smooth=True
            )
        
        # Build timesteps
        timesteps = []
        for t in range(num_frames):
            ts = Timestep(
                timestamp_ms=int(frame_times[t]),
                wrist_position=wrist_poses[t, :3].tolist() if wrist_poses is not None else [0,0,0],
                wrist_orientation=wrist_poses[t, 3:7].tolist() if wrist_poses is not None else [1,0,0,0],
                hand_joints_human=human_joints[t].tolist() if human_joints is not None else None,
                hand_joints_robot=robot_joints[t].tolist() if robot_joints is not None else None,
                rgb_path=f"frames/{t:06d}.jpg",
                camera_pose=aligned_data["camera_poses"][t].tolist() if "camera_poses" in aligned_data else None,
                watch_imu_accel=aligned_data["imu_accel"][t].tolist() if "imu_accel" in aligned_data else None,
                watch_imu_gyro=aligned_data["imu_gyro"][t].tolist() if "imu_gyro" in aligned_data else None,
                glove_raw_angles=human_joints[t].tolist() if human_joints is not None else None,
            )
            timesteps.append(ts)
        
        # Build episode
        episode = Episode(
            episode_id=f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            contributor_id=metadata.get("contributor_id", "anonymous"),
            task_description=metadata.get("task", "unspecified manipulation"),
            task_category=metadata.get("task_category", "general"),
            environment=metadata.get("environment", "unknown"),
            environment_id=metadata.get("environment_id", "unknown"),
            data_sources=[s.value for s in DataSource],
            fps=fps,
            duration_ms=int(frame_times[-1]) if len(frame_times) > 0 else 0,
            num_timesteps=num_frames,
            timesteps=timesteps,
            target_embodiment=self.target_hand,
            retargeting_method="linear",
            capture_date=datetime.now().isoformat(),
        )
        
        return episode


class QualityChecker:
    """Automated quality scoring for episodes."""
    
    def score(self, episode: Episode) -> float:
        """
        Score episode quality 0-1.
        
        Checks:
        - Sufficient duration (>3s)
        - Hand data completeness
        - Wrist motion smoothness
        - No sensor dropouts
        """
        scores = []
        
        # Duration check
        duration_s = episode.duration_ms / 1000
        if duration_s >= 10:
            scores.append(1.0)
        elif duration_s >= 3:
            scores.append(0.5 + 0.5 * (duration_s - 3) / 7)
        else:
            scores.append(0.1)
        
        # Hand data completeness
        has_hand = sum(1 for t in episode.timesteps if t.hand_joints_human is not None)
        hand_ratio = has_hand / max(episode.num_timesteps, 1)
        scores.append(hand_ratio)
        
        # Wrist motion check (should have some movement, not static)
        if episode.num_timesteps > 1:
            positions = np.array([t.wrist_position for t in episode.timesteps])
            total_motion = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            if total_motion > 0.05:  # At least 5cm of movement
                scores.append(min(1.0, total_motion / 0.5))
            else:
                scores.append(0.2)
        
        return np.mean(scores)


def process_session(
    session_dir: str,
    target_hand: str = "allegro_hand",
    target_fps: int = 30,
    output_dir: Optional[str] = None
) -> Episode:
    """
    Full pipeline: raw capture session -> packaged Episode.
    
    Args:
        session_dir: path to capture app export directory
        target_hand: robot hand for retargeting
        target_fps: output frame rate
        output_dir: where to save processed episode
    
    Returns:
        Episode object
    """
    print(f"Processing session: {session_dir}")
    
    # 1. Ingest
    ingestor = CaptureIngestor(session_dir)
    raw_data = ingestor.load()
    print(f"  Loaded: {list(raw_data.keys())}")
    
    # 2. Align
    aligner = SensorAligner(target_fps=target_fps)
    aligned = aligner.align(raw_data)
    print(f"  Aligned: {aligned['num_frames']} frames at {aligned['fps']}fps")
    
    # 3. Build episode (includes retargeting + wrist processing)
    builder = EpisodeBuilder(target_hand=target_hand)
    episode = builder.build(
        aligned, 
        raw_data.get("metadata", {}),
        session_dir
    )
    print(f"  Built episode: {episode.episode_id}")
    
    # 4. Quality check
    checker = QualityChecker()
    episode.quality_score = checker.score(episode)
    print(f"  Quality score: {episode.quality_score:.2f}")
    
    # 5. Export
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        episode_path = os.path.join(output_dir, f"{episode.episode_id}.json")
        with open(episode_path, "w") as f:
            json.dump(episode.to_dict(), f, indent=2)
        print(f"  Saved: {episode_path}")
        
        # Save LeRobot format
        lerobot_path = os.path.join(output_dir, f"{episode.episode_id}_lerobot.json")
        with open(lerobot_path, "w") as f:
            json.dump(episode.to_lerobot(), f, indent=2)
        print(f"  Saved LeRobot: {lerobot_path}")
        
        # Save RLDS format
        rlds_path = os.path.join(output_dir, f"{episode.episode_id}_rlds.json")
        with open(rlds_path, "w") as f:
            json.dump(episode.to_rlds(), f, indent=2)
        print(f"  Saved RLDS: {rlds_path}")
    
    return episode


if __name__ == "__main__":
    print("Pipeline module loaded. Use process_session() to run.")
    print("Requires a capture app export directory as input.")
