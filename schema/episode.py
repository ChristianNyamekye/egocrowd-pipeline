"""
Data schema for manipulation episodes.

Aligned with EgoScale's action representation (arxiv 2602.16710):
- Wrist-level arm motion: relative SE(3) transforms between timesteps
- Hand articulation: 21-joint angles retargeted to robot hand space
- Egocentric RGB observations

Also compatible with:
- LeRobot dataset format
- RLDS (Robotics Dataset Specification)
- Open X-Embodiment
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from enum import Enum


class DataSource(Enum):
    """Hardware source of the data."""
    IPHONE_RGB = "iphone_rgb"           # 4K@30/60fps egocentric video
    IPHONE_LIDAR = "iphone_lidar"       # LiDAR depth + SLAM
    APPLE_WATCH_IMU = "apple_watch_imu" # 6-DoF wrist IMU at 100Hz
    UDCAP_GLOVE = "udcap_glove"         # 21-joint finger angles at 120Hz


class EmbodimentType(Enum):
    """Target robot embodiment for retargeting."""
    HUMAN = "human"                     # Raw human data
    ALLEGRO_HAND = "allegro_hand"       # 16-DoF Allegro Hand
    LEAP_HAND = "leap_hand"             # 16-DoF LEAP Hand  
    SHADOW_HAND = "shadow_hand"         # 24-DoF Shadow Dexterous Hand
    ABILITY_HAND = "ability_hand"       # 10-DoF Ability Hand
    INSPIRE_HAND = "inspire_hand"       # 12-DoF INSPIRE Hand
    GENERIC = "generic"                 # Generic joint mapping


@dataclass
class Timestep:
    """Single timestep in a manipulation episode.
    
    Core fields aligned with EgoScale Section 2.1:
    - wrist_pose: relative SE(3) wrist transform from step 0
    - hand_joints: 21 joint angles (human) or retargeted robot joints
    - rgb: egocentric camera frame
    """
    # Timing
    timestamp_ms: int                   # Milliseconds from episode start
    
    # Wrist-level arm motion (EgoScale Eq. ΔW^t = (W_w^0)^-1 * W_w^t)
    wrist_position: List[float]         # [x, y, z] relative to start, meters
    wrist_orientation: List[float]      # [qw, qx, qy, qz] quaternion relative to start
    wrist_velocity: Optional[List[float]] = None  # [vx, vy, vz, wx, wy, wz]
    
    # Hand articulation (21 joints from UDCAP glove)
    hand_joints_human: Optional[List[float]] = None   # 21 joint angles, degrees
    hand_joints_robot: Optional[List[float]] = None   # Retargeted robot joint angles
    
    # Visual observation
    rgb_path: Optional[str] = None      # Path to extracted frame
    depth_path: Optional[str] = None    # Path to LiDAR depth map
    
    # Camera pose from SLAM
    camera_pose: Optional[List[float]] = None  # [x,y,z, qw,qx,qy,qz] world frame
    
    # Raw sensor data (for reprocessing)
    watch_imu_accel: Optional[List[float]] = None  # [ax, ay, az] m/s^2
    watch_imu_gyro: Optional[List[float]] = None   # [wx, wy, wz] rad/s
    glove_raw_angles: Optional[List[float]] = None  # Raw 21 angles from UDCAP


@dataclass
class ObjectAnnotation:
    """Tracked object in the scene."""
    object_id: str
    label: str                          # e.g. "mug", "apple", "drawer_handle"
    pose_6dof: Optional[List[float]] = None  # [x,y,z, qw,qx,qy,qz]
    bbox_2d: Optional[List[int]] = None      # [x1, y1, x2, y2] in pixels
    contact: bool = False               # Is the hand in contact with this object?


@dataclass 
class Episode:
    """A single manipulation episode.
    
    Represents one continuous task execution (e.g., "pick up mug",
    "open drawer", "fold cloth") captured by a contributor.
    """
    # Identity
    episode_id: str                     # Unique ID
    contributor_id: str                 # Anonymous contributor hash
    
    # Task description
    task_description: str               # Natural language: "Pick up the red mug"
    task_category: str                  # e.g. "pick_place", "articulated", "deformable"
    
    # Environment
    environment: str                    # e.g. "kitchen", "office", "workshop"
    environment_id: str                 # Unique environment hash
    
    # Hardware
    data_sources: List[str]             # Which sensors were used
    kit_version: str = "v1"             # Hardware kit version
    
    # Timing
    fps: int = 30                       # Target frame rate
    duration_ms: int = 0                # Total duration
    num_timesteps: int = 0
    
    # Data
    timesteps: List[Timestep] = field(default_factory=list)
    
    # Objects (optional, from vision pipeline)
    objects: List[ObjectAnnotation] = field(default_factory=list)
    
    # Retargeting
    target_embodiment: str = "human"    # Which robot hand space joints are in
    retargeting_method: str = "none"    # "optimization", "learned", "none"
    
    # Quality
    quality_score: Optional[float] = None  # 0-1, automated quality check
    is_validated: bool = False          # Human-validated?
    
    # Metadata
    capture_date: str = ""              # ISO 8601
    schema_version: str = "0.1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON/HDF5 storage."""
        import dataclasses
        return dataclasses.asdict(self)
    
    def to_lerobot(self) -> Dict[str, Any]:
        """Convert to LeRobot dataset format."""
        frames = []
        for t in self.timesteps:
            frame = {
                "timestamp": t.timestamp_ms / 1000.0,
                "observation.image": t.rgb_path,
                "observation.state": (
                    t.wrist_position + t.wrist_orientation + 
                    (t.hand_joints_robot or t.hand_joints_human or [])
                ),
                "action": (
                    t.wrist_velocity or [0]*6
                ) + (t.hand_joints_robot or t.hand_joints_human or []),
            }
            frames.append(frame)
        
        return {
            "episode_id": self.episode_id,
            "task": self.task_description,
            "frames": frames,
            "fps": self.fps,
            "embodiment": self.target_embodiment,
        }
    
    def to_rlds(self) -> Dict[str, Any]:
        """Convert to RLDS (TensorFlow Datasets) format."""
        steps = []
        for i, t in enumerate(self.timesteps):
            step = {
                "observation": {
                    "image": t.rgb_path,
                    "wrist_position": t.wrist_position,
                    "wrist_orientation": t.wrist_orientation,
                    "hand_joints": t.hand_joints_robot or t.hand_joints_human or [],
                },
                "action": (
                    (t.wrist_velocity or [0]*6) + 
                    (t.hand_joints_robot or t.hand_joints_human or [])
                ),
                "is_first": i == 0,
                "is_last": i == len(self.timesteps) - 1,
                "is_terminal": i == len(self.timesteps) - 1,
                "language_instruction": self.task_description,
            }
            steps.append(step)
        
        return {
            "episode_id": self.episode_id,
            "steps": steps,
        }


# Joint mapping for UDCAP 21-angle output
UDCAP_JOINT_NAMES = [
    # Thumb (4 joints)
    "thumb_cmc_flexion", "thumb_cmc_abduction", 
    "thumb_mcp_flexion", "thumb_ip_flexion",
    # Index (4 joints)
    "index_mcp_flexion", "index_mcp_abduction",
    "index_pip_flexion", "index_dip_flexion",
    # Middle (4 joints)
    "middle_mcp_flexion", "middle_mcp_abduction",
    "middle_pip_flexion", "middle_dip_flexion",
    # Ring (4 joints)
    "ring_mcp_flexion", "ring_mcp_abduction",
    "ring_pip_flexion", "ring_dip_flexion",
    # Pinky (5 joints)
    "pinky_cmc_flexion",
    "pinky_mcp_flexion", "pinky_mcp_abduction",
    "pinky_pip_flexion", "pinky_dip_flexion",
]

assert len(UDCAP_JOINT_NAMES) == 21
