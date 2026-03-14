"""
R3D Ingest — Convert Record3D .r3d files into pipeline-compatible sessions.

Record3D .r3d format (zip archive):
- metadata: JSON with w, h, dw, dh, K, poses, frameTimestamps, fps, perFrameIntrinsicCoeffs
- rgbd/N.jpg: RGB frames
- rgbd/N.depth: LiDAR depth maps (float16/32)
- rgbd/N.conf: confidence maps
- sound.m4a: audio
- icon: thumbnail

This script:
1. Extracts RGB frames as video or image sequence
2. Parses camera poses (position + quaternion per frame)
3. Extracts LiDAR depth maps
4. Generates metadata.json, arkit_poses.json compatible with pipeline
5. Runs phone_hand_tracker on extracted video for hand pose data

Usage:
    python r3d_ingest.py <input.r3d> -o <output_session_dir>
    python r3d_ingest.py --batch <dir_of_r3d_files> -o <output_dir>
"""

import argparse
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np


def parse_r3d_metadata(z: zipfile.ZipFile) -> Dict:
    """Parse the metadata blob from an R3D archive."""
    raw = z.read("metadata")
    meta = json.loads(raw)
    return meta


def extract_rgb_frames(z: zipfile.ZipFile, meta: Dict, output_dir: str) -> List[str]:
    """Extract RGB JPG frames in order, return list of paths."""
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = len(meta.get("frameTimestamps", []))
    if num_frames == 0:
        # Count jpg files
        num_frames = len([n for n in z.namelist() if n.endswith(".jpg")])
    
    paths = []
    for i in range(num_frames):
        jpg_name = f"rgbd/{i}.jpg"
        if jpg_name in z.namelist():
            out_path = os.path.join(frames_dir, f"{i:06d}.jpg")
            with open(out_path, "wb") as f:
                f.write(z.read(jpg_name))
            paths.append(out_path)
    
    print(f"  Extracted {len(paths)} RGB frames")
    return paths


def extract_depth_maps(z: zipfile.ZipFile, meta: Dict, output_dir: str) -> List[str]:
    """Extract depth maps. R3D stores them as LZ-FSE compressed float32 arrays."""
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    dw = meta.get("dw", 192)
    dh = meta.get("dh", 256)

    num_frames = len(meta.get("frameTimestamps", []))
    paths = []

    for i in range(num_frames):
        depth_name = f"rgbd/{i}.depth"
        if depth_name in z.namelist():
            raw = z.read(depth_name)
            # R3D depth is LZ-FSE compressed — decompress first
            try:
                import liblzfse
                raw = liblzfse.decompress(raw)
            except ImportError:
                pass  # liblzfse not installed, try raw
            except Exception:
                pass  # fallback if already uncompressed
            try:
                depth = np.frombuffer(raw, dtype=np.float32).reshape(dh, dw)
            except ValueError:
                # Might be float16
                try:
                    depth = np.frombuffer(raw, dtype=np.float16).reshape(dh, dw).astype(np.float32)
                except ValueError:
                    continue
            
            # Save as 16-bit PNG (millimeters)
            depth_mm = (depth * 1000).astype(np.uint16)
            out_path = os.path.join(depth_dir, f"{i:06d}.png")
            cv2.imwrite(out_path, depth_mm)
            paths.append(out_path)
    
    print(f"  Extracted {len(paths)} depth maps ({dw}x{dh})")
    return paths


def build_arkit_poses(meta: Dict) -> List[Dict]:
    """Convert R3D poses to pipeline-compatible arkit_poses format.
    
    R3D pose format: [qx, qy, qz, qw, tx, ty, tz] per frame
    Wait — actually looking at the data, it's [tx, ty, tz, qw, qx, qy, qz]
    Let me check: first pose is [-0.23, 0.05, 0.02, 0.97, 0.10, 0.04, 0.006]
    qw=0.97 makes sense as near-identity rotation. So format is [tx, ty, tz, qw, qx, qy, qz].
    """
    poses = meta.get("poses", [])
    timestamps = meta.get("frameTimestamps", [])
    
    arkit_poses = []
    for i, pose in enumerate(poses):
        if len(pose) != 7:
            continue
        
        tx, ty, tz = pose[0], pose[1], pose[2]
        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
        
        # Build 4x4 transform from quaternion + translation
        # Rotation matrix from quaternion
        r00 = 1 - 2*(qy*qy + qz*qz)
        r01 = 2*(qx*qy - qz*qw)
        r02 = 2*(qx*qz + qy*qw)
        r10 = 2*(qx*qy + qz*qw)
        r11 = 1 - 2*(qx*qx + qz*qz)
        r12 = 2*(qy*qz - qx*qw)
        r20 = 2*(qx*qz - qy*qw)
        r21 = 2*(qy*qz + qx*qw)
        r22 = 1 - 2*(qx*qx + qy*qy)
        
        transform = [
            r00, r01, r02, tx,
            r10, r11, r12, ty,
            r20, r21, r22, tz,
            0, 0, 0, 1,
        ]
        
        t_ms = timestamps[i] * 1000 if i < len(timestamps) else i * (1000 / meta.get("fps", 30))
        
        arkit_poses.append({
            "timestamp": t_ms,
            "transform_4x4": transform,
            "position": [tx, ty, tz],
            "quaternion": [qw, qx, qy, qz],
        })
    
    return arkit_poses


def frames_to_video(frames_dir: str, output_path: str, fps: float, w: int, h: int) -> str:
    """Stitch extracted frames into a video for hand tracking."""
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        return ""
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is not None:
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            writer.write(img)
    
    writer.release()
    print(f"  Created video: {output_path} ({len(frames)} frames @ {fps}fps)")
    return output_path


def ingest_r3d(
    r3d_path: str,
    output_dir: str,
    task_category: str = "general",
    task_description: str = "manipulation task",
    extract_depth: bool = True,
    create_video: bool = True,
) -> Dict:
    """
    Full R3D → pipeline session conversion.
    
    Returns dict with session info.
    """
    r3d_path = Path(r3d_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nIngesting: {r3d_path.name}")
    
    with zipfile.ZipFile(str(r3d_path), "r") as z:
        # 1. Parse metadata
        meta = parse_r3d_metadata(z)
        fps = meta.get("fps", 30)
        w = meta.get("w", 720)
        h = meta.get("h", 960)
        num_frames = len(meta.get("frameTimestamps", []))
        duration_ms = int(meta["frameTimestamps"][-1] * 1000) if meta.get("frameTimestamps") else 0
        
        print(f"  Resolution: {w}x{h} @ {fps}fps, {num_frames} frames, {duration_ms/1000:.1f}s")
        
        # 2. Extract RGB frames
        frame_paths = extract_rgb_frames(z, meta, output_dir)
        
        # 3. Extract depth maps
        depth_paths = []
        if extract_depth:
            depth_paths = extract_depth_maps(z, meta, output_dir)
        
        # 4. Build camera poses
        arkit_poses = build_arkit_poses(meta)
        arkit_path = os.path.join(output_dir, "arkit_poses.json")
        with open(arkit_path, "w") as f:
            json.dump(arkit_poses, f)
        print(f"  Saved {len(arkit_poses)} camera poses")
        
        # 5. Save camera intrinsics
        K = meta.get("K", [])
        per_frame_K = meta.get("perFrameIntrinsicCoeffs", [])
        calibration = {
            "camera_intrinsics_matrix": K,
            "per_frame_intrinsics": per_frame_K[:3] if per_frame_K else [],  # sample
            "rgb_resolution": [w, h],
            "depth_resolution": [meta.get("dw", 192), meta.get("dh", 256)],
            "camera_type": meta.get("cameraType", 1),
            "sensor_offsets": {"watch_to_wrist_cm": [0, -2, 0]},
        }
        if per_frame_K:
            calibration["camera_intrinsics"] = {
                "fx": per_frame_K[0][0],
                "fy": per_frame_K[0][1],
                "cx": per_frame_K[0][2],
                "cy": per_frame_K[0][3],
            }
        
        cal_path = os.path.join(output_dir, "calibration.json")
        with open(cal_path, "w") as f:
            json.dump(calibration, f, indent=2)
        
        # 6. Build metadata.json
        session_meta = {
            "contributor_id": "christian_001",
            "task": task_description,
            "task_category": task_category,
            "environment": "home",
            "environment_id": "env_christian_001",
            "duration_ms": duration_ms,
            "kit_version": "v0.1_record3d",
            "capture_date": r3d_path.stat().st_mtime if r3d_path.exists() else None,
            "source": "record3d",
            "fps": fps,
            "num_frames": num_frames,
            "rgb_resolution": [w, h],
            "depth_resolution": [meta.get("dw", 192), meta.get("dh", 256)],
            "has_depth": len(depth_paths) > 0,
            "has_audio": "sound.m4a" in z.namelist(),
        }
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(session_meta, f, indent=2)
        
        # 7. Create video from frames (for hand tracking)
        video_path = ""
        if create_video:
            video_path = os.path.join(output_dir, "video.mp4")
            frames_to_video(os.path.join(output_dir, "frames"), video_path, fps, w, h)
        
        # 8. Extract audio
        if "sound.m4a" in z.namelist():
            audio_path = os.path.join(output_dir, "sound.m4a")
            with open(audio_path, "wb") as f:
                f.write(z.read("sound.m4a"))
    
    return {
        "session_dir": output_dir,
        "num_frames": num_frames,
        "duration_ms": duration_ms,
        "fps": fps,
        "resolution": [w, h],
        "has_depth": len(depth_paths) > 0,
        "video_path": video_path,
        "task_category": task_category,
    }


def batch_ingest(
    r3d_dir: str,
    output_base: str,
    task_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Ingest all R3D files in a directory."""
    r3d_files = sorted(Path(r3d_dir).glob("**/*.r3d"))
    print(f"Found {len(r3d_files)} R3D files")
    
    results = []
    for r3d_path in r3d_files:
        name = r3d_path.stem
        # Infer task category from filename/parent folder
        parent = r3d_path.parent.name.lower().replace(" ", "_")
        task_cat = parent if parent != r3d_path.parent.parent.name else name
        
        # Map folder names to categories
        category_map = {
            "picknplace": "pick_place",
            "picknplace1": "pick_place",
            "picknplace2": "pick_place",
            "open_drawer": "open_drawer",
            "open_drawer1": "open_drawer",
            "pour": "pour",
            "pour1": "pour",
            "pour2": "pour",
            "stack": "stack",
            "stack1": "stack",
            "stack2": "stack",
            "sort": "sort",
            "sort1": "sort",
            "sort2": "sort",
            "assemble": "assemble",
            "assemble1": "assemble",
            "assemble2": "assemble",
            "fold": "fold_deformable",
            "fold1": "fold_deformable",
        }
        
        task_category = category_map.get(parent, parent)
        task_desc_map = {
            "pick_place": "Pick up an object and place it elsewhere",
            "open_drawer": "Open a drawer, interact with contents",
            "pour": "Pour liquid/contents from one container to another",
            "stack": "Stack multiple objects on top of each other",
            "sort": "Sort objects into groups",
            "assemble": "Assemble or fit parts together",
            "fold_deformable": "Fold a deformable object (cloth, towel)",
        }
        task_desc = task_desc_map.get(task_category, f"Manipulation task: {task_category}")
        
        out_dir = os.path.join(output_base, f"{parent}_{name}" if parent != name else name)
        
        try:
            result = ingest_r3d(
                str(r3d_path),
                out_dir,
                task_category=task_category,
                task_description=task_desc,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {r3d_path}: {e}")
            results.append({"error": str(e), "file": str(r3d_path)})
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Record3D .r3d files into pipeline format")
    parser.add_argument("input", help="Single .r3d file or directory of .r3d files")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth extraction")
    parser.add_argument("--no-video", action="store_true", help="Skip video creation")
    parser.add_argument("--task", default="general", help="Task category")
    parser.add_argument("--desc", default="manipulation task", help="Task description")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix == ".r3d":
        result = ingest_r3d(
            str(input_path), args.output,
            task_category=args.task,
            task_description=args.desc,
            extract_depth=not args.no_depth,
            create_video=not args.no_video,
        )
        print(f"\nResult: {json.dumps(result, indent=2)}")
    elif input_path.is_dir():
        results = batch_ingest(str(input_path), args.output)
        print(f"\n{'='*60}")
        print(f"Batch ingest complete: {len(results)} sessions")
        for r in results:
            if "error" in r:
                print(f"  FAIL: {r['file']} — {r['error']}")
            else:
                print(f"  OK: {r['task_category']} — {r['num_frames']} frames, {r['duration_ms']/1000:.1f}s")
    else:
        print(f"Error: {input_path} is not a .r3d file or directory")
        sys.exit(1)
