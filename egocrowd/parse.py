"""Parse Record3D .r3d files into RGB frames, depth maps, and camera poses."""

import json
import zipfile
import numpy as np
from pathlib import Path
from typing import Optional


def parse_r3d(r3d_path: str, output_dir: Optional[str] = None) -> dict:
    """Parse a Record3D .r3d file.
    
    Args:
        r3d_path: Path to .r3d file
        output_dir: Optional directory to save extracted frames
        
    Returns:
        dict with keys: rgb_frames, depth_maps, camera_poses, metadata
    """
    r3d_path = Path(r3d_path)
    if not r3d_path.exists():
        raise FileNotFoundError(f"Recording not found: {r3d_path}")
    
    # .r3d files are ZIP archives
    with zipfile.ZipFile(r3d_path, 'r') as zf:
        names = zf.namelist()
        
        # Parse metadata
        metadata = {}
        if 'metadata' in names:
            metadata = json.loads(zf.read('metadata'))
        elif 'metadata.json' in names:
            metadata = json.loads(zf.read('metadata.json'))
        
        # Find RGB frames
        rgb_files = sorted([n for n in names if n.startswith('rgbd/') and n.endswith('.jpg')])
        depth_files = sorted([n for n in names if n.startswith('rgbd/') and n.endswith('.depth')])
        conf_files = sorted([n for n in names if n.startswith('rgbd/') and n.endswith('.conf')])
        
        n_frames = len(rgb_files)
        print(f"Found {n_frames} RGB frames, {len(depth_files)} depth maps")
        
        # Extract camera intrinsics
        camera_K = None
        if 'K' in metadata:
            camera_K = np.array(metadata['K']).reshape(3, 3)
        
        # Extract camera poses
        poses = []
        if 'poses' in metadata:
            raw_poses = metadata['poses']
            for i in range(0, len(raw_poses), 7):
                if i + 7 <= len(raw_poses):
                    # qx, qy, qz, qw, tx, ty, tz
                    poses.append(raw_poses[i:i+7])
        
        result = {
            'n_frames': n_frames,
            'rgb_files': rgb_files,
            'depth_files': depth_files,
            'camera_K': camera_K,
            'camera_poses': np.array(poses) if poses else None,
            'metadata': metadata,
        }
        
        # Optionally extract to disk
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            
            (out / 'rgb').mkdir(exist_ok=True)
            (out / 'depth').mkdir(exist_ok=True)
            
            for f in rgb_files:
                data = zf.read(f)
                idx = Path(f).stem
                (out / 'rgb' / f'{idx}.jpg').write_bytes(data)
            
            for f in depth_files:
                data = zf.read(f)
                idx = Path(f).stem
                (out / 'depth' / f'{idx}.bin').write_bytes(data)
            
            # Save metadata
            with open(out / 'metadata.json', 'w') as mf:
                json.dump(metadata, mf, indent=2, default=str)
            
            print(f"Extracted to {out}")
            result['output_dir'] = str(out)
        
        return result
