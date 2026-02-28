"""
V2: Fuse GroundingDINO 2D detections with LiDAR depth to get 3D object pose.
This runs locally — no GPU needed (just depth + camera intrinsics math).

Input: RGB frames + depth frames + GroundingDINO detections
Output: 3D object poses per frame (camera frame)
"""
import os, json, sys
import numpy as np
from PIL import Image

def load_depth_frame(depth_dir, frame_idx, width=960, height=720):
    """Load depth frame (raw uint16 binary or PNG)."""
    # Try npy first (our r3d pipeline format)
    npy_path = os.path.join(depth_dir, f"{frame_idx:04d}.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32)
    
    # Try binary
    bin_path = os.path.join(depth_dir, f"{frame_idx:04d}.bin")
    if os.path.exists(bin_path):
        depth = np.fromfile(bin_path, dtype=np.float32).reshape(height, width)
        return depth
    
    # Try PNG
    png_path = os.path.join(depth_dir, f"{frame_idx:04d}.png")
    if os.path.exists(png_path):
        depth = np.array(Image.open(png_path)).astype(np.float32) / 1000.0
        return depth
    
    return None


def backproject_to_3d(cx_px, cy_px, depth_m, fx, fy, cx, cy):
    """Back-project 2D pixel + depth to 3D point in camera frame."""
    X = (cx_px - cx) * depth_m / fx
    Y = (cy_px - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z])


def fuse_detections_with_depth(detections, depth_dir, camera_k, depth_shape=(720, 960)):
    """
    Fuse 2D detections with depth to get 3D poses.
    
    detections: list of per-frame detection dicts from GroundingDINO
    camera_k: [fx, fy, cx, cy]
    """
    fx, fy, cx, cy = camera_k
    results = []
    
    for i, frame_dets in enumerate(detections):
        if not frame_dets:  # no detections
            results.append({"frame": i, "detected": False})
            continue
        
        # Use best detection
        best = max(frame_dets, key=lambda d: d['score'])
        box = best['box']  # [x1, y1, x2, y2]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # Load depth
        depth = load_depth_frame(depth_dir, i, depth_shape[1], depth_shape[0])
        if depth is None:
            results.append({
                "frame": i, "detected": True,
                "score": best['score'],
                "box_2d": box,
                "center_2d": [center_x, center_y],
                "pose_3d": None,
                "reason": "no depth frame"
            })
            continue
        
        # Scale detection box from RGB resolution to depth resolution
        rgb_w, rgb_h = 720, 960  # from PIL Image.size (W, H)
        depth_h, depth_w = depth.shape
        sx = depth_w / rgb_w
        sy = depth_h / rgb_h
        
        x1 = max(0, int(box[0] * sx))
        y1 = max(0, int(box[1] * sy))
        x2 = min(depth_w, int(box[2] * sx))
        y2 = min(depth_h, int(box[3] * sy))
        
        depth_roi = depth[y1:y2, x1:x2]
        valid_depths = depth_roi[depth_roi > 0.01]  # filter out zero/invalid
        
        if len(valid_depths) == 0:
            results.append({
                "frame": i, "detected": True,
                "score": best['score'],
                "box_2d": box,
                "center_2d": [center_x, center_y],
                "pose_3d": None,
                "reason": "no valid depth in ROI"
            })
            continue
        
        depth_m = float(np.median(valid_depths))
        pose_3d = backproject_to_3d(center_x, center_y, depth_m, fx, fy, cx, cy)
        
        results.append({
            "frame": i,
            "detected": True,
            "score": best['score'],
            "label": best.get('label', 'mug'),
            "box_2d": [float(x) for x in box],
            "center_2d": [float(center_x), float(center_y)],
            "depth_m": depth_m,
            "pose_3d": pose_3d.tolist(),  # [X, Y, Z] in camera frame
        })
    
    return results


def main():
    """Run depth fusion on our pipeline output."""
    rgb_dir = "pipeline/r3d_output/rgb"
    depth_dir = "pipeline/r3d_output/depth"
    meta_path = "pipeline/r3d_output/metadata.json"
    
    if not os.path.exists(rgb_dir):
        print("No RGB frames found. Run r3d_pipeline.py first.")
        return
    
    # Load camera intrinsics
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        K = meta.get('camera_intrinsics', {})
        fx = K.get('fx', 684.56)
        fy = K.get('fy', 684.56)
        cx = K.get('cx', 480)
        cy = K.get('cy', 360)
    else:
        # Default from Christian's Record3D
        fx = fy = 684.56
        cx, cy = 480, 360
    
    camera_k = [fx, fy, cx, cy]
    print(f"Camera K: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Simulate detections (from our GroundingDINO results)
    # In production, these come from Modal
    sample_detections = [
        [{"label": "mug cup", "score": 0.736, "box": [423, 481, 635, 802]}],
        [{"label": "mug cup", "score": 0.734, "box": [422, 482, 635, 802]}],
        [{"label": "mug cup", "score": 0.723, "box": [423, 480, 634, 800]}],
        [{"label": "mug cup", "score": 0.730, "box": [421, 479, 633, 800]}],
        [{"label": "mug cup", "score": 0.738, "box": [419, 479, 632, 800]}],
    ]
    
    # Fuse with depth
    results = fuse_detections_with_depth(sample_detections, depth_dir, camera_k)
    
    # Print results
    n_detected = sum(1 for r in results if r.get('detected'))
    n_3d = sum(1 for r in results if r.get('pose_3d'))
    print(f"\nResults: {n_detected}/{len(results)} detected, {n_3d}/{len(results)} with 3D pose")
    
    for r in results:
        if r.get('pose_3d'):
            p = r['pose_3d']
            print(f"  Frame {r['frame']}: {r['label']} score={r['score']:.3f} "
                  f"depth={r['depth_m']:.3f}m pose=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        elif r.get('detected'):
            print(f"  Frame {r['frame']}: detected but no depth ({r.get('reason', '?')})")
        else:
            print(f"  Frame {r['frame']}: not detected")
    
    # Save results
    out_path = "pipeline/r3d_output/object_poses_3d.json"
    with open(out_path, 'w') as f:
        json.dump({"camera_k": camera_k, "poses": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
