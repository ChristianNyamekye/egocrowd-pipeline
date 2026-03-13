"""
Object detection from R3D video frames using GroundingDINO.
Extracts 3D object positions using depth maps + camera intrinsics.

Pipeline:
1. Extract RGB frame at first grasp frame from R3D
2. Run GroundingDINO (zero-shot) to detect objects ("block", "cube", "box", "object")
3. Get bounding box center → pixel coords (u, v)
4. Read depth at (u, v) from R3D depth map
5. Unproject to 3D: X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy
6. Transform camera→world using R3D camera pose
"""
import json, struct, zipfile, sys
import numpy as np
from pathlib import Path
from PIL import Image
import io

RAW_CAPTURES = Path(__file__).resolve().parent / "raw_captures"
RETARGET_DIR = Path(__file__).resolve().parent / "gpu_retargeted"
OUTPUT = Path(__file__).resolve().parent / "object_detections"
OUTPUT.mkdir(exist_ok=True)


def load_r3d_metadata(r3d_path):
    """Load R3D metadata (intrinsics, poses, etc.)"""
    with zipfile.ZipFile(r3d_path) as z:
        with z.open("metadata") as f:
            return json.load(f)


def load_r3d_frame(r3d_path, frame_idx):
    """Load RGB image and depth map for a specific frame."""
    with zipfile.ZipFile(r3d_path) as z:
        # RGB
        jpg_name = f"rgbd/{frame_idx}.jpg"
        try:
            with z.open(jpg_name) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
        except KeyError:
            return None, None

        # Depth (16-bit or float16, depending on Record3D version)
        depth_name = f"rgbd/{frame_idx}.depth"
        try:
            with z.open(depth_name) as f:
                raw = f.read()
        except KeyError:
            return img, None

        # Record3D depth: LZFSE-compressed float32 array, dw x dh resolution
        try:
            import liblzfse
            raw = liblzfse.decompress(raw)
        except (ImportError, Exception):
            pass  # May already be uncompressed
        depth = np.frombuffer(raw, dtype=np.float32)

        return img, depth

    return None, None


def unproject_pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """Unproject pixel (u,v) with depth Z to 3D camera coordinates."""
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def camera_to_world(point_cam, pose):
    """Transform 3D point from camera frame to world frame using R3D pose.
    R3D pose = [x, y, z, qw, qx, qy, qz] (translation + quaternion).
    """
    tx, ty, tz = pose[0], pose[1], pose[2]
    qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]

    # Quaternion to rotation matrix
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])

    t = np.array([tx, ty, tz])
    return R @ point_cam + t


def detect_objects_groundingdino(image, text_prompt="block . cube . box . object"):
    """Run GroundingDINO locally using transformers (CPU fallback)."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch

    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"  Loading GroundingDINO ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.2, text_threshold=0.2,
        target_sizes=[image.size[::-1]]
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.cpu().numpy()
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        detections.append({
            "label": label,
            "score": float(score),
            "bbox": [float(x) for x in box],
            "center_px": [float(cx), float(cy)],
            "size_px": [float(w), float(h)]
        })
        print(f"    {label}: score={score:.3f} center=({cx:.0f},{cy:.0f}) size=({w:.0f}x{h:.0f})")

    return detections


def process_session(session_name):
    """Process one session: detect objects and compute 3D positions."""
    print(f"\n{'='*60}")
    print(f"Object Detection: {session_name}")
    print(f"{'='*60}")

    # Find R3D file
    r3d_dir = RAW_CAPTURES / session_name
    r3d_files = list(r3d_dir.glob("*.r3d"))
    if not r3d_files:
        print(f"  ERROR: No .r3d file found in {r3d_dir}")
        return None
    r3d_path = r3d_files[0]

    # Load retarget data for grasp timing
    ret_path = RETARGET_DIR / f"{session_name}_retargeted.json"
    if not ret_path.exists():
        print(f"  ERROR: No retarget data at {ret_path}")
        return None

    with open(ret_path) as f:
        ret_data = json.load(f)
    ts = ret_data["timesteps"]
    first_grasp = next((i for i, t in enumerate(ts) if t.get("grasping")), 0)
    # Use a frame BEFORE the first grasp (objects should be at rest)
    detect_frame = max(0, first_grasp - 5)
    print(f"  First grasp at frame {first_grasp}, detecting objects at frame {detect_frame}")

    # Load R3D metadata
    meta = load_r3d_metadata(r3d_path)
    fps_ratio = meta["fps"] / 30.0  # R3D is 60fps, our pipeline is ~30fps
    r3d_frame_idx = int(detect_frame * fps_ratio)

    # Camera intrinsics
    if meta.get("perFrameIntrinsicCoeffs") and r3d_frame_idx < len(meta["perFrameIntrinsicCoeffs"]):
        coeffs = meta["perFrameIntrinsicCoeffs"][r3d_frame_idx]
        fx, fy, cx, cy = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    else:
        K = meta["K"]
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    print(f"  Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"  Image size: {meta['w']}x{meta['h']}, Depth size: {meta['dw']}x{meta['dh']}")

    # Camera pose
    if r3d_frame_idx < len(meta["poses"]):
        pose = meta["poses"][r3d_frame_idx]
    else:
        pose = meta["poses"][-1]
    print(f"  Camera pose: t=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f})")

    # Load RGB + depth
    img, depth = load_r3d_frame(r3d_path, r3d_frame_idx)
    if img is None:
        print(f"  ERROR: Could not load frame {r3d_frame_idx}")
        return None

    print(f"  RGB: {img.size}, Depth: {'available' if depth is not None else 'MISSING'}")
    if depth is not None:
        print(f"  Depth stats: min={depth.min():.3f} max={depth.max():.3f} mean={depth.mean():.3f}")

    # Save the detection frame for reference
    frame_path = OUTPUT / f"{session_name}_detect_frame.jpg"
    img.save(frame_path)

    # Run GroundingDINO
    task_prompts = {
        "stack": "red block . blue block . cube",
        "picknplace": "block . cube . object",
        "sort": "block . cube . object",
        "fold": "cloth . towel . fabric",
        "open_drawer": "drawer . handle",
    }
    # Determine task
    task = "stack" if "stack" in session_name else \
           "picknplace" if "picknplace" in session_name else \
           "sort" if "sort" in session_name else \
           "fold" if "fold" in session_name else \
           "open_drawer" if "drawer" in session_name else "object"
    prompt = task_prompts.get(task, "object . block . item")
    print(f"  Task: {task}, Prompt: '{prompt}'")

    detections = detect_objects_groundingdino(img, prompt)

    if not detections:
        print("  WARNING: No objects detected!")
        return {"session": session_name, "frame": detect_frame, "detections": []}

    # Compute 3D positions using depth
    results = []
    for det in detections:
        u, v = det["center_px"]
        # Map pixel coords to depth map coords (depth is lower res)
        du = int(u * meta["dw"] / meta["w"])
        dv = int(v * meta["dh"] / meta["h"])

        if depth is not None:
            dw, dh = meta["dw"], meta["dh"]
            if 0 <= du < dw and 0 <= dv < dh:
                d_val = depth[dv * dw + du]  # depth is stored row-major
                if d_val > 0.01:  # valid depth
                    # Unproject to camera 3D
                    point_cam = unproject_pixel_to_3d(u, v, d_val, fx, fy, cx, cy)
                    # Transform to world
                    point_world = camera_to_world(point_cam, pose)
                    det["depth_m"] = float(d_val)
                    det["pos_cam"] = point_cam.tolist()
                    det["pos_world"] = point_world.tolist()
                    print(f"    3D: cam=({point_cam[0]:.3f},{point_cam[1]:.3f},{point_cam[2]:.3f}) "
                          f"world=({point_world[0]:.3f},{point_world[1]:.3f},{point_world[2]:.3f}) "
                          f"depth={d_val:.3f}m")
                else:
                    print(f"    WARNING: Invalid depth ({d_val:.3f}) at ({du},{dv})")
            else:
                print(f"    WARNING: Depth coords out of range ({du},{dv})")
        results.append(det)

    output = {
        "session": session_name,
        "task": task,
        "detect_frame": detect_frame,
        "r3d_frame": r3d_frame_idx,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "camera_pose": pose,
        "image_size": [meta["w"], meta["h"]],
        "depth_size": [meta["dw"], meta["dh"]],
        "detections": results
    }

    # Save
    out_path = OUTPUT / f"{session_name}_objects.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ Saved to {out_path}")
    return output


if __name__ == "__main__":
    sessions = sys.argv[1:] if len(sys.argv) > 1 else [
        "stack1", "stack2", "picknplace1", "picknplace2", "sort1"
    ]
    for session in sessions:
        try:
            process_session(session)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
