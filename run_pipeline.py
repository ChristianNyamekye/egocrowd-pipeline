#!/usr/bin/env python3
"""Flexa Pipeline: End-to-end R3D → robot simulation.

Processes iPhone LiDAR recordings (.r3d) through all stages and produces
a robot simulation video. Supports synthetic data mode for testing.

Usage:
    # Real R3D mode (primary):
    python run_pipeline.py --r3d path/to/recording.r3d --robot g1 --task stack

    # With manual object positions (skip GPU detection):
    python run_pipeline.py --r3d recording.r3d --robot g1 --task stack \
        --objects '[[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]'

    # Synthetic mode (testing without .r3d):
    python run_pipeline.py --synthetic --robot g1 --task stack
"""
import argparse
import json
import sys
import time
from pathlib import Path

from pipeline_config import PROJECT_ROOT, OUT_DIR, CALIB_DIR, R3D_OUTPUT, OBJECT_DET_DIR


def log_stage(stage_num, total, name, status="starting"):
    prefix = f"[{stage_num}/{total}]"
    if status == "starting":
        print(f"\n{'='*60}")
        print(f"{prefix} {name}")
        print(f"{'='*60}")
    elif status == "done":
        print(f"{prefix} {name} — done")
    elif status == "skip":
        print(f"{prefix} {name} — skipped")
    elif status == "fail":
        print(f"{prefix} {name} — FAILED")


def validate_file(path, label):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label}: {p} does not exist")
    if p.stat().st_size == 0:
        raise ValueError(f"{label}: {p} is empty")
    return True


def run_synthetic(robot, task, session_name=None):
    """Synthetic mode: generate test data and run simulation."""
    from synthetic_data import generate_synthetic

    total = 2
    session_name = session_name or f"synthetic_{task}"

    # Stage 1: Generate synthetic calibrated JSON
    log_stage(1, total, "Generate synthetic calibrated data")
    session, calib_path = generate_synthetic(robot, task)
    validate_file(calib_path, "Calibrated JSON")
    log_stage(1, total, "Generate synthetic calibrated data", "done")

    # Stage 2: Run simulation
    log_stage(2, total, f"Run {robot} simulation")
    video_path = run_simulation(robot, session, task)
    log_stage(2, total, f"Run {robot} simulation", "done")

    return video_path


def run_r3d_pipeline(r3d_path, robot, task, session_name=None, objects_manual=None):
    """Real R3D mode: full pipeline from .r3d to simulation video."""
    total = 6
    r3d_path = Path(r3d_path).resolve()

    if not r3d_path.exists():
        print(f"ERROR: R3D file not found: {r3d_path}", file=sys.stderr)
        sys.exit(1)

    if session_name is None:
        session_name = r3d_path.stem.replace(" ", "_").lower()

    session_dir = R3D_OUTPUT / session_name
    R3D_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Stage 1: Ingest R3D
    log_stage(1, total, "Ingest R3D → frames + depth + video")
    from r3d_ingest import ingest_r3d
    ingest_result = ingest_r3d(
        str(r3d_path), str(session_dir),
        task_category=task,
        task_description=f"{task} manipulation task",
    )

    # Validate
    video_path = ingest_result.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        # Try default location
        video_path = str(session_dir / "video.mp4")
    validate_file(video_path, "Extracted video")
    n_frames = ingest_result.get("num_frames", 0)
    print(f"  Extracted {n_frames} frames, video: {video_path}")
    log_stage(1, total, "Ingest R3D", "done")

    # Stage 2: Hand tracking (MediaPipe)
    log_stage(2, total, "Hand tracking (MediaPipe) → trajectory JSON")
    from hand_tracker_v2 import process_video
    traj_path = str(session_dir / f"{session_name}_hand_trajectory.json")
    trajectory = process_video(video_path, traj_path)

    # Validate detection rate
    if trajectory and trajectory.get("frames"):
        frames_with_hands = sum(1 for f in trajectory["frames"] if f.get("hands"))
        total_frames = len(trajectory["frames"])
        detection_rate = frames_with_hands / max(total_frames, 1)
        print(f"  Detection rate: {detection_rate:.0%} ({frames_with_hands}/{total_frames})")
        if detection_rate < 0.1:
            print(f"  WARNING: Very low hand detection rate ({detection_rate:.0%})")
    log_stage(2, total, "Hand tracking", "done")

    # Stage 3: Object detection (fallback)
    log_stage(3, total, "Object detection → objects JSON")
    OBJECT_DET_DIR.mkdir(parents=True, exist_ok=True)

    if objects_manual is not None:
        from detect_objects_fallback import detect_objects_manual
        obj_positions = json.loads(objects_manual) if isinstance(objects_manual, str) else objects_manual
        labels = [f"block_{chr(97+i)}" for i in range(len(obj_positions))]
        detect_objects_manual(session_name, labels, obj_positions)
        print(f"  Manual object positions: {len(obj_positions)} objects")
    else:
        # Try to infer from depth maps (simple heuristic)
        print("  No manual positions provided — using default object placement")
        from detect_objects_fallback import detect_objects_manual
        # Default positions: two blocks on the table
        default_positions = [[0.0, 0.0, 0.5], [0.1, -0.05, 0.5]]
        labels = ["block_a", "block_b"]
        detect_objects_manual(session_name, labels, default_positions)
    log_stage(3, total, "Object detection", "done")

    # Stage 4: 3D wrist reconstruction
    log_stage(4, total, "3D wrist reconstruction → wrist3d JSON")

    # Build retarget-compatible data from hand trajectory
    _build_retarget_data(session_name, session_dir, trajectory)

    from reconstruct_wrist_3d import process_session
    # Ensure raw_captures has a symlink/copy of the R3D for reconstruct_wrist_3d
    from pipeline_config import RAW_CAPTURES
    raw_session_dir = RAW_CAPTURES / session_name
    raw_session_dir.mkdir(parents=True, exist_ok=True)
    r3d_link = raw_session_dir / r3d_path.name
    if not r3d_link.exists():
        import shutil
        shutil.copy2(str(r3d_path), str(r3d_link))

    wrist_result = process_session(session_name)

    # Validate
    if wrist_result:
        n_valid = wrist_result.get("n_valid", 0)
        n_total = wrist_result.get("n_frames", 1)
        valid_rate = n_valid / max(n_total, 1)
        print(f"  Valid frames: {valid_rate:.0%} ({n_valid}/{n_total})")
        if valid_rate < 0.3:
            print(f"  WARNING: Low valid frame rate ({valid_rate:.0%})")
    else:
        print("  WARNING: Wrist reconstruction returned no result")
    log_stage(4, total, "3D wrist reconstruction", "done")

    # Stage 5: Workspace calibration
    log_stage(5, total, "Workspace calibration → calibrated JSON")
    from calibrate_workspace import calibrate_session
    calib_path = calibrate_session(session_name)

    if calib_path is None:
        print("  ERROR: Calibration failed")
        sys.exit(1)
    validate_file(calib_path, "Calibrated JSON")
    log_stage(5, total, "Workspace calibration", "done")

    # Stage 6: Simulation
    log_stage(6, total, f"Run {robot} simulation → video")
    video_path = run_simulation(robot, session_name, task)
    log_stage(6, total, f"Run {robot} simulation", "done")

    return video_path


def _build_retarget_data(session_name, session_dir, trajectory):
    """Convert hand_tracker_v2 output to retarget-compatible format for reconstruct_wrist_3d."""
    from pipeline_config import RETARGET_DIR
    RETARGET_DIR.mkdir(parents=True, exist_ok=True)

    timesteps = []
    if trajectory and trajectory.get("frames"):
        for frame in trajectory["frames"]:
            ts = {"frame": frame["frame"], "timestamp": frame.get("timestamp", 0)}
            if frame.get("hands"):
                hand = frame["hands"][0]  # primary hand
                wrist_pos = hand.get("wrist", {}).get("position", {})
                ts["wrist_pixel"] = [wrist_pos.get("x", 0), wrist_pos.get("y", 0)]
                ts["grasping"] = hand.get("grasping", False)
            else:
                ts["wrist_pixel"] = None
                ts["grasping"] = False
            timesteps.append(ts)

    retarget_data = {"session": session_name, "timesteps": timesteps}
    out_path = RETARGET_DIR / f"{session_name}_retargeted.json"
    with open(out_path, "w") as f:
        json.dump(retarget_data, f)
    print(f"  Built retarget data: {out_path} ({len(timesteps)} timesteps)")


def run_simulation(robot, session_name, task):
    """Run the appropriate simulation script for the given robot."""
    if robot == "g1":
        from mujoco_g1_v10 import render_task
        video_path = render_task(session_name)
    elif robot == "franka":
        from mujoco_franka_v9 import simulate
        simulate(session_name, task)
        video_path = OUT_DIR / f"{session_name}_franka_v9.mp4"
    elif robot == "h1":
        from mujoco_h1_shadow_v1 import render_task
        video_path = render_task(session_name)
    else:
        raise ValueError(f"Unknown robot: {robot}")

    video_path = Path(video_path) if video_path else OUT_DIR / f"{session_name}_{robot}.mp4"
    if video_path.exists() and video_path.stat().st_size > 0:
        size_kb = video_path.stat().st_size // 1024
        print(f"\n  Output video: {video_path} ({size_kb} KB)")
    else:
        print(f"\n  WARNING: Output video missing or empty: {video_path}")

    return str(video_path)


def main():
    parser = argparse.ArgumentParser(
        description="Flexa Pipeline: R3D → robot simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --synthetic --robot g1 --task stack
  python run_pipeline.py --r3d recording.r3d --robot g1 --task stack
  python run_pipeline.py --r3d recording.r3d --robot g1 --task stack \\
      --objects '[[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]'
""")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--r3d", type=str, help="Path to .r3d file (real mode)")
    group.add_argument("--synthetic", action="store_true", help="Use synthetic data (test mode)")

    parser.add_argument("--robot", default="g1", choices=["g1", "franka", "h1"],
                        help="Robot model (default: g1)")
    parser.add_argument("--task", default="stack", choices=["stack", "pick_place", "sort"],
                        help="Task type (default: stack)")
    parser.add_argument("--objects", type=str, default=None,
                        help="Manual object positions as JSON [[x,y,z], ...] (skip detection)")
    parser.add_argument("--session", type=str, default=None,
                        help="Session name (default: derived from filename)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: sim_renders/)")
    args = parser.parse_args()

    if args.output:
        import pipeline_config
        pipeline_config.OUT_DIR = Path(args.output)
        pipeline_config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Flexa Pipeline — robot={args.robot}, task={args.task}")
    print(f"{'='*60}")
    t0 = time.time()

    if args.synthetic:
        video = run_synthetic(args.robot, args.task, args.session)
    else:
        video = run_r3d_pipeline(args.r3d, args.robot, args.task, args.session, args.objects)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Output: {video}")


if __name__ == "__main__":
    main()
