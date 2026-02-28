"""EgoCrowd CLI — process recordings, download datasets, replay, audit."""

import argparse
import sys


def cmd_process(args):
    """Process a .r3d recording through the pipeline."""
    from egocrowd.parse import parse_r3d
    from egocrowd.retarget import spatial_trajectory, save_trajectory
    
    print(f"Processing: {args.input}")
    
    # Step 1: Parse
    output_dir = args.output or "egocrowd_output"
    result = parse_r3d(args.input, output_dir=f"{output_dir}/r3d_parsed")
    print(f"  Parsed {result['n_frames']} frames")
    
    # Steps 2-3 require GPU (GroundingDINO + HaMeR)
    if args.cloud:
        print("  Cloud processing not yet implemented. Run manually:")
        print(f"    modal run processing/hamer_modal.py --input {output_dir}/r3d_parsed")
    else:
        print("  Object detection + hand pose require GPU.")
        print("  Use --cloud flag or run detection/hand_pose separately.")
        print("  See: egocrowd.detect.detect_objects() and egocrowd.hand_pose.extract_hand_poses()")
    
    # If pre-computed results exist, continue
    import os
    hamer_path = f"{output_dir}/r3d_parsed/hamer_results.json"
    obj_path = f"{output_dir}/r3d_parsed/object_poses_3d.json"
    
    if os.path.exists(hamer_path) and os.path.exists(obj_path):
        print("  Found pre-computed hand pose + object detection results")
        traj = spatial_trajectory(
            hamer_results=hamer_path,
            object_poses=obj_path,
            mug_sim=(0.5, 0.0, 0.295),
        )
        save_trajectory(traj, f"{output_dir}/spatial_traj.json")
        print(f"  Trajectory: {traj['n_frames']} frames, phases: {list(traj['phases'].keys())}")
    
    print(f"Output: {output_dir}/")


def cmd_download(args):
    """Download a dataset from HuggingFace."""
    from egocrowd.export import download_dataset
    path = download_dataset(repo_id=args.dataset)
    print(f"Downloaded to: {path}")


def cmd_replay(args):
    """Replay a trajectory in MuJoCo."""
    try:
        import mujoco
    except ImportError:
        print("MuJoCo required: pip install egocrowd[sim]")
        sys.exit(1)
    
    print(f"Replaying: {args.dataset}")
    print("(Full replay requires mujoco + scene XML. See tools/replay_final.py)")


def cmd_audit(args):
    """Run video audit on a replay recording."""
    print(f"Auditing: {args.video}")
    print("(See tools/video_audit.py for full audit functionality)")


def main():
    parser = argparse.ArgumentParser(
        prog="egocrowd",
        description="EgoCrowd: Crowdsourced egocentric manipulation data pipeline",
    )
    parser.add_argument("--version", action="version", version="egocrowd 0.1.0")
    
    sub = parser.add_subparsers(dest="command")
    
    # process
    p = sub.add_parser("process", help="Process a .r3d recording")
    p.add_argument("input", help="Path to .r3d file")
    p.add_argument("--object", default="mug", help="Object to detect (default: mug)")
    p.add_argument("--output", "-o", help="Output directory")
    p.add_argument("--cloud", action="store_true", help="Use cloud GPU for detection/hand pose")
    
    # download
    d = sub.add_parser("download", help="Download dataset from HuggingFace")
    d.add_argument("--dataset", default="egocrowd/pick-mug-v5", help="HF dataset repo ID")
    
    # replay
    r = sub.add_parser("replay", help="Replay trajectory in MuJoCo")
    r.add_argument("--dataset", required=True, help="Path to dataset directory")
    
    # audit
    a = sub.add_parser("audit", help="Audit a replay video")
    a.add_argument("--video", required=True, help="Path to replay video")
    
    args = parser.parse_args()
    
    if args.command == "process":
        cmd_process(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "audit":
        cmd_audit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
