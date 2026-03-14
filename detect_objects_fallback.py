"""CPU-only object detection fallback.

Accepts manual object positions via CLI args (no GPU required).
Output format matches GroundingDINO pipeline: {detections: [{label, pos_world, ...}]}

Usage:
    python detect_objects_fallback.py <session> --objects '["red block", "blue block"]' \
        --positions '[[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]'
"""
import argparse
import json
import sys
from pathlib import Path

from pipeline_config import OBJECT_DET_DIR


def create_detections(labels, positions):
    """Create detection output matching GroundingDINO format."""
    detections = []
    for label, pos in zip(labels, positions):
        detections.append({
            "label": label,
            "pos_world": pos,
            "confidence": 1.0,
            "source": "manual",
        })
    return {"detections": detections}


def detect_objects_manual(session_name, labels, positions, output_dir=None):
    """Write manual object detections to JSON."""
    if output_dir is None:
        output_dir = OBJECT_DET_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = create_detections(labels, positions)
    out_path = output_dir / f"{session_name}_objects_clean.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {len(result['detections'])} detections to {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU fallback object detection (manual positions)")
    parser.add_argument("session", help="Session name")
    parser.add_argument("--objects", required=True, help='JSON list of object labels')
    parser.add_argument("--positions", required=True, help='JSON list of [x,y,z] positions')
    parser.add_argument("-o", "--output-dir", help="Output directory")
    args = parser.parse_args()

    labels = json.loads(args.objects)
    positions = json.loads(args.positions)

    if len(labels) != len(positions):
        print(f"Error: {len(labels)} labels but {len(positions)} positions", file=sys.stderr)
        sys.exit(1)

    detect_objects_manual(args.session, labels, positions, args.output_dir)
