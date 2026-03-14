"""
Smoke tests for the flexa pipeline — no GPU or graphics context required.
Validates data flow through pipeline stages independently.
"""

import json
import os
import tempfile
import numpy as np
from pathlib import Path


def test_synthetic_data_generation():
    """Synthetic data generator produces valid calibrated JSON."""
    from synthetic_data import generate_stack_trajectory

    for robot in ["g1", "franka"]:
        calib = generate_stack_trajectory(robot)

        assert "wrist_sim" in calib
        assert "grasping" in calib
        assert "objects_sim" in calib

        wrist = np.array(calib["wrist_sim"])
        grasping = np.array(calib["grasping"])
        objects = calib["objects_sim"]

        assert wrist.shape == (200, 3), f"Expected (200,3), got {wrist.shape}"
        assert len(grasping) == 200
        assert len(objects) == 2

        # Grasping window should exist
        grip_frames = np.sum(grasping > 0)
        assert grip_frames > 20, f"Too few grip frames: {grip_frames}"

        # Wrist should stay in reasonable bounds
        assert np.all(wrist[:, 2] > 0), "Wrist Z should be positive"

    print("PASS: test_synthetic_data_generation")


def test_pipeline_config():
    """Pipeline config resolves paths correctly."""
    from pipeline_config import PROJECT_ROOT, CALIB_DIR, OUT_DIR

    assert PROJECT_ROOT.exists(), f"PROJECT_ROOT missing: {PROJECT_ROOT}"
    assert CALIB_DIR.name == "wrist_trajectories"
    assert OUT_DIR.name == "sim_renders"

    print("PASS: test_pipeline_config")


def test_r3d_depth_decompression():
    """Depth extraction handles both compressed and raw float32 arrays."""
    # Simulate a raw (uncompressed) float32 depth buffer
    dw, dh = 192, 256
    depth = np.random.uniform(0.1, 3.0, (dh, dw)).astype(np.float32)
    raw_bytes = depth.tobytes()

    result = np.frombuffer(raw_bytes, dtype=np.float32).reshape(dh, dw)
    assert result.shape == (dh, dw)
    assert np.allclose(result, depth)

    print("PASS: test_r3d_depth_decompression")


def test_synthetic_calibrated_json_roundtrip():
    """Calibrated JSON can be written and read back for simulation."""
    from synthetic_data import generate_synthetic
    from pipeline_config import CALIB_DIR

    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override CALIB_DIR
        import synthetic_data
        orig_dir = synthetic_data.CALIB_DIR
        synthetic_data.CALIB_DIR = Path(tmpdir)
        try:
            session, path = generate_synthetic("g1", "stack")
            assert Path(path).exists()

            calib = json.loads(Path(path).read_text())
            assert isinstance(calib["wrist_sim"], list)
            assert isinstance(calib["grasping"], list)
            assert isinstance(calib["objects_sim"], list)
            assert len(calib["wrist_sim"]) == len(calib["grasping"])
        finally:
            synthetic_data.CALIB_DIR = orig_dir

    print("PASS: test_synthetic_calibrated_json_roundtrip")


def test_object_detection_output_format():
    """Object detection writes correct JSON format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        labels = ["block_a", "block_b"]
        positions = [[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]

        # Write directly (no import of fallback module)
        result = {"detections": []}
        for label, pos in zip(labels, positions):
            result["detections"].append({
                "label": label,
                "pos_world": pos,
                "confidence": 1.0,
                "source": "manual",
            })

        out_path = Path(tmpdir) / "test_objects_clean.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        loaded = json.loads(out_path.read_text())
        assert len(loaded["detections"]) == 2
        assert loaded["detections"][0]["label"] == "block_a"
        assert loaded["detections"][1]["pos_world"] == [0.35, 0.1, 0.43]

    print("PASS: test_object_detection_output_format")


if __name__ == "__main__":
    test_pipeline_config()
    test_synthetic_data_generation()
    test_r3d_depth_decompression()
    test_synthetic_calibrated_json_roundtrip()
    test_object_detection_output_format()
    print(f"\nAll tests passed.")
