"""3D hand mesh reconstruction via HaMeR."""

from typing import Optional


def extract_hand_poses(
    rgb_dir: str,
    hand_boxes: Optional[dict] = None,
    device: Optional[str] = None,
) -> dict:
    """Extract 3D hand meshes from RGB frames using HaMeR.
    
    Requires: pip install egocrowd[gpu]
    For cloud processing: egocrowd process recording.r3d --cloud
    
    Args:
        rgb_dir: Directory containing RGB frames (.jpg)
        hand_boxes: Optional pre-computed hand bounding boxes from detect_objects()
        device: torch device (auto-detected if None)
        
    Returns:
        dict with per-frame MANO parameters, joint positions, and translations
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "HaMeR requires GPU dependencies. Install with:\n"
            "  pip install egocrowd[gpu]\n"
            "Or run on Modal cloud:\n"
            "  egocrowd process recording.r3d --cloud"
        )
    
    from pathlib import Path
    
    rgb_path = Path(rgb_dir)
    frames = sorted(rgb_path.glob("*.jpg"))
    
    print(f"Processing {len(frames)} frames with HaMeR...")
    print("Note: HaMeR runs best on GPU. For cloud processing, use --cloud flag.")
    
    # HaMeR inference would go here
    # In practice, this runs on Modal (see tools/v5_hand_detect.py)
    raise NotImplementedError(
        "Local HaMeR inference not yet packaged. Use cloud processing:\n"
        "  egocrowd process recording.r3d --cloud\n"
        "Or run the Modal deployment directly:\n"
        "  modal run processing/hamer_modal.py"
    )
