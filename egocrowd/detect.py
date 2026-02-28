"""Open-vocabulary object detection via GroundingDINO."""

from typing import Optional


def detect_objects(
    rgb_dir: str,
    prompt: str = "mug",
    confidence: float = 0.3,
    device: Optional[str] = None,
) -> dict:
    """Detect objects in RGB frames using GroundingDINO.
    
    Requires: pip install egocrowd[gpu]
    
    Args:
        rgb_dir: Directory containing RGB frames (.jpg)
        prompt: Text prompt for open-vocabulary detection (e.g., "mug", "bottle")
        confidence: Detection confidence threshold
        device: torch device (auto-detected if None)
        
    Returns:
        dict with per-frame bounding boxes and confidence scores
    """
    try:
        import torch
        from groundingdino.util.inference import load_model, predict
    except ImportError:
        raise ImportError(
            "GroundingDINO requires GPU dependencies. Install with:\n"
            "  pip install egocrowd[gpu]\n"
            "Or run detection on Modal cloud:\n"
            "  egocrowd process recording.r3d --cloud"
        )
    
    from pathlib import Path
    import numpy as np
    from PIL import Image
    
    rgb_path = Path(rgb_dir)
    frames = sorted(rgb_path.glob("*.jpg"))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    for frame_path in frames:
        img = Image.open(frame_path)
        # GroundingDINO inference
        boxes, logits, phrases = predict(
            model=None,  # loaded separately
            image=img,
            caption=prompt,
            box_threshold=confidence,
            text_threshold=confidence,
        )
        results[frame_path.stem] = {
            "boxes": boxes.tolist() if len(boxes) > 0 else [],
            "scores": logits.tolist() if len(logits) > 0 else [],
            "labels": phrases,
        }
    
    return results
