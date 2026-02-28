"""
V5: Hand detection using GroundingDINO (already working on Modal).
Detects hands in egocentric frames → 2D boxes → fuse with depth → 3D hand positions.
Much simpler than whole-body pose estimation for egocentric footage.
"""
import modal
import os, json

app = modal.App("egocrowd-v5-hands")

gdino_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0",
        "transformers>=4.36.0", "numpy<2", "Pillow",
    )
)


@app.function(gpu="T4", image=gdino_image, timeout=600)
def detect_hands(frame_bytes_list: list, threshold: float = 0.2):
    """Detect hands using GroundingDINO with text prompt 'hand'."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import io, numpy as np

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).cuda()

    results = []
    for frame_bytes in frame_bytes_list:
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        inputs = processor(images=image, text="hand . finger . palm", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        dets = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        hands = []
        for score, label, box in zip(dets["scores"], dets["labels"], dets["boxes"]):
            box = box.cpu().numpy().tolist()
            hands.append({
                "label": label,
                "score": float(score.cpu()),
                "box": box,
                "center": [(box[0]+box[2])/2, (box[1]+box[3])/2]
            })

        results.append({
            "detected": len(hands) > 0,
            "n_hands": len(hands),
            "hands": hands
        })

    return results


@app.local_entrypoint()
def main():
    rgb_dir = "pipeline/r3d_output/rgb"
    if not os.path.exists(rgb_dir):
        print("No RGB frames found")
        return

    # Load all frames (up to 301)
    frames = sorted(os.listdir(rgb_dir))
    print(f"Processing {len(frames)} frames...")

    # Process in batches of 50
    batch_size = 50
    all_results = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        frame_bytes = []
        for f in batch_frames:
            with open(os.path.join(rgb_dir, f), 'rb') as fh:
                frame_bytes.append(fh.read())

        batch_results = detect_hands.remote(frame_bytes)
        all_results.extend(batch_results)
        n_det = sum(1 for r in batch_results if r["detected"])
        print(f"  Batch {i//batch_size + 1}: {n_det}/{len(batch_results)} frames with hands")

    # Summary
    total_detected = sum(1 for r in all_results if r["detected"])
    total_hands = sum(r["n_hands"] for r in all_results)
    rate = 100 * total_detected / len(all_results)
    print(f"\n{'='*50}")
    print(f"V5 HAND DETECTION: {total_detected}/{len(all_results)} frames ({rate:.1f}%)")
    print(f"Total hand instances: {total_hands}")
    print(f"{'='*50}")

    # Compare with MediaPipe baseline (50.5%)
    if rate > 50.5:
        improvement = rate - 50.5
        print(f"IMPROVEMENT over MediaPipe: +{improvement:.1f}pp ({50.5:.1f}% -> {rate:.1f}%)")
    else:
        print(f"Below MediaPipe baseline ({50.5:.1f}%). May need threshold tuning.")

    # Save results
    out = {"frames": len(all_results), "detected": total_detected,
           "rate": rate, "results": all_results}
    with open("pipeline/r3d_output/hand_detections_v5.json", "w") as f:
        json.dump(out, f)
    print(f"Saved: pipeline/r3d_output/hand_detections_v5.json")
