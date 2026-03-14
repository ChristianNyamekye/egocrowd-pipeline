# Phase 3a: HaMeR Integration -- Research

*Researched: 2026-03-14*
*Scope: TRK-04 (HaMeR >85% detection on Modal GPU), TRK-05 (HaMeR 3D wrist output used directly)*

---

## 1. Problem Statement

MediaPipe achieves only 52% hand detection on egocentric video (stack2.r3d). This forces `reconstruct_wrist_3d.py` to interpolate across 48% missing frames, producing noisy trajectories even after Savitzky-Golay smoothing and velocity clamping. The target is >85% detection, which would reduce interpolation burden to <15% of frames.

Additionally, MediaPipe provides only 2D wrist pixel coordinates, requiring depth-map unprojection via `pixel_to_world()` in `reconstruct_wrist_3d.py`. This introduces noise from depth sensor inaccuracies. HaMeR outputs 3D wrist position in camera frame directly from its MANO decoder, enabling TRK-05: bypass depth-map unprojection entirely.

---

## 2. HaMeR Architecture Overview

HaMeR (Hand Mesh Recovery) is a transformer-based model from Pavlakos et al. (CVPR 2024) that reconstructs 3D hand meshes from single RGB images.

**Pipeline:**
1. **Hand detection:** ViTDet detects hand bounding boxes in the image (HaMeR's default detector). Alternatively, GroundingDINO or any other detector can provide hand crops.
2. **Crop + resize:** Detected hands are cropped and resized to 256x192 input.
3. **ViT-H encoder:** A Vision Transformer (ViT-Huge, 1280-dim) encodes the hand crop into 16x12 tokens.
4. **Transformer decoder:** A 6-layer decoder with a single learnable query cross-attends to ViT tokens and regresses MANO parameters.
5. **MANO layer:** `smplx==0.1.28` decodes MANO params into 21 3D joint positions and 778 mesh vertices.

**Outputs per detected hand:**
- `pred_mano_params`: pose (48), shape (10), translation (3) -- MANO parameters
- `pred_keypoints_3d`: (21, 3) -- 3D joint positions in camera frame
- `pred_vertices`: (778, 3) -- mesh vertices in camera frame
- `pred_cam_t_full`: (3,) -- camera-frame translation of wrist origin
- Wrist joint (joint index 0) gives the 3D wrist position directly

**Key advantage for Flexa:** Joint index 0 (`pred_keypoints_3d[0]`) is the wrist position in camera coordinates. Combined with the R3D camera pose (available from ARKit metadata), this gives world-frame wrist position without depth-map lookup.

---

## 3. Installation Options Analysis

### Option A: Raw HaMeR repo (geopavlakos/hamer)

**Install:**
```bash
pip install git+https://github.com/geopavlakos/hamer.git
cd third-party && pip install -v -e ViTPose  # submodule
```

**Pros:** Official implementation, full control, `demo.py` provides batch inference with DataLoader.

**Cons:**
- ViTPose submodule requires `mmcv==1.3.9`, which conflicts with Python 3.12 and PyTorch 2.4+. This is a **blocking issue** for running in a Python 3.12 environment.
- `detectron2` required for ViTDet backbone loading -- fragile pip install from source.
- Heavy: pulls training code, demo scripts, pyrender.
- Model weights (~400MB ViT-H) downloaded on first run.
- Requires MANO model files (`MANO_RIGHT.pkl`) from mano.is.tue.mpg.de (academic license, manual download).

**Verdict:** Viable only in a Python 3.10 Modal container (isolating mmcv conflict). The existing `v5_hand_detect.py` already uses `python_version="3.10"` for its Modal image.

### Option B: hamer_helper (brentyi/hamer_helper)

**Install:**
```bash
pip install git+https://github.com/brentyi/hamer_helper.git
```

**What it provides:**
- Clean `HamerHelper()` class wrapping HaMeR inference
- `inference.py` script with `--input-dir` / `--output-dir` for batch processing
- Used by EgoAllo project (also egocentric hand estimation)

**Pros:** Lighter wrapper, modular API, avoids training code. Peer-depends on HaMeR underneath.

**Cons:** Still requires HaMeR + mmcv + detectron2 as peer deps. Does not solve the dependency conflict -- just wraps it. Version 0.0.0 (development stage).

**Verdict:** Slightly cleaner API but same dependency chain. Worth using IF we need the API, but for a Modal function we can call `demo.py` patterns directly.

### Option C: Combined approach (RECOMMENDED)

Use the raw HaMeR repo inside a Python 3.10 Modal container with:
- ViTPose-Pytorch (`gpastal24/ViTPose-Pytorch`) instead of HaMeR's bundled ViTPose to eliminate mmcv
- GroundingDINO (already working in `v5_hand_detect.py`) instead of ViTDet for hand detection
- `smplx==0.1.28` for MANO decoding

**Why this combination:**
1. ViTPose-Pytorch removes the mmcv dependency entirely -- the biggest conflict risk
2. GroundingDINO is already proven in the project (v5_hand_detect.py achieves good detection rates)
3. Python 3.10 Modal container isolates all GPU deps from local Python 3.12
4. Single Modal function avoids double cold-start penalty

**Dependency chain for Modal container image:**
```python
modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0",     # PyTorch (match v5)
        "transformers>=4.36.0",                       # GroundingDINO
        "numpy<2", "Pillow", "opencv-python-headless",
        "smplx==0.1.28",                              # MANO layer
        "timm", "einops",                              # ViT backbones
    )
    .pip_install("git+https://github.com/geopavlakos/hamer.git")
    .pip_install("git+https://github.com/gpastal24/ViTPose-Pytorch.git")
```

**Risk:** The raw HaMeR pip install may still try to pull mmcv transitively. If so, we can install HaMeR with `--no-deps` and manually install only the needed deps, or patch HaMeR's setup.py in the Modal image build.

### Option D: Simpler alternative -- GroundingDINO detection + MediaPipe pose on crops

If the HaMeR dependency chain proves unworkable, a simpler approach:
1. Use GroundingDINO (already working) to detect hand bounding boxes with higher recall
2. Crop detected regions and run MediaPipe on the crops (better accuracy on cropped hands vs. full egocentric frame)
3. This may achieve >85% detection by improving the detection stage without changing the pose model

**Pros:** No new model dependencies, uses proven infrastructure.
**Cons:** Still no 3D wrist output (TRK-05 not achievable), detection rate improvement is speculative.

**Verdict:** Fallback option if HaMeR integration hits blocking dependency issues.

---

## 4. Modal Deployment Analysis

### Existing Pattern (v5_hand_detect.py)

The project already has a working Modal deployment pattern:

```python
app = modal.App("egocrowd-v5-hands")
gdino_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch==2.4.0", "torchvision==0.19.0",
                 "transformers>=4.36.0", "numpy<2", "Pillow")
)

@app.function(gpu="T4", image=gdino_image, timeout=600)
def detect_hands(frame_bytes_list: list, threshold: float = 0.2):
    # Load model, process frames, return results
    ...
```

**Key patterns to reuse:**
- Frame data passed as `list[bytes]` (serialized JPEGs)
- Results returned as plain dicts (JSON-serializable)
- Model loaded inside the function (cold start per invocation)
- Batch processing with configurable batch size

### HaMeR Modal Function Design

**GPU tier:** A10G (24GB VRAM) instead of T4 (16GB).
- HaMeR ViT-H backbone: ~4GB
- GroundingDINO: ~2GB
- MANO decoder: <1GB
- Total: ~7-8GB + batch overhead
- A10G provides headroom and faster inference

**Architecture: Combined detection + HaMeR in one function.**

Research decision D3 recommends combined over separate functions:
- Avoids double cold-start penalty (~30s each)
- Avoids Modal free-tier concurrency limits
- Detection output (bounding boxes) feeds directly to HaMeR without serialization

**Estimated timing:**
- Cold start: ~30-45s (model download + loading)
- Inference per frame: ~100-200ms (detection + HaMeR)
- Stack2 (235 frames): ~25-50s inference after warm-up
- Total wall clock: ~60-90s per recording

**Model weight baking:** To reduce cold start, bake weights into the Modal image:
```python
.run_commands(
    "python -c 'from transformers import AutoProcessor; "
    "AutoProcessor.from_pretrained(\"IDEA-Research/grounding-dino-tiny\")'"
)
```

HaMeR weights can be downloaded during image build with `gdown` or cached in a Modal Volume.

### MANO Model Files

HaMeR requires `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de/ (academic license). Options:
1. **Modal Secret mount:** Store MANO files as a Modal Secret, mount at runtime
2. **Modal Volume:** Persistent storage, uploaded once via `modal volume put`
3. **Bake into image:** Download during image build (requires the file to be accessible)

**Recommended:** Modal Volume. Upload once, mount at `/data/mano/` in the container.

---

## 5. Integration Points with Existing Code

### 5.1 egocrowd/hand_pose.py (stub to replace)

Current state: `extract_hand_poses()` raises `NotImplementedError` at line 44.

**New behavior:** Call the Modal function remotely:
```python
def extract_hand_poses(rgb_dir, hand_boxes=None, device=None):
    import modal
    # Look up the deployed Modal function
    hamer_fn = modal.Function.lookup("flexa-hamer", "run_hamer_inference")
    # Load frames as bytes
    frames_dir = Path(rgb_dir)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    frame_bytes = [p.read_bytes() for p in frame_paths]
    # Call remotely in batches
    results = hamer_fn.remote(frame_bytes)
    return results
```

### 5.2 run_pipeline.py (_run_hand_tracking, lines 191-221)

The try/except pattern at lines 196-205 already handles HaMeR unavailability gracefully:
```python
try:
    from egocrowd.hand_pose import extract_hand_poses
    result = extract_hand_poses(frames_dir)
    if result:
        trajectory = _hamer_to_trajectory(result, video_path)
except (ImportError, NotImplementedError, Exception) as e:
    print(f"  HaMeR unavailable ({type(e).__name__}), using MediaPipe")
```

**Changes needed:**
1. `_hamer_to_trajectory()` (lines 224-266) must extract HaMeR-specific fields:
   - `wrist_3d_camera`: Camera-frame 3D wrist position (joint 0)
   - `landmarks_3d`: Full 21-joint positions (for future finger retargeting)
   - `grasping`: Estimated from finger curl / thumb-index distance
2. Add `wrist_3d_camera` to the trajectory dict so `reconstruct_wrist_3d.py` can use it

### 5.3 reconstruct_wrist_3d.py (downstream consumer)

Currently at line 201: `point = pixel_to_world(wp[0], wp[1], depth, meta, r3d_idx)` -- unprojecting 2D wrist pixel using depth map.

**TRK-05 change:** When `wrist_3d_camera` is available in the retarget data, skip depth lookup:
```python
if t.get("wrist_3d_camera"):
    # HaMeR provides camera-frame 3D directly
    point_cam = np.array(t["wrist_3d_camera"])
    # Transform to world using R3D camera pose
    point = R @ point_cam + np.array([tx, ty, tz])
else:
    # MediaPipe fallback: depth-map unprojection
    point = pixel_to_world(wp[0], wp[1], depth, meta, r3d_idx)
```

### 5.4 _build_retarget_data (run_pipeline.py lines 315-338)

Must pass through `wrist_3d_camera` from HaMeR trajectory to retarget data:
```python
if hand.get("wrist_3d_camera"):
    ts["wrist_3d_camera"] = hand["wrist_3d_camera"]
```

---

## 6. Grasping Signal from HaMeR

MediaPipe provides a binary `grasping` signal estimated from hand landmark geometry. HaMeR provides richer data for grasping estimation:

**Approach 1: Finger curl from MANO pose parameters**
HaMeR outputs 48-dim pose vector (16 joints x 3 axis-angle). Finger joints have known indices. When finger curl exceeds a threshold, classify as grasping.

**Approach 2: Thumb-index distance from 3D joints**
Compute distance between thumb tip (joint 4) and index tip (joint 8). When distance < threshold, classify as grasping. This is the simplest and most robust approach.

**Approach 3: Use MediaPipe's grasping logic on HaMeR joints**
The existing `hand_tracker_v2.py` computes grasping from MediaPipe landmarks. The same geometry (finger tip distances, palm closure) can be applied to HaMeR's 21 joints, which have the same topology.

**Recommended:** Approach 2 (thumb-index distance). Simple, interpretable, works with HaMeR's 3D joints.

```python
def estimate_grasping(joints_3d):
    """Estimate binary grasping from HaMeR 3D joints.

    joints_3d: (21, 3) array, MANO joint order
    Joint 4 = thumb tip, Joint 8 = index tip
    """
    thumb_tip = joints_3d[4]
    index_tip = joints_3d[8]
    dist = np.linalg.norm(thumb_tip - index_tip)
    return dist < 0.04  # 4cm threshold (in meters, camera frame)
```

---

## 7. Coordinate Frame Considerations

### HaMeR output frame
HaMeR outputs 3D positions in **camera frame** (X-right, Y-down, Z-forward for typical OpenCV convention).

### R3D camera frame
iPhone R3D uses ARKit convention (X-right, Y-up, Z-toward-viewer). The camera pose in `meta["poses"]` provides `[tx, ty, tz, qw, qx, qy, qz]` for camera-to-world transform.

### Critical warning (from research SUMMARY.md, pitfall P5)
Do NOT blend MediaPipe and HaMeR frame-by-frame. They produce systematically different wrist positions. Use one model per recording. The pipeline already does this: if HaMeR succeeds, use HaMeR for all frames; if it fails, fall back to MediaPipe for all frames.

### Camera-to-world transform
```python
# From R3D metadata pose
R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
point_world = R @ point_camera + translation
```

This is already implemented in `reconstruct_wrist_3d.py` (lines 58-67). The same transform applies to HaMeR's camera-frame output. The only question is whether HaMeR's camera convention matches R3D's camera convention -- this must be validated empirically on the first run.

---

## 8. Risk Assessment

### Risk 1: HaMeR dependency chain fails in Modal container (HIGH)

mmcv, detectron2, and HaMeR's bundled ViTPose create a fragile install. Even with ViTPose-Pytorch replacing the mmcv dep, HaMeR's setup.py may pull mmcv transitively.

**Mitigation:**
- Install HaMeR with `--no-deps`, then manually install only needed packages
- Test the Modal image build in isolation before integrating with the pipeline
- Have Option D (GroundingDINO + MediaPipe crops) ready as fallback

### Risk 2: MANO model license blocks automated deployment (MEDIUM)

MANO_RIGHT.pkl requires manual registration at mano.is.tue.mpg.de.

**Mitigation:**
- Download once manually, upload to Modal Volume
- The academic license permits research use

### Risk 3: HaMeR camera frame does not match R3D camera frame (MEDIUM)

If HaMeR uses a different camera convention than R3D, the camera-to-world transform will produce wrong world coordinates.

**Mitigation:**
- Validate on first run: compare HaMeR world-frame wrist positions with MediaPipe + depth unprojection positions for the same frames
- If offset is systematic, add a rotation correction

### Risk 4: Detection rate does not reach 85% on egocentric video (LOW-MEDIUM)

HaMeR was trained on mixed data including egocentric (VISOR, Ego4D). The EgoExo4D challenge report shows strong egocentric performance. However, our specific R3D recording quality and camera angle may differ.

**Mitigation:**
- Run on stack2 and measure detection rate before full integration
- If detection is <85% with ViTDet, try GroundingDINO as the detector (already proven to work well on these frames)
- GroundingDINO + HaMeR is the recommended approach: GroundingDINO for detection (high recall on hands), HaMeR for mesh recovery (high quality 3D pose)

### Risk 5: Cold start latency makes iteration slow (LOW)

Modal cold start for a container with PyTorch + HaMeR + GroundingDINO: ~30-45s.

**Mitigation:**
- Bake model weights into the Modal image during build
- Use `modal.Volume` to cache weights persistently
- After first invocation, subsequent calls hit warm containers (sub-second)

---

## 9. HaMeR Inference Code Pattern

Based on the official `demo.py` and research:

```python
import torch
from PIL import Image

# 1. Hand detection (GroundingDINO)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

detector_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).cuda()

def detect_hands(image):
    inputs = detector_processor(images=image, text="hand", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = detector_model(**inputs)
    results = detector_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.2,
        target_sizes=[image.size[::-1]]
    )[0]
    return results["boxes"]  # (N, 4) xyxy format

# 2. HaMeR inference on hand crops
# HaMeR model loading (from demo.py pattern)
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer

model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
model = model.cuda().eval()

def run_hamer_on_crop(crop_image, bbox, is_right=True):
    # Preprocess: resize to 256x192, normalize
    # Forward pass through ViT encoder + transformer decoder
    # Returns MANO params, 3D joints, mesh vertices
    with torch.no_grad():
        out = model(crop_tensor)
    return {
        "joints_3d": out["pred_keypoints_3d"][0].cpu().numpy(),  # (21, 3)
        "vertices": out["pred_vertices"][0].cpu().numpy(),        # (778, 3)
        "cam_t": out["pred_cam_t_full"][0].cpu().numpy(),         # (3,)
    }
```

The actual integration will need to handle:
- Batch processing across frames
- Multiple hand detections per frame (pick the best/right hand)
- Frame-level fallback when no hand is detected
- Serialization of results for return from Modal function

---

## 10. Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Installation approach | Raw HaMeR repo in Modal container (Option C) | Full control, avoid hamer_helper's 0.0.0 instability |
| ViTPose replacement | ViTPose-Pytorch (gpastal24) | Eliminates mmcv conflict |
| Hand detector | GroundingDINO (reuse from v5) | Already proven, high recall on egocentric hands |
| Modal GPU tier | A10G (24GB) | HaMeR ViT-H needs ~8GB, A10G provides headroom |
| Modal architecture | Single combined function | Avoids double cold-start, shares GPU allocation |
| MANO files | Modal Volume, uploaded once | Academic license permits research use |
| Grasping estimation | Thumb-index 3D distance | Simple, robust, uses HaMeR's 3D joints |
| Fallback strategy | MediaPipe (existing) | Already works at 52%, no code changes needed |
| Coordinate validation | Compare HaMeR vs MediaPipe world positions on first run | Catches frame convention mismatches early |
| Fallback if deps fail | GroundingDINO detection + MediaPipe on crops (Option D) | No new model deps, may still hit >85% detection |

---

*Research complete. Ready for planning.*
