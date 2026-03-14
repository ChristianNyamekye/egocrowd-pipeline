# Stack Research: Flexa Pipeline v0.3

*Researched: 2026-03-13*

## New Dependencies Required

### 1. HaMeR (Hand Mesh Recovery) — via `geopavlakos/hamer`

- **What:** Transformer-based 3D hand mesh reconstruction from single RGB images. Outputs MANO parameters (pose, shape, translation) and 21 3D joint positions per detected hand. Uses ViTPose for initial hand/body keypoint detection, then a ViT encoder for mesh regression.
- **Why:** Replaces the `egocrowd/hand_pose.py` stub (currently `raise NotImplementedError`). MediaPipe achieves only 52% detection on egocentric video; HaMeR is expected to exceed the 85% target on the same frames.
- **Version:** Latest from `main` branch (no PyPI package; install from source). The repo requires Python 3.10+, which is compatible with the project's Python 3.12.
- **Install (for Modal cloud deployment):**
  ```
  pip install git+https://github.com/geopavlakos/hamer.git
  pip install -v -e third-party/ViTPose  # bundled as git submodule
  ```
- **Key transitive dependencies (pulled by HaMeR's setup.py):**
  | Package | Version | Notes |
  |---------|---------|-------|
  | `torch` | >=2.0 | Already in `[gpu]` extras |
  | `torchvision` | (matches torch) | Already in `[gpu]` extras |
  | `smplx` | ==0.1.28 | MANO hand model layer. **Pinned version** — must not conflict |
  | `detectron2` | git+https://github.com/facebookresearch/detectron2 | Used for ViTPose backbone loading. Heavy dep. |
  | `timm` | latest | PyTorch Image Models (ViT backbones) |
  | `einops` | latest | Tensor reshaping |
  | `mmcv` | ==1.3.9 | **Pinned, old version** — only needed for ViTPose. Potential conflict risk. |
  | `chumpy` | git+https://github.com/mattloper/chumpy | MANO mesh utility |
  | `pyrender` | latest | 3D rendering (used in demo, may not be needed for inference-only) |
  | `pytorch-lightning` | latest | Training framework (used in demo scripts) |
  | `yacs` | latest | Config system |
  | `xtcocotools` | latest | Keypoint evaluation |
  | `gdown` | latest | Model weight download |
  | `pandas` | latest | Data loading |
- **MANO model files:** Must download `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de/ and place in `_DATA/data/mano/`. This is a **manual registration step** (academic license).
- **Integration notes:**
  - HaMeR runs on GPU only. The project already has Modal infrastructure (`tools/v5_hand_detect.py`) using `modal.App` + T4 GPU for GroundingDINO hand detection. HaMeR should be deployed as a second Modal function in the same or parallel app.
  - The existing `v5_hand_detect.py` uses `torch==2.4.0, torchvision==0.19.0` in its Modal image. HaMeR should use the same versions for consistency.
  - Output format: HaMeR produces per-frame MANO params + 3D joints. The `egocrowd/retarget.py` module already expects a JSON with per-frame `translation` fields — HaMeR's wrist joint (joint 0) translation maps directly to this.
  - The `mmcv==1.3.9` pin is the biggest risk — it conflicts with newer PyTorch versions. Consider using `hamer_helper` (see below) or the ViTPose-Pytorch standalone to avoid mmcv.

### 2. hamer_helper (optional wrapper)

- **What:** Lightweight wrapper around HaMeR by Brent Yi (`brentyi/hamer_helper`) that provides a clean `HamerHelper()` API for inference without needing the full training stack.
- **Why:** Avoids needing to clone the full HaMeR repo with submodules. Provides a modular inference API. Used by other egocentric hand projects (e.g., EgoAllo).
- **Version:** 0.0.0 (development package, install from git)
- **Install:** `pip install git+https://github.com/brentyi/hamer_helper.git`
- **Dependencies:** `tyro>=0.8.11`, `jaxtyping`, `imageio` (plus HaMeR itself as a peer dependency)
- **Integration notes:** Lighter touch than raw HaMeR repo. Still requires MANO model files. Good option if you want inference-only without training code.

### 3. ViTPose-Pytorch (standalone, no mmcv)

- **What:** A fork of ViTPose that removes the mmcv dependency (`gpastal24/ViTPose-Pytorch`). ViTPose is used by HaMeR for initial hand/body keypoint detection.
- **Why:** Eliminates the `mmcv==1.3.9` version pin which is the primary conflict risk in HaMeR's dependency tree. mmcv 1.3.9 is incompatible with PyTorch 2.4+ and Python 3.12.
- **Version:** Latest from GitHub
- **Install:** `pip install git+https://github.com/gpastal24/ViTPose-Pytorch.git`
- **Integration notes:** If using this, you would NOT install HaMeR's bundled `third-party/ViTPose` and instead patch the import path. This is the **recommended approach** for the Modal image to avoid mmcv conflicts.

### 4. smplx

- **What:** SMPL/MANO/SMPL-X body model layer for PyTorch. Provides differentiable forward kinematics for hand mesh vertices from pose parameters.
- **Why:** Direct dependency of HaMeR. Needed to decode MANO parameters into 3D hand joint positions and mesh vertices.
- **Version:** ==0.1.28 (pinned by HaMeR)
- **Install:** `pip install smplx==0.1.28`
- **Integration notes:** No conflicts with existing stack. Pure Python + PyTorch.

### 5. Modal (cloud GPU platform)

- **What:** Serverless GPU compute platform. Runs arbitrary Python functions on cloud GPUs with per-second billing.
- **Why:** HaMeR requires GPU (CUDA) for inference. The project already uses Modal for GroundingDINO hand detection. HaMeR inference should also run on Modal since local Mac has no NVIDIA GPU.
- **Version:** Latest (actively developed, releases weekly; current ~0.73.x as of March 2026)
- **Install:** `pip install modal` (already used in project but not in `requirements-dev.txt`)
- **Integration notes:**
  - Already proven in project (`tools/v5_hand_detect.py` uses `modal.App`, `modal.Image`, `@app.function(gpu="T4")`).
  - For HaMeR, use A10G (24GB VRAM) instead of T4 (16GB) — HaMeR's ViT-H backbone + MANO decoder needs ~8-10GB; A10G provides comfortable headroom and faster inference.
  - Modal image should pre-bake HaMeR weights and MANO model to avoid re-downloading on each cold start.
  - Pattern: define `hamer_image = modal.Image.debian_slim(python_version="3.10").pip_install(...)` with all HaMeR deps, then `@app.function(gpu="A10G", image=hamer_image)`.

## Existing Stack Sufficient For

### Trajectory Smoothing — already have `scipy>=1.10` (installed: 1.17.1)

- **Gap interpolation:** `scipy.interpolate.CubicSpline` or `scipy.interpolate.PchipInterpolator` — fill NaN/missing frames before smoothing. Current code uses naive moving-average (`smooth_vecs` in `tools/spatial_trajectory_v*.py`). CubicSpline gives C2-continuous interpolation; PCHIP prevents overshoot at gaps.
  - Note: `scipy.interpolate.interp1d` is **legacy API** — use `CubicSpline` or `PchipInterpolator` instead.
- **Temporal filtering:** `scipy.signal.savgol_filter` — Savitzky-Golay filter preserves trajectory shape better than moving average. Recommended params for 10 FPS trajectory: `window_length=11, polyorder=3`. Already available, no new install needed.
- **No new dependencies required.** The existing `scipy>=1.10` provides everything needed. The current `smooth_vecs` functions in `tools/spatial_trajectory_v*.py` should be upgraded from moving average to Savgol + gap interpolation.

### Video Trimming — already have `opencv-python` (installed: 4.11) + `numpy`

- **Action window detection:** Analyze the `grasping` signal from calibrated trajectory data to find first/last grasp frames, then pad by N frames. The 951-frame R3D starting at frame 577 can be trimmed by:
  1. Finding first non-zero grasping frame minus a pre-roll margin
  2. Finding last non-zero grasping frame plus a post-roll margin
  3. Slicing the trajectory and re-rendering only that window
- **Video output:** Already using `ffmpeg` via subprocess for MP4 encoding in all replay/render scripts. Frame subsampling is already implemented.
- **No new dependencies required.** This is a pipeline logic change, not a library addition.

### MuJoCo Grasp Visual Quality — already have `mujoco>=3.0` (installed: 3.6.0)

- **Finger pre-shaping:** The G1 model (`g1_with_hands.xml`) has thumb, index, and middle finger joints with position actuators. Current code (`mujoco_g1_v10.py`) uses `FINGER_CLOSED = [0.4, -0.5, -0.6, 0.8, 0.9, 0.8, 0.9]` — a single closed pose. Pre-shaping means ramping finger closure progressively as the hand approaches the block (already partially done with `finger_ctrl += (target - finger_ctrl) * 0.25` blend).
- **Reduced interpenetration:** Use MuJoCo `contact/exclude` pairs or `contype`/`conaffinity` bitmasks to prevent finger geoms from passing through block geoms during kinematic attachment. Current code already uses `contype=1, conaffinity=1` for blocks and table. Options:
  1. **Contact exclude pairs:** `<contact><exclude body1="right_hand_thumb_2_link" body2="block_0"/></contact>` — prevents collision detection between specific body pairs. Useful during kinematic attachment phase to avoid jitter.
  2. **Torsional friction:** Set `condim="6"` on finger geoms + increase `friction` third component for torsional stability (already available in MuJoCo 3.6).
  3. **Contype/conaffinity toggling:** Dynamically disable block-finger contacts during kinematic phase (code already does this for block settling — same pattern applies to finger geoms).
- **No new dependencies required.** All capabilities are in MuJoCo 3.6.0.

## NOT Recommended

### pytorch3d
- HaMeR's setup.py does NOT list pytorch3d as a dependency. Some forks/tutorials mention it, but the core inference path does not need it. pytorch3d is extremely difficult to install (requires matching CUDA/PyTorch versions, often fails on pip). **Skip it.**

### Dyn-HaMR (ZhengdiYu/Dyn-HaMR)
- CVPR 2025 paper extending HaMeR for dynamic cameras. Adds temporal consistency across frames. However, it's a research prototype with heavier dependencies (adds optical flow, camera motion estimation). **Too complex for v0.3 — revisit for v0.4 if temporal jitter is still an issue after Savgol smoothing.**

### WildHands / DeltaDorsal / V-HPOT
- Recent egocentric hand pose methods (ECCV 2024, January 2026). WildHands claims better than HaMeR on some metrics and is 10x smaller. However, none have mature, easy-to-integrate codebases. **Monitor for v0.4; HaMeR is the established baseline with community tooling (hamer_helper).**

### 4DHands
- Multi-view hand reconstruction. Not applicable to single egocentric camera setup. **Out of scope.**

### Full ViTPose (ViTAE-Transformer/ViTPose)
- Requires mmcv/mmpose ecosystem. The mmcv==1.3.9 pin conflicts with Python 3.12 and PyTorch 2.4. **Use ViTPose-Pytorch standalone fork instead** if integrating HaMeR directly. Or use `hamer_helper` which abstracts this away.

### mediapipe upgrade
- MediaPipe is already installed (0.10.32) and serves as the CPU fallback. No upgrade will fix the 52% detection rate on egocentric footage — it's an architectural limitation (MediaPipe targets front-facing webcam, not egocentric). **Keep as fallback, don't invest in improving it.**

## Integration Risks

1. **mmcv==1.3.9 conflicts with Python 3.12 / PyTorch 2.4:** HaMeR pins `mmcv==1.3.9` which does not support Python 3.12 or PyTorch 2.4+. **Mitigation:** Use ViTPose-Pytorch fork (no mmcv) or run HaMeR in a Python 3.10 Modal container (the Modal image in `v5_hand_detect.py` already uses `python_version="3.10"`). This isolates the conflict to cloud-side only.

2. **MANO license / manual download:** The MANO model (`MANO_RIGHT.pkl`) requires academic registration at mano.is.tue.mpg.de. Cannot be auto-downloaded. **Mitigation:** Download once, bake into Modal image volume or include as a Modal secret mount.

3. **detectron2 install complexity:** detectron2 must be installed from source (no PyPI wheel for recent PyTorch). Historically fragile. **Mitigation:** Pin specific commit in Modal image; test the image build before deploying inference.

4. **Model weight download on cold start:** HaMeR downloads ~400MB of ViT-H weights on first run. Modal cold starts will be slow (~30s). **Mitigation:** Bake weights into the Modal image using `modal.Image.run_commands("python -c 'from hamer...'")` during image build.

5. **Dual GPU functions on Modal:** Running both GroundingDINO (v5_hand_detect) and HaMeR as separate Modal functions means two GPU containers. May hit Modal concurrency limits on free tier. **Mitigation:** Combine into a single Modal function that runs detection then HaMeR sequentially, sharing the GPU allocation. Or run them as a pipeline with `.map()`.

6. **scipy smoothing on gap-heavy trajectories:** With 48% missing frames, CubicSpline interpolation over long gaps (>10 consecutive frames) may produce unrealistic oscillations. **Mitigation:** Use PchipInterpolator (monotonic, no overshoot) for gaps >5 frames, CubicSpline for shorter gaps. Cap maximum interpolation span — if gap >20 frames, hold last known position rather than interpolate.

## Summary: What to Add to pyproject.toml

```toml
[project.optional-dependencies]
gpu = [
    "torch>=2.0",
    "torchvision",
    "groundingdino-py",
    "transformers",
    "smplx==0.1.28",       # NEW: MANO hand model for HaMeR
    "timm",                 # NEW: ViT backbones for HaMeR
    "einops",               # NEW: tensor ops for HaMeR
]
cloud = [
    "modal",                # NEW: explicit dependency for Modal deployment
]
```

HaMeR itself and ViTPose-Pytorch should be installed **only in the Modal container image** (not in local pyproject.toml) because they require GPU + have complex transitive deps. The local pipeline calls Modal functions remotely and only needs the JSON output.
