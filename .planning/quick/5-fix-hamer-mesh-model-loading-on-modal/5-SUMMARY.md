# Summary: Fix HaMeR Mesh Model Loading on Modal

## What was done

Fixed 3 bugs preventing HaMeR mesh recovery from producing 3D wrist positions:

1. **np.str monkey-patch** — `np.str = np.str_` broke PyTorch tensor creation with `new(): invalid data type 'str'`. Fixed to `np.str = str` (Python builtin, matching numpy < 1.24 behavior).

2. **Model input format** — HaMeR's forward() expects `{'img': tensor}` dict, not a raw tensor. Changed `hamer_model(crop_tensor)` to `hamer_model({'img': crop_tensor})`.

3. **Input size mismatch** — ViT position embeddings expect 256×256 (16×16=256 tokens), but crop was 256×192 (16×12=192 tokens). Changed to square 256×256 crop.

Prior commits by executor fixed:
- Modal API (Function.lookup → Function.from_name)
- Real checkpoint download (hamer_demo_data.tar.gz from UT Austin)
- HaMeR + ViTPose install (--recursive clone)
- CACHE_DIR_HAMER env var
- pyrender headless stubs (PYOPENGL_PLATFORM=egl)
- chumpy numpy compat patches

## Results

| Metric | Before | After |
|--------|--------|-------|
| Model mode | gdino-only | **hamer** |
| Detection rate | 99% | **99.1%** |
| 3D wrist output | 0/951 frames | **942/951 frames** |
| wrist_3d_camera | None | **[x, y, z] per frame** |

## Known issue: Calibration mismatch

HaMeR's `wrist_3d_camera` is in camera-relative MANO coordinates, not R3D world coordinates. The current calibration maps these to X=[0.886, 1.041] — far outside G1 workspace [0.10, 0.65]. **Calibration needs to be adapted for HaMeR's coordinate frame** (separate task).

## Commits

- `0dbdd0b` Fix HaMeR model loading: real checkpoints, ViTPose, CACHE_DIR_HAMER
- `133058d`–`7761b58` (4 commits) Fix dependency chain: pyrender, renderer stubs, chumpy numpy compat
- `6c3f0e4` Fix HaMeR inference: np.str dtype, dict input format, 256x256 crop
