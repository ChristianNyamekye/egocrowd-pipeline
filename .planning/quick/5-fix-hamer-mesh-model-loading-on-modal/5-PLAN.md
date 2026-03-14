# QT-005: Fix HaMeR Mesh Model Loading on Modal

## Problem

`hamer_modal.py` builds a Modal image that creates empty checkpoint directories but never downloads actual model weights. The HaMeR pip install failure is swallowed by `|| echo 'WARN...'`, so the image builds "successfully" but `_try_load_hamer()` always fails at runtime, forcing gdino-only mode. Result: no `wrist_3d_camera` output.

## Root Causes

1. **HaMeR install failure silenced** -- `|| echo` swallows pip errors; image builds without HaMeR installed
2. **No checkpoint download** -- lines 51-57 create empty dirs but never download `hamer_demo_data.tar.gz`
3. **ViTPose not installed** -- HaMeR requires `third-party/ViTPose` as a separate pip install; not present in image
4. **CACHE_DIR_HAMER mismatch** -- HaMeR defaults to `./_DATA` (relative), but MANO files sit at `/root/_DATA`

## Solution

Rewrite the Modal image build in `hamer_modal.py` to:

1. **Clone HaMeR repo with submodules** (gets ViTPose third-party code)
2. **Install HaMeR + ViTPose** without `|| echo` fallback -- let build fail if broken
3. **Download actual checkpoints** via `wget` from `https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz` and extract to `/root/_DATA/`
4. **Set `CACHE_DIR_HAMER`** env var or workdir so HaMeR finds checkpoints at `/root/_DATA`
5. **Verify** `_try_load_hamer()` produces `wrist_3d_camera` output

---

## Tasks

### Task 1: Rewrite Modal image build with real checkpoints (hamer_modal.py)

**File:** `processing/hamer_modal.py`

**Changes to the `hamer_image` definition:**

1. **Replace the HaMeR install block** (lines 26-29): Instead of `pip install --no-deps git+... || echo`, clone the repo with `--recursive` to get ViTPose, then `pip install -e .` and `pip install -v -e third-party/ViTPose`. No `|| echo` -- let it fail loudly.

2. **Replace empty checkpoint block** (lines 50-57): Download actual weights:
   ```
   wget -q https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
   tar --warning=no-unknown-keyword --exclude=".*" -xf hamer_demo_data.tar.gz
   mv _DATA/* /root/_DATA/ || true
   rm hamer_demo_data.tar.gz
   ```
   This gives us:
   - `/root/_DATA/hamer_ckpts/checkpoints/hamer.ckpt` + `model_config.yaml`
   - `/root/_DATA/vitpose_ckpts/` (ViTPose weights)
   - Any other required data files

3. **Set environment variable** so HaMeR finds checkpoints at `/root/_DATA`:
   ```python
   .env({"CACHE_DIR_HAMER": "/root/_DATA"})
   ```
   OR override in `_try_load_hamer()` by setting the path before import.

4. **Set workdir to `/root`** (already done at line 58) so relative `_DATA` resolves to `/root/_DATA`.

5. **Update `_try_load_hamer()`** to explicitly set `CACHE_DIR_HAMER` before importing HaMeR:
   ```python
   import hamer.configs
   hamer.configs.CACHE_DIR_HAMER = "/root/_DATA"
   ```

6. **Remove the `|| echo` pattern** entirely -- if HaMeR can't install, the build should fail so we know immediately.

### Task 2: Deploy and verify wrist_3d_camera output

1. Run `modal deploy processing/hamer_modal.py` -- confirm image builds without errors
2. Run `modal run processing/hamer_modal.py` with stack2 frames
3. Verify output shows:
   - `Model: hamer` (not `gdino-only`)
   - `3D wrist output: >0` frames (wrist_3d_camera populated)
4. If successful, update STATE.md

### Task 3: Handle ViTPose/HaMeR install edge cases

If Task 1 build fails due to dependency conflicts:
- Pin specific versions of conflicting packages (mmcv, detectron2, etc.)
- Use `--no-deps` for HaMeR but ensure `hamer` package IS installed (check with `python -c "import hamer"`)
- Install ViTPose separately with `--no-deps` if needed
- The key insight: checkpoints must exist even if we use `--no-deps`; the pip install just needs to succeed enough for `import hamer` to work

**Success criteria:** `run_hamer_inference` returns `"model": "hamer"` and frames have non-null `wrist_3d_camera` values.
