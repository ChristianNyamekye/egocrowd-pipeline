# Phase 3 Plan 01 Summary: HaMeR Hand Pose Estimation on Modal GPU

**Status:** Code complete. Deployment testing deferred (requires Modal credentials + MANO model files).
**Date:** 2026-03-14
**Requirements:** TRK-04 (>85% detection), TRK-05 (direct 3D wrist output)

---

## What Was Done

### Task 1: processing/hamer_modal.py (NEW)
Created Modal function combining GroundingDINO hand detection + HaMeR mesh recovery in a single A10G GPU container.

- Modal app `"flexa-hamer"` with `debian_slim(python_version="3.10")` image
- GroundingDINO for hand detection (text prompt "hand", threshold 0.2)
- HaMeR installed with `--no-deps` to avoid mmcv conflicts
- Graceful fallback: if HaMeR loading fails, runs in `"gdino-only"` detection mode
- Returns per-frame: `wrist_pixel`, `wrist_3d_camera` (MANO joint 0), `joints_3d` (21 joints), `grasping` (thumb-index distance < 4cm), `confidence`
- GroundingDINO weights pre-baked into image for faster cold start
- Local entrypoint for testing: `modal run processing/hamer_modal.py`

### Task 2: egocrowd/hand_pose.py (MODIFIED)
Replaced `NotImplementedError` stub with actual Modal remote call.

- Loads frame bytes from `rgb_dir/*.jpg`
- Calls `modal.Function.lookup("flexa-hamer", "run_hamer_inference")` remotely
- Import-time safe: `modal` imported inside function body
- Backward compatible: same function signature, caller catches exceptions for MediaPipe fallback

### Task 3: run_pipeline.py (MODIFIED)
Updated trajectory conversion and retarget data to pass through HaMeR 3D wrist data.

- `_hamer_to_trajectory()`: passes through `wrist_3d_camera` and `joints_3d` fields, uses `confidence` from HaMeR result
- `_build_retarget_data()`: passes through `wrist_3d_camera` to retarget JSON for `reconstruct_wrist_3d.py`
- Added `--hamer` / `--no-hamer` CLI flags (default: `--hamer`)
- `_run_hand_tracking()` respects `use_hamer` parameter, skips HaMeR try block when `--no-hamer`
- `run_r3d_pipeline()` accepts `use_hamer` parameter

### Task 4: reconstruct_wrist_3d.py (MODIFIED)
Added code path to use HaMeR's camera-frame 3D wrist directly, skipping depth-map unprojection.

- New `cam_to_world(point_cam, pose)` helper: quaternion-to-rotation + translate (extracted from duplicated code in `pixel_to_world`)
- `process_session()` detects `wrist_3d_camera` in retarget data and prints mode (HaMeR 3D vs depth-map)
- Frame loop: when `wrist_3d_camera` present, transforms camera-frame point to world via R3D pose and `continue`s (skips depth loading entirely)
- Fallback: frames without `wrist_3d_camera` use existing depth-map unprojection path

### Task 5: Deployment & Validation (DEFERRED)
Modal deployment and stack2 validation require:
1. Modal CLI authentication (`modal token set`)
2. MANO model files (`MANO_RIGHT.pkl` from mano.is.tue.mpg.de, academic license)
3. Stack2 R3D frames extracted to `r3d_output/stack2/frames/`

These are runtime requirements that cannot be validated in this environment.

---

## Key Design Decisions

1. **Combined function (not separate detect + mesh):** Avoids double cold-start penalty (~30s each). Single A10G handles both models (~8GB total VRAM).
2. **HaMeR `--no-deps` install:** Prevents mmcv/detectron2 transitive dependency conflicts in Python 3.10 container.
3. **Graceful degradation chain:** HaMeR mesh recovery -> GroundingDINO detection-only -> MediaPipe (CPU). Each level provides less data but still works.
4. **`cam_to_world` helper:** Extracted shared quaternion math between `pixel_to_world` and the new HaMeR path. Same transform, different inputs.
5. **No frame-level model mixing:** When HaMeR succeeds, all frames use HaMeR. When it fails, all frames use MediaPipe. Avoids coordinate frame inconsistencies (research pitfall P5).

---

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `processing/hamer_modal.py` | NEW | ~250 |
| `egocrowd/hand_pose.py` | Replaced stub | 50 -> 50 |
| `run_pipeline.py` | HaMeR passthrough + CLI flags | +25 |
| `reconstruct_wrist_3d.py` | HaMeR 3D path + cam_to_world | +35 |

---

## Verification Status

- [x] `processing/hamer_modal.py` exists and defines `run_hamer_inference` Modal function
- [x] `egocrowd/hand_pose.py` no longer raises `NotImplementedError`
- [x] `egocrowd/hand_pose.py` calls Modal function via `modal.Function.lookup`
- [x] `run_pipeline.py` `_hamer_to_trajectory()` passes through `wrist_3d_camera`
- [x] `run_pipeline.py` `_build_retarget_data()` passes through `wrist_3d_camera`
- [x] `reconstruct_wrist_3d.py` has `wrist_3d_camera` code path that skips depth lookup
- [x] `reconstruct_wrist_3d.py` has `cam_to_world` helper function
- [x] `--hamer` / `--no-hamer` CLI flags work
- [x] All files pass syntax validation
- [ ] `modal deploy processing/hamer_modal.py` succeeds (DEFERRED: needs Modal auth)
- [ ] Detection rate on stack2 > 85% (DEFERRED: needs deployment)
- [ ] End-to-end pipeline with HaMeR data (DEFERRED: needs deployment)

---

## Next Steps

1. **Deploy:** `modal deploy processing/hamer_modal.py` (requires Modal token)
2. **Upload MANO:** Download `MANO_RIGHT.pkl`, upload to Modal Volume
3. **Test:** `modal run processing/hamer_modal.py` on stack2 frames
4. **Validate TRK-04:** Detection rate > 85%
5. **Run full pipeline:** `python run_pipeline.py --r3d raw_captures/stack2/stack2.r3d --robot g1 --task stack --objects '[[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]' --trim`
6. **Compare:** Run with `--no-hamer` to compare MediaPipe vs HaMeR trajectories
7. **If detection < 85%:** Lower GroundingDINO threshold to 0.15, try expanded text prompts

---

*Completed: 2026-03-14*
