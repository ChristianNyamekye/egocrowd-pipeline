# Architecture Research: Flexa Pipeline v0.3

## Current Architecture Summary

The pipeline processes iPhone LiDAR R3D captures through 6 sequential stages, producing a MuJoCo G1 humanoid simulation video:

```
[R3D file] → ingest → hand_tracking → object_detection → reconstruction_3d → calibration → simulation → [MP4]
```

**Data flow between stages (JSON contracts):**

1. **Ingest** (`r3d_ingest.py`): R3D zip → extracted frames, video.mp4, arkit_poses.json, depth maps
2. **Hand tracking** (`hand_tracker_v2.py` / `egocrowd/hand_pose.py`): video.mp4 → `{session}_hand_trajectory.json` (per-frame wrist_pixel, grasping, 21 landmarks)
3. **Object detection** (`detect_objects.py` / manual): frames → `object_detections/{session}_objects_clean.json` (label, pos_world, confidence)
4. **3D reconstruction** (`reconstruct_wrist_3d.py`): retargeted JSON + R3D depth → `wrist_trajectories/{session}_wrist3d.json` (wrist_world_smooth [N,3], grasping [N])
5. **Calibration** (`calibrate_workspace.py`): wrist3d + objects → `wrist_trajectories/{session}_calibrated.json` (wrist_sim [N,3], grasping [N], objects_sim)
6. **Simulation** (`mujoco_g1_v10.py`): calibrated JSON → `sim_renders/{session}_g1_v10.mp4`

**Critical data contract — calibrated JSON:**
```json
{
  "session": "stack2",
  "wrist_sim": [[x, y, z], ...],   // N frames, sim coordinates
  "grasping": [0, 0, 1, ...],      // N frames, boolean-like
  "objects_sim": [[x, y, z], ...],  // or dict {name: [x,y,z]}
  "r3d_to_sim": {"scale": float, "obj_centroid_r3d": [...], "obj_centroid_sim": [...]}
}
```

**Current stack2 example:** 951 frames, grasping from frame 88-726 (72 grasping frames out of 951), rendering at 10 FPS = 95-second video. Action doesn't start until ~frame 577.

**Key architectural detail in simulation (`mujoco_g1_v10.py`):**
The `p` progress variable (lines 324) maps the entire N-frame trajectory into a normalized 0.0-1.0 grasp window:
- `p = 0.0` before grasp start (`i < ls`)
- `p = (i-ls)/(le-ls)` during grasp window
- `p = 1.0` after grasp end (`i > le`)

This `p` drives a **hardcoded choreography** (lines 337-390): hover→descend→dwell→lift→carry→descend→release. The real wrist trajectory is only used for pre-grasp arm motion (`i < ls`, line 339-341). During the grasp window, IK targets are computed from fixed waypoints (pick_xy0, place_xy0) with smoothstep interpolation — the human's actual hand motion is ignored.

---

## Integration Analysis

### 1. True Wrist Retargeting

**Problem:** Lines 337-390 of `mujoco_g1_v10.py` replace the human's real trajectory with a choreographed pick-and-place sequence during the grasp window. Only pre-grasp motion (lines 339-341) uses `wrist[i]`.

**Files to modify:**
- `mujoco_g1_v10.py` — lines 323-390 (the main frame loop's target computation). This is the primary change.
- `calibrate_workspace.py` — lines 115-141 (scale/offset transform). May need adjustment to preserve trajectory shape better.

**New files:** None.

**Data flow changes:**
- The calibrated JSON `wrist_sim` already contains the full trajectory for all N frames. No upstream data flow change needed.
- The `grasping` array already marks grasp/release timing. No change needed.
- The simulation needs to use `wrist[i]` as the IK target for ALL frames, not just pre-grasp.

**Integration points:**
- `render_task()` reads `wrist` and `grasping` from calibrated JSON (line 138-139)
- `ik_solve()` takes a target position (line 396) — this function is agnostic to how the target was computed
- Grasp/release logic (lines 399-449) uses `want_grip` which currently comes from the `p`-based choreography — should come from `grasping[i]` directly
- Block attachment still uses kinematic attach (palm proximity check) — this can stay

**Suggested approach:**
1. Replace the `p`-based target computation block (lines 337-390) with direct trajectory following: `target = wrist[i]` with Z clamping to stay above table.
2. Derive `want_grip` from `grasping[i]` instead of `p`-based phases.
3. Keep the existing grasp/release kinematic attachment logic (lines 399-449) — it's already proximity-based and doesn't depend on choreography.
4. Keep `pick_xy0` / `place_xy0` only for object placement (determining which block to grasp and where the support block is), not for arm trajectory.
5. Add a height safety floor: `target[2] = max(target[2], G1_TABLE_HEIGHT + 0.05)` to prevent table collision.
6. The `ls`/`le` variables (grasp start/end frame indices, lines 183-187) can still be computed from `grasping` array for logging, but should not drive the trajectory.

**Risk:** Raw trajectories from MediaPipe (52% detection) have large gaps and noise. This feature has a hard dependency on features 2 (HaMeR) and 3 (trajectory smoothing) to work well.

---

### 2. HaMeR Integration

**Problem:** `egocrowd/hand_pose.py` raises `NotImplementedError` at line 44. The pipeline falls back to MediaPipe (52% detection rate). `tools/v5_hand_detect.py` uses GroundingDINO for hand bounding boxes (not full pose), and `processing/hand_pose.py` is a placeholder class.

**Files to modify:**
- `egocrowd/hand_pose.py` — lines 42-48 (replace `NotImplementedError` with actual HaMeR inference or Modal call)
- `run_pipeline.py` — lines 172-202 (`_run_hand_tracking`). The try/except pattern is already correct, but `_hamer_to_trajectory` (lines 205-247) needs to handle full HaMeR output (MANO params, 21 joints, wrist 3D position).

**New files:**
- `processing/hamer_modal.py` — Modal deployment for HaMeR inference (does not exist yet despite being referenced in `egocrowd/hand_pose.py` line 48). Should wrap HaMeR model in a Modal function with GPU.

**Data flow changes:**
- HaMeR produces richer output than MediaPipe: MANO mesh parameters, 3D joint positions in camera frame, wrist translation. The `_hamer_to_trajectory()` converter (lines 205-247) currently only extracts `wrist_pixel` and `grasping` — should also output `landmarks_3d` (21 joints) and `wrist_3d_camera` for direct 3D reconstruction.
- If HaMeR provides wrist 3D positions directly (in camera frame), `reconstruct_wrist_3d.py` could skip the depth-lookup step and use HaMeR's output, improving accuracy.

**Integration points:**
- `run_pipeline.py` line 178: `from egocrowd.hand_pose import extract_hand_poses` — this import path stays the same
- `_hamer_to_trajectory()` at line 205: converts HaMeR output → hand_tracker_v2-compatible dict
- Downstream: `_build_retarget_data()` at line 296 reads the trajectory dict — format must stay compatible
- `tools/v5_hand_detect.py`: provides GroundingDINO hand detection boxes that could serve as input to HaMeR (HaMeR needs hand crops). This is a potential two-stage pipeline: GroundingDINO detect → HaMeR pose.

**Suggested approach:**
1. Create `processing/hamer_modal.py` with a Modal function that:
   - Takes a batch of RGB frames (or frame bytes)
   - Runs ViTDet/GroundingDINO for hand detection boxes
   - Runs HaMeR on detected hand crops
   - Returns per-frame: wrist_pixel, wrist_3d_camera, MANO params, 21 joint positions, grasping estimate
2. Update `egocrowd/hand_pose.py` to call the Modal function via `modal.Function.lookup()` or `modal run`.
3. Extend `_hamer_to_trajectory()` to pass through 3D joint data.
4. Optionally update `reconstruct_wrist_3d.py` to use HaMeR's camera-frame 3D wrist when available (skip depth lookup).

---

### 3. Trajectory Smoothing

**Problem:** 48% frame gaps from MediaPipe cause jerky motion. `reconstruct_wrist_3d.py` already has a `smooth_trajectory()` function (lines 69-85) using bidirectional EMA with `alpha=0.12`, plus NaN gap interpolation (lines 146-159). But the smoothing is minimal and only applied to wrist world coordinates, not to the final sim-space trajectory.

**Files to modify:**
- `reconstruct_wrist_3d.py` — lines 69-85 (`smooth_trajectory()`). Replace or augment with a better filter (Savitzky-Golay or Butterworth low-pass).
- `reconstruct_wrist_3d.py` — lines 146-162 (gap interpolation). Current linear interpolation is basic; could use spline interpolation for smoother gap filling.
- `calibrate_workspace.py` — lines 134-141 (post-calibration). Apply a second smoothing pass after coordinate transform to sim space.
- `mujoco_g1_v10.py` — could add frame-to-frame velocity limiting to prevent sudden jumps.

**New files:** None. Smoothing is a modification of existing functions.

**Data flow changes:**
- `wrist_trajectories/{session}_wrist3d.json` already stores both `wrist_world_raw` and `wrist_world_smooth`. No schema change needed.
- Calibrated JSON `wrist_sim` would contain smoother data. No schema change needed.
- Potentially add a `smoothing_params` field to the JSON metadata for reproducibility.

**Integration points:**
- `smooth_trajectory()` at `reconstruct_wrist_3d.py:69` — called at line 162
- Gap interpolation at `reconstruct_wrist_3d.py:146-159` — runs before smoothing
- `calibrate_workspace.py:135` — `wrist_sim = wrist_sim_axes * scale + offset` — smoothing could be applied here too
- `mujoco_g1_v10.py` IK loop — velocity clamping could be added at the target level

**Suggested approach:**
1. Replace `smooth_trajectory()` with a configurable filter: Savitzky-Golay (preserves trajectory shape better than EMA) or Butterworth low-pass.
2. Upgrade gap interpolation from linear to cubic spline (`scipy.interpolate.CubicSpline`).
3. Add outlier rejection before smoothing: flag points where frame-to-frame displacement exceeds a threshold (e.g., 0.1m between consecutive frames at 30fps).
4. Apply a final smoothing pass in `calibrate_workspace.py` after the coordinate transform.
5. In `mujoco_g1_v10.py`, add per-frame velocity clamping: `target = prev_target + np.clip(target - prev_target, -max_vel, max_vel)`.

---

### 4. Video Trimming

**Problem:** stack2 has 951 frames (95 seconds at 10 FPS rendering) but the actual stacking action occurs in roughly frames 577-726 (15 seconds). The pipeline currently processes and renders ALL frames. PROJECT.md notes "action starts at frame 577."

**Files to modify:**
- `run_pipeline.py` — add a `--trim` flag and trimming stage. Could be inserted between calibration (stage 5) and simulation (stage 6), or as a post-processing step on the calibrated JSON.
- `calibrate_workspace.py` — add optional trim to the calibrated output (slice `wrist_sim` and `grasping` arrays).

**New files:**
- `trim_trajectory.py` — Auto-detect action window from trajectory data. Analyzes wrist velocity and grasping events to find the meaningful segment.

**Data flow changes:**
- The calibrated JSON `wrist_sim` and `grasping` arrays would be trimmed from N frames to a shorter window.
- Downstream simulation receives fewer frames → shorter, more focused output video.
- Need to preserve frame index mapping if other stages reference original frame numbers.

**Integration points:**
- After `calibrate_session()` returns the calibrated JSON path (line 156 of `run_pipeline.py`)
- Before `run_simulation()` call (line 166 of `run_pipeline.py`)
- The simulation's `n = len(wrist)` (line 181 of `mujoco_g1_v10.py`) automatically adapts to shorter input
- `grasping` array indexing for `ls`/`le` computation (lines 183-187) works on any length

**Suggested approach:**
1. Create `trim_trajectory.py` with `auto_trim(wrist_sim, grasping, margin_frames=30)`:
   - Find first/last grasping frame
   - Compute wrist velocity; find sustained motion onset
   - Return `(start_frame, end_frame)` with configurable margin
2. Apply trimming in `run_pipeline.py` between stages 5 and 6: load calibrated JSON, slice arrays, save back.
3. Add `--trim` / `--no-trim` CLI flag (default: trim).
4. Add `--trim-margin` for configurable margin around the action window.
5. Target output: 15-30 seconds (150-300 frames at 10 FPS).

---

### 5. Grasp Visual Quality

**Problem:** Fingers snap between open/closed states. Finger control uses a simple exponential blend (`finger_ctrl += (target - finger_ctrl) * 0.25` at line 457). No finger pre-shaping before contact. Potential block interpenetration during kinematic attachment.

**Files to modify:**
- `mujoco_g1_v10.py` — lines 39-43 (FINGER_OPEN/FINGER_CLOSED constants), line 457 (finger blend), lines 303-319 (grasp state initialization), lines 399-449 (attachment logic).

**New files:** None.

**Data flow changes:** None. This is purely a simulation rendering improvement. No upstream data changes.

**Integration points:**
- `FINGER_OPEN` / `FINGER_CLOSED` arrays at lines 39-40 — define finger target positions
- `finger_ctrl` blend at line 457 — `0.25` blend factor controls closing speed
- `want_grip` boolean drives finger state — currently binary, could be graduated
- `BLEND_FRAMES = 12` at line 43 — controls kinematic attachment blend smoothness
- `FINGER_GAIN_MULTIPLIER = 25.0` at line 42 — controls finger actuator force

**Suggested approach:**
1. **Finger pre-shaping:** Start curling fingers slightly (~30% of FINGER_CLOSED) when palm is within 0.15m of a block, before `want_grip` becomes True. Use distance-to-nearest-block as a continuous signal.
2. **Graduated finger closure:** Replace binary FINGER_OPEN/FINGER_CLOSED with a 3-stage sequence: open → pre-shape (partial curl) → full grasp. Map proximity to curl amount.
3. **Slower, smoother finger blend:** Reduce blend factor from 0.25 to ~0.12 for opening, keep 0.20 for closing. Add per-finger timing offsets (thumb leads, pinky lags) for more natural motion.
4. **Reduce interpenetration:** Increase `BLOCK_HALF` collision margin or adjust `grasp_offset` computation to keep block surface outside finger geometry. The `grasp_offset = block_pos - pc` at line 420 could be adjusted to add a small outward bias.
5. **Visual polish:** Add slight wrist rotation during grasp (tilt palm downward during approach, level during carry).

---

## Build Order

Dependencies between features dictate the build order:

```
Phase 1 (Independent foundations):
  3. Trajectory Smoothing  ← no dependencies, improves everything downstream
  4. Video Trimming         ← no dependencies, speeds up iteration

Phase 2 (Quality stack):
  2. HaMeR Integration      ← benefits from smoothing (Phase 1)
  5. Grasp Visual Quality    ← independent but best tested after trimming

Phase 3 (Capstone):
  1. True Wrist Retargeting  ← depends on 2 (HaMeR) + 3 (smoothing) for usable results
```

**Rationale:**
- **Trajectory smoothing first:** It's a contained change in existing files, has no dependencies, and improves all downstream features. Even with MediaPipe's 52% detection, smoothed trajectories are more usable for testing retargeting.
- **Video trimming second:** Also independent, and dramatically speeds up iteration (95s → 15-30s render time). Essential for rapid testing of subsequent features.
- **HaMeR third:** Requires Modal deployment (new file), but the integration points in `run_pipeline.py` are already structured for it (try HaMeR / catch / fallback to MediaPipe). With smoothing already in place, even partial HaMeR results will be usable.
- **Grasp visual quality fourth:** Pure rendering improvement, best validated after trimming (faster iteration) and can be tested with either MediaPipe or HaMeR data.
- **True wrist retargeting last:** The capstone feature. With 85%+ HaMeR detection and smooth trajectories, replacing the `p` choreography with real trajectory following should produce quality results. Without those foundations, raw trajectories are too noisy/gappy to drive the simulation.

## Data Flow Changes

### Before (v0.2):

```
wrist_sim[N,3]  ─┐
                  ├──► mujoco_g1_v10.py ──► p-based choreography ──► IK target
grasping[N]     ─┘    (only uses wrist[i] for i < ls)
                       (ignores wrist[i] during grasp window)
                       (hardcoded pick/carry/place waypoints)
```

### After (v0.3):

```
                     ┌─── trim_trajectory.py ───┐
                     │   (auto-detect action     │
                     │    window, 15-30s)         │
                     └────────┬──────────────────┘
                              │
wrist_sim[M,3]  ─┐           │  M << N (trimmed)
  (HaMeR 85%+)   ├──► calibrated JSON (trimmed + smoothed)
  (smoothed)      │           │
grasping[M]     ─┘           │
                              ▼
                  mujoco_g1_v10.py
                  ├── target = wrist[i]  (ALL frames, not just pre-grasp)
                  ├── want_grip = grasping[i]  (from data, not p-choreography)
                  ├── finger pre-shaping  (distance-based gradual closure)
                  └── IK solve → render
```

### Key contract changes:
- **Calibrated JSON:** Same schema, but arrays are trimmed (M frames instead of N). New optional field: `trim_info: {original_frames, start_frame, end_frame}`.
- **HaMeR trajectory dict:** Extended with `landmarks_3d` (21 joints) and `wrist_3d_camera` fields. Backward-compatible — MediaPipe fallback still works.
- **No breaking changes to pipeline_config.py or directory structure.**
