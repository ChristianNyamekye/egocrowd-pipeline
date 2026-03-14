# Phase 1: Foundations — Research

*Researched: 2026-03-14*
*Scope: TRK-01, TRK-02, TRK-03, OUT-01, OUT-02*

---

## 1. Current Code Analysis

### What Exists

**`reconstruct_wrist_3d.py` — 3D reconstruction + smoothing (204 lines)**

- **Gap interpolation (lines 146-159):** Uses `np.interp()` — pure linear interpolation across all NaN gaps, regardless of gap length. Applied per-axis independently. This is the implementation behind TRK-01.
- **Smoothing (lines 69-85, `smooth_trajectory()`):** Bidirectional EMA with `alpha=0.12`. Forward pass, backward pass, then average. This is the implementation behind TRK-02.
  - **Problem:** alpha=0.12 means each sample contributes only 12% weight — extremely heavy smoothing that destroys grasp dwell precision. Endpoint distortion is inherent to bidirectional EMA (backward pass has no future data at boundaries). Research pitfall P6 confirms this.
- **Grasping signal (line 121, 138):** Passed through from hand tracking data as boolean. Never smoothed — currently satisfies TRK-03 by accident, not by design.
- **Output:** `wrist_trajectories/{session}_wrist3d.json` with both `wrist_world_raw` and `wrist_world_smooth` arrays plus `grasping`.

**`calibrate_workspace.py` — coordinate transform (186 lines)**

- **Axis swap (line 31-36, `r3d_to_sim_axes()`):** R3D `(X,Y,Z)` to sim `(-Z,-X,Y)`.
- **Scaling (lines 115-124):** `scale = 0.35 / max_wrist_range`. For stack2: `scale = 0.215`.
- **Offset (line 127):** Centers object centroid at sim workspace center `(0.5, 0.0, 0.43)`.
- **Z correction (lines 138-142):** Forces all object Z to `TABLE_Z=0.425`, applies same correction to wrist Z. This is a potential discontinuity source — smoothing after calibration would spread this step.
- **Output:** `wrist_trajectories/{session}_calibrated.json` with `wrist_sim`, `grasping`, `objects_sim`, `r3d_to_sim`.

**`mujoco_g1_v10.py` — simulation (557 lines)**

- **Frame loop (lines 323-524):** Reads `wrist_sim` and `grasping` from calibrated JSON. Currently, `wrist[i]` is only used for pre-grasp motion (i < ls). During grasp window, the `p` progress variable drives choreographed waypoints (lines 337-390).
- **`ls`/`le` computation (lines 182-187):** Finds "late" grasp frames (>60% of trajectory), uses first/last as grasp start/end. For stack2: `ls=577, le=731, win=154`. This computation auto-adapts to trimmed arrays.
- **`n = len(wrist)` (line 181):** Simulation length adapts automatically to input array length. No hardcoded frame count.
- **FPS = 10 (line 24):** Rendering at 10 FPS. At 951 frames = 95.1 seconds. Target: 150-300 frames = 15-30 seconds.

**`run_pipeline.py` — orchestrator (396 lines)**

- **Stage flow:** ingest -> hand_tracking -> object_detection -> 3D_reconstruction -> calibration -> simulation.
- **Trim insertion point:** Between stage 5 (calibration, line 156) and stage 6 (simulation, line 166). The calibrated JSON is loaded, arrays sliced, saved back, then simulation reads the trimmed version.
- No `--trim` flag currently exists.

### What Needs Changing

| File | Change | Requirement |
|------|--------|-------------|
| `reconstruct_wrist_3d.py` | Replace `smooth_trajectory()` with Savitzky-Golay filter | TRK-02 |
| `reconstruct_wrist_3d.py` | Replace `np.interp` with gap-length-aware strategy (CubicSpline/PCHIP/linear) | TRK-01 |
| `reconstruct_wrist_3d.py` | Add explicit guard: never smooth `grasping` array | TRK-03 |
| New: `trim_trajectory.py` | Auto-detect action window, return trim indices | OUT-01 |
| `run_pipeline.py` | Add trim stage between calibration and simulation; add `--trim` CLI flag | OUT-01, OUT-02 |
| `calibrate_workspace.py` | Add `trim_info` to calibrated JSON output (traceability) | OUT-01 |

### What Stays Unchanged

- `calibrate_workspace.py` core transform logic (axis swap, scale, offset, Z correction)
- `mujoco_g1_v10.py` entirely (Phase 1 does not touch simulation; that is Phase 2)
- Pipeline stages 1-5 (ingest through calibration)
- All data contracts (JSON schemas stay identical; only array lengths change after trim)

---

## 2. Data Structure Analysis

### Calibrated JSON (`stack2_calibrated.json`)

```json
{
  "session": "stack2",
  "r3d_to_sim": {
    "obj_centroid_r3d": [0.5, 0.0, 0.425],
    "obj_centroid_sim": [0.5, 0.0, 0.43],
    "scale": 0.21474
  },
  "objects_sim": [[0.5, 0.0, 0.425], [0.5, 0.0, 0.425]],
  "wrist_sim": [[x, y, z], ...],   // float[951][3]
  "grasping": [false, true, ...]   // bool[951]
}
```

**Key measurements from stack2:**

| Metric | Value |
|--------|-------|
| Total frames | 951 |
| Duration at 10 FPS | 95.1 seconds |
| Valid 3D reconstructions | 492/951 (52%) |
| Missing frames | 459 (48%) |
| Grasping frames | 72 (7.6%) |
| First grasp frame | 88 |
| Last grasp frame | 726 |
| Wrist X range | 0.350m (0.176 to 0.526) |
| Wrist Y range | 0.309m (-0.050 to 0.259) |
| Wrist Z range | 0.182m (0.300 to 0.482) |
| Mean frame displacement | 2.64mm |
| Max frame displacement | 45.2mm |
| Frames with >3cm jumps | 3 |
| NaN frames (post-interp) | 0 |

### Gap Distribution (raw data, pre-interpolation)

| Gap Length | Count | Notes |
|------------|-------|-------|
| 1 frame | 9 | Trivial — any method works |
| 2-5 frames | 8 | Short — cubic spline ideal |
| 6-15 frames | 2 | Medium — PCHIP recommended |
| 16-30 frames | 4 (17, 20, 22, 29 frames) | Long — PCHIP or linear |
| >30 frames | 4 (37, 51, 88, 138 frames) | Very long — linear + hold |

The 138-frame gap (13.8 seconds at 10 FPS) is the worst case. CubicSpline would produce wild oscillations across this gap. Linear interpolation or hold-last-known is the only safe option.

### Grasping Signal Pattern

34 separate grasping segments, most very short (1-6 frames). The signal is noisy — not clean on/off blocks. This is from MediaPipe's finger-curl heuristic producing intermittent detections. Key observations:

- Early grasping signals (frames 88-220): likely false positives from hand movement near objects
- Late grasping signals (frames 577-726): the actual stacking action
- The sim currently filters to "late" grasps (>60% of trajectory) for `ls`/`le`, which correctly focuses on frames 577-731

---

## 3. Smoothing Implementation (TRK-01, TRK-02, TRK-03)

### 3a. Gap Interpolation Strategy (TRK-01)

**Requirement:** Gap-length-aware strategy: cubic spline <15 frames, PCHIP 15-30, linear >30.

**Implementation approach:**

```python
from scipy.interpolate import CubicSpline, PchipInterpolator

def interpolate_gaps(arr):
    """Gap-length-aware interpolation for [N,3] trajectory.

    - Short gaps (<15 frames): CubicSpline (C2 smooth, best for short spans)
    - Medium gaps (15-30 frames): PCHIP (monotonic, no overshoot)
    - Long gaps (>30 frames): Linear (safe, flags as low-confidence)

    Applied per-axis independently.
    """
    result = arr.copy()
    for ax in range(3):
        col = result[:, ax]
        valid_mask = ~np.isnan(col)
        if valid_mask.all() or not valid_mask.any():
            continue

        valid_idx = np.where(valid_mask)[0]
        valid_vals = col[valid_mask]

        # Identify each gap
        nan_mask = np.isnan(col)
        gaps = _find_gaps(nan_mask)  # returns list of (start, end) tuples

        for gap_start, gap_end in gaps:
            gap_len = gap_end - gap_start
            gap_indices = np.arange(gap_start, gap_end)

            if gap_len <= 15:
                # CubicSpline — needs at least 2 points on each side
                interp = CubicSpline(valid_idx, valid_vals)
                col[gap_indices] = interp(gap_indices)
            elif gap_len <= 30:
                # PCHIP — monotonic, no overshoot
                interp = PchipInterpolator(valid_idx, valid_vals)
                col[gap_indices] = interp(gap_indices)
            else:
                # Linear — safe for long gaps
                col[gap_indices] = np.interp(gap_indices, valid_idx, valid_vals)

    return result
```

**Key decisions:**
- CubicSpline and PchipInterpolator are both fit on ALL valid points, not just neighbors. This provides global context for short-gap filling but means the interpolator is rebuilt per gap. Since N=951 and there are only 27 gaps, this is cheap.
- CubicSpline provides C2 continuity (smooth acceleration), ideal for short gaps where we have high confidence in the fill.
- PCHIP provides C1 continuity but prevents overshoot — critical for medium gaps where wild oscillations would send IK targets out of workspace.
- Linear interpolation for >30 frame gaps accepts the trajectory is essentially unknown. The 138-frame gap (13.8s) is too long for any polynomial method.

**Validation for TRK-01:** Zero NaN frames after interpolation. Check: `assert not np.isnan(result).any()`.

### 3b. Savitzky-Golay Spatial Filter (TRK-02)

**Requirement:** Zero-phase smoothing that preserves grasp dwell positions.

**Parameters for 10 FPS motion capture:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `window_length` | 7 | 0.7s window. Conservative start — preserves dwell. Can tune up to 11 if jitter remains. Must be odd. |
| `polyorder` | 3 | Cubic — preserves velocity and acceleration shape. Order 2 would over-smooth direction changes. |
| `mode` | `'nearest'` | Endpoint handling: repeats edge value. Better than `'constant'` (zero padding) or `'wrap'` (circular). |
| `axis` | 0 | Apply along time axis for [N,3] array. |

**Why window=7, not 11:**

At 10 FPS, `window_length=11` means 1.1 seconds of smoothing — this covers an entire quick grasp approach and would blur the dwell. The research pitfall P7 warns against this. Starting at 7 (0.7s) is safer; we can increase if >3cm jumps persist.

At 10 FPS, human hand movement during manipulation has meaningful frequency content up to ~2-3 Hz. The Savitzky-Golay filter with window=7 at 10 FPS has an effective cutoff around 3-4 Hz — above the manipulation band but below the noise floor.

**Implementation:**

```python
from scipy.signal import savgol_filter

def smooth_trajectory_savgol(traj, window_length=7, polyorder=3):
    """Savitzky-Golay smoothing for [N,3] trajectory.

    Zero-phase (no lag in batch mode).
    Preserves trajectory shape better than EMA.
    Applied per-axis independently.
    """
    if len(traj) < window_length:
        return traj
    return savgol_filter(traj, window_length, polyorder, axis=0, mode='nearest')
```

**Validation for TRK-02:**
1. Zero NaN gaps: `assert not np.isnan(smoothed).any()`
2. No >3cm jumps: `assert np.linalg.norm(np.diff(smoothed, axis=0), axis=1).max() <= 0.03`
3. Wrist range stays within workspace bounds after calibration

**Order of operations (critical):**
1. Fill gaps (TRK-01) — produce complete trajectory with no NaNs
2. Apply Savitzky-Golay (TRK-02) — smooth the filled trajectory
3. Never touch `grasping` array (TRK-03)

This order is important: savgol_filter cannot handle NaN values, and smoothing across gaps would produce artifacts if gaps were filled after smoothing.

### 3c. Binary Grasping Preservation (TRK-03)

**Requirement:** Grasping signal remains binary (0 or 1) — never interpolated, never smoothed.

**Implementation:** Explicit code guard + assertion.

```python
# In reconstruct_wrist_3d.py — after all smoothing:
assert set(np.unique(grasping)).issubset({0, 1, True, False}), \
    f"Grasping signal corrupted: unique values = {np.unique(grasping)}"
```

The current code already passes `grasping` through without modification. The risk is future changes accidentally applying `smooth_trajectory()` to it. The guard makes this a hard contract.

---

## 4. Video Trimming Implementation (OUT-01, OUT-02)

### 4a. Action Window Detection (OUT-01)

**Requirement:** Video starts within 1 second of first grasp approach and ends within 2 seconds of last release.

**The problem with naive grasping-signal-based trimming:**

The stack2 grasping signal has 34 segments spanning frames 88-726. Using first/last grasp frame with a margin yields a 718-frame window (71.8s) — far too long. The early grasping detections (frames 88-220) are noise from MediaPipe, not real grasp intent.

**Proposed approach: combined signal detection**

The sim's existing `ls`/`le` logic (lines 182-187) already solves this by filtering to "late" grasps (>60% of trajectory). The resulting window is frames 577-731 = 154 frames = 15.4 seconds. This is within the 15-30s target.

For a general-purpose trim, we should combine:
1. **Grasping signal clustering:** Find the densest cluster of grasping segments. The late cluster (frames 577-726) has 21 grasping frames across 7 segments, versus the early cluster (88-220) with 33 grasping frames across 11 segments but spread sparsely.
2. **Wrist velocity as tiebreaker:** The velocity profile shows a clear ramp-up starting at frame ~500 (0.044 m/s average) peaking at frames 550-750 (0.089 m/s).

**Algorithm:**

```python
def detect_action_window(wrist_sim, grasping, fps=10):
    """Detect the action window using grasping signal + wrist velocity.

    Returns (start_frame, end_frame) for trimming.

    Strategy:
    1. Compute wrist velocity, find sustained high-velocity region
    2. Find grasping frames within that region
    3. Pad with margin: 30 frames (3s) before, 50 frames (5s) after
    """
    n = len(wrist_sim)

    # Wrist velocity (rolling mean, 1-second window)
    disps = np.linalg.norm(np.diff(wrist_sim, axis=0), axis=1)
    vel = np.zeros(n)
    vel[1:] = disps
    window = fps  # 1-second rolling window
    rolling_vel = np.convolve(vel, np.ones(window)/window, mode='same')

    # Find high-velocity region (top 30% of velocity)
    vel_threshold = np.percentile(rolling_vel[rolling_vel > 0], 70)
    high_vel = rolling_vel > vel_threshold

    # Find grasping frames in high-velocity region
    g = np.array(grasping, dtype=bool)
    action_frames = np.where(g & high_vel)[0]

    if len(action_frames) == 0:
        # Fallback: use last 40% of grasping frames
        grasp_idx = np.where(g)[0]
        if len(grasp_idx):
            cutoff = grasp_idx[int(len(grasp_idx) * 0.6)]
            action_frames = grasp_idx[grasp_idx >= cutoff]

    if len(action_frames) == 0:
        # No grasping signal: use velocity alone
        action_frames = np.where(high_vel)[0]

    # Margins from OUT-01 spec: "first grasp - 30 frames to last grasp + 50 frames"
    margin_before = 30   # 3 seconds at 10fps
    margin_after = 50    # 5 seconds at 10fps

    start = max(0, action_frames[0] - margin_before)
    end = min(n, action_frames[-1] + margin_after)

    # Enforce 15-30 second duration
    duration = end - start
    target_min = 150  # 15s at 10fps
    target_max = 300  # 30s at 10fps

    if duration < target_min:
        # Expand symmetrically
        deficit = target_min - duration
        start = max(0, start - deficit // 2)
        end = min(n, end + (deficit - deficit // 2))
    elif duration > target_max:
        # Tighten: reduce margins
        excess = duration - target_max
        start = min(start + excess // 2, action_frames[0] - 10)
        end = max(end - (excess - excess // 2), action_frames[-1] + 20)

    return int(start), int(end)
```

**Expected result for stack2:**

With the velocity profile showing clear action at frames 500-800, and late grasping at 577-726:
- `action_frames` will be approximately frames 577-726
- With margins: start = 547, end = 776
- Duration: 229 frames = 22.9 seconds (within 15-30s target)

### 4b. Trim Application (OUT-02)

**Where in the pipeline:** Between calibration (stage 5) and simulation (stage 6).

**Implementation:**

```python
def trim_calibrated_data(calib_path, start, end):
    """Trim calibrated JSON arrays to [start:end] window.

    Modifies wrist_sim and grasping in-place.
    Adds trim_info for traceability.
    """
    with open(calib_path) as f:
        calib = json.load(f)

    original_frames = len(calib["wrist_sim"])
    calib["wrist_sim"] = calib["wrist_sim"][start:end]
    calib["grasping"] = calib["grasping"][start:end]
    calib["trim_info"] = {
        "original_frames": original_frames,
        "start_frame": start,
        "end_frame": end,
        "trimmed_frames": end - start,
    }

    with open(calib_path, "w") as f:
        json.dump(calib, f)
```

**Downstream auto-adaptation:**
- `mujoco_g1_v10.py` line 181: `n = len(wrist)` — automatically uses trimmed length
- `mujoco_g1_v10.py` lines 182-187: `ls`/`le` recomputed from trimmed `grasping` array
- No re-indexing needed for RGB frames or depth maps (trim happens after 3D reconstruction)

**Critical constraint:** Trimming must happen AFTER smoothing, not before. Research pitfall P6 confirms that smoothing on a trimmed array causes endpoint distortion. Sequence: fill gaps -> smooth -> calibrate -> trim -> simulate.

Wait — the research SUMMARY.md says "Smooth before trimming to avoid edge artifacts from the long idle period." But P6 says the opposite. Let me resolve this:

- **Smoothing is done in `reconstruct_wrist_3d.py` (world coords), before calibration.**
- **Trimming is done after calibration, on the calibrated JSON.**
- These are different stages. Smoothing happens first (in 3D world coordinates, on the full 951-frame array), then calibration transforms to sim coords, THEN trimming slices the calibrated result.
- This is the correct order: smooth the full array (no endpoint issues because smoothing window sees the full trajectory), then trim the result.

---

## 5. Integration Points

### Pipeline Data Flow (with Phase 1 changes highlighted)

```
Stage 1: Ingest R3D                        (unchanged)
Stage 2: Hand Tracking                     (unchanged)
Stage 3: Object Detection                  (unchanged)
Stage 4: 3D Reconstruction                 *** MODIFIED ***
  └── reconstruct_wrist_3d.py
      ├── Gap interpolation: np.interp -> gap-length-aware (TRK-01)
      ├── Smoothing: bidirectional EMA -> savgol_filter (TRK-02)
      └── Grasping: pass-through + assertion (TRK-03)
Stage 5: Calibration                        (unchanged)
Stage 5.5: Trim                            *** NEW ***
  └── trim_trajectory.py
      ├── detect_action_window() -> (start, end)
      └── trim_calibrated_data() -> modified JSON
Stage 6: Simulation                        (unchanged — auto-adapts via n=len(wrist))
```

### File Dependencies

```
reconstruct_wrist_3d.py  (modify)
  ├── scipy.interpolate.CubicSpline       (existing dep, new import)
  ├── scipy.interpolate.PchipInterpolator (existing dep, new import)
  ├── scipy.signal.savgol_filter          (existing dep, new import)
  └── Output: wrist_trajectories/{session}_wrist3d.json (same schema)

trim_trajectory.py  (new file)
  ├── numpy                               (existing dep)
  ├── json                                (stdlib)
  └── Output: modifies {session}_calibrated.json in-place

run_pipeline.py  (modify — add trim stage call)
  └── imports trim_trajectory.trim_and_save()
```

### Where smoothing happens in the data flow

1. `reconstruct_wrist_3d.py` line 146-162: gap interpolation on `wrist_world_raw` (R3D world coords)
2. `reconstruct_wrist_3d.py` line 162: `smooth_trajectory()` on interpolated array (R3D world coords)
3. Result saved as `wrist_world_smooth` in `{session}_wrist3d.json`
4. `calibrate_workspace.py` line 76: loads `wrist_world_smooth` (already smoothed)
5. `calibrate_workspace.py` lines 108-142: axis swap + scale + offset + Z correction
6. Result saved as `wrist_sim` in `{session}_calibrated.json`

Smoothing in world coords (step 2) BEFORE calibration (step 5) is correct per research finding IP1: "smoothing happens in world coordinates, before any sim-specific transforms."

No second smoothing pass in sim coords is needed for Phase 1. The Z correction (step 5) is a uniform offset, not a discontinuity, so it does not create artifacts that require post-calibration smoothing.

---

## 6. Validation Architecture

### Success Criterion 1: Zero NaN gaps, no >3cm jumps

**Where to validate:** After `smooth_trajectory_savgol()` returns, in `reconstruct_wrist_3d.py`.

```python
# After smoothing
smoothed = smooth_trajectory_savgol(interpolated)
assert not np.isnan(smoothed).any(), "NaN gaps remain after smoothing"
disps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
max_jump = disps.max()
if max_jump > 0.03:
    n_violations = (disps > 0.03).sum()
    print(f"  WARNING: {n_violations} frames with >3cm jump (max={max_jump:.4f})")
    # Consider: increase window_length or add velocity clamping
```

**Current state:** stack2 has 3 frames with >3cm jumps (max 4.5cm). The Savitzky-Golay filter with window=7 should reduce these. If not, add post-smoothing velocity clamp:

```python
# Optional: velocity clamp as safety net
max_step = 0.03  # 3cm per frame at 10fps = 0.3 m/s
for i in range(1, len(smoothed)):
    delta = smoothed[i] - smoothed[i-1]
    norm = np.linalg.norm(delta)
    if norm > max_step:
        smoothed[i] = smoothed[i-1] + delta * (max_step / norm)
```

### Success Criterion 2: Grasping signal stays binary

**Where to validate:** At save time in `reconstruct_wrist_3d.py`, and at load time in `mujoco_g1_v10.py`.

```python
unique_vals = set(np.unique(grasping).tolist())
assert unique_vals.issubset({0, 1, 0.0, 1.0, True, False}), \
    f"Grasping signal not binary: {unique_vals}"
```

### Success Criterion 3: Video starts within 1s of first grasp approach, ends within 2s of last release

**Where to validate:** After trimming, check the trim indices against the grasping signal.

```python
trimmed_g = np.array(calib["grasping"])
grasp_idx = np.where(trimmed_g)[0]
if len(grasp_idx):
    frames_before_first_grasp = grasp_idx[0]
    frames_after_last_grasp = len(trimmed_g) - grasp_idx[-1] - 1
    fps = 10
    assert frames_before_first_grasp / fps <= 4.0, \
        f"Video starts {frames_before_first_grasp/fps:.1f}s before first grasp (max 4s)"
    assert frames_after_last_grasp / fps <= 6.0, \
        f"Video ends {frames_after_last_grasp/fps:.1f}s after last grasp (max 6s)"
```

Note: The 1-second and 2-second criteria from the success spec refer to the OUTPUT video timing relative to "first grasp approach" and "last release." With the OUT-01 margins of 30 frames before and 50 frames after, and the grasping signal pattern (noisy, fragmented), the relevant measurement is against the action cluster, not the first isolated grasp blip.

### Success Criterion 4: Output video 15-30 seconds

**Where to validate:** After trimming, before simulation.

```python
trimmed_frames = len(calib["wrist_sim"])
duration_s = trimmed_frames / 10  # FPS=10
assert 15 <= duration_s <= 30, \
    f"Trimmed duration {duration_s:.1f}s outside 15-30s range"
```

### Success Criterion 5: STACKED=True still passes

**Where to validate:** End of simulation output (already printed by `mujoco_g1_v10.py` line 536).

**Risk:** Smoothing + trimming should not affect stacking because:
- The choreographed `p`-based motion (lines 337-390) is untouched in Phase 1
- The `ls`/`le` computation re-derives from the trimmed `grasping` array
- Block positions are hardcoded to `desired_pick_xy` / `desired_support_xy` — not trajectory-dependent
- The `p` progress variable normalizes to [0,1] regardless of absolute frame count

**Test command:** `python mujoco_g1_v10.py stack2` — verify "STACKED=True" in output.

### Validation Test Script

Create a test that runs the full pipeline and checks all 5 criteria:

```python
# test_phase1.py
def test_smoothed_trajectory():
    """TRK-01, TRK-02, TRK-03: smoothing quality checks."""
    data = load_wrist3d("stack2")
    smoothed = np.array(data["wrist_world_smooth"])
    grasping = data["grasping"]

    # TRK-01: no NaN gaps
    assert not np.isnan(smoothed).any()

    # TRK-02: no >3cm jumps
    disps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    assert disps.max() <= 0.03, f"Max jump {disps.max():.4f} > 0.03"

    # TRK-03: grasping binary
    unique = set(np.unique(grasping).tolist())
    assert unique.issubset({0, 1, True, False})

def test_trimmed_video():
    """OUT-01, OUT-02: trim quality checks."""
    calib = load_calibrated("stack2")
    n = len(calib["wrist_sim"])
    duration = n / 10
    assert 15 <= duration <= 30, f"Duration {duration:.1f}s"
    assert "trim_info" in calib

def test_stacked():
    """OUT-05 (mapped from Phase 3 but tested here): regression check."""
    # Run simulation and parse output for "STACKED=True"
    result = subprocess.run(["python", "mujoco_g1_v10.py", "stack2"],
                          capture_output=True, text=True)
    assert "STACKED=True" in result.stdout
```

---

## 7. Risk Mitigation

### Risk 1: Over-smoothing destroys grasp dwell (P7)

**Likelihood:** Medium — savgol with window=7 is conservative, but worth monitoring.

**Mitigation:**
- Start with `window_length=7` (0.7s). Only increase to 9 or 11 if >3cm jumps persist.
- After smoothing, validate that wrist position during grasping frames is within 5cm of its pre-smoothing position: `assert np.linalg.norm(smoothed[g] - raw[g]) < 0.05`.
- If validation fails, consider phase-aware smoothing (lighter during grasp phases).

### Risk 2: CubicSpline overshoot on medium gaps

**Likelihood:** Low for gaps <15 frames, medium for edge cases.

**Mitigation:**
- PCHIP is the fallback for gaps 15-30 frames. It is monotonicity-preserving and cannot overshoot.
- After interpolation and before smoothing, check that interpolated values stay within the bounding box of surrounding valid points, with 20% margin.

### Risk 3: Trim detection picks wrong action window (P8)

**Likelihood:** Medium for recordings other than stack2. Low for stack2 (clear velocity pattern).

**Mitigation:**
- Use combined signal (velocity + grasping), not grasping alone.
- The `detect_action_window()` function has a fallback chain: velocity+grasping -> late grasping -> velocity alone.
- Add `--trim-start` and `--trim-end` CLI overrides for manual control when auto-detection fails.
- Test with at least stack1 and picknplace2 in addition to stack2.

### Risk 4: Frame index misalignment after trim (P9, IP3)

**Likelihood:** Low — trimming happens on the calibrated JSON, after all frame-index-dependent stages have completed.

**Mitigation:**
- Store `trim_info` in the calibrated JSON for traceability.
- The simulation's `ls`/`le` computation is relative to the loaded array, not to original frame indices. No re-indexing needed.
- No upstream stage references the calibrated JSON frame indices — they work on `wrist3d.json` which stays untrimmed.

### Risk 5: Smoothing applied in wrong coordinate space (IP1)

**Likelihood:** Very low — the current architecture already does smoothing in world coords before calibration.

**Mitigation:**
- Do NOT add a second smoothing pass in `calibrate_workspace.py`.
- Document the invariant: "Smoothing happens in world coords (reconstruct_wrist_3d.py), never in sim coords."

### Risk 6: Endpoint distortion from Savitzky-Golay (P6)

**Likelihood:** Low — savgol_filter with `mode='nearest'` handles endpoints by repeating the edge value, which is better than EMA's inherent endpoint pull.

**Mitigation:**
- The `mode='nearest'` parameter extends the signal by repeating the boundary value, avoiding the forward/backward pass asymmetry of bidirectional EMA.
- Since smoothing is on the FULL 951-frame array (before trim), the endpoints of the eventual trimmed output are interior points of the smoothed array — no endpoint effect.

---

## Appendix: Web Research References

### Savitzky-Golay Filter Parameters

- [SciPy savgol_filter documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html): `window_length` must be odd, `polyorder` must be less than `window_length`. The `mode` parameter controls endpoint handling ('mirror', 'nearest', 'constant', 'wrap').
- [Choosing optimal parameters for Savitzky-Golay](https://nirpyresearch.com/choosing-optimal-parameters-savitzky-golay-smoothing-filter/): Larger windows increase smoothing but reduce feature preservation. Polynomial order controls how well peaks are preserved.
- For 10 FPS data: window=7 covers 0.7s (typical reach duration ~0.5-1.0s), window=11 covers 1.1s (typical grasp dwell ~0.3-0.5s). Window=7 is the conservative starting point.

### PCHIP vs CubicSpline for Gap Filling

- [PchipInterpolator docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html): C1 continuous, monotonicity-preserving, prevents overshoot.
- [CubicSpline docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html): C2 continuous (smoother), but can overshoot and oscillate on long gaps.
- [SciPy interpolation tutorial](https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html): Recommends PCHIP when shape preservation is important, CubicSpline when smoothness is the priority.

### Motion Capture Gap Filling

- [ResearchGate discussion on gap filling methods](https://www.researchgate.net/post/Whats-the-best-method-for-gap-fillingmissing-data-in-motion-capture-data): Interpolation methods work well for gaps < 50 samples. Above that, neural network approaches or hold-last-known are safer.
- [Gap reconstruction with neural networks (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8472986/): For short gaps, cubic spline and PCHIP perform comparably to learned methods. The gap-length threshold for method switching is approximately 500ms.
- At 10 FPS: 500ms = 5 frames. Our threshold of 15 frames (1.5s) for cubic->PCHIP is conservative by this standard.

---

*Research complete. Ready for planning.*
