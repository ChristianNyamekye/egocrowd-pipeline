---
phase: 01-foundations
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [reconstruct_wrist_3d.py]
autonomous: true
requirements: [TRK-01, TRK-02, TRK-03]

must_haves:
  truths:
    - "Smoothed trajectory has zero NaN values"
    - "No consecutive-frame displacement exceeds 3cm (0.03m)"
    - "Grasping signal contains only binary values (0/1/True/False) after processing"
    - "Savitzky-Golay filter is used instead of bidirectional EMA"
    - "Gap interpolation uses different methods based on gap length"
  artifacts:
    - path: "reconstruct_wrist_3d.py"
      provides: "Gap-aware interpolation + Savitzky-Golay smoothing + binary grasping guard"
      contains: "savgol_filter"
    - path: "wrist_trajectories/stack2_wrist3d.json"
      provides: "Smoothed trajectory output for stack2"
  key_links:
    - from: "reconstruct_wrist_3d.py"
      to: "wrist_trajectories/*_wrist3d.json"
      via: "process_session() writes smoothed output"
      pattern: "savgol_filter"
---

<objective>
Replace the trajectory smoothing pipeline in `reconstruct_wrist_3d.py` with gap-length-aware interpolation and Savitzky-Golay filtering, while preserving the binary grasping signal.

Purpose: The current bidirectional EMA (alpha=0.12) over-smooths and destroys grasp dwell precision. Linear `np.interp` for gap filling produces artifacts on long gaps. These must be fixed before retargeting (Phase 2) can work.
Output: Modified `reconstruct_wrist_3d.py` with three improvements: gap-aware interpolation (TRK-01), Savitzky-Golay filter (TRK-02), binary grasping guard (TRK-03).
</objective>

<execution_context>
@/Users/christian/.claude/get-shit-done/workflows/execute-plan.md
@/Users/christian/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/01-foundations/01-RESEARCH.md

Relevant source file:
@reconstruct_wrist_3d.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace gap interpolation with gap-length-aware strategy</name>
  <files>reconstruct_wrist_3d.py</files>
  <action>
Replace the linear interpolation block (lines 146-159) and the `smooth_trajectory()` function (lines 69-85) with two new functions: `_find_gaps()` and `interpolate_gaps()`.

**Step 1: Add imports at the top of the file (after `import numpy as np`):**
```python
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import savgol_filter
```

**Step 2: Replace the `smooth_trajectory()` function (lines 69-85) with these three functions:**

```python
def _find_gaps(nan_mask):
    """Find contiguous NaN gap regions. Returns list of (start, end) index tuples."""
    gaps = []
    in_gap = False
    start = 0
    for i, is_nan in enumerate(nan_mask):
        if is_nan and not in_gap:
            start = i
            in_gap = True
        elif not is_nan and in_gap:
            gaps.append((start, i))
            in_gap = False
    if in_gap:
        gaps.append((start, len(nan_mask)))
    return gaps


def interpolate_gaps(arr):
    """Gap-length-aware interpolation for [N,3] trajectory.

    Strategy per gap length:
    - Short gaps (<= 15 frames): CubicSpline (C2 smooth, best for short spans)
    - Medium gaps (16-30 frames): PCHIP (monotonic, no overshoot)
    - Long gaps (> 30 frames): Linear (safe, no wild oscillations)

    Applied per-axis independently. All valid points are used to fit the
    interpolator, giving global context even for local gap fills.
    """
    result = arr.copy()
    for ax in range(3):
        col = result[:, ax]
        valid_mask = ~np.isnan(col)
        if valid_mask.all() or not valid_mask.any():
            continue

        valid_idx = np.where(valid_mask)[0]
        valid_vals = col[valid_mask]

        # Identify each contiguous NaN gap
        nan_mask = np.isnan(col)
        gaps = _find_gaps(nan_mask)

        for gap_start, gap_end in gaps:
            gap_len = gap_end - gap_start
            gap_indices = np.arange(gap_start, gap_end)

            # Need at least 2 valid points for any interpolation
            if len(valid_idx) < 2:
                col[gap_indices] = valid_vals[0] if len(valid_vals) > 0 else 0.0
                continue

            if gap_len <= 15:
                # CubicSpline: C2 continuous, smooth acceleration
                interp = CubicSpline(valid_idx, valid_vals)
                col[gap_indices] = interp(gap_indices)
            elif gap_len <= 30:
                # PCHIP: C1 continuous, monotonicity-preserving, no overshoot
                interp = PchipInterpolator(valid_idx, valid_vals)
                col[gap_indices] = interp(gap_indices)
            else:
                # Linear: safe for long gaps (e.g., 138-frame gap)
                col[gap_indices] = np.interp(gap_indices, valid_idx, valid_vals)

    return result


def smooth_trajectory_savgol(traj, window_length=7, polyorder=3):
    """Savitzky-Golay smoothing for [N,3] trajectory.

    Zero-phase (no lag in batch mode). Preserves trajectory shape and
    grasp dwell positions better than bidirectional EMA.

    Parameters:
    - window_length=7: 0.7s at 10 FPS. Conservative — preserves dwell.
    - polyorder=3: Cubic — preserves velocity and acceleration shape.
    - mode='nearest': Repeats edge value for endpoint handling.
    """
    if len(traj) < window_length:
        return traj.copy() if isinstance(traj, np.ndarray) else np.array(traj)
    return savgol_filter(traj, window_length, polyorder, axis=0, mode='nearest')
```

**Step 3: Replace the gap-filling + smoothing code in `process_session()` (lines 145-163).** Find this block:
```python
    # Fill gaps with linear interpolation
    arr = []
    for w in wrist_3d_world:
        arr.append(w if w is not None else [np.nan, np.nan, np.nan])
    arr = np.array(arr)

    # Interpolate NaN gaps
    for ax in range(3):
        col = arr[:, ax]
        nans = np.isnan(col)
        if nans.all(): continue
        good = ~nans
        indices = np.arange(len(col))
        col[nans] = np.interp(indices[nans], indices[good], col[good])
        arr[:, ax] = col

    # Smooth
    smoothed = smooth_trajectory(arr.tolist(), alpha=0.12)
    smoothed = np.array(smoothed)
```

Replace with:
```python
    # Convert to array with NaN for missing frames
    arr = []
    for w in wrist_3d_world:
        arr.append(w if w is not None else [np.nan, np.nan, np.nan])
    arr = np.array(arr, dtype=float)

    # TRK-01: Gap-length-aware interpolation (cubic < 15, PCHIP 15-30, linear > 30)
    interpolated = interpolate_gaps(arr)
    assert not np.isnan(interpolated).any(), "NaN gaps remain after interpolation"

    # TRK-02: Savitzky-Golay spatial filter (zero-phase, preserves grasp dwell)
    smoothed = smooth_trajectory_savgol(interpolated, window_length=7, polyorder=3)
    assert not np.isnan(smoothed).any(), "NaN values after smoothing"

    # Validate: check for large jumps
    disps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    max_jump = disps.max()
    n_large_jumps = (disps > 0.03).sum()
    if n_large_jumps > 0:
        print(f"  WARNING: {n_large_jumps} frames with >3cm jump (max={max_jump:.4f}m)")

    # TRK-03: Verify grasping signal remains binary
    unique_grasp = set(np.unique(np.array(grasping, dtype=float)).tolist())
    assert unique_grasp.issubset({0.0, 1.0}), \
        f"Grasping signal corrupted: unique values = {unique_grasp}"
```

**Do NOT touch:** The `grasping` list — it is never smoothed, only passed through. The rest of `process_session()` (loading, reconstruction loop, saving) stays unchanged.
  </action>
  <verify>
Run: `python -c "from reconstruct_wrist_3d import interpolate_gaps, smooth_trajectory_savgol, _find_gaps; print('imports OK')"` to verify the new functions are importable and scipy dependencies resolve.
  </verify>
  <done>
- `smooth_trajectory()` (bidirectional EMA) replaced by `smooth_trajectory_savgol()` (Savitzky-Golay window=7 polyorder=3)
- `np.interp` linear gap-fill replaced by `interpolate_gaps()` with CubicSpline/PCHIP/linear based on gap length
- `_find_gaps()` helper identifies contiguous NaN regions
- Assertions guard: no NaN after interpolation, no NaN after smoothing, grasping is binary
- Warning printed if any frame-to-frame jump exceeds 3cm
  </done>
</task>

<task type="auto">
  <name>Task 2: Validate smoothing with stack2 data</name>
  <files>reconstruct_wrist_3d.py</files>
  <action>
Run the modified `reconstruct_wrist_3d.py` on the stack2 session to verify the new smoothing pipeline produces valid output.

**Step 1:** Run reconstruction:
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python reconstruct_wrist_3d.py stack2
```

**Step 2:** Verify output by running an inline validation:
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python -c "
import json, numpy as np

# Load output
with open('wrist_trajectories/stack2_wrist3d.json') as f:
    data = json.load(f)

smoothed = np.array(data['wrist_world_smooth'])
grasping = data['grasping']

# TRK-01: Zero NaN
assert not np.isnan(smoothed).any(), 'FAIL: NaN gaps remain'
print(f'TRK-01 PASS: Zero NaN gaps ({smoothed.shape[0]} frames)')

# TRK-02: Check jumps
disps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
max_jump = disps.max()
n_big = (disps > 0.03).sum()
print(f'TRK-02: max jump={max_jump:.4f}m, frames >3cm={n_big}')
if n_big == 0:
    print('TRK-02 PASS: No jumps >3cm')
else:
    print(f'TRK-02 WARNING: {n_big} jumps >3cm (may need velocity clamp)')

# TRK-03: Binary grasping
unique = set(np.unique(np.array(grasping, dtype=float)).tolist())
assert unique.issubset({0.0, 1.0}), f'FAIL: grasping not binary: {unique}'
print(f'TRK-03 PASS: Grasping signal binary, unique={unique}')

print('ALL CHECKS PASSED')
"
```

**Step 3:** If TRK-02 still shows >3cm jumps, add a post-smoothing velocity clamp after the savgol line in `process_session()`. Insert this block right after the `smoothed = smooth_trajectory_savgol(...)` line and before the assertion:
```python
    # Velocity clamp: limit frame-to-frame displacement to 3cm
    MAX_STEP = 0.03  # 3cm per frame at 10fps = 0.3 m/s
    for i in range(1, len(smoothed)):
        delta = smoothed[i] - smoothed[i-1]
        norm = np.linalg.norm(delta)
        if norm > MAX_STEP:
            smoothed[i] = smoothed[i-1] + delta * (MAX_STEP / norm)
```
Only add this if the validation in Step 2 shows jumps >3cm. If all jumps are <=3cm, skip this step.

**Step 4:** Re-run validation if velocity clamp was added to confirm all checks pass.
  </action>
  <verify>
The inline Python validation script prints "ALL CHECKS PASSED" — zero NaN, grasping binary, and ideally zero >3cm jumps.
  </verify>
  <done>
- `stack2_wrist3d.json` regenerated with new smoothing pipeline
- TRK-01 validated: zero NaN gaps
- TRK-02 validated: no >3cm jumps (or velocity clamp added if needed)
- TRK-03 validated: grasping signal binary
  </done>
</task>

<task type="auto">
  <name>Task 3: Verify downstream pipeline still works (calibration + STACKED=True)</name>
  <files>reconstruct_wrist_3d.py</files>
  <action>
Run calibration and simulation on the smoothed stack2 data to verify the full pipeline still produces STACKED=True.

**Step 1:** Run calibration:
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python calibrate_workspace.py stack2
```
Verify it completes without error and produces `wrist_trajectories/stack2_calibrated.json`.

**Step 2:** Run simulation:
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python mujoco_g1_v10.py stack2
```
Verify the output contains "STACKED=True".

**Step 3:** If STACKED=False, this is a regression. Check:
- Is the wrist range similar to before? (X ~0.35m, Y ~0.31m, Z ~0.18m in world coords)
- Did the calibration scale change significantly?
- Are there new artifacts in the smoothed trajectory that break the choreographed motion?

The smoothing change should NOT affect STACKED status because:
- The `p` progress variable choreography (lines 337-390 of mujoco_g1_v10.py) is unchanged
- `ls`/`le` are recomputed from the grasping array which is untouched
- Block positions are hardcoded, not trajectory-dependent

If STACKED=False persists after investigation, document the issue but do NOT modify mujoco_g1_v10.py or calibrate_workspace.py — those are out of scope for this plan.
  </action>
  <verify>
Simulation output includes "STACKED=True". Calibrated JSON exists at `wrist_trajectories/stack2_calibrated.json`.
  </verify>
  <done>
- Calibration runs successfully on smoothed data
- Simulation produces STACKED=True (regression check passes)
- Full pipeline integrity confirmed: smoothing changes are backward-compatible
  </done>
</task>

</tasks>

<verification>
Before declaring plan complete:
- [ ] `python -c "from reconstruct_wrist_3d import interpolate_gaps, smooth_trajectory_savgol; print('OK')"` succeeds
- [ ] `wrist_trajectories/stack2_wrist3d.json` exists and has zero NaN in `wrist_world_smooth`
- [ ] Grasping signal in output is binary (only 0.0 and 1.0)
- [ ] No >3cm consecutive-frame jumps (or velocity clamp added)
- [ ] `wrist_trajectories/stack2_calibrated.json` exists
- [ ] Simulation output contains "STACKED=True"
</verification>

<success_criteria>
- All three tasks completed
- All verification checks pass
- No errors or warnings introduced (except tolerable >3cm jump warnings if velocity clamp handles them)
- `reconstruct_wrist_3d.py` uses Savitzky-Golay instead of bidirectional EMA
- `reconstruct_wrist_3d.py` uses gap-length-aware interpolation instead of `np.interp`
- Grasping signal explicitly guarded with assertion
- STACKED=True regression test passes
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundations/01-01-SUMMARY.md`
</output>
