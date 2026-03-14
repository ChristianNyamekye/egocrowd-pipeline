---
phase: 01-foundations
plan: 02
type: execute
wave: 1
depends_on: []
files_modified: [trim_trajectory.py, run_pipeline.py]
autonomous: true
requirements: [OUT-01, OUT-02]

must_haves:
  truths:
    - "Trimmed video duration is between 15 and 30 seconds for stack2"
    - "Trimmed output starts within ~30 frames of first action-cluster grasp"
    - "Trimmed output ends within ~50 frames of last action-cluster grasp"
    - "Pipeline orchestrator supports --trim flag"
    - "Simulation auto-adapts to trimmed array length (n=len(wrist))"
    - "STACKED=True still passes after trimming"
  artifacts:
    - path: "trim_trajectory.py"
      provides: "Action window detection and calibrated data trimming"
      contains: "detect_action_window"
    - path: "run_pipeline.py"
      provides: "Trim stage integration between calibration and simulation"
      contains: "trim"
  key_links:
    - from: "trim_trajectory.py"
      to: "wrist_trajectories/*_calibrated.json"
      via: "trim_calibrated_data() modifies calibrated JSON in-place"
      pattern: "trim_info"
    - from: "run_pipeline.py"
      to: "trim_trajectory.py"
      via: "import and call trim_and_save()"
      pattern: "from trim_trajectory import"
---

<objective>
Create a video trimming module that auto-detects the action window from grasping signal + wrist velocity, and integrate it into the pipeline between calibration and simulation.

Purpose: The current 95-second renders waste iteration time. Trimming to the 15-30 second action window speeds up every subsequent test cycle and focuses the output on the meaningful manipulation.
Output: New `trim_trajectory.py` module + modified `run_pipeline.py` with trim stage and `--trim` CLI flag.
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

Relevant source files:
@run_pipeline.py
@calibrate_workspace.py
@mujoco_g1_v10.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create trim_trajectory.py with action window detection</name>
  <files>trim_trajectory.py</files>
  <action>
Create a new file `trim_trajectory.py` in the project root (`/Users/christian/Documents/ai_dev/flexa-pipeline/trim_trajectory.py`) with the following exact content:

```python
"""Trim calibrated trajectory data to the action window.

Detects the meaningful manipulation segment using grasping signal + wrist velocity,
then slices the calibrated JSON arrays to that window.

Inserted between calibration (stage 5) and simulation (stage 6) in the pipeline.
"""
import json
import numpy as np
from pathlib import Path


def detect_action_window(wrist_sim, grasping, fps=10):
    """Detect the action window using grasping signal + wrist velocity.

    Returns (start_frame, end_frame) for trimming.

    Strategy:
    1. Compute wrist velocity with rolling mean (1-second window)
    2. Find sustained high-velocity region (top 30% of non-zero velocity)
    3. Find grasping frames within that region
    4. Pad with margins: 30 frames before, 50 frames after
    5. Enforce 15-30 second duration (150-300 frames at 10fps)

    Fallback chain:
    - velocity + grasping -> late grasping (>60%) -> velocity alone
    """
    wrist_sim = np.array(wrist_sim, dtype=float)
    grasping = np.array(grasping, dtype=float)
    n = len(wrist_sim)

    if n == 0:
        return 0, 0

    # Wrist velocity: frame-to-frame displacement with 1-second rolling mean
    disps = np.linalg.norm(np.diff(wrist_sim, axis=0), axis=1)
    vel = np.zeros(n)
    vel[1:] = disps
    window = max(1, fps)  # 1-second rolling window
    kernel = np.ones(window) / window
    rolling_vel = np.convolve(vel, kernel, mode='same')

    # High-velocity region: top 30% of non-zero velocity
    nonzero_vel = rolling_vel[rolling_vel > 0]
    if len(nonzero_vel) > 0:
        vel_threshold = np.percentile(nonzero_vel, 70)
        high_vel = rolling_vel > vel_threshold
    else:
        high_vel = np.ones(n, dtype=bool)

    # Find grasping frames in high-velocity region
    g = grasping > 0
    action_frames = np.where(g & high_vel)[0]

    if len(action_frames) == 0:
        # Fallback 1: use last 40% of grasping frames (mimics sim's late-grasp filter)
        grasp_idx = np.where(g)[0]
        if len(grasp_idx) > 0:
            cutoff = grasp_idx[int(len(grasp_idx) * 0.6)]
            action_frames = grasp_idx[grasp_idx >= cutoff]

    if len(action_frames) == 0:
        # Fallback 2: velocity alone (no grasping signal)
        action_frames = np.where(high_vel)[0]

    if len(action_frames) == 0:
        # Fallback 3: use entire trajectory
        print("  TRIM WARNING: No action window detected, using full trajectory")
        return 0, n

    # Margins from OUT-01 spec: "first grasp - 30 frames to last grasp + 50 frames"
    margin_before = 30   # 3 seconds at 10fps
    margin_after = 50    # 5 seconds at 10fps

    start = max(0, int(action_frames[0]) - margin_before)
    end = min(n, int(action_frames[-1]) + margin_after)

    # Enforce 15-30 second duration (150-300 frames at 10fps)
    duration = end - start
    target_min = 150  # 15s
    target_max = 300  # 30s

    if duration < target_min:
        # Expand symmetrically
        deficit = target_min - duration
        expand_before = deficit // 2
        expand_after = deficit - expand_before
        start = max(0, start - expand_before)
        end = min(n, end + expand_after)
        # Re-check if still short (hit array boundary)
        if (end - start) < target_min:
            # Expand from whichever side has room
            remaining = target_min - (end - start)
            if start > 0:
                start = max(0, start - remaining)
            else:
                end = min(n, end + remaining)
    elif duration > target_max:
        # Tighten: reduce margins but keep at least 10 frames before and 20 after action
        excess = duration - target_max
        reduce_before = min(excess // 2, max(0, start - max(0, int(action_frames[0]) - 10)))
        reduce_after = min(excess - reduce_before, max(0, end - (int(action_frames[-1]) + 20)))
        start += reduce_before
        end -= reduce_after

    return int(start), int(end)


def trim_calibrated_data(calib_path, start=None, end=None, fps=10):
    """Trim calibrated JSON arrays to the action window.

    If start/end not provided, auto-detects using detect_action_window().
    Modifies wrist_sim and grasping arrays in-place.
    Adds trim_info for traceability.

    Args:
        calib_path: Path to calibrated JSON file
        start: Start frame index (auto-detected if None)
        end: End frame index (auto-detected if None)
        fps: Frames per second (for duration calculation)

    Returns:
        dict with trim_info: {original_frames, start_frame, end_frame, trimmed_frames, duration_s}
    """
    calib_path = Path(calib_path)
    with open(calib_path) as f:
        calib = json.load(f)

    wrist_sim = calib["wrist_sim"]
    grasping = calib["grasping"]
    original_frames = len(wrist_sim)

    # Auto-detect window if not specified
    if start is None or end is None:
        start, end = detect_action_window(wrist_sim, grasping, fps=fps)

    # Validate bounds
    start = max(0, min(start, original_frames))
    end = max(start, min(end, original_frames))
    trimmed_frames = end - start

    if trimmed_frames == 0:
        print("  TRIM WARNING: Zero frames after trim, keeping full trajectory")
        return {"original_frames": original_frames, "start_frame": 0,
                "end_frame": original_frames, "trimmed_frames": original_frames,
                "duration_s": original_frames / fps}

    # Slice arrays
    calib["wrist_sim"] = wrist_sim[start:end]
    calib["grasping"] = grasping[start:end]

    # Add traceability metadata
    trim_info = {
        "original_frames": original_frames,
        "start_frame": start,
        "end_frame": end,
        "trimmed_frames": trimmed_frames,
        "duration_s": round(trimmed_frames / fps, 1),
    }
    calib["trim_info"] = trim_info

    # Write back
    with open(calib_path, "w") as f:
        json.dump(calib, f)

    return trim_info


def trim_and_save(session_name, calib_dir=None, start=None, end=None, fps=10):
    """High-level trim function for pipeline integration.

    Args:
        session_name: Session name (e.g., 'stack2')
        calib_dir: Directory containing calibrated JSONs (default: wrist_trajectories/)
        start: Manual start frame (auto-detect if None)
        end: Manual end frame (auto-detect if None)
        fps: Frames per second

    Returns:
        trim_info dict, or None on failure
    """
    if calib_dir is None:
        from pipeline_config import CALIB_DIR
        calib_dir = CALIB_DIR

    calib_path = Path(calib_dir) / f"{session_name}_calibrated.json"
    if not calib_path.exists():
        print(f"  ERROR: Calibrated file not found: {calib_path}")
        return None

    print(f"  Trimming: {calib_path}")
    trim_info = trim_calibrated_data(calib_path, start=start, end=end, fps=fps)

    print(f"  Original: {trim_info['original_frames']} frames ({trim_info['original_frames']/fps:.1f}s)")
    print(f"  Trimmed:  {trim_info['trimmed_frames']} frames ({trim_info['duration_s']}s)")
    print(f"  Window:   [{trim_info['start_frame']}, {trim_info['end_frame']})")

    return trim_info


if __name__ == "__main__":
    import sys
    session = sys.argv[1] if len(sys.argv) > 1 else "stack2"

    # Support optional manual start/end
    start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    end = int(sys.argv[3]) if len(sys.argv) > 3 else None

    result = trim_and_save(session, start=start, end=end)
    if result:
        print(f"\n  Trim complete: {result['trimmed_frames']} frames ({result['duration_s']}s)")
    else:
        print("\n  Trim failed")
        sys.exit(1)
```

**Key design decisions:**
- `detect_action_window()` uses combined velocity + grasping signal with a 3-level fallback chain
- `trim_calibrated_data()` modifies the calibrated JSON in-place (no new file)
- `trim_info` added to JSON for traceability (IP3 risk mitigation)
- `trim_and_save()` is the pipeline entry point, handles path resolution and logging
- CLI support via `__main__` for standalone testing with optional manual start/end overrides
  </action>
  <verify>
Run: `python -c "from trim_trajectory import detect_action_window, trim_and_save; print('imports OK')"` to verify the module loads.
  </verify>
  <done>
- `trim_trajectory.py` created with `detect_action_window()`, `trim_calibrated_data()`, and `trim_and_save()`
- Action window detection uses velocity + grasping with fallback chain
- Trim enforces 15-30 second duration (150-300 frames at 10fps)
- `trim_info` metadata added to calibrated JSON for traceability
  </done>
</task>

<task type="auto">
  <name>Task 2: Integrate trim stage into run_pipeline.py</name>
  <files>run_pipeline.py</files>
  <action>
Modify `run_pipeline.py` to add a trim stage between calibration (stage 5) and simulation (stage 6), controlled by a `--trim` CLI flag.

**Step 1: Update the total stage count in `run_r3d_pipeline()`.** Find `total = 6` (line 73) and change to `total = 7`.

**Step 2: Add the trim stage between calibration and simulation.** Find this block in `run_r3d_pipeline()` (approximately lines 162-167):
```python
    validate_file(calib_path, "Calibrated JSON")
    log_stage(5, total, "Workspace calibration", "done")

    # Stage 6: Simulation
    log_stage(6, total, f"Run {robot} simulation -> video")
    video_path = run_simulation(robot, session_name, task)
    log_stage(6, total, f"Run {robot} simulation", "done")
```

Replace with:
```python
    validate_file(calib_path, "Calibrated JSON")
    log_stage(5, total, "Workspace calibration", "done")

    # Stage 6: Trim trajectory (optional, enabled by --trim)
    if trim_enabled:
        log_stage(6, total, "Trim trajectory -> action window")
        from trim_trajectory import trim_and_save
        trim_info = trim_and_save(
            session_name,
            start=trim_start,
            end=trim_end,
        )
        if trim_info:
            duration = trim_info['duration_s']
            print(f"  Trimmed to {trim_info['trimmed_frames']} frames ({duration}s)")
        else:
            print("  WARNING: Trim failed, using full trajectory")
        log_stage(6, total, "Trim trajectory", "done")
    else:
        log_stage(6, total, "Trim trajectory", "skip")

    # Stage 7: Simulation
    log_stage(7, total, f"Run {robot} simulation -> video")
    video_path = run_simulation(robot, session_name, task)
    log_stage(7, total, f"Run {robot} simulation", "done")
```

**Step 3: Update `run_r3d_pipeline()` function signature.** Find:
```python
def run_r3d_pipeline(r3d_path, robot, task, session_name=None, objects_manual=None):
```
Replace with:
```python
def run_r3d_pipeline(r3d_path, robot, task, session_name=None, objects_manual=None,
                     trim_enabled=False, trim_start=None, trim_end=None):
```

**Step 4: Add CLI arguments in `main()`.** Find this block in `main()` (approximately lines 366-371):
```python
    parser.add_argument("--session", type=str, default=None,
                        help="Session name (default: derived from filename)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: sim_renders/)")
```

Add after it:
```python
    parser.add_argument("--trim", action="store_true", default=False,
                        help="Trim trajectory to action window (15-30s)")
    parser.add_argument("--trim-start", type=int, default=None,
                        help="Manual trim start frame (overrides auto-detection)")
    parser.add_argument("--trim-end", type=int, default=None,
                        help="Manual trim end frame (overrides auto-detection)")
```

**Step 5: Pass trim args to `run_r3d_pipeline()`.** Find:
```python
        video = run_r3d_pipeline(args.r3d, args.robot, args.task, args.session, args.objects)
```

Replace with:
```python
        video = run_r3d_pipeline(
            args.r3d, args.robot, args.task, args.session, args.objects,
            trim_enabled=args.trim,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
        )
```

**Do NOT modify:** The synthetic pipeline (`run_synthetic()`), any existing stages (1-5), the simulation function, or any other part of the file.
  </action>
  <verify>
Run: `python run_pipeline.py --help` and verify the output includes `--trim`, `--trim-start`, and `--trim-end` arguments.
  </verify>
  <done>
- `run_pipeline.py` has trim stage (stage 6) between calibration and simulation
- `--trim` flag enables trimming (default: off for backward compatibility)
- `--trim-start` and `--trim-end` provide manual override
- Stage numbering updated: total=7, simulation is now stage 7
- Synthetic pipeline unchanged
  </done>
</task>

<task type="auto">
  <name>Task 3: Validate trimming with stack2 end-to-end</name>
  <files>trim_trajectory.py, run_pipeline.py</files>
  <action>
Test the trim module standalone and verify the full pipeline still produces STACKED=True after trimming.

**Step 1: Re-run calibration to get a fresh (untrimmed) calibrated JSON.** The calibrated JSON may have been modified by Plan 01's validation. Re-calibrate:
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python calibrate_workspace.py stack2
```

**Step 2: Run trim standalone on stack2:**
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python trim_trajectory.py stack2
```
Verify output shows:
- Original frames: ~951
- Trimmed frames: 150-300 (15-30 seconds)
- Window: approximately [547, 776] or similar based on action detection

**Step 3: Validate the trimmed calibrated JSON:**
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python -c "
import json, numpy as np

with open('wrist_trajectories/stack2_calibrated.json') as f:
    calib = json.load(f)

n = len(calib['wrist_sim'])
duration = n / 10
print(f'Frames: {n}')
print(f'Duration: {duration:.1f}s')
assert 15 <= duration <= 30, f'FAIL: Duration {duration:.1f}s outside 15-30s'
print(f'OUT-02 PASS: Duration {duration:.1f}s is within 15-30s')

assert 'trim_info' in calib, 'FAIL: No trim_info in calibrated JSON'
ti = calib['trim_info']
print(f'trim_info: {ti}')
print(f'OUT-01 PASS: trim_info present with start={ti[\"start_frame\"]}, end={ti[\"end_frame\"]}')

# Verify grasping still has grasp frames (not all trimmed away)
g = np.array(calib['grasping'], dtype=float)
n_grasp = (g > 0).sum()
print(f'Grasping frames in trimmed window: {n_grasp}')
assert n_grasp > 0, 'FAIL: No grasping frames in trimmed window'

print('ALL TRIM CHECKS PASSED')
"
```

**Step 4: Run simulation on the trimmed data:**
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python mujoco_g1_v10.py stack2
```
Verify:
- Output contains "STACKED=True"
- Simulation runs with the trimmed frame count (150-300 frames, not 951)

**Step 5: If STACKED=False after trimming:**
The simulation's `ls`/`le` computation auto-adapts to the trimmed array. If STACKED=False:
- Check that the trimmed window actually contains the grasping cluster
- The `late` filter (frames > 60% of n) may select different frames in the trimmed array
- If the trim window is correct but `ls`/`le` are wrong, adjust `margin_before` or `margin_after` in `detect_action_window()` — NOT the simulation code

**Step 6: Verify the trim also works via the pipeline flag (if the R3D file is available):**
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python run_pipeline.py --help | grep -A1 trim
```
Confirm the `--trim` flag is documented in help output. Do NOT run the full pipeline (it requires the .r3d file and takes minutes) — the standalone test above is sufficient.
  </action>
  <verify>
- Trimmed duration is 15-30 seconds
- `trim_info` present in calibrated JSON
- Simulation output includes "STACKED=True" with trimmed data
- `--trim` flag visible in pipeline help
  </verify>
  <done>
- Trim module produces valid 15-30 second output for stack2
- `trim_info` traceability metadata present in calibrated JSON
- STACKED=True regression test passes with trimmed data
- Pipeline CLI includes --trim, --trim-start, --trim-end flags
  </done>
</task>

</tasks>

<verification>
Before declaring plan complete:
- [ ] `trim_trajectory.py` exists and imports cleanly
- [ ] `python trim_trajectory.py stack2` produces 150-300 frame output
- [ ] Trimmed calibrated JSON contains `trim_info` with `original_frames`, `start_frame`, `end_frame`
- [ ] Duration of trimmed output is 15-30 seconds
- [ ] `python mujoco_g1_v10.py stack2` outputs "STACKED=True" on trimmed data
- [ ] `python run_pipeline.py --help` shows `--trim` flag
- [ ] No existing pipeline stages (1-5) were modified
</verification>

<success_criteria>
- All three tasks completed
- All verification checks pass
- Output video duration for stack2 is 15-30 seconds (not 95 seconds)
- STACKED=True regression test passes
- `trim_trajectory.py` created as new standalone module
- `run_pipeline.py` updated with trim stage (stage 6) and CLI flags
- Backward compatible: without `--trim` flag, pipeline behaves exactly as before
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundations/01-02-SUMMARY.md`
</output>
