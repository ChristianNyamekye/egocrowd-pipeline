# QT-008: Fix trim to include approach phase before grasp onset

## Problem

`detect_action_window` in `trim_trajectory.py` picks the densest grasping cluster via sliding window, but with HaMeR's 632 grasping frames (vs MediaPipe's 72), `focus_start` is already deep inside a continuous grasping block. Subtracting `margin_before` (30 frames) from `focus_start` still lands inside the grasping region, producing a trim window where ALL frames have `grip=True` and no approach phase exists.

**stack2_hamer data:** 951 frames, grasping starts ~frame 200, trim window [568, 837) is 100% grip=True.

## Tasks

### Task 1: Add grasp-onset backtracking to detect_action_window

**File:** `/Users/christian/Documents/ai_dev/flexa-pipeline/trim_trajectory.py`

**Changes to the sliding-window branch (lines 109-126):**

After line 120 (`focus_start = int(action_frames[best_start_idx])`), add grasp-onset backtracking:

1. From `focus_start`, walk backwards through the `grasping` array to find the first non-grasping frame (grasp onset boundary)
2. Set `approach_start = onset_frame - approach_pad` where `approach_pad = 30` (3s of approach motion)
3. Use `approach_start` instead of `focus_start - margin_before` as the window start
4. Keep `focus_end + margin_after` as the window end
5. If total duration exceeds `target_max`, truncate the **end** (keep the approach, sacrifice late grip)

**Also fix the non-sliding-window path (lines 75-76):**

Same logic: if `action_frames[0]` is already a grasping frame, find the actual onset boundary before it and pad from there instead of blindly subtracting `margin_before`.

**Logic pseudocode:**
```python
def find_approach_start(grasping, action_start, approach_pad=30):
    """Walk backwards from action_start to find where grasping begins, then pad."""
    onset = action_start
    while onset > 0 and grasping[onset - 1] > 0:
        onset -= 1
    return max(0, onset - approach_pad)
```

Apply this in both the sliding-window branch and the normal-margin branch when the computed start would still be inside a grasping region.

### Task 2: Verify with stack2_hamer pipeline run

```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && source .venv/bin/activate
python run_pipeline.py --r3d raw_captures/stack2/stack2.r3d \
  --robot g1 --task stack --session stack2_hamer \
  --hamer --trim \
  --objects '[[0.5, 0.01, 0.425], [0.5, -0.01, 0.425]]'
python mujoco_g1_v10.py stack2_hamer
```

**Success criteria:**
- Trim window starts before grasp onset (~frame 170 or earlier)
- First N frames of trimmed trajectory have `grip=False` (approach phase present)
- Simulation still runs (robot approaches, grasps, moves block)
- Duration stays within 15-30s target range

### Task 3: Commit

Commit with concise message describing the fix.

## Key Constraints

- Do not change fallback chains or grasping detection logic
- Keep `target_min`/`target_max` duration enforcement
- When exceeding `target_max`, prefer keeping approach phase and truncating the end
- The fix must also work for the MediaPipe case (fewer grasping frames) without regression
