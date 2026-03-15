# QT-011: Fix grasping signal — post-hoc temporal filter + tighten HaMeR threshold

## Problem

HaMeR's grasping detection (`thumb-index tip < 4cm`) fires on 66% of frames (630/951). After trimming to 300 frames, 100% are `grip=True`. The robot fingers never open — there's no open-approach -> close-grasp transition, which physics grasping requires.

## Tasks

### Task 1: Add `clean_grasping_signal()` to `trim_trajectory.py`

Post-process raw grasping array BEFORE action window detection:

1. **Debounce**: Require minimum 5 consecutive grasping frames (0.5s at 10fps) — short bursts become `False`
2. **Find sustained onset**: Locate the FIRST run of 5+ consecutive `True` frames — this is the real grasp onset
3. **Clear pre-onset**: Set all frames before that onset to `grip=False`
4. **Call it early**: Insert `clean_grasping_signal()` at the top of `detect_action_window()` before any other processing, AND in `trim_calibrated_data()` before slicing the grasping array

**File**: `trim_trajectory.py`

**Acceptance**:
- After cleaning, the grasping array has a clear `False -> True` transition
- Approach phase (first ~30-40 frames of trimmed window) has `grip=False`
- Existing fallback chain in `detect_action_window` still works

### Task 2: Tighten HaMeR grasping threshold from 0.04 -> 0.02

Update `_estimate_grasping_from_joints()` threshold constant from `0.04` (4cm) to `0.02` (2cm). This is a complementary fix for future Modal runs — doesn't affect current data without re-running HaMeR, but prevents the problem from recurring.

**File**: `processing/hamer_modal.py` line ~206

**Acceptance**:
- Threshold changed to 0.02
- Docstring updated to reflect 2cm

### Task 3: Add diagnostic logging

Add a print statement in `clean_grasping_signal()` showing before/after stats:
- Original grip=True count and percentage
- Cleaned grip=True count and percentage
- Onset frame index

This helps verify the filter is working on current and future data.

**File**: `trim_trajectory.py` (inside the new function)
