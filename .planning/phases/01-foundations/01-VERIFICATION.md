---
phase: 1
status: passed
date: 2026-03-14
---

# Phase 1: Foundations — Verification

## Must-Haves

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Smoothed wrist trajectory has zero NaN gaps and no jumps >3cm between consecutive frames | PASS | `stack2_wrist3d.json`: 951 frames, 0 NaN. `stack2_calibrated.json`: 229 frames, 0 NaN, max jump 0.0064m. Velocity clamp caps at 0.03m/frame. |
| 2 | Grasping signal remains binary (0 or 1) after smoothing — never interpolated | PASS | `unique(grasping) = {0.0, 1.0}` in both wrist3d and calibrated JSONs. Runtime assertion in `reconstruct_wrist_3d.py:242` guards this. |
| 3 | Output video starts within 1s of first grasp approach and ends within 2s of last release | PARTIAL | Trimmed data starts 30 frames (3.0s) before first grasp and ends 49 frames (4.9s) after last release. This matches the OUT-01 requirement spec ("first grasp - 30 frames, last grasp + 50 frames") but **exceeds the roadmap success criterion** (1s before, 2s after). See Gaps section. |
| 4 | Output video duration is between 15 and 30 seconds for the test recording (stack2.r3d) | PASS | `trim_info.duration_s = 22.9` (229 frames at 10fps). Within 15-30s target. |
| 5 | STACKED=True still passes with smoothed + trimmed data | PASS | Per 01-01-SUMMARY: `STACK CHECK: top_z=0.870, bot_z=0.810, gap=0.060, expected=0.060 STACKED=True`. Output: `sim_renders/stack2_g1_v10.mp4` (194 KB). |

## Requirement Coverage

| REQ-ID | Description | Status | Evidence |
|--------|-------------|--------|----------|
| TRK-01 | Trajectory gaps interpolated using gap-length-aware strategy (cubic spline <15, PCHIP 15-30, linear >30) | PASS | `interpolate_gaps()` in `reconstruct_wrist_3d.py:88-134` implements all three strategies with `_find_gaps()` helper. `CubicSpline`, `PchipInterpolator` imported from scipy. Runtime assertion at line 218 confirms zero NaN post-interpolation. |
| TRK-02 | Spatial trajectory smoothed with Savitzky-Golay filter (zero-phase, preserves grasp dwell) | PASS | `smooth_trajectory_savgol()` in `reconstruct_wrist_3d.py:137-150` uses `savgol_filter(window=7, polyorder=3, mode='nearest')`. Velocity clamp (MAX_STEP=0.03m) at lines 224-229 prevents teleportation. |
| TRK-03 | Grasping signal preserved as binary (never smoothed) | PASS | Grasping array bypasses all smoothing functions. Assertion at `reconstruct_wrist_3d.py:242` guards binary constraint. Verified: `unique(grasping) = {0.0, 1.0}`. |
| OUT-01 | Video auto-trimmed to action window (first grasp - 30 frames to last grasp + 50 frames) | PASS | `detect_action_window()` in `trim_trajectory.py:13-128` implements velocity+grasping detection with 3-level fallback. Margins: 30 frames before, 50 frames after. Sliding-window density search for duration enforcement. |
| OUT-02 | Output video is 15-30 seconds (not 95 seconds) | PASS | stack2 trimmed from 951 frames (95.1s) to 229 frames (22.9s). `trim_info` in calibrated JSON confirms. Duration enforcement logic at `trim_trajectory.py:78-127`. |

## Human Verification

- **Criterion 3 margin discrepancy**: The roadmap success criterion says "starts within 1s of first grasp approach, ends within 2s of last release." The OUT-01 requirement says "first grasp - 30 frames to last grasp + 50 frames." These are contradictory at 10fps (30 frames = 3s, 50 frames = 5s). Decide which spec is authoritative. If the tighter criterion is desired, reduce `margin_before` from 30 to 10 and `margin_after` from 50 to 20 in `detect_action_window()`.
- **Visual inspection**: Confirm `sim_renders/stack2_g1_v10.mp4` shows correct block-stacking behavior with no visual artifacts from the velocity clamp.

## Gaps

**1. Success Criterion 3 vs OUT-01 margin mismatch**
- The roadmap success criterion (1s before first grasp, 2s after last release) conflicts with the OUT-01 requirement definition (30 frames = 3s before, 50 frames = 5s after).
- Current implementation follows OUT-01 spec faithfully.
- **Resolution needed**: Either relax criterion 3 to match OUT-01 margins, or tighten `detect_action_window()` margins to 10/20 frames.
- **Impact**: Low — the trimmed duration (22.9s) is well within the 15-30s target regardless. The extra margin provides useful context for retargeting.

**2. REQUIREMENTS.md traceability not updated**
- All five Phase 1 requirement IDs (TRK-01, TRK-02, TRK-03, OUT-01, OUT-02) still show `Pending` in REQUIREMENTS.md traceability table.
- Should be updated to `Complete` with plan references.
