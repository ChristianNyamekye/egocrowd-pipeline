# QT-008 Summary: Fix trim to include approach phase before grasp onset

## Code Change

**File:** `trim_trajectory.py`

Added `_find_approach_start(grasping, action_start, approach_pad, gap_tolerance)` helper that walks backwards from a grasping frame through the grasping signal, skipping small non-grasping gaps (< 10 frames), until it finds a sustained non-grasping region. This handles HaMeR's fragmented grasping signal (many small on/off bursts vs MediaPipe's cleaner signal).

Applied in two places:
1. **Normal-margin path** (line 97): When `action_frames[0]` is a grasping frame, uses `_find_approach_start` instead of blind `- margin_before`.
2. **Sliding-window path** (line 151): Uses `_find_approach_start` instead of `focus_start - margin_before`. If the resulting window exceeds `target_max`, truncates the END (not the start), preserving the approach phase.

## New Trim Window

| Metric | Before (QT-007) | After (QT-008) |
|--------|-----------------|-----------------|
| Window | [568, 837) | [403, 703) |
| Duration | 269 frames (26.9s) | 300 frames (30.0s) |
| F000 grip | True | **False** |
| Approach frames | 0 | ~40 (4 seconds) |

## Approach Phase

The first 4 seconds (F000-F039) now have `grip=False`. The approach phase is present, with a large non-grasping gap from offset 12-29 (frames 415-432, 18 frames) providing clear pre-grasp approach motion.

## STACKED Result

**STACKED=False.** Blocks remain on the table side-by-side (separation=0.120m). This is a pre-existing issue unrelated to trimming:
- IK RMS error = 0.0796m (above 0.05m threshold)
- 271/300 frames have >2cm IK error
- Root cause: calibration scale (0.222) compresses the workspace, Z-floor clamping creates offset

## Remaining Issues

- STACKED=False persists -- calibration/IK accuracy is the bottleneck, not trim window selection
- RMS tracking error increased slightly (0.067 -> 0.080) with wider window, likely because earlier frames have less precise wrist positions
- Grasping debounce reduces 232 raw -> 266 debounced grip frames (some fragmentation remains in the trimmed window)

## Commit

`a72ec7b` on branch `fix/e2e-pipeline`
