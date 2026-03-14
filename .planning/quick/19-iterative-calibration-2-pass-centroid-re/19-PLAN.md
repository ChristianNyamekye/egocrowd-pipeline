# QT-019: Iterative calibration — 2-pass centroid refinement with proximity filter

## Problem
QT-018 showed debounce-only centroid (558 frames) is nearly identical to raw centroid (630 frames). Both span the whole trajectory, producing the same biased offset. Need to filter to only frames where the wrist is actually near the objects.

## Task 1: Add 2-pass calibration refinement

**File:** calibrate_workspace.py

After the initial (pass 1) offset application, add pass 2:
1. For each grasping frame, compute XY distance from rough wrist_sim to nearest object
2. Keep only frames within 20cm (CALIB_PROXIMITY)
3. If >= 10 frames pass, recompute centroid and offset from filtered frames
4. Replace wrist_sim with refined-offset version

**Acceptance:** Centroid refined from much fewer, proximity-verified frames. Better alignment should reduce RMS error and palm-to-block minimum distance.
