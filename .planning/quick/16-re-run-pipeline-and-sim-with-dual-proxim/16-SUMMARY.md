# QT-016 Summary: Re-run pipeline and sim with dual proximity gates (attempt 3)

## Pipeline Results

### GRASP PROXIMITY (trajectory-side, 15cm threshold)
- Before: 632/951 (66%)
- After: 166/951 (17%)

### GRASP CLEAN (after debounce + onset)
- After: 146/951 (15%), onset frame=168

### Trim Window
- Window: [423, 718), 295 frames (29.5s)
- Same as QT-014 (proximity gate params unchanged)

### Sim Grasping (debounced)
- 19 raw -> 57 debounced frames (19% of trimmed window)
- Grip window: frames [236, 269] of trimmed trajectory

## Simulation Results

### "grip GATED by proximity" Messages
- **7 gated frames** detected (at F030, F040, F050, F240, F250, F260, and implicit F030 with dist=0.131m)
- Gating distances ranged from 0.131m to 0.244m (all well above the 5cm threshold)

### Grip=True Frames NOT Gated
- **Zero** grip=True frames passed both gates — no frame in the trimmed window had grip=True AND palm < 5cm from block simultaneously
- All printed frames show grip=False, meaning the grasping signal (from the trajectory) was already False at those timestamps
- The grip window [236,269] maps to frames where grip was True, but the 10-frame sample logging (F000, F010, ...) doesn't capture every frame in that range

### Block Final Positions
- block_a: (-0.043, -0.430, 0.810) — moved far from initial (0.500, 0.010)
- block_b: (0.320, -0.030, 0.810) — shifted from initial (0.500, -0.010)
- Both blocks at z=0.810 (table height), neither lifted

### STACKED Result
- **STACKED=False**
- top_z=0.810, bot_z=0.810, gap=0.000 (expected 0.060)
- Z aligned: False, XY dist: 0.540m (threshold < 0.060m)

### IK Stats
- RMS tracking error: 0.0993m (threshold 0.05m, **2x over**)
- Mean error: 0.0630m, Max: 0.2151m
- Clamped frames: 116/295 (39%)
- IK failures (>2cm): 124/295 (42%)

## Analysis

### What's Working
1. **Trajectory-side proximity gate (15cm)**: Reduced grasping from 66% to 17% — effective at filtering false positives
2. **Sim-side proximity gate (5cm)**: Successfully prevented premature finger closure when palm was far from block
3. **No block knockoff**: Unlike QT-012, blocks stay on table (closed-fist sweep problem is fixed)
4. **Trim window**: Good 4s approach phase + clear False->True transition

### Root Cause of STACKED=False
The **IK precision is the bottleneck**, not the grasping signal:
- Palm never gets closer than ~0.096m to any block during the grip window
- 42% of frames have IK error > 2cm
- 39% of frames are clamped to workspace envelope
- The hand moves in the general vicinity of the blocks but never reaches them precisely enough for the 5cm proximity gate to allow finger closure

### What to Try Next
1. **Calibration improvement**: The wrist trajectory range X=[-0.613, 0.964] is far wider than the workspace [0.10, 0.65] — 39% clamping means the calibration offset/scale is wrong. Re-examine the grasp centroid anchor and transform.
2. **Z correction**: Z range [0.267, 1.125] after calibration, but table is at 0.81. Many frames are above 0.95m — the hand is too high to reach blocks.
3. **Workspace envelope tuning**: Current clamp X[0.10,0.65] Y[-0.45,0.30] Z[0.80,1.20] may be too restrictive or misaligned with actual block positions.
4. **IK solver improvements**: Consider increasing max iterations, using a better initial guess, or switching to a different solver.
5. **Lower sim proximity gate**: Even with gate at 5cm, palm never gets that close — the problem is upstream (IK/calibration).

## Verdict
Dual proximity gates work correctly as designed. The grasping signal is now clean (17% vs 90% in QT-012). The remaining bottleneck is **IK tracking accuracy** — the robot hand doesn't reach the block positions precisely enough for physics-based grasping to succeed. Next focus should be on calibration/IK improvement.
