# QT-018 Summary: Re-run pipeline and sim with cleaned calibration anchor (attempt 4)

## Pipeline Results

### Calibration Step
- **GRASP CLEAN:** 630/951 (66%) -> 558/951 (59%), onset frame=86
- **Grasp centroid (558 frames):** (-0.678, 0.267, -0.011)
- **Offset:** (1.178, -0.267, 0.436)
- Debounce reduced frame count from 630 to 558 (72 frames removed, 11.4% reduction)

### Trim Step
- **GRASP PROXIMITY:** 558/951 (59%) -> 147/951 (15%) (threshold=0.15m)
- **GRASP CLEAN:** 558/951 (59%) -> 138/951 (15%), onset frame=168
- **Trim window:** [424, 715), 291 frames (29.1s)

### Simulation Results
- **Grasping debounce:** 15 raw -> 53 debounced frames
- **Grip window:** [235, 265] (30-frame grip zone)
- **grip GATED by proximity:** 6 events (F030, F040, F050, F240, F250, F260)
- **Grip passed gate (palm < 5cm):** 0 — palm never reached close enough
- **Minimum palm-block distance:** ~0.087m at F100 (block_a), ~0.121m at F200 (block_b)
- **block_a:** knocked off table at ~F100 (z drops to 0.030), never recovered
- **block_b:** remained on table at z=0.810 throughout
- **RMS tracking error:** 0.0984m (mean=0.0629m, max=0.2168m)
- **IK failures:** 131/291 frames with >2cm error (45%)
- **Clamped frames:** 113/291 (39%)
- **STACKED:** False

### Final Block Positions
- block_a: (1.195, -0.377, 0.030) — fell off table
- block_b: (0.320, -0.030, 0.810) — on table but displaced

## Comparison Table

| Metric | QT-016 (before) | QT-018 (after) | Delta |
|--------|-----------------|----------------|-------|
| Calib grasp frames | 630 | 558 | -72 (-11.4%) |
| RMS error | 0.099m | 0.098m | -0.001m (negligible) |
| Min palm-block dist | ~9.6cm | ~8.7cm | -0.9cm (marginal) |
| Grip passed gate | 0 | 0 | no change |
| Clamped frames | 39% | 39% | no change |
| IK failures (>2cm) | 42% | 45% | +3% (slightly worse) |
| STACKED | False | False | no change |

## Analysis

The calibration debounce from QT-017 had **minimal impact** on end-to-end results:

1. **Grasp centroid shift was small:** 558 vs 630 frames produced nearly the same centroid, so the calibration offset barely changed.
2. **RMS error unchanged:** 0.098m vs 0.099m — the IK/calibration bottleneck is not caused by noisy grasp frames in the centroid calculation.
3. **Palm never reaches block:** Minimum distance ~8.7cm (was ~9.6cm) — the 5cm proximity gate threshold is still never crossed.
4. **block_a still knocked off table** at ~F100 by the sweeping arm motion, same failure mode as earlier attempts.

## Root Cause (unchanged)

The fundamental issue remains **IK/calibration precision**. The robot's palm trajectories don't track the human's closely enough to bring the hand within grasping distance of the blocks. With 45% IK failures and 39% workspace clamping, the trajectory is severely distorted from the original human motion.

**Next steps would need to address:**
- Calibration offset accuracy (the transform from R3D world coords to sim coords)
- IK solver convergence (possibly switching to a better solver or relaxing workspace constraints)
- Z-axis alignment (systematic Z offset between human wrist and sim table height)
