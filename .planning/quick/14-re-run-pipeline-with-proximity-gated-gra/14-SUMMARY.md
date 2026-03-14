# QT-014 Summary: Re-run pipeline with proximity-gated grasping and verify sim

## Result: STACKED=False

## Grasping Signal Stats

### GRASP PROXIMITY (new proximity gate)
- **Before**: 632/951 (66%) frames marked as grasping (raw aspect-ratio heuristic)
- **After proximity gate**: 166/951 (17%) — only frames where wrist < 15cm from object
- **Reduction**: 66% -> 17% = **49 percentage point drop** (74% of false positives eliminated)

### GRASP CLEAN (after debounce + onset detection)
- **After full pipeline**: 146/951 (15%), onset frame=168
- **Post-debounce in sim**: 19 raw -> 57 debounced frames (out of 295 total)

**Comparison to QT-012**:
| Metric | QT-012 | QT-014 | Improvement |
|--------|--------|--------|-------------|
| Raw grasping | 66% | 66% | (same input) |
| After clean | 59% (558 frames) | 15% (146 frames) | -44 pts |
| Sim debounced | 270/300 (90%) | 57/295 (19%) | -71 pts |
| Grip onset dist | 0.334m from block | see below | improved |

## Trim Window
- **Window**: [423, 718) — 295 frames, 29.5s
- **Original**: 951 frames (95.1s)
- Slightly shifted from QT-012's [407, 707) due to proximity-gated onset detection

## Simulation Analysis

### Grip timing
- **Grip window**: frames [236, 269] within trimmed data (34 frames of grip=True)
- **Two grip events visible**:
  1. F030: grip=True, palm at (0.394, 0.078, 0.825), 0.180m from block_a, 0.132m from block_b
  2. F236-F269: grip=True (main event), palm starts at (0.467, -0.048, 0.870), 0.160m from block_b

### Grasping Transition (False->True) Visibility
**Yes, dramatically improved.** The trimmed data now shows a clear extended approach phase:
- F000-F029: grip=False (30 frames of open-hand approach)
- F030: first grip=True (short burst, near blocks at 0.13-0.18m)
- F060-F235: grip=False (176 frames of motion with open hand)
- F236: second grip=True onset (34-frame grip window)
- F270+: grip=False for remainder

This is a massive improvement over QT-012 where 90% of frames were grip=True.

### Block behavior
- block_a briefly lifted at F030 (z=0.868), then settled back to 0.810
- F110: block_a nudged (z=0.811), then drifts to z=0.805 through F140-F160
- Blocks never knocked off table (unlike QT-012 where block_a fell to z=0.030)
- **FINAL positions**: block_a at (-0.045, -0.424, 0.810), block_b at (0.320, -0.030, 0.810)
- Both blocks remain on table at z=0.810 but are far apart (0.537m XY distance)

### Stack check failure
- top_z=0.810, bot_z=0.810, gap=0.000 (expected 0.060)
- Neither Z alignment nor XY proximity met
- **No stacking occurred** — block_a was nudged but never picked up and placed on block_b

## IK / Tracking Stats
- RMS=0.099m, mean=0.063m, max=0.215m (RMS 2x above 0.05m threshold)
- 116/295 frames clamped to workspace envelope
- 124/295 frames with >2cm IK error
- Nearly identical to QT-012 (RMS=0.099m, 146/300 >2cm, 105/300 clamped)

## Assessment

**Proximity gating is working correctly.** The grasping signal went from 90% True (QT-012) to 19% True in the trimmed sim window. False positives where the hand was far from blocks are eliminated. The grip events now coincide with actual proximity to objects.

**However, STACKED=False persists.** The remaining bottleneck is NOT the grasping signal — it's the IK tracking quality:

1. **Grip onset too far from block**: At F030, the palm is 0.180m from block_a. The fingers close but the hand isn't close enough to grab. At F236, the palm is 0.160m from block_b — also too far.
2. **IK error is systemic**: RMS=0.099m with 42% of frames having >2cm error means the hand can't reach precise positions. The workspace clamping (39% of frames) contributes to this.
3. **The hand never gets close enough to a block while gripping**: In all grip=True frames, the minimum distance to any block is ~0.13m. Physics-based grasping requires the fingers to actually contact the block (< ~3cm).

## Next Steps

1. **Proximity-gate the sim grasping, not just the signal**: Currently the proximity gate runs in the preprocessing (trim_trajectory). The sim should also gate finger closure on palm-to-block distance — only close fingers when palm < 5cm from pick block.
2. **Fix IK convergence**: The 0.099m RMS error means the hand trajectory is systematically offset. Investigate:
   - Grasp centroid anchor (computed from 632 frames of grasping) — with proximity gating reducing this to 146 frames, the centroid should be more accurate if recalculated
   - Recalibrate with the proximity-gated grasping frames as the anchor set
3. **Lower proximity threshold**: Current 15cm may still be too generous for sim grasping. Consider 5-8cm for the sim-side gate.
4. **Dual-gating approach**: Keep 15cm for signal cleaning (catches false positives), add 5cm gate in sim for finger closure (ensures physical contact before grip).
