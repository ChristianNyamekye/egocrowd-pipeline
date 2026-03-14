# QT-015 Summary: Sim-side proximity gate

## Changes (all in `mujoco_g1_v10.py`)

1. **Added constant** `SIM_GRIP_PROXIMITY = 0.05` (line 45) — 5cm threshold for sim-side finger closure.

2. **Inserted proximity gate** (lines 481-488) between IK convergence tracking and finger control block:
   - Computes `min_block_dist_grip` = min distance from palm center (`pc`) to any block body
   - If `want_grip` is True but distance > 5cm, overrides `want_grip = False`
   - Uses actual sim-space palm position (not trajectory-space wrist), so it's more accurate than the trajectory-side 15cm gate (QT-013)

3. **Added diagnostic logging** (lines 545-547) in the F%10 block:
   - Detects when grip was requested by grasping signal but suppressed by proximity gate
   - Prints `grip GATED by proximity` with actual distance vs threshold

## How it works

The two-layer gating stack:
- **Layer 1 (trajectory-side, QT-013):** Grasping signal suppressed when wrist > 15cm from object centroid during calibration
- **Layer 2 (sim-side, QT-015):** Even if grasping signal says grip, fingers stay open until sim palm < 5cm from nearest block

This prevents the closed-fist sweep problem (QT-012) where fingers closed 13-18cm from blocks, knocking them off the table.

## Commit

- `5fc98f7` — Add sim-side proximity gate

## Verification needed

Run `python mujoco_g1_v10.py stack2` to confirm:
- F%10 logs show `grip GATED` messages during approach
- Fingers stay open until palm is within 5cm
- Blocks not knocked off table by premature finger closure
