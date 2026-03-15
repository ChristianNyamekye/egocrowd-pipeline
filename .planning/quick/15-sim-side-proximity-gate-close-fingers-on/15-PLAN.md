# QT-015: Sim-side proximity gate — close fingers only when palm < 5cm from block

## Context

QT-014 showed grip events fire at 0.13-0.18m from blocks — too far for physics contact. The trajectory-side proximity gate (QT-013, 15cm threshold) cleaned up the grasping signal (66%->17%) but the remaining grip events still occur before the palm is close enough for the sim's fingers to actually wrap around a block. A sim-side gate using the actual palm body position (`pc` from line 474) will suppress finger closure until the palm is within 5cm of the nearest block in simulation space.

## Task

**Add a sim-side proximity gate that overrides `want_grip` to False when the palm is more than 5cm from any block.**

### Changes (all in `mujoco_g1_v10.py`)

1. **Add constant** `SIM_GRIP_PROXIMITY = 0.05` near line 44 (after `PRESHAPE_DIST_FULL`).

2. **Insert proximity gate** between line 478 (IK error tracking) and line 481 (`if want_grip:`):
   ```python
   # Sim-side proximity gate: only close fingers when palm is near a block
   if want_grip:
       min_block_dist = min(
           np.linalg.norm(data.xpos[bid] - pc) for bid in obj_body_ids
       )
       if min_block_dist > SIM_GRIP_PROXIMITY:
           want_grip = False
   ```

3. **Add diagnostic print** in the F%10 logging block (~line 524-534). Include proximity gate status:
   - Show `prox_gate=True/False` (whether the gate overrode grip) and `min_dist` when grip was requested.

### Verification

- Run `python mujoco_g1_v10.py stack2` and check:
  - F%10 logs show `prox_gate=True` (gate suppressing) until palm is within 5cm
  - Fingers stay open during approach, close only near block contact
  - Compare grip timing vs QT-014 (grip events were at 0.13-0.18m, should now be deferred)

## Acceptance

- [ ] `SIM_GRIP_PROXIMITY = 0.05` constant added
- [ ] Proximity gate inserted before finger control block
- [ ] Diagnostic logging shows gate status
- [ ] Simulation runs without errors
