# Quick Task 4: Replace kinematic attachment with physics-based grasping

**Status:** DONE
**Commit:** 4e94895
**Date:** 2026-03-14

## What was done

### Task 1: Remove kinematic attachment, implement contact-based grasping

All kinematic block attachment code was gutted from `mujoco_g1_v10.py` and replaced with pure contact friction grasping:

**Removed:**
- Kinematic state variables: `grasped_obj`, `grasped_bid`, `grasped_jnt_adr`, `grasp_offset`, `grasp_start_pos`, `grasp_quat`, `grasp_age`
- Post-release settle state: `placed_jnt_adr`, `placed_dof_adr`, `placed_pos`, `placed_countdown`, `SETTLE_FRAMES`
- ATTACH block (palm proximity + kinematic grasp initiation + collision exclusion)
- RELEASE block (stack-position teleport + collision restoration)
- `grasp_age` increment
- Substep kinematic block tracking (palm-following with smooth blend + velocity zeroing)
- Settle countdown (placed_countdown decrement + state clear)
- Dead code: `smoothstep()` function, `BLEND_FRAMES` constant

**Added/Modified:**
- Block geom friction: `3.0 0.02 0.002` with `condim="4"` (torsional + rolling friction)
- Right-hand finger geoms: friction `3.0 0.02 0.002` + `condim=4` set at runtime
- `FINGER_CLOSED` scaled 1.2x for tighter wrap: `[0.48, -0.60, -0.72, 0.96, 1.08, 0.96, 1.08]`
- `FINGER_GAIN_MULTIPLIER`: 25.0 -> 40.0
- `blend_rate` for grip: 0.25 -> 0.4 (faster finger closure)
- `SUBSTEPS` minimum: 10 -> 25 (more stable contact dynamics)
- Docstring updated to reflect contact-based grasping
- Debug print cleaned (removed `attached=` reference to deleted state)

### Task 2: Validation

Simulation ran successfully: `python mujoco_g1_v10.py stack2`

**Result: STACKED=False** -- This is the expected honest result.

**Observations:**
- No crashes, no kinematic cheats, no collision exclusion during grasp
- Both blocks remain stable on table at z=0.810
- Neither block was lifted -- the hand trajectory doesn't bring the palm close enough to block_a (pick block) for contact grip. Minimum distance to pick block was ~0.15m (never within contact range)
- The palm gets closer to block_b (support, ~0.044m at frame 680) but that block is pinned
- This is an honest signal: the current trajectory/block placement doesn't produce a successful physical grasp

**Why the grasp fails (root cause):**
The trajectory's grip phase moves the wrist through a path that passes near block_b (support) but never achieves close-enough contact with block_a (pick). With kinematic attachment, a 0.15m proximity threshold was enough to "teleport attach" the block. With pure physics, the fingers need to physically wrap around the block, requiring much closer approach (~0.03m).

**No tuning was applied** because the gap is fundamental (trajectory-block alignment), not a friction/gain issue. Per user decision: "Let it fail visibly -- STACKED=False is honest signal that trajectory needs work."

## Acceptance criteria

- [x] Zero kinematic qpos writes to block during grasp
- [x] Zero collision exclusion during grasp (contype/conaffinity stay at 1)
- [x] Fingers physically wrap around block (FINGER_CLOSED values with 1.2x scaling)
- [x] Block held by MuJoCo friction solver only (no kinematic cheats)
- [x] Simulation runs without crash
- [x] STACKED=False is honest (no reintroduced kinematic cheats)
