# Quick Task 4: Replace kinematic attachment with physics-based grasping

**Status:** DONE
**File:** `mujoco_g1_v10.py`
**Estimated context:** ~25%

---

## Task 1: Remove kinematic attachment system and implement contact-based grasping

**What:** Gut the kinematic block teleportation code and replace it with pure friction-based grasping where fingers physically hold the block.

### Remove (lines ~426-624)

1. **Delete kinematic state variables** (lines 426-439):
   - `grasped_obj`, `grasped_bid`, `grasped_jnt_adr`, `grasp_offset`, `grasp_start_pos`, `grasp_quat`, `grasp_age`
   - `placed_jnt_adr`, `placed_dof_adr`, `placed_pos`, `placed_countdown`, `SETTLE_FRAMES`

2. **Delete ATTACH block** (lines 483-510):
   - Palm proximity detection + kinematic grasp initiation
   - Collision exclusion on grasp (contype/conaffinity = 0)

3. **Delete RELEASE block** (lines 512-546):
   - Stack-position placement teleport
   - Collision restoration
   - Placed state reset

4. **Delete grasp_age increment** (lines 549-550)

5. **Delete kinematic block tracking in substep loop** (lines 596-613):
   - `grasped_jnt_adr` palm-tracking with smooth blend
   - Velocity zeroing during kinematic hold

6. **Delete settle countdown** (lines 618-624):
   - `placed_countdown` decrement + state clear

7. **Delete collision exclusion hack** (settle phase lines 363-375):
   - The 2-phase settle that disables block collision is still useful for initial settling, but the save/restore of contype/conaffinity for kinematic purposes goes away. Keep the settle steps but remove collision toggling.

### Add / Modify

8. **Tune friction for reliable contact grip:**
   - Block geom friction: increase to `3.0 0.02 0.002` (high sliding friction)
   - Add `condim="4"` to block geoms (enable torsional + rolling friction)
   - Add finger geom friction override: find right-hand finger geoms, set friction to `3.0 0.02 0.002` and `condim="4"` at runtime via `model.geom_friction` and `model.geom_condim`

9. **Increase SUBSTEPS** for contact stability:
   - Current formula yields ~50 (at FPS=10, TIMESTEP=0.002). Increase minimum to 25 to ensure stable contact dynamics. Consider `SUBSTEPS = max(25, min(50, ...))`.

10. **Faster finger closure during grasp:**
    - Increase `blend_rate` for `want_grip=True` from 0.25 to 0.4 (fingers need to close quickly before block slips)
    - Increase `FINGER_CLOSED` values slightly for tighter wrap: scale by ~1.2x
    - Increase `FINGER_GAIN_MULTIPLIER` from 25.0 to 40.0 for stronger grip force

11. **Support block: start pinned, unpin on release** (per user decision to use Claude's discretion):
    - Keep support block pinned (current behavior at lines 584-588) during the entire simulation for stability. This is already implemented.
    - The pick block is fully dynamic (free joint) and must be held by friction alone.

12. **Remove support block settle-pin in substep if placed block was support** — simplify: support stays pinned always, pick block is physics-only.

13. **Update docstring** at top of file: replace "Kinematic block attachment" description with "Contact-based physics grasping".

### Verification

- Run `python mujoco_g1_v10.py stack2`
- Block should be lifted by finger contact (no teleportation)
- If block drops, that's an honest STACKED=False (acceptable per user decision)
- If block is carried and placed, STACKED=True
- Visual check: no block-through-finger clipping, no sudden teleportation

### Acceptance criteria

- Zero kinematic qpos writes to block during grasp (no `data.qpos[grasped_jnt_adr] = ...`)
- Zero collision exclusion during grasp (contype/conaffinity stay at 1)
- Fingers physically wrap around block (FINGER_CLOSED values)
- Block held by MuJoCo friction solver only
- Simulation runs without crash

---

## Task 2: Validate and tune (if Task 1 block drops)

**What:** If the block consistently drops during grasp, iteratively tune friction/gain parameters.

### Tuning levers (in order of priority)

1. `FINGER_CLOSED` angles — tighter wrap
2. `FINGER_GAIN_MULTIPLIER` — stronger actuator force
3. Block `mass` — reduce from 0.3 to 0.1 if needed
4. `friction` coefficients on block + finger geoms
5. `solimp` / `solref` on block geoms for softer contact (more grip surface)
6. `SUBSTEPS` — increase if contact is unstable

### Acceptance criteria

- Block stays in hand for majority of grip phase
- No kinematic cheats reintroduced
