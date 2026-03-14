# Phase 3, Plan 02: Grasp Visual Quality — Summary

**Status:** Code complete, pending verification and commits
**Date:** 2026-03-14
**Files modified:** `mujoco_g1_v10.py`

## What was done

### Task 1: Add finger pre-shaping constants
- Added `FINGER_PRESHAPE = np.array([0.2, -0.25, -0.3, 0.4, 0.45, 0.4, 0.45])` — 50% of CLOSED for anticipatory curl
- Added `PRESHAPE_DIST_START = 0.20` — begin pre-shaping at 20cm from nearest block
- Added `PRESHAPE_DIST_FULL = 0.06` — full pre-shape at 6cm (block diameter)

### Task 2: Distance-based finger pre-shaping
- Replaced binary `FINGER_CLOSED if want_grip else FINGER_OPEN` with three-phase control:
  - `want_grip=True`: Full closure with fast blend (0.25)
  - `want_grip=False`, near block: Graduated pre-shaping via quadratic ease-in (`preshape_t^2`)
  - `want_grip=False`, far from block: Return to open with medium blend (0.20)
- Pre-shaping computes `min_block_dist` from palm center to nearest block body
- `preshape_t` interpolates linearly between `PRESHAPE_DIST_START` and `PRESHAPE_DIST_FULL`, squared for ease-in

### Task 3: Collision exclusion during kinematic hold
- On GRASP: Set `model.geom_contype[gi] = 0` and `model.geom_conaffinity[gi] = 0` for all geoms belonging to the grasped block body
- On RELEASE: Restore `contype=1` and `conaffinity=1` before clearing `grasped_bid`
- Follows the existing pattern from the settle-phase collision disable (lines 254-265)

### Task 4: Validation
- **Pending:** Simulation must be run (`python mujoco_g1_v10.py stack2`) to verify STACKED=True

## Requirements addressed

| Requirement | Description | Status |
|-------------|-------------|--------|
| OUT-03 | Fingers visibly begin closing during descent approach | Code complete |
| OUT-04 | No visible finger-block interpenetration during hold | Code complete |
| OUT-05 | STACKED=True still passes | Pending verification |

## Verification checklist

- [x] `FINGER_PRESHAPE` constant defined (line 41)
- [x] `PRESHAPE_DIST_START` and `PRESHAPE_DIST_FULL` defined (lines 44-45)
- [x] `preshape_t` distance-based computation in finger control block (lines 447-453)
- [x] `geom_contype[gi] = 0` in ATTACH block (line 399)
- [x] `geom_contype[gi] = 1` in RELEASE block (line 421)
- [ ] STACKED=True in simulation output (pending run)
- [ ] Output video shows graduated finger closure

## Manual steps required

Run these commands to complete the plan:

```bash
# 1. Run simulation test
source .venv/bin/activate && python mujoco_g1_v10.py stack2

# 2. Commit Task 1-3 changes
node "$HOME/.claude/get-shit-done/bin/gsd-tools.cjs" commit "feat(sim): add finger pre-shaping and collision exclusion for grasp visual quality (OUT-03, OUT-04)" --files mujoco_g1_v10.py

# 3. Commit planning files
node "$HOME/.claude/get-shit-done/bin/gsd-tools.cjs" commit "docs(03-02): summary, state, and roadmap updates for grasp quality plan" --files .planning/phases/03-quality-uplift/03-02-SUMMARY.md .planning/STATE.md .planning/ROADMAP.md
```

## Key design decisions

1. **Dynamic contype toggle over static XML exclude** — toggling block contype/conaffinity to 0 during hold is simpler, follows existing code pattern, and only affects the grasped block
2. **Quadratic ease-in for pre-shaping** — `preshape_t^2` creates gentle initial finger curl that accelerates as hand approaches, matching natural human grasping
3. **Separate blend rates** — 0.25 for grasp closure (responsive), 0.15 for pre-shaping (anticipatory), 0.20 for return-to-open (medium)
4. **Collision restore before state clear** — `contype=1` is set before `grasped_bid = None` to ensure correct geom identification
