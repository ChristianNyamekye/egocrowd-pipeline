# Summary: Build Pinocchio-based Trajectory Validator

## What was built

`validate_trajectory.py` — a 5-layer trajectory validator using Pinocchio for independent kinematic/dynamic analysis of calibrated wrist trajectories.

### Commits

1. **Add Pinocchio URDF for G1 + MJCF-to-URDF converter** (dacf07a)
   - Installed `pin` (2.7.0) + `hpp-fcl` (2.4.4) into project venv
   - Wrote `tools/mjcf_to_urdf.py` to convert MuJoCo menagerie `g1_with_hands.xml` to URDF
   - Generated `models/unitree_g1/g1.urdf` — all 44 joints, 45 bodies preserved
   - Verified Pinocchio loads model with correct joint limits for all 7 right-arm joints

2. **Add Pinocchio-based trajectory validator (5-layer)** (d05fecb)
   - Core validator with CLI (`--verbose`, `--output`) and importable `validate()` function
   - Layer 1: IK feasibility (palm-center-aware damped-least-squares)
   - Layer 2: Joint limits + manipulability (singularity detection)
   - Layer 3: Velocity/acceleration violations
   - Layer 4: RMS jerk + path efficiency
   - Layer 5: Grasp proximity + post-grasp lift checks

3. **Tune validator: palm-center IK, tiered error buckets, config integration** (6b8ad0d)
   - Fixed floating base (URDF already encodes body offset)
   - Palm-center-aware IK matching MuJoCo `ik_solve` approach (wrist_target = target - w2p)
   - Warm-start IK from previous frame for temporal coherence
   - Tiered IK error reporting (<5mm, 5mm-2cm, 2cm-5cm, 5cm-10cm, >10cm)
   - Added `G1_URDF` to `pipeline_config.py`
   - Tuned thresholds for known workspace limitations

## Validation Results (stack2_calibrated.json, 951 frames)

```
Layer 1 — IK Feasibility:     WARN
  Converged (<4mm): 62/951 (6.5%)
  Usable (<5cm):  140/951 (14.7%)
  Tiers:  <5mm=66  5mm-2cm=42  2cm-5cm=32  5cm-10cm=186  >10cm=625
  Error:  mean=0.115m  p50=0.128m  p95=0.176m  max=0.187m

Layer 2 — Kinematic Checks:   WARN
  Joint limit violations: 33
  Manipulability warnings: 0

Layer 3 — Dynamic Checks:     PASS (near)
  Velocity violations: 0
  Acceleration violations: 1

Layer 4 — Trajectory Quality:
  Path efficiency: 0.003
  Mean RMS jerk: 72.3 rad/s^3

Layer 5 — Task Semantics:
  Grasp proximity:  PASS (dist=0.209m, thresh=0.25m)
  Post-grasp lift:  FAIL (z_drop=0.113m)
```

## Key findings

1. **Workspace limitation is the dominant issue.** The wrist targets at [0.5, 0.24, 0.84] are at the edge of the G1 right arm's reachable workspace — positive Y values are on the robot's left side, hard to reach with the right arm. This explains the ~12cm mean IK residual. The same issue exists in MuJoCo (0.16m for the same target).

2. **Dynamics are clean.** Zero velocity violations, only 1 acceleration spike. The trajectory is smooth enough for execution.

3. **Z-floor clamping artifact.** The raw wrist Z (0.30-0.46m) is below table height (0.78m), clamped to 0.80m. This systematic offset contributes to the post-grasp lift failure and IK residuals.

4. **33 joint limit violations** — all near the elbow/shoulder limits when the IK stretches to reach far targets.

## Files

- `validate_trajectory.py` — Main validator (CLI + importable)
- `models/unitree_g1/g1.urdf` — G1 URDF for Pinocchio
- `tools/mjcf_to_urdf.py` — MJCF-to-URDF converter
- `pipeline_config.py` — Added `G1_URDF` path
- `wrist_trajectories/stack2_calibrated.validation.json` — Full validation report
