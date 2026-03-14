# Plan: Build Pinocchio-based Trajectory Validator

## Context

Build `validate_trajectory.py` — a standalone multi-layer trajectory validator using Pinocchio for kinematics/dynamics checks. Takes calibrated JSON from `wrist_trajectories/` and produces a per-frame pass/fail report.

**Critical blocker:** No URDF exists for G1 in `mujoco_menagerie/unitree_g1/` (only MJCF `.xml`). Pinocchio requires URDF. Unitree publishes URDFs in their `unitree_ros` / `unitree_ros2` repos. We'll download the official G1 URDF from GitHub or convert from the MJCF XML if needed.

**Dependencies to install:** `pin` (Pinocchio), `hpp-fcl` (collision checking) — neither is in the project venv.

## Tasks

### Task 1: Setup — Install Pinocchio + Obtain G1 URDF
**File:** (no code file — environment setup)

1. Install Pinocchio and HPP-FCL into project venv:
   ```
   .venv/bin/pip install pin hpp-fcl
   ```
2. Obtain G1 URDF: Download from Unitree's official `unitree_ros2` GitHub repo (`unitree_ros2/unitree_ros2/robots/g1_description/urdf/`). Place at `models/unitree_g1/g1.urdf` (with mesh assets).
   - If official URDF unavailable, use `mujoco` Python API to export URDF from `g1.xml` via `mujoco.save_last_xml` or use the `mujoco2urdf` tool.
3. Verify Pinocchio can load the model:
   ```python
   import pinocchio as pin
   model = pin.buildModelFromUrdf("models/unitree_g1/g1.urdf")
   print(model.nq, model.nv, [model.names[i] for i in range(model.njoints)])
   ```
4. Confirm the 7 right-arm joints from `mujoco_g1_v10.py` exist in the Pinocchio model:
   - `right_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `right_shoulder_yaw_joint`
   - `right_elbow_joint`
   - `right_wrist_roll_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint`

**Exit:** `import pinocchio` works, model loads, 7 arm joints identified by name.

### Task 2: Build validate_trajectory.py — Core Validator
**File:** `validate_trajectory.py`

Build the full validator with layered checks. Architecture:

```
validate_trajectory.py <calibrated_json> [--verbose] [--output report.json]
```

**Input format** (from `stack2_calibrated.json`):
- `wrist_sim`: list of [x, y, z] positions (951 frames)
- `grasping`: list of booleans (951 frames)
- `objects_sim`: list of [x, y, z] object positions
- `r3d_to_sim`: calibration metadata

**Implementation layers:**

**Layer 1 — IK solve per frame:**
- For each `wrist_sim[i]`, solve IK using `pin.computeFrameJacobian` + damped-least-squares
- Use same 7 DOF right arm joints and `SEED` configuration as `mujoco_g1_v10.py`
- Record: joint angles `q[i]`, IK residual error, convergence status
- EE frame: `right_wrist_yaw_link` (matching MuJoCo script's `wrist_bid`)

**Layer 2 — Kinematic checks (per-frame):**
- **Joint limits:** Compare `q[i]` against `model.lowerPositionLimit` / `model.upperPositionLimit`
- **Self-collision:** Use `pin.computeCollisions(model, data, geom_model, geom_data, q)` with HPP-FCL geometry
- **Manipulability:** Compute `det(J @ J.T)` at each frame; flag if near-singular (< threshold)

**Layer 3 — Dynamic checks (frame-to-frame):**
- **Velocity:** `dq = (q[i] - q[i-1]) * FPS` — check against `model.velocityLimit`
- **Acceleration:** `ddq = (dq[i] - dq[i-1]) * FPS` — flag if exceeds reasonable bounds (e.g., 50 rad/s^2)

**Layer 4 — Trajectory quality (aggregate):**
- **Smoothness (jerk):** `d3q = diff(q, 3) * FPS^3` — compute RMS jerk per joint
- **Path length efficiency:** ratio of Euclidean distance (start-to-end) vs. total arc length of EE trajectory

**Layer 5 — Task semantics:**
- **Grasp onset:** At first `grasping[i] == True`, check EE is within `GRASP_RADIUS` (e.g., 8cm) of nearest `objects_sim`
- **Post-grasp lift:** For frames after grasp onset, check EE Z increases (or at least doesn't drop significantly)
- **Pre-release approach:** Before `grasping` transitions False->True again (place), check EE approaches the target location

**Output:** JSON report with:
```json
{
  "summary": { "total_frames": 951, "ik_converged": 940, "joint_limit_violations": 3, ... },
  "per_frame": [ { "frame": 0, "ik_error": 0.002, "joint_limits_ok": true, ... }, ... ],
  "trajectory_quality": { "rms_jerk": [...], "path_efficiency": 0.43 },
  "task_semantics": { "grasp_proximity_ok": true, "post_grasp_lift_ok": true }
}
```

Also print a human-readable summary to stdout with pass/fail per layer.

**Exit:** `python validate_trajectory.py wrist_trajectories/stack2_calibrated.json` runs and produces report.

### Task 3: Run Validation + Fix Issues
**File:** `validate_trajectory.py` (refinements)

1. Run the validator on `stack2_calibrated.json`
2. Interpret results — expect some known issues:
   - Z-floor clamping at 0.80m (table height) will show up in IK residuals
   - 108/229 frames had >2cm IK error in MuJoCo (known from STATE.md)
3. Tune thresholds if defaults are too aggressive/lenient:
   - IK convergence threshold
   - Manipulability singularity threshold
   - Acceleration bounds
4. Add `pipeline_config.py` integration for model paths
5. Ensure the script is importable (for future pipeline integration): expose `validate(json_path) -> report_dict`

**Exit:** Clean run producing meaningful report. Script is both CLI-usable and importable.

## Verification

```bash
# Full validation run
python validate_trajectory.py wrist_trajectories/stack2_calibrated.json --verbose

# Quick smoke test
python -c "from validate_trajectory import validate; r = validate('wrist_trajectories/stack2_calibrated.json'); print(r['summary'])"
```
