# QT-007: Pinocchio Validator Diagnostics + Full Pipeline Re-test

**Type:** diagnostics / verification (no code changes)
**Created:** 2026-03-14

## Context

QT-006 fixed HaMeR calibration by switching from crop-relative wrist_3d_camera to depth unprojection. The calibrated trajectory now exists at `wrist_trajectories/stack2_hamer_calibrated.json`. This task validates the fix by:
1. Running the Pinocchio validator on the HaMeR trajectory
2. Re-running the full 7-stage pipeline end-to-end with `--hamer`
3. Running MuJoCo simulation and checking STACKED status

**Baseline:** MediaPipe trajectory had 0.115m mean IK error.

---

## Task 1: Pinocchio Validator + Full Pipeline Re-test + Simulation Analysis

**Goal:** Run all three diagnostics sequentially and document findings.

### Steps

1. **Run Pinocchio validator** on existing HaMeR calibrated trajectory:
   ```bash
   cd /Users/christian/Documents/ai_dev/flexa-pipeline && source .venv/bin/activate
   python validate_trajectory.py wrist_trajectories/stack2_hamer_calibrated.json --verbose
   ```
   Capture: mean IK error, max error, frame-by-frame breakdown, comparison to MediaPipe baseline (0.115m).

2. **Re-run full pipeline** end-to-end with fixed calibration:
   ```bash
   cd /Users/christian/Documents/ai_dev/flexa-pipeline && source .venv/bin/activate
   python run_pipeline.py --r3d raw_captures/stack2/stack2.r3d \
     --robot g1 --task stack --session stack2_hamer \
     --hamer --trim \
     --objects '[[0.5, 0.01, 0.425], [0.5, -0.01, 0.425]]'
   ```
   This runs all 7 stages: ingest, HaMeR hand tracking (Modal GPU), object detection, 3D wrist reconstruction, calibration, trim, simulation. **Expect 5-10 min for Modal remote call.**

3. **Run MuJoCo simulation** on the freshly generated trajectory:
   ```bash
   cd /Users/christian/Documents/ai_dev/flexa-pipeline && source .venv/bin/activate
   python mujoco_g1_v10.py stack2_hamer
   ```
   Capture: STACKED=True/False result.

4. **Analyze results:**
   - Compare HaMeR IK error to MediaPipe baseline (0.115m)
   - Check STACKED=True or STACKED=False
   - If STACKED=False: analyze wrist-to-object distance at grasp onset, IK error patterns, Z offset
   - Document all findings in SUMMARY.md

### Success Criteria
- Pinocchio validator output captured with IK error metrics
- Full pipeline completes all 7 stages without error
- Simulation result (STACKED status) documented
- All findings written to SUMMARY.md

### Estimated Context
~25% (command output capture and analysis, no code reading/editing)
