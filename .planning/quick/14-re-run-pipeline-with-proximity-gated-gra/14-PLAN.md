# QT-014: Re-run pipeline with proximity-gated grasping and verify sim

## Context
QT-013 added proximity gating to `clean_grasping_signal()` — grasping only counts when wrist is < 0.15m from an object. Previous runs showed 66% raw grasping (aspect-ratio heuristic unreliable), with grip onset 0.334m from block. Proximity gate should dramatically cut false positives.

## Tasks

### Task 1: Run pipeline + analyze grasping stats
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && source .venv/bin/activate
python run_pipeline.py --r3d raw_captures/stack2/stack2.r3d \
  --robot g1 --task stack --session stack2_hamer \
  --hamer --trim \
  --objects '[[0.5, 0.01, 0.425], [0.5, -0.01, 0.425]]'
```
**Check:** `GRASP PROXIMITY:` and `GRASP CLEAN:` log lines. Expect grasping % to drop well below 59% (QT-012 value). Note trim window, frame counts, IK stats.

### Task 2: Run simulation + report results
```bash
python mujoco_g1_v10.py stack2_hamer
```
**Check:** STACKED=True/False. Observe whether robot approaches block before gripping, whether grip timing aligns with proximity to block.

## Success Criteria
- Pipeline completes all stages
- Proximity gate reduces grasping frames significantly vs QT-012 (was 59%)
- Document STACKED result and full diagnostics

## Expected Output
- 14-FINDINGS.md with grasping stats, sim result, and next steps
