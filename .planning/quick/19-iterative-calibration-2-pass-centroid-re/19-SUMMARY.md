# QT-019 Summary: Iterative calibration — 2-pass centroid refinement

## Result: STACKED=False

## Calibration Pass 2 Stats
- Pass 1: 565 grasping frames → centroid at (-0.682, 0.262, -0.012)
- Pass 2: 247/565 frames within 20cm of objects
- Centroid shift: (-0.011, -0.041, -0.031) — minimal change
- RMS: 0.0975m (was 0.098m in QT-018)

## Simulation
- block_a knocked off table at ~F290 (same failure as previous attempts)
- Palm minimum distance to any block: 9.3cm (F270)
- Zero grip=True frames passed the 5cm sim gate
- 112/293 (38%) frames clamped to workspace, 117/293 (40%) with >2cm IK error

## Analysis
The 2-pass refinement barely shifted the centroid because 247 frames (44% of grasping) were within 20cm after rough alignment — still a large, spread-out set.

**Root cause identified:** The palm trajectory systematically misses the blocks in Y. Palm Y ranges from -0.4 to +0.2, spending most time at Y=-0.2 to -0.4. Blocks are at Y=0.01 and Y=-0.01. There's a ~20-30cm systematic Y offset between where the palm goes and where the blocks are.

The arm makes incidental contact (sweeping through) but never deliberately reaches a block. This is a calibration alignment issue AND a reach mismatch issue (human arm range > G1 workspace).

## Next Step
Need **attraction bias during grasping** — when grip=True and palm is within ~15cm of a block, bias the IK target toward the block. This preserves the motion character while ensuring contact for physics grasping. Implement in mujoco_g1_v10.py's frame loop.
