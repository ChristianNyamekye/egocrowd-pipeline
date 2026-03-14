# Pitfalls Research: Flexa Pipeline v0.3

Research context: Adding true wrist retargeting, HaMeR hand pose estimation, trajectory smoothing, video trimming, and grasp visual quality to the existing v0.2 pipeline that already produces end-to-end block stacking simulations with choreographed motion.

---

## Critical Pitfalls

### P1: Choreographed-to-Real Trajectory Switch Breaks Grasp Timing

**Feature:** True wrist retargeting
**What goes wrong:** The current v10 simulation uses a hardcoded progress variable `p` (lines 324-390 of `mujoco_g1_v10.py`) to orchestrate approach/descend/dwell/lift/carry/place phases. When switching to real human trajectory data, the hand does not follow this choreographed timeline. The real wrist may approach the block from unexpected angles, hover at wrong heights, or move erratically. The grasp trigger (`want_grip`) is currently tied to `p` thresholds (e.g., `p > 0.25` for grip), not to actual proximity or grasp intent from the tracking data. Result: the block attachment fires when the hand is nowhere near the block, or never fires at all.
**Warning signs:** Block teleports to hand position mid-air; block never gets picked up despite hand being above it; GRASP log message appears at wrong frame.
**Prevention:** Decouple grasp triggering from the progress variable entirely. Use a state machine driven by (a) palm-to-block proximity from the real trajectory + (b) grasping signal from hand tracking data. Keep the `want_grip` logic based on the `grasping` array from calibrated data, not from `p` thresholds. Test with at least 3 different R3D recordings that have different timing.
**When to address:** Phase 1 (true wrist retargeting) -- this is the core architectural change.

### P2: Human Workspace Exceeds Robot Reachable Workspace

**Feature:** True wrist retargeting
**What goes wrong:** The current `calibrate_workspace.py` scales wrist trajectories to a `target_range` of 0.35m and offsets them to the sim workspace center. But humans have ~0.7m arm reach while the G1's right arm has roughly 0.5m effective reach from shoulder. After calibration, some wrist positions may land outside the IK-solvable volume -- especially lateral (Y) and high-Z motions. The IK solver in v10 (damped least squares with `0.005 * np.eye(3)` damping) will silently converge to the nearest reachable point, causing the palm to "stick" at workspace boundaries while the trajectory data keeps moving. Retargeting research confirms that scaling artifacts are the primary source of motion distortion.
**Warning signs:** IK residual (`np.linalg.norm(err)`) never drops below the 0.004 threshold; arm locks in fully extended pose; wrist position logs show the palm is not tracking the target.
**Prevention:** After calibration, clamp all wrist trajectory points to a validated reachable envelope (e.g., sphere of radius 0.45m centered at shoulder). Log IK convergence failures per frame. Add a workspace boundary check before IK solve -- if target is unreachable, project it onto the boundary and flag it.
**When to address:** Phase 1, immediately after switching to real trajectories.

### P3: Coordinate Frame Convention Mismatch at R3D-to-Sim Boundary

**Feature:** True wrist retargeting
**What goes wrong:** The pipeline has a critical axis swap in `calibrate_workspace.py` line 35: `r3d_to_sim_axes()` maps R3D `(X_right, Y_up, Z_toward)` to sim `(-Z, -X, Y)`. This transform was validated for the v0.2 choreographed approach where only the pre-grasp phase used real wrist data (v10 line 339). When the full trajectory drives the sim, any error in this convention (e.g., a sign flip or wrong axis) produces mirrored or rotated motion. Retargeting literature confirms that initial orientation offsets between human and robot frames are a leading cause of artifacts like "toed-in" or mirrored motion.
**Warning signs:** Robot arm moves backward when human moves forward; lateral motion is inverted; wrist trajectory visualization in sim coordinates looks mirrored vs. the raw R3D video.
**Prevention:** Add a visual validation step: render the calibrated wrist trajectory as a 3D scatter plot overlaid on the sim workspace, and compare against the video. Verify the convention with a known "reach forward" motion. Unit-test `r3d_to_sim_axes()` with canonical vectors.
**When to address:** Phase 1, before any IK work.

### P4: HaMeR Model Exhausts GPU Memory on Full Video

**Feature:** HaMeR integration
**What goes wrong:** HaMeR is a ViT-based transformer model. Running inference on a 951-frame R3D recording at full resolution will likely exceed 8GB VRAM on consumer GPUs. The current stub (`egocrowd/hand_pose.py`) raises `NotImplementedError` with no batch control. If implemented naively (loading all frames, running in one batch), it will OOM. Even with batching, loading the model weights (~1-2GB) plus intermediate activations for batch_size=48 (HaMeR's default demo setting) at 224x224 can use 4-6GB. With the pipeline's 960x720 R3D frames, naive full-resolution inference will fail.
**Warning signs:** `torch.cuda.OutOfMemoryError`; process killed by OS; Modal container timeout.
**Prevention:** (1) Resize input crops to HaMeR's expected resolution (224x224) before inference. (2) Process frames in batches of 16-32 with explicit `torch.cuda.empty_cache()` between batches. (3) Use `torch.no_grad()` context. (4) If using Modal, set GPU memory tier to A10G (24GB) minimum. (5) Pre-crop hands using existing MediaPipe bounding boxes from the hand tracker to reduce input size.
**When to address:** Phase 2 (HaMeR integration), before any inference code.

### P5: HaMeR Output Format Incompatible with Pipeline's Wrist Representation

**Feature:** HaMeR integration
**What goes wrong:** HaMeR outputs MANO parameters (pose, shape, translation) in camera coordinates. The pipeline expects `wrist_pixel` (2D pixel location) and uses `reconstruct_wrist_3d.py` to unproject via depth maps. If HaMeR's 3D wrist translation is used directly, it lives in a different coordinate frame than the depth-unprojected world points. Mixing the two representations (HaMeR camera-space 3D vs. depth-map world-space 3D) will produce trajectory jumps or drift. Additionally, HaMeR's translation is relative to the crop center, not the full image.
**Warning signs:** Wrist trajectory has sudden jumps between frames processed by MediaPipe vs. HaMeR; wrist positions are systematically offset; 3D positions are in camera frame instead of world frame.
**Prevention:** Convert HaMeR outputs to the same world coordinate system used by `reconstruct_wrist_3d.py`. Use HaMeR's 3D wrist joint as a replacement for the depth-based unprojection (which is noisy anyway), but apply the same R3D camera-to-world transform (pose quaternion + translation from R3D metadata). Do not mix MediaPipe 2D + depth with HaMeR 3D in the same trajectory.
**When to address:** Phase 2, during integration design.

### P6: Bidirectional EMA Endpoint Distortion

**Feature:** Trajectory smoothing
**What goes wrong:** The current smoother in `reconstruct_wrist_3d.py` (line 69, `smooth_trajectory()`) uses bidirectional EMA with `alpha=0.12`. Bidirectional EMA averages forward and backward passes. At trajectory endpoints, the backward pass has no future data, causing the first and last ~10 frames to be pulled toward the endpoint value. For the current pipeline where action starts at frame 577 of 951, this means the critical grasp-initiation frames get distorted if smoothing is applied after trimming. For IIR filters like EMA, each frequency component experiences different lag, so the "zero-lag" claim of bidirectional averaging is only approximately true.
**Warning signs:** Wrist position at start/end of trimmed clip snaps toward a single point; first few frames of motion look artificially slow; block approach trajectory curves unnaturally at boundaries.
**Prevention:** (1) Smooth before trimming, not after. (2) Pad endpoints with the edge value (mirror padding) before smoothing, then crop the padding off. (3) Consider a Savitzky-Golay filter which handles endpoints better and preserves velocity/acceleration features. (4) Validate that peak velocity and trajectory shape are preserved after smoothing.
**When to address:** Phase 3 (trajectory smoothing), design phase.

### P7: Over-Smoothing Destroys Grasp Intent Signal

**Feature:** Trajectory smoothing
**What goes wrong:** The grasping signal is a binary array derived from finger curl detection. Smoothing the wrist trajectory can blur the precise moment when the hand pauses at the grasp location -- the "dwell" that signals intentional grasping. With `alpha=0.12` (very aggressive smoothing), a 5-frame pause becomes a slow glide-through, making grasp timing detection unreliable. Additionally, if trajectory smoothing is applied independently to X, Y, Z channels, rapid direction changes during grasp/release get rounded off, making the hand appear to follow a longer arc than the human actually performed.
**Warning signs:** The frame where `grasping` transitions from False to True no longer coincides with the wrist being near the block; the wrist "slides through" the block position without pausing; grasp dwell time in smoothed data is shorter than in raw data.
**Prevention:** (1) Do not smooth the `grasping` signal -- keep it binary. (2) Use a lower smoothing strength during detected grasp phases (adaptive alpha). (3) Validate grasp timing: check that the wrist is within 5cm of the block position when `grasping=True`. (4) Consider smoothing velocity rather than position to preserve spatial accuracy at waypoints.
**When to address:** Phase 3, after basic smoothing works.

### P8: Video Trim Auto-Detection Triggers on Non-Action Motion

**Feature:** Video trimming
**What goes wrong:** The R3D recordings include ~577 frames of non-action (walking, looking around, adjusting phone) before the actual block stacking starts. Auto-detection of the action window likely uses wrist velocity or proximity to objects. But the human may gesture, scratch, or wave during non-action periods, producing false positives. Simple velocity thresholds will fire on camera shake (the egocentric camera moves with the person's head). Research on online action detection confirms that the first 0-30% of an action is the hardest to classify correctly.
**Warning signs:** Trimmed video starts mid-action or includes long idle periods; different recordings trim at different quality levels; the trim point changes if threshold is tweaked slightly.
**Prevention:** (1) Use object proximity, not just velocity: action starts when the wrist first enters a radius around any detected object. (2) Require sustained proximity (e.g., 10+ consecutive frames within 0.3m of an object) to avoid false triggers from passing near objects. (3) Add a margin: start the trim 30 frames before the detected action start to preserve approach context. (4) Make the trim parameters configurable per-recording, not hardcoded.
**When to address:** Phase 4 (video trimming).

### P9: Off-by-One Between R3D Frame Index and Pipeline Frame Index

**Feature:** Video trimming
**What goes wrong:** The R3D files run at 60fps, the pipeline downsamples to 30fps (line 111 of `reconstruct_wrist_3d.py`: `fps_ratio = meta["fps"] / 30.0`), and the simulation runs at 10fps (`FPS = 10` in v10). Frame index `i` in the pipeline maps to `int(i * fps_ratio)` in the R3D. After trimming, if the trim offset is applied at the wrong level (R3D indices vs. pipeline indices vs. sim indices), all downstream data (wrist positions, grasping signals, object detections) will be misaligned. A 1-frame offset at 10fps sim = 100ms of timing error, which is enough to miss a grasp.
**Warning signs:** Grasping signal is offset from actual hand closure in the video; wrist position data does not match the RGB frame at the same index; block attachment triggers slightly early or late.
**Prevention:** (1) Define a single canonical frame index (pipeline frames at 30fps) and convert everything to/from it. (2) Store the trim offset in the calibration JSON so downstream stages know about it. (3) After trimming, verify alignment by spot-checking 3-5 frames: overlay wrist pixel position on the RGB frame at the same index. (4) Use 0-indexed frames consistently everywhere.
**When to address:** Phase 4, and verify in integration testing.

### P10: Finger Closure Fights Block Collision, Causing Visual Judder

**Feature:** Grasp visual quality
**What goes wrong:** In v10, fingers close via position-controlled actuators (`FINGER_CLOSED` values, line 40). When fingers reach the block surface, the physics engine detects collision (block has `contype=1 conaffinity=1`, finger geoms also have collision enabled). The finger actuators keep pushing (gain multiplied by 25x at line 159) while the contact solver pushes back, causing rapid oscillation -- visible as finger juddering or the block vibrating. The current code mitigates this by kinematically pinning the block during grasp, but the finger-block collision force still causes visible artifacts.
**Warning signs:** Fingers visually vibrate when touching the block; block shakes despite being kinematically held; simulation becomes unstable (NaN velocities) near grasp.
**Prevention:** (1) Disable finger-block collisions during grasp phase by setting `contype=0` on finger geoms when `grasped_obj is not None`. (2) Re-enable collisions on release. (3) Use the existing `block_geom_ids` set (line 206) pattern for finger geoms too. (4) Tune `FINGER_CLOSED` values so fingers stop just short of block surface rather than pushing into it.
**When to address:** Phase 5 (grasp quality).

### P11: Finger Pre-Shaping Timing Creates "Claw" Artifact

**Feature:** Grasp visual quality
**What goes wrong:** If fingers begin closing too early (during approach), the hand looks like a claw reaching for the block, which is visually unnatural. If too late, the block appears to jump into an open hand. The current `finger_ctrl` interpolation (line 457, 25% blend per frame) means fingers take ~12 frames to fully close at 10fps = 1.2 seconds. With real trajectory data (vs. choreographed), the approach speed varies, so a fixed blend rate may be too fast or too slow for different recordings.
**Warning signs:** Fingers are half-closed during approach (claw look); fingers are still opening when block should be grasped; finger closure timing looks different across recordings.
**Prevention:** (1) Tie finger pre-shaping to palm-to-block distance, not frame count. Start pre-shape when palm is <0.08m from block, complete when <0.03m. (2) Make the blend rate proportional to approach velocity. (3) Test with fastest and slowest recordings to validate timing looks natural.
**When to address:** Phase 5, after basic grasp works with real trajectories.

---

## Integration Pitfalls

### IP1: Smoothing Applied to Wrong Coordinate Space

**What goes wrong:** If trajectory smoothing is applied in R3D world coordinates (before `calibrate_workspace.py`) and then the calibration applies a nonlinear transform (scale + offset + Z correction), the smoothing is correct. But if smoothing is applied after calibration (in sim coordinates), the Z correction (line 140-142 of `calibrate_workspace.py`, which forces all object Z to `TABLE_Z`) creates a discontinuity at the correction boundary that smoothing will spread. Currently smoothing happens in `reconstruct_wrist_3d.py` (world coords) before calibration -- this is correct and should stay this way.
**Prevention:** Document and enforce: smoothing happens in world coordinates, before any sim-specific transforms. If adding a second smoothing pass in sim coords, do not smooth Z independently from XY.

### IP2: HaMeR Detection Rate Mismatch Creates Trajectory Gaps at Different Locations than MediaPipe

**What goes wrong:** MediaPipe detects hands in 52% of egocentric frames. HaMeR targets >85%. The gap interpolation in `reconstruct_wrist_3d.py` (line 152) uses linear interpolation across NaN gaps. With MediaPipe, gaps are frequent but short (1-3 frames). With HaMeR, gaps are rare but may be longer (occlusion events). If the interpolation strategy is tuned for frequent-short gaps (current) and encounters rare-long gaps (HaMeR misses during heavy occlusion), the linear interpolation will produce straight-line segments that look unnatural during complex hand motions.
**Prevention:** After switching to HaMeR, profile the gap distribution. For gaps >10 frames, use cubic spline interpolation instead of linear. For gaps >30 frames, consider marking those segments as unreliable and freezing the sim at the last known position rather than interpolating.

### IP3: Video Trimming Invalidates Frame Indices in All Downstream Stages

**What goes wrong:** Trimming removes frames from the beginning and end. But `calibrate_workspace.py` stores wrist trajectories and grasping arrays at the original (untrimmed) frame indices. If trimming is applied to the RGB frames but not to the wrist/grasping data (or vice versa), the sim will index into misaligned arrays. The `grasping` array (which drives grasp triggering) will be offset from the actual hand state.
**Prevention:** Trimming must be the FIRST pipeline stage (or must produce a frame-index mapping that all downstream stages consume). Store `trim_start` and `trim_end` in the pipeline metadata. All subsequent stages index relative to the trimmed range. Never re-index arrays without the trim offset.

### IP4: IK Solver Seed Diverges Between Choreographed and Real Trajectories

**What goes wrong:** The IK solver uses `SEED = [0.0, 0.0, 0.0, 2.4, 0.0, -1.5, 0.0]` and a null-space projection that pulls toward this seed (line 108). With choreographed trajectories, targets move smoothly and the IK always starts near the previous solution. With real trajectories, sudden hand motions can cause the IK to jump to a different configuration (elbow flip), producing jarring arm motion. The null-space term `(SEED - qcur) * 0.05` biases toward a fixed pose, which may fight the natural arm configuration needed for certain reach directions.
**Prevention:** (1) Use the previous frame's IK solution as the seed instead of the fixed `SEED` array (the code already does this via `last_arm_q`). (2) Add joint velocity limits: clamp per-frame joint changes to a max angular velocity (e.g., 2 rad/s at 10fps = 0.2 rad/frame). (3) Detect elbow flips by checking if any joint changes by more than 0.5 rad in one frame, and if so, re-solve from the previous configuration with stronger damping.

### IP5: Kinematic Block Pin Hides Physics Issues Until Release

**What goes wrong:** The current approach kinematically pins the block to the palm during grasp and pins it at the release position for `SETTLE_FRAMES` (set to `n`, the entire remaining trajectory). This means physics issues (finger-block collision, block weight, friction) are completely masked during grasp AND after release. When improving grasp visual quality, you may disable the post-release pin to allow the block to settle naturally -- but then all the masked physics issues emerge at once: block bounces off fingers, slides off the support block, or falls through the table.
**Prevention:** Reduce the scope of kinematic overrides gradually: (1) First, reduce `SETTLE_FRAMES` from `n` to 20-30 frames. (2) Verify the block stays stacked. (3) Then work on finger collision filtering. (4) Do not attempt all physics improvements simultaneously.

---

## Anti-Patterns to Avoid

- **"Fix it in smoothing"**: Tempting to add heavy smoothing to mask trajectory noise, IK failures, and frame drops. Wrong because it destroys the timing and spatial precision needed for grasp coordination. Instead, fix the root cause of each type of noise: improve detection (HaMeR), improve calibration (workspace clamping), improve IK (convergence checking).

- **Universal smoothing alpha**: Tempting to use one smoothing parameter for the entire trajectory. Wrong because approach phase needs less smoothing (preserve intent) while carry phase tolerates more (hand tremor is noise, not signal). Use phase-aware smoothing with different parameters for pre-grasp, grasp, carry, and release.

- **Testing with only the "good" recording**: Tempting to develop against `stack2` (the one that currently works end-to-end) and validate later. Wrong because different recordings have different timing, workspace extents, and occlusion patterns. Test every feature addition against at least `stack1`, `stack2`, and `picknplace2` to catch recording-specific assumptions.

- **Disabling all collision during development**: Tempting to set `contype=0` on everything to avoid physics interference while debugging IK and trajectories. Wrong because collision filtering bugs compound -- when you re-enable collisions, multiple things break simultaneously and it is impossible to diagnose which interaction is the root cause. Instead, disable specific collision pairs one at a time with clear documentation.

- **Running HaMeR locally "just to test"**: Tempting to try running HaMeR on a CPU or low-VRAM GPU to avoid Modal setup. Wrong because HaMeR on CPU is 100-1000x slower, will time out on a full recording, and gives a false impression that the model does not work. Use Modal from the start, even for development iteration.

- **Trimming by absolute frame count**: Tempting to hardcode "skip first 577 frames" based on the `stack2` recording. Wrong because each recording has a different idle period. The trim detection must be relative to motion/proximity signals, not absolute frame numbers.

- **Blending MediaPipe and HaMeR results frame-by-frame**: Tempting to use HaMeR where available and fall back to MediaPipe per-frame. Wrong because the two models produce systematically different wrist positions (different 3D estimation approaches), causing the trajectory to jump between two "tracks." Use one model per recording, with MediaPipe as the fallback only when HaMeR is entirely unavailable (no GPU).

---

*Researched 2026-03-13. Sources: project codebase analysis, MuJoCo documentation, motion retargeting literature, HaMeR repository.*

*Key source files analyzed:*
- `mujoco_g1_v10.py` -- current simulation (557 lines, choreographed motion with kinematic block attachment)
- `calibrate_workspace.py` -- R3D-to-sim coordinate transform (axis swap, scale, offset, Z correction)
- `reconstruct_wrist_3d.py` -- 3D wrist reconstruction with bidirectional EMA smoothing
- `egocrowd/hand_pose.py` -- HaMeR stub (raises NotImplementedError)
- `hand_tracker_v2.py` -- MediaPipe hand tracking (52% detection rate on egocentric video)
- `egocrowd/retarget.py` -- spatial trajectory with phase-based expert motion
