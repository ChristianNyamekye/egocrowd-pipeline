# Features Research: Flexa Pipeline v0.3

## Feature Analysis

### 1. True Wrist Retargeting

**Category:** Table stakes — this is the core value proposition stated in PROJECT.md ("the robot must faithfully reproduce the human's actual hand motion"). Without it, the pipeline is producing choreographed animation, not retargeting.

**Complexity:** Medium-High

**How it typically works:**

Motion retargeting from human to robot can be subdivided into two paradigms:

1. **Cartesian-space (end-effector) mapping** — Track the human wrist/hand position in 3D, transform it into the robot's coordinate frame, and use IK to solve for joint angles that place the robot's end-effector at the corresponding position. This is the approach already partially in place in `mujoco_g1_v10.py`.

2. **Joint-space mapping** — Map human joint angles directly to robot joint angles via kinematic correspondence. More complex, requires full arm tracking (not just wrist), and handles proportion differences poorly without additional optimization.

For this pipeline, Cartesian-space end-effector tracking is the right approach. The current code already does this for pre-grasp frames (`i < ls`) but falls back to hardcoded choreography during the grasp/carry/place phases (`p` variable with smoothstep interpolation through hover/descend/dwell/lift/carry/descend/release).

**The v0.3 change:** Replace the entire `p`-based choreography block (lines 337-390 of `mujoco_g1_v10.py`) with continuous wrist trajectory following for ALL frames. The IK target at each frame should come from the calibrated `wrist_sim` data, not from computed waypoints.

**Best practices:**

- **Workspace scaling:** The current `calibrate_workspace.py` already handles R3D-to-sim coordinate transform with axis swap (R3D X,Y,Z to sim -Z,-X,Y) and scale factor derived from wrist motion range. This is adequate, but the target range (0.35m) is hardcoded — consider making it adaptive to the robot's actual reachable workspace (~0.4-0.6m for G1 right arm).
- **Safety clamping:** Clamp IK targets to a bounding box within the robot's reachable workspace. The current code only clamps Z to `G1_TABLE_HEIGHT + 0.15` during pre-grasp. For true retargeting, clamp all axes: X in [0.15, 0.65], Y in [-0.45, 0.15], Z in [table_height, table_height + 0.4]. This prevents IK from diverging on unreachable targets.
- **IK null-space bias:** Already implemented (line 106-108, bias toward SEED posture). Keep this — it prevents elbow flipping and maintains natural arm configuration.
- **Temporal coherence:** Use previous frame's joint solution as the IK seed (already done via `last_arm_q`). This is critical for smooth motion.
- **Grasp trigger from data:** Instead of deriving grasp timing from the `p` progress variable, use the `grasping` signal from hand tracking. The calibrated JSON already contains per-frame `grasping` flags.

**Dependencies:**
- Calibrated wrist trajectory (`calibrate_workspace.py` output) — already exists
- Per-frame grasping signal from hand tracking — already exists
- Working IK solver — already exists in `ik_solve()`
- Kinematic block attachment — already exists (Franka v9 pattern)

**Risks:**
- **Noisy trajectory causes jerky motion** — directly mitigated by Feature 3 (trajectory smoothing). These two features are tightly coupled.
- **48% frame gaps** — frames with no wrist detection produce NaN/zero positions. Must interpolate before retargeting. Again, Feature 3.
- **Workspace mismatch** — human motion range may not map well to robot workspace after scaling. The current calibration forces objects to fixed positions (`desired_pick_xy`, `desired_support_xy`) which breaks the spatial relationship between wrist and objects. True retargeting should preserve relative wrist-to-object geometry.
- **IK failure on extreme targets** — the solver has 150 iterations with a 4mm convergence threshold. May need fallback (hold last good solution) for frames where the target is unreachable.

---

### 2. HaMeR Integration

**Category:** Differentiator — MediaPipe at 52% detection is functional but produces too many gaps. HaMeR at >85% detection is the quality threshold that makes true retargeting viable. Without it, half the trajectory is interpolated guesswork.

**Complexity:** High (GPU dependency, model integration, potential Modal deployment)

**How it typically works:**

HaMeR (Hand Mesh Recovery) is a transformer-based 3D hand reconstruction model from CVPR 2024 by Pavlakos et al.

**Architecture:**
- ViT-H (Vision Transformer, Huge) backbone for feature extraction
- Transformer decoder head that regresses MANO hand model parameters (pose theta, shape beta, camera pi)
- Trained on consolidated multi-dataset corpus with both 2D and 3D supervision

**Performance numbers (published benchmarks):**
- FreiHAND: PA-MPVPE 5.7mm, F@15mm 0.990
- HO3Dv2: PA-MPJPE 7.7mm, F@15mm 0.980
- 2nd place in EgoExo4D Ego-Pose Hands challenge (June 2024)

**Why it beats MediaPipe for this use case:**
- MediaPipe is designed for real-time webcam use (front-facing, well-lit, close range). Egocentric (head-mounted) video has heavy self-occlusion, motion blur, unusual angles, and hands at frame edges — all cases where MediaPipe's lightweight CNN struggles.
- HaMeR's ViT-H backbone has much higher capacity for handling these difficult cases.
- HaMeR outputs full 3D hand mesh (MANO parameters), not just 2D/3D keypoints. This gives wrist position, orientation, and finger pose in a single inference.

**Realistic detection rates for egocentric video:**
- MediaPipe: 50-65% on egocentric (observed 52% in this pipeline)
- HaMeR: 80-92% on egocentric, depending on occlusion severity and preprocessing
- Target of >85% is realistic with proper preprocessing

**Preprocessing that helps:**
- **Hand detection first:** HaMeR expects a cropped hand image. Use ViTDet or a dedicated hand detector to localize hands, then crop with margin before feeding to HaMeR.
- **Distortion correction:** Egocentric cameras (especially wide-angle like iPhone R3D) distort hands at frame edges. Applying perspective warp correction before HaMeR improves edge-case detection.
- **Frame subsampling:** At 30fps, consecutive frames are nearly identical. Process every 2-4th frame and interpolate between detections. The pipeline already subsamples 4:1 (951 to ~235 frames).
- **Confidence filtering:** HaMeR outputs reconstruction confidence. Set a threshold (e.g., 0.5) and treat low-confidence frames as gaps for interpolation.

**Dependencies:**
- GPU inference (ViT-H is ~600M parameters, needs >=8GB VRAM). Modal cloud deployment already anticipated in PROJECT.md.
- Hand detector for bounding box crops (ViTDet or MediaPipe detection-only mode as fallback)
- MANO model for hand mesh parameterization
- Existing pipeline stage slot (HaMeR stub already exists, raises NotImplementedError)

**Risks:**
- **GPU cost:** ViT-H inference is ~100-200ms per frame on A100. At 235 frames, that's 25-50 seconds per session — acceptable for batch processing.
- **MANO-to-wrist extraction:** Need to extract wrist position from MANO mesh output. The wrist joint is the root of the MANO kinematic chain, so this is straightforward.
- **Integration complexity:** Model weights (~2.5GB), dependencies (PyTorch, timm, ViT), and potential version conflicts. Modal deployment isolates this.
- **Fallback path:** MediaPipe fallback should remain for CPU-only environments. The pipeline already has this architecture.

---

### 3. Trajectory Smoothing

**Category:** Table stakes — with 48% frame gaps from MediaPipe (and even 10-15% from HaMeR), raw trajectories are unusable for IK. Smoothing is not optional; it is a prerequisite for Feature 1 (true retargeting).

**Complexity:** Low-Medium

**How it typically works:**

Trajectory smoothing for motion capture data with gaps is a two-step process:

**Step 1: Gap interpolation (fill missing frames)**

Methods ranked by suitability for this pipeline:

1. **Cubic spline interpolation** (recommended for gaps < 15 frames) — fits a smooth curve through known points. Preserves velocity continuity at boundaries. `scipy.interpolate.CubicSpline` or `interp1d(kind='cubic')`.

2. **Linear interpolation** (for gaps > 15 frames or as fallback) — simple but produces velocity discontinuities at gap boundaries. Acceptable for long gaps where the true trajectory is unknown anyway.

3. **PCHIP (Piecewise Cubic Hermite)** — monotone-preserving variant of cubic spline. Prevents overshoot in gaps, which can cause IK targets outside the workspace. Good default choice.

4. **Skeleton-based / physics-based** — uses kinematic constraints to fill gaps. Overkill for wrist-only trajectory.

**Strategy for gap classification:**
- Short gaps (1-5 frames): cubic spline, high confidence
- Medium gaps (6-20 frames): PCHIP interpolation, moderate confidence
- Long gaps (>20 frames): linear interpolation + flag as low-confidence region
- Very long gaps (>50 frames): hold last known position, flag for review

**Step 2: Temporal smoothing (remove noise from filled trajectory)**

**Savitzky-Golay filter** (recommended):
- Fits a polynomial to a sliding window, evaluates at center point
- Preserves peak shapes and sharp transitions better than moving average
- Key parameters for human motion at 10 FPS (this pipeline's rate):
  - **Window size:** 7-15 frames (0.7-1.5 seconds). Start with 11.
  - **Polynomial order:** 2-3. Order 2 (quadratic) for smooth motions, order 3 (cubic) if preserving acceleration peaks matters.
  - Rule: window_size > polynomial_order + 1, and window_size must be odd.

**Butterworth low-pass filter** (alternative):
- Better for removing high-frequency noise at known cutoff
- Typical cutoff: 3-6 Hz for hand manipulation movements
- At 10 FPS, Nyquist is 5 Hz, so cutoff ~2-3 Hz is safe
- Caution: Butterworth can ring at gap boundaries; apply after gap filling

**Best practices:**
- Always fill gaps BEFORE smoothing (smoothing across NaN gaps produces artifacts)
- Apply smoothing per-axis independently (X, Y, Z)
- Validate that smoothed trajectory stays within robot workspace bounds
- Preserve the grasping signal alignment — smoothing must not shift the temporal position of grasp events
- Consider adaptive smoothing: more aggressive in high-gap-density regions, lighter in clean regions

**Dependencies:**
- Wrist trajectory from hand tracking (MediaPipe or HaMeR) — exists
- `scipy.signal.savgol_filter` and `scipy.interpolate` — standard dependencies
- Grasping signal for temporal alignment — exists in calibrated JSON

**Risks:**
- **Over-smoothing:** Too-large window erases the actual manipulation motion, making the robot move in vague arcs instead of precise reaches. Start conservative (window=7).
- **Interpolation artifacts at long gaps:** Spline interpolation can overshoot wildly in long gaps. PCHIP or linear is safer for gaps > 15 frames.
- **Phase shift:** Some filters introduce lag. Savitzky-Golay is zero-phase (no lag) when applied offline, which is correct for this batch pipeline.

---

### 4. Video Trimming

**Category:** Differentiator (quality-of-life) — the 951-frame R3D with action starting at frame 577 produces 57 seconds of the robot standing still before doing anything. Trimming to the 15-30s action window makes output watchable and reduces wasted computation. Not blocking for the core retargeting goal.

**Complexity:** Low

**How it typically works:**

For this pipeline, the action window is the period of actual hand manipulation. Two approaches:

**Approach A: Signal-based (recommended for this pipeline)**

Use the wrist velocity and grasping signal already available from hand tracking:

1. Compute wrist velocity: `v[i] = ||wrist[i+1] - wrist[i]|| / dt`
2. Compute rolling mean velocity over a window (e.g., 10 frames = 1 second)
3. Define "active" frames: rolling mean velocity > threshold (e.g., 0.01 m/frame)
4. OR use the grasping signal directly: action window = first_grasp_frame - margin to last_grasp_frame + margin
5. Add padding: 1-2 seconds before first active frame, 1-2 seconds after last active frame
6. Clamp to valid frame range

This is simple, requires no ML, and uses data already computed by the pipeline.

**Approach B: Learned action detection (overkill for this use case)**

Temporal action detection models (ActionFormer, DETR-style) can detect action boundaries in untrimmed video. These require:
- Pre-trained feature extractors (I3D, SlowFast)
- Fine-tuning on manipulation task data
- GPU inference

Not justified here — the signal-based approach will work because the pipeline already knows when the hand is detected and when it's grasping.

**Practical implementation for this pipeline:**

```
# Pseudocode
valid = ~np.isnan(wrist[:, 0])  # frames with wrist detection
active = valid & (velocity > threshold)
# OR simply:
active = np.array(grasping, dtype=bool)

first_active = np.argmax(active) - padding_frames
last_active = len(active) - np.argmax(active[::-1]) + padding_frames
trimmed_wrist = wrist[first_active:last_active]
```

**Best practices:**
- Use generous padding (2-3 seconds) — viewers need context to understand the action
- For the sim video, include a brief "approach" phase before the grasp starts
- Save trim indices in the calibration JSON so downstream stages know the mapping
- Don't discard the original data — trimming is a view/window, not destructive

**Dependencies:**
- Wrist trajectory — exists
- Grasping signal — exists
- These signals are already per-frame in the calibrated JSON

**Risks:**
- **False starts:** Hand might move early (adjusting, fidgeting) before the real manipulation. Use the grasping signal as the primary anchor, velocity as secondary.
- **Multiple action segments:** If the recording contains multiple pick-and-place cycles, need to decide: trim to the first one, or include all? For v0.3, trim to the dominant action.
- **Interaction with trajectory smoothing:** Trim BEFORE smoothing to avoid edge artifacts from the long idle period.

---

### 5. Grasp Visual Quality

**Category:** Differentiator — the kinematic attachment already achieves STACKED=True. This feature is about making the simulation look convincing: fingers wrap around the block, no interpenetration, visually plausible grasp. Does not affect functional correctness.

**Complexity:** Medium

**How it typically works:**

In MuJoCo, kinematic grasping (as opposed to physics-based grasping) means the block is attached to the hand programmatically rather than held by contact forces. The current v10 code already does this via the "Franka v9 pattern" (smooth-blend kinematic attachment). The visual quality improvements are about making the fingers look right during this kinematic grasp.

**Key techniques:**

1. **Finger pre-shaping (approach shaping)**
   - Before the palm reaches the block, gradually curl fingers into a pre-grasp pose
   - Current code: binary FINGER_OPEN / FINGER_CLOSED with 0.25 blend rate (line 457)
   - Improvement: Define 3-4 finger pose keyframes:
     - FINGER_OPEN (default)
     - FINGER_PRESHAPE (fingers slightly curved, ~30% closed, triggered 5-10 frames before grasp)
     - FINGER_GRASP (wrapped around block, ~70% closed)
     - FINGER_PINCH (tighter grip during carry, ~80% closed)
   - Transition between keyframes using smoothstep blending

2. **Contact exclusion to prevent interpenetration**

   MuJoCo's `contype` and `conaffinity` bitmasks control which geoms can collide:
   - Two geoms collide only if `(geom1.contype & geom2.conaffinity) || (geom2.contype & geom1.conaffinity)` is nonzero
   - Current code: blocks have `contype="1" conaffinity="1"`, same as floor and table
   - Problem: finger geoms also have `contype="1"`, so finger-block collisions push the block away during close approach, fighting the kinematic attachment

   **Solution:** Use bitmask separation:
   - Finger geoms: `contype="2" conaffinity="1"` (fingers can collide with table/floor via conaffinity, but blocks with `contype="1"` don't trigger against finger `contype="2"`)
   - OR use `<contact><exclude body1="right_hand_*" body2="block_a"/></contact>` in the XML for explicit exclusion
   - This lets fingers visually close around the block without physics interference

3. **Grasp offset calibration**
   - Current `grasp_offset` is computed as `block_pos - palm_center` at grasp time (line 420)
   - If this offset is too large, the block floats visibly away from the fingers
   - Improvement: Clamp the offset magnitude to block_half + small margin. Force the block center to be within the finger envelope.

4. **Finger gain tuning**
   - Current: `FINGER_GAIN_MULTIPLIER = 25.0` applied to actuator gains (lines 157-160)
   - Too high: fingers snap shut unrealistically. Too low: fingers lag behind the control signal.
   - Tune per-finger: thumb needs different gain than index/middle for natural wrap appearance.

5. **Visual-only finger geoms**
   - Add visual-only (non-colliding) finger geoms that are slightly larger/rounder than the collision geoms
   - `contype="0" conaffinity="0"` makes them purely visual
   - Creates appearance of soft finger pads wrapping around the block

**Best practices:**
- Use `<option cone="elliptic">` (already set) for better friction cone modeling
- Set block friction high (`friction="2.0"`) for stable kinematic attachment appearance (already done)
- Render from multiple camera angles to verify no interpenetration is visible
- The current `FINGER_CLOSED = [0.4, -0.5, -0.6, 0.8, 0.9, 0.8, 0.9]` with comment "partial closure -- wrap around block, not through it" shows this has already been manually tuned. Further tuning should be data-driven (render, inspect, adjust).

**Dependencies:**
- G1 hand model from mujoco_menagerie (finger joints, actuators) — exists
- Kinematic attachment logic — exists (Franka v9 pattern)
- Scene XML generation — exists in `scene_xml()` and `build_objects()`

**Risks:**
- **Model-specific tuning:** Finger joint limits and actuator properties are specific to the G1 model. Any changes to the menagerie model will break these tunings.
- **Performance:** More finger geoms = more collision pairs = slower simulation. Keep visual-only geoms minimal.
- **Diminishing returns:** At some point, kinematic grasping will always look slightly "off" compared to real physics grasping. The out-of-scope note in PROJECT.md correctly defers dynamic grasping physics.

---

## Feature Dependencies

```
                    ┌─────────────────┐
                    │  R3D Ingest     │ (exists)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
               ┌────│  Hand Tracking  │────┐
               │    └────────┬────────┘    │
               │             │             │
      ┌────────▼────────┐    │    ┌────────▼────────┐
      │ 2. HaMeR        │    │    │  MediaPipe      │ (exists,
      │   (new)         │    │    │  (fallback)     │  fallback)
      └────────┬────────┘    │    └────────┬────────┘
               │             │             │
               └──────┬──────┘─────────────┘
                      │
             ┌────────▼────────┐
             │ 4. Video Trim   │ ◄── uses grasping signal + velocity
             │   (new)         │
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │ 3. Smoothing    │ ◄── fills gaps, filters noise
             │   (new)         │
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │ Calibration     │ (exists)
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │ 1. True Wrist   │ ◄── replaces choreography with
             │   Retargeting   │     real trajectory + IK
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │ 5. Grasp Visual │ ◄── finger shaping, contact
             │   Quality       │     exclusion
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │  MuJoCo Sim     │ (exists)
             └─────────────────┘
```

**Critical path:** HaMeR (2) -> Trim (4) -> Smoothing (3) -> True Retargeting (1)

Feature 5 (Grasp Visual Quality) is independent and can be developed in parallel.

## Recommendations

### Build Order

1. **Trajectory Smoothing (3) first** — lowest complexity, immediately improves existing pipeline even with MediaPipe data. Unblocks Feature 1.
2. **True Wrist Retargeting (1) second** — the core value. With smoothed MediaPipe trajectories, this can be validated before HaMeR is ready.
3. **Video Trimming (4) third** — quick win, signal-based approach uses existing data.
4. **HaMeR Integration (2) fourth** — highest complexity, GPU dependency. Can be developed on Modal in parallel but integrated after 1/3/4 are working.
5. **Grasp Visual Quality (5) last** — polish. Independent of the trajectory pipeline.

### Key Gotchas

- **Features 1 and 3 are inseparable.** Do not attempt true retargeting without smoothing; raw trajectories will produce violent arm motion. Build and test them together.
- **The `p` variable elimination is the key refactor.** The entire choreography block in `mujoco_g1_v10.py` (lines 337-390) must be replaced with: `target = smoothed_wrist[i]` plus safety clamping. The grasp/release logic stays but triggers from the data's `grasping` signal instead of `p` thresholds.
- **Calibration needs adjustment for true retargeting.** The current calibration forces blocks to hardcoded positions (`desired_pick_xy`, `desired_support_xy`), overriding the actual spatial relationship. For true retargeting, the wrist trajectory should be scaled/offset to match where the blocks actually are in sim, not the other way around.
- **HaMeR's 85% target is achievable but requires proper hand detection.** The ViT-H model alone does not detect hands — it reconstructs them from crops. A hand detector (ViTDet or even MediaPipe's palm detector) must provide bounding boxes first.
- **Video trimming should happen early in the pipeline** (before 3D reconstruction) to save compute. Don't reconstruct 577 frames of idle.

### Anti-Features (things to NOT build)

- **Learned action detection for trimming** — signal-based is sufficient, a neural approach would add complexity and GPU dependency for minimal gain.
- **Physics-based grasping** — correctly scoped as out-of-scope in PROJECT.md. Kinematic attachment with visual finger shaping is the right level of fidelity for research output.
- **Adaptive workspace scaling per frame** — scale should be computed once per session from the calibration, not per-frame. Per-frame scaling would introduce jitter.
- **Real-time streaming retargeting** — batch processing is fine per PROJECT.md. Don't add latency constraints.
