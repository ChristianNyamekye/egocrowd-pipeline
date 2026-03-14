# Research Summary: Flexa Pipeline v0.3

*Synthesized: 2026-03-14*

## Stack Additions

Only three new dependencies are truly required for v0.3:

| Dependency | Why | Install Location |
|---|---|---|
| **HaMeR** (via `hamer_helper` or direct) | Replaces MediaPipe's 52% detection with >85% on egocentric video. Outputs MANO params + 3D wrist position directly. | Modal container only (not local) |
| **ViTPose-Pytorch** (`gpastal24/ViTPose-Pytorch`) | mmcv-free fork of ViTPose. Eliminates the `mmcv==1.3.9` pin that conflicts with Python 3.12 / PyTorch 2.4. | Modal container only |
| **smplx==0.1.28** | MANO hand model layer, required by HaMeR for 3D joint decoding. | `pyproject.toml` `[gpu]` extras |

Everything else is already in the stack: `scipy` (smoothing, interpolation), `opencv-python` (trimming), `mujoco>=3.0` (grasp quality), `modal` (cloud GPU). Add `modal` explicitly to `pyproject.toml` under a new `[cloud]` extra.

**Skip:** pytorch3d (not needed by HaMeR core), Dyn-HaMR (too complex for v0.3), full ViTPose with mmcv (conflict risk), WildHands/4DHands (immature).

## Feature Table Stakes

These must work for v0.3 to deliver its stated goal ("the robot must faithfully reproduce the human's actual hand motion"):

1. **Trajectory Smoothing** — Replace bidirectional EMA (`alpha=0.12`) with Savitzky-Golay filter. Upgrade gap interpolation from linear to PCHIP/cubic spline with gap-length-aware strategy (cubic <15 frames, PCHIP 15-30, linear+hold >30). Without this, raw trajectories are unusable for IK.

2. **True Wrist Retargeting** — Eliminate the `p` progress variable choreography (lines 337-390 of `mujoco_g1_v10.py`). Replace with `target = wrist[i]` for ALL frames. Derive `want_grip` from the `grasping` signal, not from `p` thresholds. Keep the existing IK solver, null-space bias, and kinematic block attachment.

These two features are inseparable — do not attempt retargeting without smoothing.

## Feature Differentiators

These elevate v0.3 from functional to impressive:

3. **HaMeR Integration** — 85%+ detection vs. MediaPipe's 52%. Provides direct 3D wrist position (can skip noisy depth-map unprojection). Requires Modal deployment on A10G GPU. Creates `processing/hamer_modal.py` as a new file.

4. **Video Trimming** — Signal-based action window detection using `grasping` signal + wrist velocity. Cuts 95s render down to 15-30s. No ML needed.

5. **Grasp Visual Quality** — Finger pre-shaping (distance-based graduated closure), contact exclusion to prevent interpenetration, per-finger gain tuning. Pure rendering polish.

## Recommended Build Order

Synthesized from dependency analysis, risk profile, and iteration speed:

```
Phase 1: Foundations (unblock everything, speed up iteration)
  ├── 1a. Trajectory Smoothing
  │     Savgol filter + spline gap interpolation in reconstruct_wrist_3d.py
  │     Immediately improves even MediaPipe data
  │     Files: reconstruct_wrist_3d.py, calibrate_workspace.py
  │
  └── 1b. Video Trimming (parallel with 1a)
        Signal-based trim using grasping array
        Cuts iteration time from 95s to 15-30s renders
        Files: new trim_trajectory.py, run_pipeline.py

Phase 2: Core Retargeting
  └── 2. True Wrist Retargeting
        Replace p-choreography with wrist[i] trajectory following
        Validate with smoothed MediaPipe data first (before HaMeR)
        Files: mujoco_g1_v10.py (primary), calibrate_workspace.py (minor)

Phase 3: Quality Uplift (can run in parallel)
  ├── 3a. HaMeR Integration
  │     Modal function for GPU inference, ViTPose-Pytorch for detection
  │     Files: new processing/hamer_modal.py, egocrowd/hand_pose.py, run_pipeline.py
  │
  └── 3b. Grasp Visual Quality (parallel with 3a)
        Finger pre-shaping, contact exclusion, gain tuning
        Files: mujoco_g1_v10.py
```

**Rationale:** Smoothing first because it has zero dependencies and unblocks retargeting. Trimming in parallel because it speeds up every subsequent test cycle. Retargeting before HaMeR because it can be validated with smoothed MediaPipe data — proving the architecture works before adding GPU complexity. HaMeR and grasp quality are independent and can be developed in parallel.

## Watch Out For

Ranked by likelihood x impact (highest first):

### 1. Choreography-to-Real Trajectory Switch Breaks Grasp Timing (P1)
- **Likelihood:** Near-certain on first attempt
- **Impact:** Block never gets picked up, or teleports to hand mid-air
- **Prevention:** Decouple `want_grip` from the `p` variable entirely. Use a state machine driven by `grasping[i]` from tracking data + palm-to-block proximity. Test with at least 3 different R3D recordings.

### 2. Human Workspace Exceeds Robot Reachable Workspace (P2)
- **Likelihood:** High — human arm reach (0.7m) exceeds G1's effective reach (0.5m)
- **Impact:** IK solver converges silently to wrong position; arm locks at full extension
- **Prevention:** Clamp all IK targets to a validated reachable envelope after calibration. Log IK convergence failures per frame. Workspace bounds: X [0.15, 0.65], Y [-0.45, 0.15], Z [table+0.05, table+0.4].

### 3. Over-Smoothing Destroys Grasp Intent Signal (P7)
- **Likelihood:** Medium-high — default Savgol window of 11 frames at 10 FPS = 1.1s of smoothing
- **Impact:** Wrist "slides through" grasp position without dwelling; grasp timing misaligns with block proximity
- **Prevention:** Never smooth the `grasping` signal (keep binary). Use phase-aware smoothing: lighter during grasp phases, heavier during carry. Validate that wrist is within 5cm of block when `grasping=True`.

### 4. HaMeR Output Coordinate Frame Mismatch (P5)
- **Likelihood:** Medium — HaMeR outputs camera-space 3D, pipeline expects world-space
- **Impact:** Trajectory jumps or systematic offset; mixed MediaPipe/HaMeR frames cause track-switching artifacts
- **Prevention:** Use one model per recording (never blend frame-by-frame). Apply R3D camera-to-world transform to HaMeR's camera-space output. Bypass depth-map unprojection entirely when using HaMeR.

### 5. Video Trim Invalidates Downstream Frame Indices (IP3 + P9)
- **Likelihood:** Medium — three different frame rates (60fps R3D, 30fps pipeline, 10fps sim)
- **Impact:** Grasping signal offset from actual hand state by 100ms+; grasp triggers at wrong time
- **Prevention:** Define one canonical frame index (pipeline 30fps). Store `trim_start`/`trim_end` in calibration JSON. Verify alignment by spot-checking wrist overlay on RGB at 3-5 frames.

## Key Technical Decisions

### D1: HaMeR installation approach
- **Options:** (a) Raw HaMeR repo + submodules, (b) `hamer_helper` wrapper, (c) custom minimal integration
- **Recommended:** `hamer_helper` — provides clean inference API, avoids training code baggage, used by other egocentric projects (EgoAllo). Still requires MANO model files (manual download, academic license).

### D2: Modal GPU tier for HaMeR
- **Options:** T4 (16GB, cheaper), A10G (24GB, faster)
- **Recommended:** A10G — HaMeR's ViT-H backbone + MANO decoder needs 8-10GB; A10G provides headroom and faster inference. ~25-50s per recording at 235 frames.

### D3: Single vs. dual Modal functions
- **Options:** (a) Separate GroundingDINO + HaMeR functions (two GPU containers), (b) Combined pipeline function
- **Recommended:** Combined — avoids double cold-start penalty and Modal free-tier concurrency limits. Run detection then HaMeR sequentially in one GPU allocation.

### D4: Smoothing filter type
- **Options:** (a) Bidirectional EMA (current), (b) Savitzky-Golay, (c) Butterworth low-pass
- **Recommended:** Savitzky-Golay — zero-phase (no lag in batch mode), preserves trajectory shape and peaks better than EMA, handles endpoints better than Butterworth. Start with `window_length=7, polyorder=3` and tune up.

### D5: Where to apply trimming in the pipeline
- **Options:** (a) Before 3D reconstruction (saves compute), (b) Between calibration and simulation (simpler), (c) Post-hoc on calibrated JSON
- **Recommended:** Between calibration and simulation (option b) — simplest integration, avoids re-indexing upstream stages, simulation automatically adapts to shorter input via `n = len(wrist)`.

### D6: Coordinate frame validation
- **Options:** (a) Trust existing `r3d_to_sim_axes()` mapping, (b) Add visual validation step
- **Recommended:** Add visual validation — render calibrated wrist trajectory as 3D scatter overlaid on sim workspace. Unit-test `r3d_to_sim_axes()` with canonical vectors. This is cheap insurance against the most confusing class of bugs (mirrored/rotated motion).

## Integration Points

### Hand Tracking -> Trajectory Smoothing
- HaMeR changes the gap distribution: fewer but longer gaps vs. MediaPipe's frequent short gaps. The interpolation strategy must be gap-length-aware (not one-size-fits-all).
- Do NOT blend MediaPipe and HaMeR frame-by-frame — they produce systematically different wrist positions. Use one model per recording.

### Trajectory Smoothing -> True Retargeting
- Smoothing must happen BEFORE calibration (in world coords, not sim coords) to avoid spreading Z-correction discontinuities.
- The `grasping` signal must NOT be smoothed — it stays binary. Only spatial trajectory (X, Y, Z) gets filtered.
- Smoothed trajectory must stay within the robot's reachable workspace after calibration — add a post-calibration workspace clamp.

### Video Trimming -> All Downstream Stages
- Trimming slices `wrist_sim` and `grasping` arrays. The simulation's `n = len(wrist)` auto-adapts, but `ls`/`le` (grasp start/end indices) must be recomputed relative to trimmed arrays.
- Store `trim_info: {original_frames, start_frame, end_frame}` in calibrated JSON for traceability.

### True Retargeting -> Grasp Visual Quality
- Once `want_grip` comes from `grasping[i]` instead of `p` thresholds, finger pre-shaping must be distance-based (palm-to-block proximity), not time-based. The approach speed varies per recording.
- Finger-block collision exclusion (`contype`/`conaffinity` toggling) must be synchronized with the new data-driven grasp state, not the old `p`-based phases.

### HaMeR -> 3D Reconstruction
- If HaMeR provides camera-frame 3D wrist, `reconstruct_wrist_3d.py` can skip depth-map unprojection entirely — use HaMeR's 3D output + R3D camera pose for world-frame conversion. This is more accurate than depth lookup.
- The `_hamer_to_trajectory()` converter in `run_pipeline.py` must output `wrist_3d_camera` as an additional field so downstream stages can detect and use it.

---

*Synthesized from: STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md (researched 2026-03-13)*
