# Phase 3b: Grasp Visual Quality -- Research

*Researched: 2026-03-14*
*Scope: OUT-03, OUT-04, OUT-05*

---

## 1. Current Code Analysis

### File: `mujoco_g1_v10.py` (post-Phase 2)

Phase 2 eliminated the `p` choreography and replaced it with `target = wrist[i].copy()` for all frames and `want_grip = bool(grasping[i] > 0)` for grasp intent. The grasp logic is now fully data-driven. This research builds on that foundation.

### Finger Control (Lines 39-40, 422-424)

Current finger control is binary with exponential blending:

```python
FINGER_OPEN   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
FINGER_CLOSED = np.array([0.4, -0.5, -0.6, 0.8, 0.9, 0.8, 0.9])

# In frame loop (line 422-424):
target_ctrl = FINGER_CLOSED if want_grip else FINGER_OPEN
finger_ctrl += (target_ctrl - finger_ctrl) * 0.25
data.ctrl[HAND_CTRL_START:HAND_CTRL_START+7] = finger_ctrl
```

The blend rate of 0.25 means fingers go from open to ~63% closed in 4 frames (0.4s at 10 FPS). The transition is a binary switch: when `want_grip` flips from False to True, fingers immediately start closing toward `FINGER_CLOSED`. There is no distance-based pre-shaping -- the fingers are fully open until the exact frame where `grasping[i]` becomes True.

**Problem:** In real grasping, fingers begin curling during the approach phase, well before contact. The current binary switch looks robotic and unrealistic.

### Finger Joint Mapping (from G1 model XML)

The 7 right-hand actuators (indices `HAND_CTRL_START` to `HAND_CTRL_START+6`) map to:

| Index | Joint Name | Axis | Range | Role |
|-------|-----------|------|-------|------|
| 0 | `right_hand_thumb_0_joint` | Y | [-1.05, 1.05] | Thumb abduction |
| 1 | `right_hand_thumb_1_joint` | Z | [-1.05, 0.72] | Thumb proximal flexion |
| 2 | `right_hand_thumb_2_joint` | Z | [-1.75, 0] | Thumb distal flexion |
| 3 | `right_hand_index_0_joint` | Z | [0, 1.57] | Index proximal flexion |
| 4 | `right_hand_index_1_joint` | Z | [0, 1.75] | Index distal flexion |
| 5 | `right_hand_middle_0_joint` | Z | [0, 1.57] | Middle proximal flexion |
| 6 | `right_hand_middle_1_joint` | Z | [0, 1.75] | Middle distal flexion |

Current `FINGER_CLOSED = [0.4, -0.5, -0.6, 0.8, 0.9, 0.8, 0.9]` uses partial closure values (not joint limits) to wrap around the block without interpenetration. The comment "partial closure -- wrap around block, not through it" confirms this was manually tuned.

### G1 Right Hand Body Hierarchy (from `g1_with_hands.xml`)

```
right_wrist_yaw_link
  +-- right_hand_palm_link (geom: visual + collision)
  +-- right_hand_thumb_0_link
  |     +-- right_hand_thumb_1_link
  |           +-- right_hand_thumb_2_link
  +-- right_hand_middle_0_link
  |     +-- right_hand_middle_1_link
  +-- right_hand_index_0_link
        +-- right_hand_index_1_link
```

Each body has two geoms: one `class="visual"` (contype=0, conaffinity=0 -- no collision) and one `class="collision"` (default contype/conaffinity from the mesh collision class). The collision geoms are what interact with the blocks.

### Block Geom Properties (Lines 126-131)

```python
f'<geom type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" rgba="{c}" mass="0.3" '
f'friction="2.0 0.01 0.001" contype="1" conaffinity="1"/>'
```

Blocks have `contype="1" conaffinity="1"`. The G1 collision class does not explicitly set contype/conaffinity, so they inherit the MuJoCo defaults: `contype=1, conaffinity=1`.

### The Collision Problem

MuJoCo collision rule: two geoms collide if `(geom1.contype & geom2.conaffinity) || (geom2.contype & geom1.conaffinity)` is nonzero.

- Finger collision geoms: contype=1, conaffinity=1 (default)
- Block geoms: contype=1, conaffinity=1

`(1 & 1) || (1 & 1) = True` -- fingers collide with blocks.

During kinematic hold, the block is programmatically attached to the palm (lines 447-463). The block tracks `palm_center + grasp_offset` via direct qpos writes. But finger collision geoms are also in contact with the block. The physics solver generates contact forces that push the block away from the fingers, fighting the kinematic attachment. This causes:

1. **Visual interpenetration:** Fingers pass through the block because the kinematic attachment overrides collision forces
2. **Jitter:** Contact forces create micro-oscillations between substeps
3. **Post-release bounce:** When the block is released, accumulated contact forces can launch it

The current code already handles post-release bounce with `SETTLE_FRAMES = n` (line 324) which pins the block through the end of the trajectory. But the visual interpenetration during hold is still present.

### Kinematic Attachment (Lines 447-463)

```python
if grasped_jnt_adr is not None:
    pc_sub = palm_center(model, data)
    target_pos = pc_sub + grasp_offset
    blend_t = min(1.0, grasp_age / BLEND_FRAMES)
    blend_t = blend_t * blend_t * (3 - 2 * blend_t)  # smoothstep
    blended_pos = np.array([...])
    data.qpos[grasped_jnt_adr:grasped_jnt_adr+3] = blended_pos
    data.qpos[grasped_jnt_adr+3:grasped_jnt_adr+7] = grasp_quat
    # Zero velocity
    data.qvel[dof_adr:dof_adr+6] = 0
```

The block position is set kinematically every substep. Contact forces from fingers are computed but overridden by the position write. This means collision exclusion won't change physics behavior, but it WILL prevent contact forces from being computed (reducing jitter) and prevent visual artifacts in the contact visualization.

### `block_geom_ids` and Existing contype Manipulation (Lines 211-260)

The code already identifies block geom IDs and temporarily sets their contype/conaffinity to 0 during the settle phase (lines 248-260). This pattern can be reused for finger-block exclusion during kinematic hold.

```python
block_geom_ids = set()
for nm in obj_names:
    bid = jid(model, nm, mujoco.mjtObj.mjOBJ_BODY)
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == bid:
            block_geom_ids.add(gi)
```

---

## 2. MuJoCo Contact Filtering Techniques

### Option A: contype/conaffinity Bitmask Separation

Set finger collision geoms to a different bitmask bit than blocks:

```python
# Finger geoms: contype=2, conaffinity=1
# Block geoms: contype=1, conaffinity=1

# Check: (finger.contype & block.conaffinity) || (block.contype & finger.conaffinity)
#       = (2 & 1) || (1 & 1) = 0 || 1 = 1  --> STILL COLLIDES
```

This doesn't work because conaffinity=1 on fingers means blocks (contype=1) still trigger. To fully disable:

```python
# Finger geoms: contype=2, conaffinity=2  (only collide with contype=2 things)
# Block geoms: contype=1, conaffinity=1   (only collide with contype=1 things)

# Check: (2 & 1) || (1 & 2) = 0 || 0 = 0  --> NO COLLISION
```

But this also disables finger-table and finger-floor collision (table/floor have contype=1, conaffinity=1). To keep finger-table collision:

```python
# Floor/table geoms: contype=1, conaffinity=3 (collide with bit 0 OR bit 1)
# Block geoms: contype=1, conaffinity=1       (collide with bit 0 only)
# Finger geoms: contype=2, conaffinity=2      (collide with bit 1 only)

# Finger vs floor: (2 & 3) || (1 & 2) = 2 || 0 = COLLIDES
# Finger vs block: (2 & 1) || (1 & 2) = 0 || 0 = NO COLLISION
# Block vs floor:  (1 & 3) || (1 & 1) = 1 || 1 = COLLIDES
```

This works, but requires modifying floor/table conaffinity and is fragile.

### Option B: Dynamic contype Toggle (Recommended)

Toggle block contype to 0 during kinematic hold, restore on release. The code already does this during settle (lines 248-260).

```python
# When GRASP triggers:
for gi in block_geom_ids:
    if model.geom_bodyid[gi] == grasped_bid:
        model.geom_contype[gi] = 0    # disable block collision
        model.geom_conaffinity[gi] = 0

# When RELEASE triggers:
for gi in block_geom_ids:
    if model.geom_bodyid[gi] == grasped_bid:
        model.geom_contype[gi] = 1    # restore block collision
        model.geom_conaffinity[gi] = 1
```

**Advantages:**
- Simple, follows existing pattern in the code
- Only affects the grasped block, not all blocks
- No need to modify floor/table/finger geom properties
- Easy to synchronize with the grasp state machine

**Disadvantages:**
- Block doesn't collide with anything during hold (including the table if the hand passes through table height). Acceptable because the block position is kinematically controlled during hold.
- Must remember to restore on release

### Option C: `<contact><exclude>` XML Element

Add explicit exclusion pairs in the scene XML:

```xml
<contact>
  <exclude body1="right_hand_thumb_0_link" body2="block_a"/>
  <exclude body1="right_hand_thumb_1_link" body2="block_a"/>
  <exclude body1="right_hand_thumb_2_link" body2="block_a"/>
  <exclude body1="right_hand_index_0_link" body2="block_a"/>
  <exclude body1="right_hand_index_1_link" body2="block_a"/>
  <exclude body1="right_hand_middle_0_link" body2="block_a"/>
  <exclude body1="right_hand_middle_1_link" body2="block_a"/>
  <exclude body1="right_hand_palm_link" body2="block_a"/>
  <!-- Repeat for block_b -->
</contact>
```

**Problem:** `right_hand_palm_link` is not a body; it's a geom attached to `right_wrist_yaw_link`. The palm geom shares a body with the wrist, so excluding `right_wrist_yaw_link` would also exclude wrist-block collision. Additionally, the `<exclude>` element is NOT inherited by child bodies -- each finger link needs its own exclusion line.

**Larger problem:** The `<exclude>` element is static -- it applies for the entire simulation. We only want to disable finger-block collision during kinematic hold, not during approach (where finger-block contact provides visual feedback) or after release (where block must rest on table/other blocks). A static exclusion permanently removes finger-block interaction.

### Recommendation: Option B (Dynamic contype Toggle)

Dynamic toggling is the right approach because:
1. It follows the existing code pattern (lines 248-260)
2. It only applies during kinematic hold (when the block position is overridden anyway)
3. It's trivially synchronized with the grasp/release state machine
4. On release, collision is restored so the block interacts normally with the table and other blocks
5. No XML changes needed

---

## 3. Finger Pre-Shaping Design

### The Problem

Currently, `want_grip` drives finger state as a binary: open or closed. With the Phase 2 change (`want_grip = bool(grasping[i] > 0)`), fingers snap from open to closing the instant the tracking grasping signal goes True. There is no anticipatory pre-shaping.

### Distance-Based Pre-Shaping

Compute palm-to-block distance each frame. When the palm is within a "pre-shape zone" (but before `want_grip` is True), begin gradually curling fingers:

```
Distance > 0.20m:  FINGER_OPEN         (fully open)
0.20m >= d > 0.12m: FINGER_PRESHAPE    (30% closed, fingers starting to curl)
0.12m >= d > 0.06m: FINGER_APPROACH    (50% closed, preparing to wrap)
d <= 0.06m:         FINGER_CLOSED      (full grasp closure)
```

**Key insight:** Pre-shaping should happen BEFORE `want_grip=True`, based purely on palm-to-block proximity. This creates the visual effect of the hand preparing to grasp as it approaches, regardless of the tracking grasping signal timing.

### Integration with `want_grip`

The finger target should be the maximum of:
1. Distance-based pre-shape (anticipatory)
2. `want_grip`-based closure (data-driven)

```python
# Distance-based pre-shape (always active, based on proximity to nearest block)
pc = palm_center(model, data)
min_block_dist = min(np.linalg.norm(data.xpos[bid] - pc) for bid in obj_body_ids)

# Compute pre-shape factor: 0.0 (far) to 1.0 (contact)
preshape_t = np.clip(1.0 - (min_block_dist - 0.06) / (0.20 - 0.06), 0.0, 1.0)
preshape_t = preshape_t * preshape_t  # quadratic ease-in (gentle start, faster close)

# Determine finger target
if want_grip:
    # Full closure for data-driven grasp
    finger_target = FINGER_CLOSED
elif preshape_t > 0:
    # Distance-based pre-shaping (partial closure)
    finger_target = FINGER_OPEN + (FINGER_PRESHAPE - FINGER_OPEN) * preshape_t
else:
    finger_target = FINGER_OPEN

# Smooth blend (existing logic)
finger_ctrl += (finger_target - finger_ctrl) * 0.25
```

### Pre-Shape Pose

The pre-shape pose should be a natural "reaching to grasp" configuration:

```python
FINGER_PRESHAPE = np.array([0.2, -0.25, -0.3, 0.4, 0.45, 0.4, 0.45])
```

This is roughly 50% of `FINGER_CLOSED`, with:
- Thumb slightly abducted and flexed (preparing to oppose)
- Index and middle fingers partially curled (preparing to wrap)

### Distance Thresholds

The thresholds should be calibrated to the block size and approach speed:

- **Pre-shape start (0.20m):** About 3x block diameter (6cm). At 10 FPS with typical approach speed, this gives ~5-10 frames of pre-shaping. Visible but not premature.
- **Full pre-shape (0.06m):** Block diameter. At this distance, the hand is essentially touching the block. Fingers should be at maximum pre-shape (not full grasp -- that waits for `want_grip`).

### Blend Rate Tuning

The current blend rate of 0.25 per frame is too aggressive for pre-shaping (reaches 95% in ~11 frames = 1.1s). For pre-shaping, a slower blend creates more natural anticipation:

```python
# Different blend rates for different phases
if want_grip:
    blend_rate = 0.25    # fast closure for actual grasp
elif preshape_t > 0:
    blend_rate = 0.15    # slower for anticipatory pre-shaping
else:
    blend_rate = 0.20    # medium for returning to open
```

---

## 4. G1 Finger Geom Identification

To implement Option B (dynamic contype toggle), we need to identify finger geom IDs at model load time. The finger collision geoms belong to these bodies:

**Right hand finger bodies (collision geoms that may contact blocks):**
- `right_hand_thumb_0_link`
- `right_hand_thumb_1_link` (has a box collision geom, not mesh)
- `right_hand_thumb_2_link`
- `right_hand_index_0_link`
- `right_hand_index_1_link`
- `right_hand_middle_0_link`
- `right_hand_middle_1_link`

**Right hand palm (also has collision geom):**
- `right_wrist_yaw_link` (contains the palm geom: `right_hand_palm_link`)

Note: The palm collision geom is attached to the `right_wrist_yaw_link` body, not a separate palm body. This body also contains wrist collision geometry. Toggling the palm geom's contype would require identifying the specific geom (by mesh name or position), not by body.

**Practical simplification:** Since we're toggling the BLOCK's contype/conaffinity to 0 (Option B), we don't need to identify finger geoms at all. Disabling block collision affects all geom pairs involving the block, including finger-block pairs. This is the simplest approach.

---

## 5. STACKED=True Regression Risk (OUT-05)

### Current State

Phase 2 already passes STACKED=True:
> STACKED=True still passes. RMS=0.0555m.

### What Could Break

1. **Pre-shaping changes finger positions during approach:** Fingers may push the pick block before grasp if collision is still active during pre-shape. Mitigation: Pre-shaping only affects control targets, not collision properties. The existing FINGER_CLOSED values were already tuned to avoid pushing the block.

2. **Collision exclusion during hold changes block trajectory:** Disabling block collision means the block doesn't collide with anything during kinematic hold. Since the block position is kinematically overridden every substep, this doesn't change its trajectory. No regression expected.

3. **Collision restoration on release:** When contype/conaffinity are restored, the block must be in a valid position (not interpenetrating other geometry). The existing release logic places the block at `stack_z = support_settled_pos[2] + 2 * BLOCK_HALF` and pins it with `SETTLE_FRAMES = n`. This should work correctly.

4. **Blend rate changes affect grasp timing:** Slower blend rates mean fingers take longer to close, potentially delaying the visual closure. But the kinematic attachment is triggered by `want_grip + proximity`, not by finger position. Finger visual state doesn't affect attachment.

### Validation Plan

Run `python mujoco_g1_v10.py stack2` after all changes and verify:
- `STACKED=True` in output
- Block final Z values show correct stacking
- No error messages or anomalies

---

## 6. Implementation Strategy

### Order of Changes

1. **Add `FINGER_PRESHAPE` constant and pre-shape distance thresholds** -- new constants only, no behavior change yet.

2. **Replace binary finger control with distance-based pre-shaping** -- modifies the finger control block (lines 422-424) to use proximity-based graduated closure. This is purely visual; does not affect kinematic attachment.

3. **Add dynamic contype toggle on grasp/release** -- inside the ATTACH block (line 380-390), disable grasped block collision. Inside the RELEASE block (line 393-414), restore it. Follows the existing pattern from lines 248-260.

4. **Tune finger closure values and blend rates** -- iterate on `FINGER_PRESHAPE`, `FINGER_CLOSED`, distance thresholds, and blend rates by rendering and inspecting.

5. **Validate STACKED=True** -- run on stack2, verify no regression.

### What NOT to Change

- IK solver -- no changes
- Kinematic attachment logic -- no changes to the attach/release/blend mechanism
- Block positioning -- no changes
- `want_grip` derivation -- stays as `bool(grasping[i] > 0)`
- `SETTLE_FRAMES` / post-release pinning -- stays as is

---

## 7. Code Change Map

### Constants to ADD (after `FINGER_CLOSED`, before `FINGER_GAIN_MULTIPLIER`)

```python
FINGER_PRESHAPE = np.array([0.2, -0.25, -0.3, 0.4, 0.45, 0.4, 0.45])
PRESHAPE_DIST_START = 0.20   # start pre-shaping at 20cm from block
PRESHAPE_DIST_FULL  = 0.06   # full pre-shape at 6cm (block diameter)
```

### Lines to MODIFY (finger control, lines 422-424)

Replace:
```python
target_ctrl = FINGER_CLOSED if want_grip else FINGER_OPEN
finger_ctrl += (target_ctrl - finger_ctrl) * 0.25
```

With distance-based pre-shaping logic (~12 lines).

### Lines to ADD (inside ATTACH block, after line 390)

After `print(f"  F{i:03d} GRASP: {best_name} (dist={best_dist:.3f})")`:
```python
# Disable grasped block collision (OUT-04)
for gi in block_geom_ids:
    if model.geom_bodyid[gi] == grasped_bid:
        model.geom_contype[gi] = 0
        model.geom_conaffinity[gi] = 0
```

### Lines to ADD (inside RELEASE block, before `grasped_obj = None`)

Before `print(f"  F{i:03d} RELEASE: ...")`:
```python
# Restore grasped block collision (OUT-04)
for gi in block_geom_ids:
    if model.geom_bodyid[gi] == grasped_bid:
        model.geom_contype[gi] = 1
        model.geom_conaffinity[gi] = 1
```

---

## 8. Sources

- [MuJoCo XML Reference (contype/conaffinity)](https://mujoco.readthedocs.io/en/stable/XMLreference.html) -- authoritative reference for geom contact attributes
- [MuJoCo Computation (collision detection)](https://mujoco.readthedocs.io/en/stable/computation/index.html) -- collision filtering pipeline details
- [MuJoCo Discussion #1941 -- avoiding penetration](https://github.com/google-deepmind/mujoco/discussions/1941) -- community discussion on contype/conaffinity usage patterns
- [MuJoCo GitHub Issue #104 -- canceling contact detection](https://github.com/google-deepmind/mujoco/issues/104) -- explicit exclude usage
- [Allegro Hand Grasping in MuJoCo](https://github.com/premtc/Human_Robot_Hand_Grasping_Mujoco) -- reference implementation for multi-finger grasping
- [MuJoCo Overview (contact exclude)](https://mujoco.readthedocs.io/en/3.1.6/overview.html) -- exclude element documentation

---

*Research complete. Ready for planning.*
