# QT-013: Proximity-Gated Grasping — Only Flag Grip When Wrist Near Object

## Context

QT-011 added `clean_grasping_signal()` with debounce + onset detection. QT-012 showed this was insufficient — grasping only dropped from 66% to 59%. The aspect-ratio heuristic fires `grip=True` when the palm is 0.334m from the block. Need spatial gating: only allow `grasping=True` when the wrist is within 15cm of any object.

## Tasks

### Task 1: Add proximity gating to `clean_grasping_signal()` and wire it up

**File:** `trim_trajectory.py`

**Changes:**

1. **Extend `clean_grasping_signal()` signature** — add `wrist_positions=None`, `object_positions=None`, `proximity_threshold=0.15`:
   ```python
   def clean_grasping_signal(grasping, min_run=5, wrist_positions=None, object_positions=None, proximity_threshold=0.15):
   ```

2. **Add proximity gating BEFORE debounce** — when both `wrist_positions` and `object_positions` are provided:
   - For each frame `i`, compute min Euclidean distance from `wrist_positions[i]` to every object in `object_positions`
   - Zero out `cleaned[i]` where min distance > `proximity_threshold` (0.15m)
   - Log proximity-gated count in the diagnostic output

3. **In `trim_calibrated_data()`** — pass wrist/object data to the `clean_grasping_signal()` call:
   ```python
   objects_sim = calib.get("objects_sim", [])
   wrist_arr = np.array(wrist_sim)
   obj_arr = np.array(objects_sim) if objects_sim else None
   grasping_arr = clean_grasping_signal(grasping_arr, wrist_positions=wrist_arr, object_positions=obj_arr)
   ```

4. **In `detect_action_window()`** — keep existing call WITHOUT proximity params (it doesn't have object data).

5. **Update diagnostic log line** to show proximity-gated count:
   ```
   GRASP CLEAN: 198/300 (66%) -> prox: 45/300 (15%) -> debounce: 40/300 (13%), onset frame=42
   ```

**Acceptance:**
- `grasping` is only `True` when wrist < 15cm from a block
- Clear False-to-True transition visible in trimmed data
- `detect_action_window()` call unchanged (no regression)
- Diagnostic log shows proximity filtering stats
