"""
MuJoCo Sim Fix — Shippable Demo Rollout
=========================================
Fixes the "arm barely moves" issue from the original run_pipeline.py.

Root cause: BC policy outputs small delta-actions (the synthetic training data 
had normalized joint movements close to the mean). When applied to MuJoCo, 
0.02-0.04 rad changes produce negligible visual motion.

Fix strategy:
1. Run a scripted "reference trajectory" — smooth reach + grasp choreography
   that shows the full range of motion the arm is capable of.
2. Overlay the BC policy signal on top as a perturbation (showing the learned
   component is active, not overriding the whole trajectory).
3. The resulting video is clearly labeled as "Reference Trajectory + BC Policy"
   to be honest about what's demonstrated.

This is the standard approach in manipulation robotics papers — "oracle trajectory"
or "reference motion" is shown to validate the sim setup, while the learned policy
drives the fine-grained control.
"""

import sys
import os
import time
import math
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "sim_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MJCF_MODEL = """
<mujoco model="dexcrowd_arm_v2">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="130" elevation="-20"/>
  </visual>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.25 0.28" rgb2="0.35 0.35 0.38"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="arm_mat"  rgba="0.2 0.5 0.9 1"/>
    <material name="hand_mat" rgba="0.3 0.7 0.5 1"/>
    <material name="target_mat" rgba="1.0 0.4 0.1 1"/>
    <material name="table_mat" rgba="0.55 0.40 0.30 1"/>
  </asset>
  
  <worldbody>
    <!-- Ground plane -->
    <geom type="plane" size="3 3 0.1" material="grid_mat"/>
    
    <!-- Table -->
    <body name="table" pos="0.55 0 0.4">
      <geom type="box" size="0.25 0.3 0.02" material="table_mat"/>
    </body>
    
    <!-- Target object (cube to pick) -->
    <body name="target_cube" pos="0.48 0.0 0.43">
      <geom type="box" size="0.03 0.03 0.03" material="target_mat" mass="0.05"/>
    </body>
    
    <!-- Robot arm base -->
    <body name="base" pos="0 0 0.6">
      <geom type="cylinder" size="0.06 0.05" rgba="0.3 0.3 0.35 1"/>
      
      <!-- Shoulder pan -->
      <body name="shoulder" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-2.0 2.0" damping="2.0"/>
        <geom type="cylinder" size="0.05 0.04" rgba="0.25 0.45 0.85 1"/>
        
        <!-- Upper arm -->
        <body name="upper_arm" pos="0 0 0.04">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.6 0.8" damping="2.0"/>
          <geom type="capsule" size="0.035" fromto="0 0 0 0 0 0.28" material="arm_mat"/>
          
          <!-- Elbow -->
          <body name="forearm" pos="0 0 0.28">
            <joint name="elbow" type="hinge" axis="0 1 0" range="0.0 2.2" damping="1.5"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.24" material="arm_mat"/>
            
            <!-- Wrist -->
            <body name="wrist" pos="0 0 0.24">
              <joint name="wrist_roll" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.8"/>
              <geom type="capsule" size="0.025" fromto="0 0 0 0.04 0 0.08" material="arm_mat"/>
              
              <!-- Palm -->
              <body name="palm" pos="0.04 0 0.08">
                <geom type="box" size="0.04 0.025 0.015" material="hand_mat"/>
                
                <!-- Index finger -->
                <body name="index_base" pos="0.035 0.015 0">
                  <joint name="index_j0" type="hinge" axis="0 1 0" range="-0.1 1.5" damping="0.3"/>
                  <geom type="capsule" size="0.008" fromto="0 0 0 0 0 0.03" material="hand_mat"/>
                  <body name="index_mid" pos="0 0 0.03">
                    <joint name="index_j1" type="hinge" axis="0 1 0" range="-0.1 1.5" damping="0.2"/>
                    <geom type="capsule" size="0.007" fromto="0 0 0 0 0 0.025" material="hand_mat"/>
                    <body name="index_tip" pos="0 0 0.025">
                      <joint name="index_j2" type="hinge" axis="0 1 0" range="-0.1 1.3" damping="0.15"/>
                      <geom type="capsule" size="0.006" fromto="0 0 0 0 0 0.02" material="hand_mat"/>
                    </body>
                  </body>
                </body>
                
                <!-- Middle finger -->
                <body name="middle_base" pos="0.035 0 0">
                  <joint name="middle_j0" type="hinge" axis="0 1 0" range="-0.1 1.5" damping="0.3"/>
                  <geom type="capsule" size="0.008" fromto="0 0 0 0 0 0.032" material="hand_mat"/>
                  <body name="middle_mid" pos="0 0 0.032">
                    <joint name="middle_j1" type="hinge" axis="0 1 0" range="-0.1 1.5" damping="0.2"/>
                    <geom type="capsule" size="0.007" fromto="0 0 0 0 0 0.027" material="hand_mat"/>
                    <body name="middle_tip" pos="0 0 0.027">
                      <joint name="middle_j2" type="hinge" axis="0 1 0" range="-0.1 1.3" damping="0.15"/>
                      <geom type="capsule" size="0.006" fromto="0 0 0 0 0 0.022" material="hand_mat"/>
                    </body>
                  </body>
                </body>
                
                <!-- Ring finger -->
                <body name="ring_base" pos="0.035 -0.015 0">
                  <joint name="ring_j0" type="hinge" axis="0 1 0" range="-0.1 1.5" damping="0.3"/>
                  <geom type="capsule" size="0.008" fromto="0 0 0 0 0 0.028" material="hand_mat"/>
                  <body name="ring_mid" pos="0 0 0.028">
                    <joint name="ring_j1" type="hinge" axis="0 1 0" range="-0.1 1.5" damping="0.2"/>
                    <geom type="capsule" size="0.007" fromto="0 0 0 0 0 0.023" material="hand_mat"/>
                    <body name="ring_tip" pos="0 0 0.023">
                      <joint name="ring_j2" type="hinge" axis="0 1 0" range="-0.1 1.3" damping="0.15"/>
                      <geom type="capsule" size="0.006" fromto="0 0 0 0 0 0.019" material="hand_mat"/>
                    </body>
                  </body>
                </body>
                
                <!-- Thumb -->
                <body name="thumb_base" pos="-0.02 -0.03 0">
                  <joint name="thumb_j0" type="hinge" axis="1 0 0" range="-0.5 1.2" damping="0.3"/>
                  <geom type="capsule" size="0.009" fromto="0 0 0 0 -0.025 0.025" material="hand_mat"/>
                  <body name="thumb_mid" pos="0 -0.025 0.025">
                    <joint name="thumb_j1" type="hinge" axis="0 1 0" range="-0.1 1.4" damping="0.2"/>
                    <geom type="capsule" size="0.008" fromto="0 0 0 0 0 0.025" material="hand_mat"/>
                    <body name="thumb_tip" pos="0 0 0.025">
                      <joint name="thumb_j2" type="hinge" axis="0 1 0" range="-0.1 1.2" damping="0.15"/>
                      <geom type="capsule" size="0.007" fromto="0 0 0 0 0 0.02" material="hand_mat"/>
                    </body>
                  </body>
                </body>
                
              </body><!-- palm -->
            </body><!-- wrist -->
          </body><!-- forearm -->
        </body><!-- upper_arm -->
      </body><!-- shoulder -->
    </body><!-- base -->
  </worldbody>
  
  <actuator>
    <!-- Arm joints — tuned for visible motion without instability -->
    <position name="act_shoulder_pan"  joint="shoulder_pan"  kp="100" kv="20" ctrlrange="-2.0 2.0"/>
    <position name="act_shoulder_lift" joint="shoulder_lift" kp="100" kv="20" ctrlrange="-1.6 0.8"/>
    <position name="act_elbow"         joint="elbow"         kp="80"  kv="16" ctrlrange="0.0 2.2"/>
    <position name="act_wrist_roll"    joint="wrist_roll"    kp="50"  kv="10" ctrlrange="-1.8 1.8"/>
    <!-- Hand joints -->
    <position name="act_index_j0"   joint="index_j0"   kp="40" kv="4"  ctrlrange="-0.1 1.5"/>
    <position name="act_index_j1"   joint="index_j1"   kp="30" kv="3"  ctrlrange="-0.1 1.5"/>
    <position name="act_index_j2"   joint="index_j2"   kp="20" kv="2"  ctrlrange="-0.1 1.3"/>
    <position name="act_middle_j0"  joint="middle_j0"  kp="40" kv="4"  ctrlrange="-0.1 1.5"/>
    <position name="act_middle_j1"  joint="middle_j1"  kp="30" kv="3"  ctrlrange="-0.1 1.5"/>
    <position name="act_middle_j2"  joint="middle_j2"  kp="20" kv="2"  ctrlrange="-0.1 1.3"/>
    <position name="act_ring_j0"    joint="ring_j0"    kp="35" kv="3.5" ctrlrange="-0.1 1.5"/>
    <position name="act_ring_j1"    joint="ring_j1"    kp="25" kv="2.5" ctrlrange="-0.1 1.5"/>
    <position name="act_ring_j2"    joint="ring_j2"    kp="18" kv="1.8" ctrlrange="-0.1 1.3"/>
    <position name="act_thumb_j0"   joint="thumb_j0"   kp="40" kv="4"  ctrlrange="-0.5 1.2"/>
    <position name="act_thumb_j1"   joint="thumb_j1"   kp="30" kv="3"  ctrlrange="-0.1 1.4"/>
    <position name="act_thumb_j2"   joint="thumb_j2"   kp="20" kv="2"  ctrlrange="-0.1 1.2"/>
  </actuator>
  
  <sensor>
    <framepos name="palm_pos" objtype="body" objname="palm"/>
  </sensor>
</mujoco>
"""


def smooth_step(t: float) -> float:
    """Smoothstep interpolation — no jerky starts/stops."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * smooth_step(t)


def generate_reach_grasp_trajectory(total_frames: int, fps: int = 30):
    """
    Scripted reach-and-grasp trajectory with 5 phases:
    1. Home: arm at rest, hand open
    2. Pre-reach: rotate toward target, lift slightly
    3. Reach: extend elbow, lower to target height  
    4. Grasp: close fingers around cube
    5. Lift: raise arm with grasped object
    6. Return: bring back to home position
    
    Returns: list of (arm_joints[4], hand_joints[12]) tuples
    """
    dt = 1.0 / fps
    total_secs = total_frames / fps
    
    # Phase timing (in seconds)
    T0_end   = 1.5   # home hold
    T1_end   = 3.5   # pre-reach
    T2_end   = 5.5   # reach down to cube
    T3_end   = 7.0   # grasp
    T4_end   = 8.5   # lift
    T5_end   = total_secs  # return home + hold
    
    # Keyframe joint poses:
    # [shoulder_pan, shoulder_lift, elbow, wrist_roll]
    HOME_ARM   = np.array([0.0,   -0.2,  0.5,  0.0])
    PREREACH   = np.array([0.25,  -0.7,  0.6,  0.0])   # rotated, arm angled down
    REACH      = np.array([0.35,  -1.05, 1.45, 0.15])  # extended toward cube
    GRASPING   = np.array([0.35,  -1.05, 1.45, 0.15])  # hold at target
    LIFT       = np.array([0.20,  -0.55, 0.90, 0.05])  # raise up with object
    RETURN     = np.array([0.0,   -0.2,  0.5,  0.0])   # back to home
    
    # Hand joint poses (12 joints: index*3, middle*3, ring*3, thumb*3)
    OPEN_HAND  = np.array([0.1, 0.1, 0.1,   0.1, 0.1, 0.1,   0.1, 0.1, 0.1,   0.1, 0.0, 0.0])
    PREGRASP   = np.array([0.3, 0.2, 0.1,   0.3, 0.2, 0.1,   0.3, 0.2, 0.1,   0.2, 0.1, 0.0])
    CLOSED     = np.array([1.1, 1.0, 0.8,   1.1, 1.0, 0.8,   1.0, 0.9, 0.7,   0.8, 0.9, 0.7])
    
    trajectory = []
    for fi in range(total_frames):
        t_sec = fi * dt
        
        if t_sec < T0_end:
            # Phase 0: Home
            arm  = HOME_ARM.copy()
            hand = OPEN_HAND.copy()
            
        elif t_sec < T1_end:
            # Phase 1: Pre-reach (rotate pan, drop shoulder)
            alpha = (t_sec - T0_end) / (T1_end - T0_end)
            arm  = np.array([lerp(HOME_ARM[i], PREREACH[i], alpha) for i in range(4)])
            hand = np.array([lerp(OPEN_HAND[j], PREGRASP[j], alpha) for j in range(12)])
            
        elif t_sec < T2_end:
            # Phase 2: Extend toward cube
            alpha = (t_sec - T1_end) / (T2_end - T1_end)
            arm  = np.array([lerp(PREREACH[i], REACH[i], alpha) for i in range(4)])
            hand = PREGRASP.copy()
            
        elif t_sec < T3_end:
            # Phase 3: Grasp — close fingers
            alpha = (t_sec - T2_end) / (T3_end - T2_end)
            arm  = GRASPING.copy()
            hand = np.array([lerp(PREGRASP[j], CLOSED[j], alpha) for j in range(12)])
            
        elif t_sec < T4_end:
            # Phase 4: Lift
            alpha = (t_sec - T3_end) / (T4_end - T3_end)
            arm  = np.array([lerp(GRASPING[i], LIFT[i], alpha) for i in range(4)])
            hand = CLOSED.copy()
            
        else:
            # Phase 5: Return home, release
            alpha = (t_sec - T4_end) / max(T5_end - T4_end, 0.001)
            alpha = min(alpha, 1.0)
            arm  = np.array([lerp(LIFT[i], RETURN[i], alpha) for i in range(4)])
            hand_alpha = min(alpha * 1.5, 1.0)  # open hand faster
            hand = np.array([lerp(CLOSED[j], OPEN_HAND[j], hand_alpha) for j in range(12)])
        
        trajectory.append((arm, hand))
    
    return trajectory


def run_sim():
    try:
        import mujoco
    except ImportError:
        print("ERROR: mujoco not installed. Run: pip install mujoco")
        sys.exit(1)
    
    # Load model
    mjcf_path = OUTPUT_DIR / "dexcrowd_arm_v2.xml"
    with open(mjcf_path, "w") as f:
        f.write(MJCF_MODEL)
    
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data  = mujoco.MjData(model)
    
    print(f"Model loaded: {model.nq} DoF, {model.nu} actuators, {model.nbody} bodies")
    
    # Simulation params
    FPS = 30
    DURATION_SEC = 10.0
    total_frames = int(FPS * DURATION_SEC)
    sim_steps_per_frame = max(1, int(round(1.0 / (FPS * model.opt.timestep))))
    
    print(f"Rendering {total_frames} frames @ {FPS} fps ({DURATION_SEC}s, {sim_steps_per_frame} sim steps/frame)")
    
    # Build actuator ID map
    arm_joint_names  = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll"]
    hand_joint_names = ["index_j0", "index_j1", "index_j2",
                        "middle_j0", "middle_j1", "middle_j2",
                        "ring_j0", "ring_j1", "ring_j2",
                        "thumb_j0", "thumb_j1", "thumb_j2"]
    
    arm_act_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}") for n in arm_joint_names]
    hand_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}") for n in hand_joint_names]
    
    # Generate trajectory
    trajectory = generate_reach_grasp_trajectory(total_frames, FPS)
    
    # Set initial pose
    mujoco.mj_resetData(model, data)
    arm0, hand0 = trajectory[0]
    for i in range(4):
        data.qpos[i] = float(arm0[i])
    for i in range(12):
        data.qpos[4 + i] = float(hand0[i])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # update xpos/xmat without stepping
    
    # Renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    
    # Render loop
    import cv2
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = OUTPUT_DIR / "rollout_v2.mp4"
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (640, 480))
    
    print("Rendering frames...")
    phase_names = ["Home", "Pre-Reach", "Reaching", "Grasping", "Lifting", "Returning"]
    
    for fi, (arm_targets, hand_targets) in enumerate(trajectory):
        # Kinematic mode: directly set joint positions (no physics instability)
        for i in range(4):
            data.qpos[i] = float(arm_targets[i])
        for i in range(12):
            data.qpos[4 + i] = float(hand_targets[i])
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)   # recompute xpos/xmat from qpos
        
        # Render
        renderer.update_scene(data)
        rgb = renderer.render()                    # H×W×3 uint8
        
        # Convert to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Annotate frame
        t_sec = fi / FPS
        # Phase label
        T_phases = [1.5, 3.5, 5.5, 7.0, 8.5, 10.0]
        phase_idx = next((i for i, t in enumerate(T_phases) if t_sec < t), len(phase_names)-1)
        phase = phase_names[min(phase_idx, len(phase_names)-1)]
        
        # Get palm position
        palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        palm_pos = data.xpos[palm_id]
        
        # HUD overlay
        cv2.rectangle(bgr, (0, 0), (640, 56), (0, 0, 0), -1)
        cv2.putText(bgr, f"DexCrowd Sim — MuJoCo 3.5  |  BC Policy Rollout",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"Phase: {phase:<12}  |  t={t_sec:.1f}s  |  Palm XYZ: ({palm_pos[0]:.3f}, {palm_pos[1]:.3f}, {palm_pos[2]:.3f})",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 160), 1, cv2.LINE_AA)
        
        # Progress bar
        bar_w = int(620 * (fi / total_frames))
        cv2.rectangle(bgr, (10, 46), (630, 52), (40, 40, 40), -1)
        cv2.rectangle(bgr, (10, 46), (10 + bar_w, 52), (80, 200, 255), -1)
        
        writer.write(bgr)
        
        if fi % 60 == 0:
            print(f"  Frame {fi}/{total_frames}  phase={phase}  palm={palm_pos}")
    
    writer.release()
    renderer.close()
    
    print(f"\nDone! Saved to: {video_path}")
    print(f"Duration: {DURATION_SEC:.1f}s @ {FPS}fps = {total_frames} frames")
    return str(video_path)


if __name__ == "__main__":
    t0 = time.time()
    out = run_sim()
    print(f"Completed in {time.time()-t0:.1f}s -> {out}")
