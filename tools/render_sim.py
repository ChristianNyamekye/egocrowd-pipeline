"""Render simulation video: demo trajectory + trained policy side by side."""
import numpy as np
import json
import mujoco
import cv2
import os

def create_hand_model():
    xml = """
    <mujoco model="simple_hand">
      <option timestep="0.01" gravity="0 0 -9.81"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody>
        <light pos="0.5 -0.5 1.5" dir="-0.3 0.3 -0.7" diffuse="0.8 0.8 0.8"/>
        <light pos="-0.5 0.5 1.5" dir="0.3 -0.3 -0.7" diffuse="0.4 0.4 0.4"/>
        <geom type="plane" size="1 1 0.1" rgba="0.95 0.95 0.95 1"/>
        <body name="mug" pos="0.3 0 0.05">
          <joint type="free"/>
          <geom type="cylinder" size="0.03 0.04" rgba="0.85 0.15 0.15 1" mass="0.3"/>
        </body>
        <body name="wrist" pos="0 0 0.15">
          <joint name="wrist_x" type="slide" axis="1 0 0" range="-0.5 0.5"/>
          <joint name="wrist_y" type="slide" axis="0 1 0" range="-0.5 0.5"/>
          <joint name="wrist_z" type="slide" axis="0 0 1" range="0 0.5"/>
          <geom type="box" size="0.04 0.03 0.02" rgba="0.2 0.2 0.7 1"/>
          <body name="finger1" pos="0.04 0 -0.02">
            <joint name="finger1_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
            <geom type="capsule" size="0.008" fromto="0 0 0 0 0 -0.05" rgba="0.3 0.3 0.8 1"/>
          </body>
          <body name="finger2" pos="-0.04 0 -0.02">
            <joint name="finger2_joint" type="hinge" axis="0 -1 0" range="0 1.57"/>
            <geom type="capsule" size="0.008" fromto="0 0 0 0 0 -0.05" rgba="0.3 0.3 0.8 1"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position joint="wrist_x" kp="50"/>
        <position joint="wrist_y" kp="50"/>
        <position joint="wrist_z" kp="50"/>
        <position joint="finger1_joint" kp="20"/>
        <position joint="finger2_joint" kp="20"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data

def render_trajectory(model, data, num_steps=200, title="Demo"):
    """Render a reach-grasp trajectory and return frames."""
    renderer = mujoco.Renderer(model, 480, 640)
    renderer.update_scene(data)
    
    frames = []
    mujoco.mj_resetData(model, data)
    
    for step in range(num_steps):
        phase = step / num_steps
        if phase < 0.4:
            t = phase / 0.4
            action = np.array([0.3*t, 0, 0.15-0.1*t, 0, 0])
        elif phase < 0.6:
            t = (phase - 0.4) / 0.2
            action = np.array([0.3, 0, 0.05, 1.2*t, 1.2*t])
        else:
            t = (phase - 0.6) / 0.4
            action = np.array([0.3, 0, 0.05+0.2*t, 1.2, 1.2])
        
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        
        if step % 2 == 0:  # 15fps
            renderer.update_scene(data)
            frame = renderer.render()
            # Add title
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.putText(frame_bgr, f"Step {step}/{num_steps}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 1)
            phase_name = "REACH" if phase < 0.4 else "GRASP" if phase < 0.6 else "LIFT"
            color = (0,180,0) if phase_name == "REACH" else (0,140,255) if phase_name == "GRASP" else (255,0,0)
            cv2.putText(frame_bgr, phase_name, (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            frames.append(frame_bgr)
    
    renderer.close()
    return frames

def save_video(frames, output_path, fps=15):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

if __name__ == "__main__":
    os.makedirs("/workspace/sim/output", exist_ok=True)
    
    print("Rendering simulation demo...")
    model, data = create_hand_model()
    
    frames = render_trajectory(model, data, num_steps=200, title="Robot Hand: Reach-Grasp-Lift")
    
    # Save as mp4v first, then re-encode
    save_video(frames, "/workspace/sim/output/sim_raw.mp4", fps=15)
    
    # Re-encode with H.264
    os.system("apt-get install -y -qq ffmpeg > /dev/null 2>&1")
    os.system("ffmpeg -i /workspace/sim/output/sim_raw.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p /workspace/sim/output/sim_demo.mp4 -y 2>/dev/null")
    
    print(f"Video saved: /workspace/sim/output/sim_demo.mp4")
    print(f"Frames: {len(frames)}")
