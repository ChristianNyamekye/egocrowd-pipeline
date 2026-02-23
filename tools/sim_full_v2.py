"""
Simulation v2: Static mug (no free joint instability), cleaner BC training.
Run on RunPod with: MUJOCO_GL=osmesa python3 /tmp/sim_full_v2.py
"""
import numpy as np
import json
import os

HAND_XML = """
<mujoco model="simple_hand">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0.5 -0.5 1.5" dir="-0.3 0.3 -0.7" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.5 0.5 1.5" dir="0.3 -0.3 -0.7" diffuse="0.4 0.4 0.4"/>
    <geom type="plane" size="1 1 0.1" rgba="0.95 0.95 0.95 1"/>

    <!-- Static target (no free joint = no instability) -->
    <body name="mug" pos="0.3 0 0.05">
      <geom type="cylinder" size="0.03 0.04" rgba="0.85 0.15 0.15 1"/>
    </body>

    <!-- Robot hand -->
    <body name="wrist" pos="0 0 0.15">
      <joint name="wrist_x" type="slide" axis="1 0 0" range="-0.5 0.5"/>
      <joint name="wrist_y" type="slide" axis="0 1 0" range="-0.5 0.5"/>
      <joint name="wrist_z" type="slide" axis="0 0 1" range="-0.2 0.5"/>
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
    <position joint="wrist_x" kp="100"/>
    <position joint="wrist_y" kp="100"/>
    <position joint="wrist_z" kp="100"/>
    <position joint="finger1_joint" kp="30"/>
    <position joint="finger2_joint" kp="30"/>
  </actuator>
</mujoco>
"""

# qpos is now just [wrist_x, wrist_y, wrist_z, finger1, finger2] — 5 DOF, no free joint
MUG_POS = np.array([0.3, 0.0, 0.05])
ACT_LOW = np.array([-0.5, -0.5, -0.2, 0.0, 0.0])
ACT_HIGH = np.array([0.5, 0.5, 0.5, 1.57, 1.57])

def make_env():
    import mujoco
    model = mujoco.MjModel.from_xml_string(HAND_XML)
    data = mujoco.MjData(model)
    return model, data

def get_obs(data):
    """Observation: [wrist_xyz, finger1, finger2, target_xyz, dist_to_target]"""
    wrist = data.qpos[:3].copy()
    fingers = data.qpos[3:5].copy()
    dist = np.linalg.norm(wrist - MUG_POS)
    return np.concatenate([wrist, fingers, MUG_POS, [dist]]).astype(np.float32)

def scripted_action(step, total=300):
    phase = step / total
    if phase < 0.4:
        t = phase / 0.4
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3 * s, 0.0, -0.10 * s, 0.0, 0.0])
    elif phase < 0.6:
        t = (phase - 0.4) / 0.2
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3, 0.0, -0.10, 1.2 * s, 1.2 * s])
    else:
        t = (phase - 0.6) / 0.4
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3, 0.0, -0.10 + 0.25 * s, 1.2, 1.2])

def generate_demos(model, data, num_demos=10, steps_per=300):
    import mujoco
    all_obs, all_acts = [], []
    for d in range(num_demos):
        mujoco.mj_resetData(model, data)
        # Run a few warmup steps for physics to settle
        for _ in range(50):
            mujoco.mj_step(model, data)
        for step in range(steps_per):
            action = scripted_action(step, steps_per)
            if d > 0:
                action = action + np.random.normal(0, 0.003, size=action.shape)
                action = np.clip(action, ACT_LOW, ACT_HIGH)
            obs = get_obs(data)
            all_obs.append(obs)
            all_acts.append(action.astype(np.float32))
            data.ctrl[:] = action
            # Multiple physics substeps for stability
            for _ in range(5):
                mujoco.mj_step(model, data)
    return np.array(all_obs), np.array(all_acts)

def train_bc(observations, actions, epochs=1000, lr=1e-3, batch_size=256):
    import torch
    import torch.nn as nn

    class Policy(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, act_dim),
            )
        def forward(self, x):
            return self.net(x)

    obs_t = torch.FloatTensor(observations)
    act_t = torch.FloatTensor(actions)
    obs_mean, obs_std = obs_t.mean(0), obs_t.std(0) + 1e-8
    act_mean, act_std = act_t.mean(0), act_t.std(0) + 1e-8
    obs_n = (obs_t - obs_mean) / obs_std
    act_n = (act_t - act_mean) / act_std

    policy = Policy(observations.shape[1], actions.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = len(obs_n)

    print(f"Training BC: {n} samples, obs={observations.shape[1]}, act={actions.shape[1]}")
    best_loss = float('inf')
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        epoch_loss, batches = 0, 0
        for i in range(0, n, batch_size):
            bi = idx[i:i+batch_size]
            pred = policy(obs_n[bi])
            loss = loss_fn(pred, act_n[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        scheduler.step()
        avg = epoch_loss / batches
        if avg < best_loss:
            best_loss = avg
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.6f} (best={best_loss:.6f})")

    print(f"  Final: {best_loss:.6f}")
    return policy, obs_mean, obs_std, act_mean, act_std

def rollout_policy(model, data, policy, obs_mean, obs_std, act_mean, act_std, num_steps=300):
    import mujoco, torch
    mujoco.mj_resetData(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)

    trajectory = []
    for step in range(num_steps):
        obs = get_obs(data)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        obs_n = (obs_t - obs_mean) / obs_std
        with torch.no_grad():
            act_n = policy(obs_n)
        action = (act_n * act_std + act_mean).numpy()[0]
        action = np.clip(action, ACT_LOW, ACT_HIGH)
        data.ctrl[:] = action
        for _ in range(5):
            mujoco.mj_step(model, data)
        trajectory.append({"step": step, "wrist_pos": data.qpos[:3].copy().tolist(),
                           "fingers": data.qpos[3:5].copy().tolist(), "action": action.tolist()})

    final_wrist = np.array(trajectory[-1]["wrist_pos"])
    dist = np.linalg.norm(final_wrist - MUG_POS)
    results = {
        "reached_target": bool(dist < 0.06),
        "final_dist": float(dist),
        "total_movement": float(sum(
            np.linalg.norm(np.array(trajectory[i+1]["wrist_pos"]) - np.array(trajectory[i]["wrist_pos"]))
            for i in range(len(trajectory)-1)
        )),
        "final_wrist_pos": trajectory[-1]["wrist_pos"],
        "final_fingers": trajectory[-1]["fingers"],
    }
    return trajectory, results

def render_rollout(model, data, actions_list, num_steps, output_path, title):
    import mujoco, cv2
    mujoco.mj_resetData(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, 480, 640)
    frames = []
    for step in range(num_steps):
        data.ctrl[:] = actions_list[step]
        for _ in range(5):
            mujoco.mj_step(model, data)
        if step % 2 == 0:
            renderer.update_scene(data)
            frame = renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            phase = step / num_steps
            phase_name = "REACH" if phase < 0.4 else "GRASP" if phase < 0.6 else "LIFT"
            colors = {"REACH": (0,180,0), "GRASP": (0,140,255), "LIFT": (255,0,0)}
            cv2.putText(frame_bgr, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(frame_bgr, f"Step {step}/{num_steps} | {phase_name}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[phase_name], 2)
            wrist = data.qpos[:3]
            dist = np.linalg.norm(wrist - MUG_POS)
            cv2.putText(frame_bgr, f"Dist: {dist:.3f}m", (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            frames.append(frame_bgr)
    renderer.close()

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path + ".raw.mp4", fourcc, 15, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    os.system(f"ffmpeg -y -i '{output_path}.raw.mp4' -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p '{output_path}' 2>/dev/null")
    if os.path.exists(f"{output_path}.raw.mp4"):
        os.remove(f"{output_path}.raw.mp4")
    print(f"  Saved: {output_path} ({len(frames)} frames)")

if __name__ == "__main__":
    import mujoco
    NUM_STEPS = 300
    os.makedirs("/workspace/sim/output", exist_ok=True)

    print("=" * 60)
    print("SIMULATION v2: Fixed physics, better training")
    print("=" * 60)

    print("\n[1/5] Creating environment (static mug, no free joint)...")
    model, data = make_env()
    print(f"  DOF: {model.nq}, actuators: {model.nu}")

    print("\n[2/5] Generating 10 expert demos × 300 steps...")
    obs, acts = generate_demos(model, data, num_demos=10, steps_per=NUM_STEPS)
    print(f"  Total samples: {len(obs)}")

    # Quick sanity: check action variance
    print(f"  Action mean: {acts.mean(0)}")
    print(f"  Action std:  {acts.std(0)}")

    print("\n[3/5] Training BC policy (1000 epochs)...")
    policy, om, os_, am, as_ = train_bc(obs, acts, epochs=1000, lr=1e-3, batch_size=256)

    print("\n[4/5] Evaluating policy...")
    mujoco.mj_resetData(model, data)
    traj, results = rollout_policy(model, data, policy, om, os_, am, as_, num_steps=NUM_STEPS)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open("/workspace/sim/output/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[5/5] Rendering videos...")
    demo_actions = [scripted_action(s, NUM_STEPS) for s in range(NUM_STEPS)]
    render_rollout(model, data, demo_actions, NUM_STEPS,
                   "/workspace/sim/output/demo_scripted.mp4", "Expert Demo")

    policy_actions = [np.array(t["action"]) for t in traj]
    render_rollout(model, data, policy_actions, NUM_STEPS,
                   "/workspace/sim/output/policy_rollout.mp4", "Trained BC Policy")

    if results["reached_target"]:
        print(f"\n✅ SUCCESS: Hand reached target! (dist={results['final_dist']:.3f}m)")
    else:
        print(f"\n⚠️ Not quite (dist={results['final_dist']:.3f}m). May need more training.")
    print("Done.")
