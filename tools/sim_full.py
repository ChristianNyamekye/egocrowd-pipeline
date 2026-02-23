"""
Full simulation pipeline: generate demo → train BC → evaluate → render video.
Renders the TRAINED POLICY (not just the scripted demo).
Run on RunPod with: MUJOCO_GL=osmesa python3 /tmp/sim_full.py
"""
import numpy as np
import json
import os

# ── MuJoCo model ──────────────────────────────────────────────
HAND_XML = """
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

MUG_INIT = np.array([0.3, 0.0, 0.05])

def make_env():
    import mujoco
    model = mujoco.MjModel.from_xml_string(HAND_XML)
    data = mujoco.MjData(model)
    return model, data


# ── Demo generation (scripted expert) ─────────────────────────
def scripted_action(step, total=300):
    """Smooth reach→grasp→lift with more steps for better learning."""
    phase = step / total
    if phase < 0.35:
        t = phase / 0.35
        # Smooth cosine interpolation
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3 * s, 0.0, 0.15 - 0.10 * s, 0.0, 0.0])
    elif phase < 0.55:
        t = (phase - 0.35) / 0.20
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3, 0.0, 0.05, 1.2 * s, 1.2 * s])
    else:
        t = (phase - 0.55) / 0.45
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3, 0.0, 0.05 + 0.20 * s, 1.2, 1.2])


def generate_demos(model, data, num_demos=5, steps_per=300):
    """Generate multiple demonstrations with slight noise for diversity."""
    import mujoco
    all_obs, all_acts = [], []

    for d in range(num_demos):
        mujoco.mj_resetData(model, data)
        for step in range(steps_per):
            action = scripted_action(step, steps_per)
            # Add small noise for diversity (except first demo)
            if d > 0:
                action = action + np.random.normal(0, 0.005, size=action.shape)
                action = np.clip(action, [-0.5, -0.5, 0, 0, 0], [0.5, 0.5, 0.5, 1.57, 1.57])

            obs = np.concatenate([data.qpos[:5], MUG_INIT])  # 5 joints + 3 mug pos
            all_obs.append(obs.copy())
            all_acts.append(action.copy())

            data.ctrl[:] = action
            mujoco.mj_step(model, data)

    return np.array(all_obs, dtype=np.float32), np.array(all_acts, dtype=np.float32)


# ── Behavioral Cloning ────────────────────────────────────────
def train_bc(observations, actions, epochs=500, lr=3e-4, batch_size=128):
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

    # Normalize
    obs_mean, obs_std = obs_t.mean(0), obs_t.std(0) + 1e-8
    act_mean, act_std = act_t.mean(0), act_t.std(0) + 1e-8
    obs_n = (obs_t - obs_mean) / obs_std
    act_n = (act_t - act_mean) / act_std

    policy = Policy(observations.shape[1], actions.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    n = len(obs_n)
    print(f"Training BC: {n} samples, obs_dim={observations.shape[1]}, act_dim={actions.shape[1]}")

    for epoch in range(epochs):
        # Mini-batch SGD
        idx = np.random.permutation(n)
        epoch_loss = 0
        batches = 0
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            pred = policy(obs_n[batch_idx])
            loss = loss_fn(pred, act_n[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            avg = epoch_loss / batches
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.6f}")

    final_loss = epoch_loss / batches
    print(f"  Final loss: {final_loss:.6f}")
    return policy, obs_mean, obs_std, act_mean, act_std


# ── Evaluation ────────────────────────────────────────────────
def rollout_policy(model, data, policy, obs_mean, obs_std, act_mean, act_std, num_steps=300):
    """Roll out the trained policy and return per-step data."""
    import mujoco
    import torch

    mujoco.mj_resetData(model, data)
    trajectory = []

    for step in range(num_steps):
        obs = np.concatenate([data.qpos[:5], MUG_INIT])
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        obs_n = (obs_t - obs_mean) / obs_std

        with torch.no_grad():
            act_n = policy(obs_n)
        action = (act_n * act_std + act_mean).numpy()[0]
        action = np.clip(action, [-0.5, -0.5, 0, 0, 0], [0.5, 0.5, 0.5, 1.57, 1.57])

        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        wrist_pos = data.qpos[:3].copy()
        mug_pos = data.qpos[5:8].copy() if len(data.qpos) > 7 else MUG_INIT.copy()

        trajectory.append({
            "step": step,
            "wrist_pos": wrist_pos.tolist(),
            "mug_pos": mug_pos.tolist(),
            "action": action.tolist(),
            "fingers": data.qpos[3:5].copy().tolist(),
        })

    # Metrics
    final = trajectory[-1]
    wrist_xy = np.array(final["wrist_pos"][:2])
    mug_xy = MUG_INIT[:2]
    dist = np.linalg.norm(wrist_xy - mug_xy)
    reached = dist < 0.06
    max_mug_h = max(t["mug_pos"][2] for t in trajectory)
    total_move = sum(
        np.linalg.norm(np.array(trajectory[i+1]["wrist_pos"]) - np.array(trajectory[i]["wrist_pos"]))
        for i in range(len(trajectory)-1)
    )

    results = {
        "reached_target": bool(reached),
        "final_dist_to_mug": float(dist),
        "mug_lifted": bool(max_mug_h > 0.08),
        "max_mug_height": float(max_mug_h),
        "total_wrist_movement": float(total_move),
        "final_wrist_pos": final["wrist_pos"],
        "final_fingers": final["fingers"],
    }
    return trajectory, results


# ── Rendering ─────────────────────────────────────────────────
def render_rollout(model, data, trajectory_actions, num_steps, output_path, title="Policy Rollout"):
    """Render a video from a list of actions."""
    import mujoco
    import cv2

    mujoco.mj_resetData(model, data)
    renderer = mujoco.Renderer(model, 480, 640)

    frames = []
    for step in range(num_steps):
        action = trajectory_actions[step]
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        if step % 2 == 0:
            renderer.update_scene(data)
            frame = renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Overlay text
            phase = step / num_steps
            phase_name = "REACH" if phase < 0.35 else "GRASP" if phase < 0.55 else "LIFT"
            colors = {"REACH": (0,180,0), "GRASP": (0,140,255), "LIFT": (255,0,0)}
            cv2.putText(frame_bgr, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(frame_bgr, f"Step {step}/{num_steps}  |  {phase_name}",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[phase_name], 2)
            frames.append(frame_bgr)

    renderer.close()

    # Write video
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path + ".raw.mp4", fourcc, 15, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    # Re-encode H.264
    os.system(f"ffmpeg -y -i '{output_path}.raw.mp4' -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p '{output_path}' 2>/dev/null")
    os.remove(f"{output_path}.raw.mp4")
    print(f"  Saved: {output_path} ({len(frames)} frames)")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import mujoco
    NUM_STEPS = 300

    os.makedirs("/workspace/sim/output", exist_ok=True)

    print("=" * 60)
    print("SIMULATION: Human Demo → BC Policy → Robot Execution")
    print("=" * 60)

    # 1. Environment
    print("\n[1/5] Creating MuJoCo environment...")
    model, data = make_env()
    print(f"  DOF: {model.nq}, actuators: {model.nu}")

    # 2. Generate demos
    print("\n[2/5] Generating expert demonstrations (5 demos × 300 steps)...")
    obs, acts = generate_demos(model, data, num_demos=5, steps_per=NUM_STEPS)
    print(f"  Total samples: {len(obs)}")

    # 3. Train
    print("\n[3/5] Training behavioral cloning policy...")
    policy, om, os_, am, as_ = train_bc(obs, acts, epochs=500, lr=3e-4, batch_size=128)

    # 4. Evaluate
    print("\n[4/5] Evaluating trained policy...")
    mujoco.mj_resetData(model, data)
    traj, results = rollout_policy(model, data, policy, om, os_, am, as_, num_steps=NUM_STEPS)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")

    with open("/workspace/sim/output/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 5. Render videos
    print("\n[5/5] Rendering videos...")

    # 5a. Scripted demo (ground truth)
    print("  Rendering scripted demo...")
    demo_actions = [scripted_action(s, NUM_STEPS) for s in range(NUM_STEPS)]
    render_rollout(model, data, demo_actions, NUM_STEPS,
                   "/workspace/sim/output/demo_scripted.mp4", "Expert Demo (Ground Truth)")

    # 5b. Trained policy
    print("  Rendering trained policy...")
    policy_actions = [np.array(t["action"]) for t in traj]
    render_rollout(model, data, policy_actions, NUM_STEPS,
                   "/workspace/sim/output/policy_rollout.mp4", "Trained BC Policy")

    if results["reached_target"]:
        print("\n✅ SUCCESS: Policy reached the target!")
    else:
        print(f"\n⚠️  Policy didn't reach target (dist={results['final_dist_to_mug']:.3f}m). Needs tuning.")

    print("\nDone. Videos at /workspace/sim/output/")
