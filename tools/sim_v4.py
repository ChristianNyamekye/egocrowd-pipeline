"""
Sim v4: Direct action prediction (no normalization), bigger network, DAgger-style.
The key insight: normalized actions were losing the absolute position signal.
MUJOCO_GL=osmesa python3 -u /tmp/sim_v4.py
"""
import numpy as np
import json
import os

HAND_XML = """
<mujoco model="simple_hand">
  <option timestep="0.005" gravity="0 0 -9.81" solver="Newton" iterations="50"/>
  <default>
    <joint armature="0.1"/>
    <geom friction="1 0.5 0.5"/>
  </default>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0.5 -0.5 1.5" dir="-0.3 0.3 -0.7" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.5 0.5 1.5" dir="0.3 -0.3 -0.7" diffuse="0.4 0.4 0.4"/>
    <geom type="plane" size="1 1 0.1" rgba="0.95 0.95 0.95 1"/>
    <body name="mug" pos="0.3 0 0.05">
      <geom type="cylinder" size="0.03 0.04" rgba="0.85 0.15 0.15 1"/>
    </body>
    <body name="wrist" pos="0 0 0.15">
      <joint name="wrist_x" type="slide" axis="1 0 0" range="-0.5 0.5" damping="1"/>
      <joint name="wrist_y" type="slide" axis="0 1 0" range="-0.5 0.5" damping="1"/>
      <joint name="wrist_z" type="slide" axis="0 0 1" range="-0.2 0.5" damping="1"/>
      <geom type="box" size="0.04 0.03 0.02" rgba="0.2 0.2 0.7 1" mass="0.1"/>
      <body name="finger1" pos="0.04 0 -0.02">
        <joint name="f1" type="hinge" axis="0 1 0" range="0 1.57" damping="2"/>
        <geom type="capsule" size="0.008" fromto="0 0 0 0 0 -0.05" rgba="0.3 0.3 0.8 1" mass="0.05"/>
      </body>
      <body name="finger2" pos="-0.04 0 -0.02">
        <joint name="f2" type="hinge" axis="0 -1 0" range="0 1.57" damping="2"/>
        <geom type="capsule" size="0.008" fromto="0 0 0 0 0 -0.05" rgba="0.3 0.3 0.8 1" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="wrist_x" kp="40" kv="3"/>
    <position joint="wrist_y" kp="40" kv="3"/>
    <position joint="wrist_z" kp="40" kv="3"/>
    <position joint="f1" kp="10" kv="1"/>
    <position joint="f2" kp="10" kv="1"/>
  </actuator>
</mujoco>
"""

MUG_POS = np.array([0.3, 0.0, 0.05])
ACT_LOW = np.array([-0.5, -0.5, -0.2, 0.0, 0.0])
ACT_HIGH = np.array([0.5, 0.5, 0.5, 1.57, 1.57])

def make_env():
    import mujoco
    return mujoco.MjModel.from_xml_string(HAND_XML), None

def reset(m):
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return d

def step_sim(m, d, action, substeps=20):
    import mujoco
    d.ctrl[:] = action
    for _ in range(substeps):
        mujoco.mj_step(m, d)

def get_obs(d, step, total):
    """Rich observation: joint state + target + phase progress"""
    wrist = d.qpos[:3].copy()
    fingers = d.qpos[3:5].copy()
    vel = d.qvel[:3].copy()
    delta = MUG_POS - wrist
    phase = np.array([step / total])
    return np.concatenate([wrist, fingers, vel, delta, phase]).astype(np.float32)

def scripted_action(step, total=200):
    p = step / total
    if p < 0.4:
        t = p / 0.4
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3*s, 0.0, 0.15 - 0.10*s, 0.0, 0.0])
    elif p < 0.6:
        t = (p - 0.4) / 0.2
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3, 0.0, 0.05, 1.2*s, 1.2*s])
    else:
        t = (p - 0.6) / 0.4
        s = 0.5 * (1 - np.cos(np.pi * t))
        return np.array([0.3, 0.0, 0.05 + 0.15*s, 1.2, 1.2])

def generate_demos(m, n_demos=30, steps=200):
    all_obs, all_acts = [], []
    for i in range(n_demos):
        d = reset(m)
        for s in range(steps):
            obs = get_obs(d, s, steps)
            act = scripted_action(s, steps)
            if i > 0:
                act = act + np.random.normal(0, 0.003, size=act.shape)
                act = np.clip(act, ACT_LOW, ACT_HIGH)
            all_obs.append(obs)
            all_acts.append(act.astype(np.float32))
            step_sim(m, d, act)
    return np.array(all_obs), np.array(all_acts)

def train_bc(obs, acts, epochs=3000, lr=1e-3, bs=512):
    """NO normalization on actions — predict raw values directly."""
    import torch, torch.nn as nn

    # Only normalize observations, NOT actions
    ot = torch.FloatTensor(obs)
    at = torch.FloatTensor(acts)
    om, os = ot.mean(0), ot.std(0) + 1e-8
    on = (ot - om) / os

    class Policy(nn.Module):
        def __init__(self, od, ad):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(od, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, ad)
            )
        def forward(self, x):
            return self.net(x)

    pol = Policy(obs.shape[1], acts.shape[1])
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(on)

    print(f"Training BC: {n} samples, obs={obs.shape[1]}, act={acts.shape[1]}")
    print(f"Action ranges: min={acts.min(0)}, max={acts.max(0)}")

    for ep in range(epochs):
        idx = np.random.permutation(n)
        el, nb = 0, 0
        for i in range(0, n, bs):
            bi = idx[i:i+bs]
            pred = pol(on[bi])
            # MSE on RAW actions
            l = ((pred - at[bi])**2).mean()
            opt.zero_grad(); l.backward(); opt.step()
            el += l.item(); nb += 1
        sch.step()
        if (ep+1) % 300 == 0:
            avg = el/nb
            # Test: predict action for a mid-reach obs
            with torch.no_grad():
                test_obs = on[100:101]  # early reach
                test_pred = pol(test_obs).numpy()[0]
                test_true = acts[100]
            print(f"  Epoch {ep+1}/{epochs}: loss={avg:.6f} | sample pred={test_pred[:3]} true={test_true[:3]}")

    final = el/nb
    print(f"  Final loss: {final:.6f}")
    return pol, om, os

def rollout(m, pol, om, os, steps=200):
    import torch
    d = reset(m)
    traj = []
    for s in range(steps):
        obs = get_obs(d, s, steps)
        ot = torch.FloatTensor(obs).unsqueeze(0)
        on = (ot - om) / os
        with torch.no_grad():
            act = pol(on).numpy()[0]
        act = np.clip(act, ACT_LOW, ACT_HIGH)
        step_sim(m, d, act)
        traj.append({"step": s, "wrist": d.qpos[:3].copy().tolist(),
                      "fingers": d.qpos[3:5].copy().tolist(), "action": act.tolist()})

    dists = [np.linalg.norm(np.array(t["wrist"]) - MUG_POS) for t in traj]
    min_dist = min(dists)
    min_step = dists.index(min_dist)
    mv = sum(np.linalg.norm(np.array(traj[i+1]["wrist"])-np.array(traj[i]["wrist"])) for i in range(len(traj)-1))
    return traj, {
        "reached": bool(min_dist < 0.06),
        "min_dist": float(min_dist), "min_dist_step": min_step,
        "final_dist": float(dists[-1]), "movement": float(mv),
        "final_wrist": traj[-1]["wrist"], "final_fingers": traj[-1]["fingers"]
    }

def render(m, actions, steps, path, title):
    import mujoco, cv2
    d = reset(m)
    renderer = mujoco.Renderer(m, 480, 640)
    frames = []
    for s in range(steps):
        step_sim(m, d, actions[s])
        if s % 2 == 0:
            renderer.update_scene(d)
            f = renderer.render().copy()
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            p = s/steps
            pn = "REACH" if p<0.4 else "GRASP" if p<0.6 else "LIFT"
            c = {"REACH":(0,180,0),"GRASP":(0,140,255),"LIFT":(255,0,0)}[pn]
            cv2.putText(f, title, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(f, f"Step {s}/{steps} | {pn}", (15,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            w = d.qpos[:3]
            dist = np.linalg.norm(w - MUG_POS)
            cv2.putText(f, f"Dist: {dist:.3f}m", (15,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            frames.append(f)
    renderer.close()
    h, w2 = frames[0].shape[:2]
    wr = cv2.VideoWriter(path+".raw.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (w2, h))
    for f in frames: wr.write(f)
    wr.release()
    os.system(f"ffmpeg -y -i '{path}.raw.mp4' -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p '{path}' 2>/dev/null")
    if os.path.exists(f"{path}.raw.mp4"): os.remove(f"{path}.raw.mp4")
    print(f"  Video: {path} ({len(frames)} frames)")

if __name__ == "__main__":
    S = 200
    os.makedirs("/workspace/sim/output", exist_ok=True)
    print("="*60)
    print("SIM v4 — Raw action prediction, bigger net, phase input")
    print("="*60)
    m, _ = make_env()

    # Verify scripted
    d = reset(m)
    for s in range(S):
        step_sim(m, d, scripted_action(s, S))
    fw = d.qpos[:3]
    print(f"Scripted final: {fw}, dist={np.linalg.norm(fw-MUG_POS):.4f}")
    dists = []
    d = reset(m)
    for s in range(S):
        step_sim(m, d, scripted_action(s, S))
        dists.append(np.linalg.norm(d.qpos[:3]-MUG_POS))
    print(f"Scripted min_dist: {min(dists):.4f} at step {dists.index(min(dists))}")

    print(f"\nGenerating 30 demos...")
    obs, acts = generate_demos(m, n_demos=30, steps=S)
    print(f"Total: {len(obs)} samples, NaN: {np.isnan(obs).any() or np.isnan(acts).any()}")

    print(f"\nTraining (3000 epochs, raw actions)...")
    pol, om, os_ = train_bc(obs, acts, epochs=3000, lr=1e-3, bs=512)

    print(f"\nEvaluating...")
    traj, res = rollout(m, pol, om, os_, steps=S)
    print(f"\nRESULTS: {json.dumps(res, indent=2)}")
    with open("/workspace/sim/output/results.json","w") as f: json.dump(res,f,indent=2)

    print(f"\nRendering...")
    demo_acts = [scripted_action(s,S) for s in range(S)]
    render(m, demo_acts, S, "/workspace/sim/output/demo_scripted.mp4", "Expert Demo")
    pol_acts = [np.array(t["action"]) for t in traj]
    render(m, pol_acts, S, "/workspace/sim/output/policy_rollout.mp4", "BC Policy (v4)")
    print(f"\n{'✅ SUCCESS' if res['reached'] else '⚠️ MISS'}: min_dist={res['min_dist']:.3f}m at step {res['min_dist_step']}")
