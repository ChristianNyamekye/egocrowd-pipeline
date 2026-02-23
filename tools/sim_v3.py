"""
Simulation v3: Stable physics. Lower gains, damping, larger timestep.
MUJOCO_GL=osmesa python3 -u /tmp/sim_v3.py
"""
import numpy as np
import json
import os

HAND_XML = """
<mujoco model="simple_hand">
  <option timestep="0.005" gravity="0 0 -9.81" solver="Newton" iterations="50"/>
  <default>
    <joint damping="5" armature="0.1"/>
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

def make_env():
    import mujoco
    m = mujoco.MjModel.from_xml_string(HAND_XML)
    d = mujoco.MjData(m)
    return m, d

def get_obs(d):
    wrist = d.qpos[:3].copy()
    fingers = d.qpos[3:5].copy()
    delta = MUG_POS - wrist
    dist = np.linalg.norm(delta)
    return np.concatenate([wrist, fingers, delta, [dist]]).astype(np.float32)

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

def step_sim(m, d, action, substeps=20):
    import mujoco
    d.ctrl[:] = action
    for _ in range(substeps):
        mujoco.mj_step(m, d)

def generate_demos(m, d, n_demos=20, steps=200):
    import mujoco
    all_obs, all_acts = [], []
    for i in range(n_demos):
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        for s in range(steps):
            obs = get_obs(d)
            act = scripted_action(s, steps)
            if i > 0:
                act += np.random.normal(0, 0.002, size=act.shape)
                act = np.clip(act, [-0.5,-0.5,-0.2,0,0], [0.5,0.5,0.5,1.57,1.57])
            all_obs.append(obs)
            all_acts.append(act.astype(np.float32))
            step_sim(m, d, act)
    print(f"  Sanity: first obs = {all_obs[0][:3]}, last obs = {all_obs[-1][:3]}")
    print(f"  Sanity: any NaN in obs? {any(np.isnan(o).any() for o in all_obs)}")
    print(f"  Sanity: any NaN in acts? {any(np.isnan(a).any() for a in all_acts)}")
    return np.array(all_obs), np.array(all_acts)

def train_bc(obs, acts, epochs=2000, lr=5e-4, bs=256):
    import torch, torch.nn as nn
    # Filter NaN
    mask = ~(np.isnan(obs).any(1) | np.isnan(acts).any(1))
    obs, acts = obs[mask], acts[mask]
    print(f"  After NaN filter: {len(obs)} samples (removed {(~mask).sum()})")

    ot = torch.FloatTensor(obs)
    at = torch.FloatTensor(acts)
    om, os = ot.mean(0), ot.std(0)+1e-8
    am, ast = at.mean(0), at.std(0)+1e-8
    on = (ot-om)/os
    an = (at-am)/ast

    class P(nn.Module):
        def __init__(s, od, ad):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(od,256),nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(256,128),nn.ReLU(),
                                  nn.Linear(128,ad))
        def forward(s,x): return s.net(x)

    pol = P(obs.shape[1], acts.shape[1])
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = len(on)

    for ep in range(epochs):
        idx = np.random.permutation(n)
        el, nb = 0, 0
        for i in range(0, n, bs):
            bi = idx[i:i+bs]
            l = loss_fn(pol(on[bi]), an[bi])
            opt.zero_grad(); l.backward(); opt.step()
            el += l.item(); nb += 1
        sch.step()
        if (ep+1) % 200 == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={el/nb:.6f}")
    print(f"  Final loss: {el/nb:.6f}")
    return pol, om, os, am, ast

def rollout(m, d, pol, om, os, am, ast, steps=200):
    import mujoco, torch
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    traj = []
    for s in range(steps):
        obs = get_obs(d)
        ot = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            an = pol((ot-om)/os)
        act = (an*ast+am).numpy()[0]
        act = np.clip(act, [-0.5,-0.5,-0.2,0,0], [0.5,0.5,0.5,1.57,1.57])
        step_sim(m, d, act)
        traj.append({"step":s, "wrist": d.qpos[:3].copy().tolist(),
                      "fingers": d.qpos[3:5].copy().tolist(), "action": act.tolist()})
    fw = np.array(traj[-1]["wrist"])
    dist = np.linalg.norm(fw - MUG_POS)
    min_dist = min(np.linalg.norm(np.array(t["wrist"]) - MUG_POS) for t in traj)
    mv = sum(np.linalg.norm(np.array(traj[i+1]["wrist"])-np.array(traj[i]["wrist"])) for i in range(len(traj)-1))
    return traj, {"reached": bool(min_dist<0.06), "min_dist": float(min_dist), "final_dist": float(dist),
                  "movement": float(mv), "final_wrist": traj[-1]["wrist"], "final_fingers": traj[-1]["fingers"]}

def render(m, d, actions, steps, path, title):
    import mujoco, cv2
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
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
            dist = np.linalg.norm(w-MUG_POS)
            cv2.putText(f, f"Dist: {dist:.3f}m", (15,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            frames.append(f)
    renderer.close()
    h,w2 = frames[0].shape[:2]
    wr = cv2.VideoWriter(path+".raw.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (w2,h))
    for f in frames: wr.write(f)
    wr.release()
    os.system(f"ffmpeg -y -i '{path}.raw.mp4' -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p '{path}' 2>/dev/null")
    if os.path.exists(f"{path}.raw.mp4"): os.remove(f"{path}.raw.mp4")
    print(f"  Video: {path} ({len(frames)} frames)")

if __name__ == "__main__":
    import mujoco
    S = 200
    os.makedirs("/workspace/sim/output", exist_ok=True)
    print("="*60)
    print("SIM v3 — Stable physics, 20 demos, 2000 epochs")
    print("="*60)

    m, d = make_env()
    print(f"DOF={m.nq}, actuators={m.nu}")

    # Verify scripted demo works
    print("\nVerifying scripted demo...")
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    for s in range(S):
        step_sim(m, d, scripted_action(s, S))
    final_w = d.qpos[:3].copy()
    print(f"  Scripted final wrist: {final_w} (target: {MUG_POS})")
    print(f"  Scripted dist: {np.linalg.norm(final_w - MUG_POS):.4f}")
    if np.isnan(final_w).any():
        print("  FATAL: NaN in scripted demo! Aborting.")
        exit(1)

    print("\nGenerating 20 demos...")
    obs, acts = generate_demos(m, d, n_demos=20, steps=S)
    print(f"  Total: {len(obs)} samples")

    print("\nTraining BC (2000 epochs)...")
    pol, om, os_, am, ast = train_bc(obs, acts, epochs=2000)

    print("\nEvaluating...")
    mujoco.mj_resetData(m, d)
    traj, res = rollout(m, d, pol, om, os_, am, ast, steps=S)
    print(f"\nRESULTS: {json.dumps(res, indent=2)}")

    with open("/workspace/sim/output/results.json","w") as f: json.dump(res,f,indent=2)

    print("\nRendering...")
    demo_acts = [scripted_action(s,S) for s in range(S)]
    render(m, d, demo_acts, S, "/workspace/sim/output/demo_scripted.mp4", "Expert Demo")
    pol_acts = [np.array(t["action"]) for t in traj]
    render(m, d, pol_acts, S, "/workspace/sim/output/policy_rollout.mp4", "Trained BC Policy")

    print(f"\n{'✅ SUCCESS' if res['reached'] else '⚠️ MISS'}: min_dist={res['min_dist']:.3f}m, final_dist={res['final_dist']:.3f}m")
