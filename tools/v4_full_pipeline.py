"""
V4: Full pipeline — recollect better data + train + eval.
Key fixes:
1. Larger network (512 hidden)
2. Higher eval controller gain (0.5 instead of 0.3)
3. More training epochs (2000, chunked)
4. More episodes (100) for better coverage
"""
import os, json, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
import mujoco, imageio

PANDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mujoco_menagerie", "franka_emika_panda")

SCENE_XML = """
<mujoco model="panda_loop">
  <include file="{panda_dir}/mjx_panda.xml"/>
  <statistic center="0.3 0 0.4" extent="1"/>
  <option timestep="0.002" iterations="50" ls_iterations="20" integrator="implicitfast" gravity="0 0 -9.81">
    <flag eulerdamp="disable"/>
  </option>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="150" elevation="-25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.5 1.5" dir="0 0.3 -1" directional="true"/>
    <light pos="-0.2 0.5 1.0" dir="0.2 -0.2 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
    <body name="table" pos="0.5 0 0.125">
      <geom type="box" size="0.25 0.3 0.125" rgba="0.55 0.35 0.18 1" contype="1" conaffinity="1"/>
    </body>
    <body name="mug" pos="0.5 0 0.295">
      <freejoint name="mug_joint"/>
      <geom name="mug_body" type="cylinder" size="0.03 0.04" mass="0.25"
        rgba="0.85 0.15 0.15 1" contype="2" conaffinity="1"
        friction="1.5 0.05 0.01" condim="4" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home"
      qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04   0.5 0 0.295 1 0 0 0"
      ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"/>
  </keyframe>
</mujoco>
""".replace("{panda_dir}", PANDA_DIR)


class Net(nn.Module):
    def __init__(self, obs=11, act=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, act))
    def forward(self, x): return self.net(x)


def eval_policy(pt_path, n_evals=10):
    xml_path = os.path.join(PANDA_DIR, '_v4e.xml')
    with open(xml_path, 'w') as f: f.write(SCENE_XML)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    gs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripper')
    mb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'mug')
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{j+1}') for j in range(7)]

    ck = torch.load(pt_path, weights_only=False)
    net = Net(11, 4); net.load_state_dict(ck['m']); net.eval()
    om, os_, am, as_ = ck['om'], ck['os'], ck['am'], ck['as']

    fps = 30; duration = 10.0
    successes = 0
    all_frames = []

    for trial in range(n_evals):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if trial > 0:
            dx = np.random.uniform(-0.05, 0.05)
            dy = np.random.uniform(-0.05, 0.05)
            data.qpos[9] += dx
            data.qpos[10] += dy
        mujoco.mj_forward(model, data)

        mug0 = data.xpos[mb].copy()
        n_steps = int(duration * fps)
        spf = int(1.0 / (fps * model.opt.timestep))
        ctrl = np.array([0, 0.3, 0, -1.57079, 0, 2.0, -0.7853], dtype=np.float64)
        max_mug_z = mug0[2]; min_d = 999

        renderer = mujoco.Renderer(model, 480, 640) if trial < 3 else None
        cam = mujoco.MjvCamera()
        cam.azimuth = 160; cam.elevation = -25; cam.distance = 1.4
        cam.lookat[:] = [0.4, 0.05, 0.25]

        for fi in range(n_steps):
            t = fi / fps; phase = t / duration
            mujoco.mj_forward(model, data)
            ee = data.site_xpos[gs].copy()
            mug = data.xpos[mb].copy()
            grip_now = float(data.ctrl[7])

            obs = np.array([phase, ee[0], ee[1], ee[2], grip_now,
                            mug[0], mug[1], mug[2],
                            mug[0]-ee[0], mug[1]-ee[1], mug[2]-ee[2]], dtype=np.float32)
            ob_n = (obs - om) / os_
            with torch.no_grad():
                ac_n = net(torch.from_numpy(ob_n).unsqueeze(0)).numpy()[0]
            act = ac_n * as_ + am
            ee_t = act[:3]; grip_t = float(np.clip(act[3], 0, 0.04))

            err = ee_t - ee
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, gs)
            J = jacp[:, :7]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(3), err)
            ctrl += dq * 0.5  # Higher gain!
            for i in range(7):
                ctrl[i] = np.clip(ctrl[i], *model.jnt_range[jids[i]])

            data.ctrl[:7] = ctrl; data.ctrl[7] = grip_t
            for _ in range(spf): mujoco.mj_step(model, data)

            d = np.linalg.norm(ee - mug); min_d = min(min_d, d)
            max_mug_z = max(max_mug_z, data.xpos[mb][2])

            if renderer and fi % 2 == 0:
                renderer.update_scene(data, cam)
                all_frames.append(renderer.render().copy())

        if renderer: renderer.close()
        lift = (max_mug_z - mug0[2]) * 100
        ok = lift > 3
        successes += int(ok)
        print(f"  Trial {trial}: mug@[{mug0[0]:.3f},{mug0[1]:.3f}] lift={lift:+.1f}cm min_d={min_d:.4f}m {'OK' if ok else 'FAIL'}")
        sys.stdout.flush()

    rate = 100 * successes / n_evals
    print(f"\n  V4 ROBUST: {successes}/{n_evals} ({rate:.0f}%)")

    if all_frames:
        vid = 'pipeline/policy/pick_lift_robust_rollout.mp4'
        writer = imageio.get_writer(vid, fps=fps, codec='libx264', quality=8)
        for fr in all_frames: writer.append_data(fr)
        writer.close()
        print(f"  Video: {vid} ({os.path.getsize(vid)//1024}KB)")

    try: os.remove(xml_path)
    except: pass
    return rate


def train_and_eval():
    DS = 'pipeline/policy/dr_dataset.json'
    PT = 'pipeline/policy/pick_lift_robust_v2.pt'

    # Load data
    d = json.load(open(DS))
    steps = d['steps']
    obs = np.array([s['obs'] for s in steps], np.float32)
    act = np.array([s['action'] for s in steps], np.float32)
    print(f"Data: {len(obs)} transitions")

    # Light augmentation
    obs_l, act_l = [obs], [act]
    for _ in range(5):
        n = np.random.randn(*obs.shape).astype(np.float32) * 0.005
        n[:, 0] = 0
        obs_l.append(obs + n)
        act_l.append(act)
    all_obs = np.concatenate(obs_l)
    all_act = np.concatenate(act_l)
    
    om = all_obs.mean(0); os_ = all_obs.std(0) + 1e-6
    am = all_act.mean(0); as_ = all_act.std(0) + 1e-6
    
    obs_n = torch.from_numpy((all_obs - om) / os_)
    act_n = torch.from_numpy((all_act - am) / as_)
    
    ds = torch.utils.data.TensorDataset(obs_n, act_n)
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    
    net = Net(11, 4)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1500)
    
    best = 1e9
    for ep in range(1500):
        tot = 0; n = 0
        for ob, ac in dl:
            lo = nn.functional.mse_loss(net(ob), ac)
            opt.zero_grad(); lo.backward(); opt.step()
            tot += lo.item() * ob.shape[0]; n += ob.shape[0]
        avg = tot / n; sch.step()
        if avg < best:
            best = avg
            torch.save({'m': net.state_dict(), 'om': om, 'os': os_, 'am': am, 'as': as_,
                        'epoch': ep, 'best_loss': best}, PT)
        if ep % 200 == 0 or ep == 1499:
            print(f"  ep {ep:4d} loss={avg:.6f} best={best:.6f}")
            sys.stdout.flush()
    
    print(f"\nTraining done. Evaluating...")
    eval_policy(PT, n_evals=10)


if __name__ == '__main__':
    if '--eval-only' in sys.argv:
        pt = 'pipeline/policy/pick_lift_robust_v2.pt'
        if not os.path.exists(pt):
            pt = 'pipeline/policy/pick_lift_robust.pt'
        eval_policy(pt, n_evals=10)
    else:
        train_and_eval()
