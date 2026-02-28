#!/usr/bin/env python3
"""Video audit: replay a spatial trajectory in MuJoCo and produce diagnostic plots + keyframes.

Usage:
  python tools/video_audit.py <traj_json> [--out_dir DIR]

Reads spatial_traj.json (ee_targets, grippers) and replays in sim,
measuring EE→mug distance, EE z, mug z, gripper state over time.
Outputs: keyframes/*.png, audit_plots.png, audit_summary.txt
"""

import os, sys, json, argparse
import numpy as np

PANDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mujoco_menagerie", "franka_emika_panda")

SCENE_XML = """
<mujoco model="panda_audit">
  <include file="{panda_dir}/mjx_panda.xml"/>
  <statistic center="0.3 0 0.4" extent="1"/>
  <option timestep="0.002" iterations="50" ls_iterations="20" integrator="implicitfast" gravity="0 0 -9.81">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody>
    <light pos="0.5 -0.5 1.5" dir="0 0.3 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" contype="1" conaffinity="1"/>
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


def solve_ik(model, target, site_id, start_q=None):
    import mujoco
    d2 = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d2, 0)
    if start_q is not None:
        d2.qpos[:7] = start_q
    q = d2.qpos[:7].copy()
    for _ in range(1200):
        d2.qpos[:7] = q
        mujoco.mj_forward(model, d2)
        err = target - d2.site_xpos[site_id].copy()
        if np.linalg.norm(err) < 0.003:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, None, site_id)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05*np.eye(3), err)
        q += dq * min(0.35, 0.07/(np.linalg.norm(dq)+1e-8))
        for j in range(7):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{j+1}')
            q[j] = np.clip(q[j], *model.jnt_range[jid])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_json", help="Path to spatial_traj.json")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    traj = json.load(open(args.traj_json))
    ee_targets = np.array(traj["ee_targets"])
    grippers = np.array(traj["grippers"])

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.traj_json), "audit")
    os.makedirs(out_dir, exist_ok=True)
    kf_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)

    import mujoco, cv2

    xml_path = os.path.join(PANDA_DIR, "_audit.xml")
    with open(xml_path, "w") as f:
        f.write(SCENE_XML)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    gs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    mb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    mug_start = data.xpos[mb].copy()

    spf = int(1.0 / (args.fps * model.opt.timestep))

    # Metrics arrays
    n = len(ee_targets)
    ee_pos_log = np.zeros((n, 3))
    mug_pos_log = np.zeros((n, 3))
    ee_mug_dist = np.zeros(n)
    grip_log = np.zeros(n)

    prev_q = None
    keyframe_indices = set()
    # Save keyframes at: start, 25%, 50%, 75%, end, and detected events
    for pct in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        keyframe_indices.add(min(n-1, int(pct * (n-1))))

    # Detect gripper close/open events
    for i in range(1, n):
        if grippers[i] < 0.3 and grippers[i-1] >= 0.3:
            keyframe_indices.add(i)
        if grippers[i] >= 0.7 and grippers[i-1] < 0.7:
            keyframe_indices.add(i)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    for i in range(n):
        q = solve_ik(model, ee_targets[i], gs, prev_q)
        prev_q = q
        data.ctrl[:7] = q
        data.ctrl[7] = 0.04 * float(np.clip(grippers[i], 0, 1))
        for _ in range(spf):
            mujoco.mj_step(model, data)

        ee_pos_log[i] = data.site_xpos[gs].copy()
        mug_pos_log[i] = data.xpos[mb].copy()
        ee_mug_dist[i] = np.linalg.norm(ee_pos_log[i] - mug_pos_log[i])
        grip_log[i] = grippers[i]

        if i in keyframe_indices:
            renderer.update_scene(data)
            img = renderer.render()
            cv2.imwrite(os.path.join(kf_dir, f"frame_{i:04d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if i % 50 == 0:
            print(f"Audit step {i}/{n}")

    # Compute metrics
    mug_lift = mug_pos_log[:, 2] - mug_start[2]
    max_lift = float(np.max(mug_lift))
    min_ee_z = float(np.min(ee_pos_log[:, 2]))
    min_dist = float(np.min(ee_mug_dist))
    table_top = 0.25  # table height
    ee_below_table = int(np.sum(ee_pos_log[:, 2] < table_top))

    # Detect phases
    close_frame = -1
    for i in range(1, n):
        if grippers[i] < 0.3 and grippers[i-1] >= 0.3:
            close_frame = i
            break

    lift_start = -1
    if max_lift > 0.01:
        lift_start = int(np.argmax(mug_lift > 0.01))

    # Verdict
    verdicts = []
    if min_dist > 0.08:
        verdicts.append("MISS: EE never got close to mug")
    if ee_below_table > 5:
        verdicts.append(f"TABLE_CRASH: EE went below table {ee_below_table} frames")
    if max_lift < 0.01:
        verdicts.append("NO_LIFT: mug never lifted")
    elif max_lift < 0.05:
        verdicts.append(f"WEAK_LIFT: only {max_lift*100:.1f}cm")
    else:
        verdicts.append(f"LIFT_OK: {max_lift*100:.1f}cm")

    # Check if mug dropped after lift
    if max_lift > 0.02:
        peak_frame = int(np.argmax(mug_lift))
        if peak_frame < n - 10 and mug_lift[-1] < 0.01:
            verdicts.append("DROPPED: mug lifted then fell back")

    # Descent speed
    if close_frame > 10:
        descent_region = ee_pos_log[max(0, close_frame-20):close_frame, 2]
        if len(descent_region) > 2:
            dz = np.diff(descent_region)
            max_dz = float(np.max(np.abs(dz)))
            if max_dz > 0.01:
                verdicts.append(f"FAST_DESCENT: max dz/step = {max_dz*100:.2f}cm")

    verdict_str = " | ".join(verdicts) if verdicts else "UNKNOWN"

    summary = f"""=== VIDEO AUDIT: {args.label or args.traj_json} ===
Frames: {n}
EE-to-mug min distance: {min_dist*100:.2f}cm (frame {int(np.argmin(ee_mug_dist))})
EE min Z: {min_ee_z*100:.2f}cm (table top = 25.0cm)
EE below table frames: {ee_below_table}
Mug max lift: {max_lift*100:.2f}cm (frame {int(np.argmax(mug_lift))})
Gripper close frame: {close_frame}
Lift start frame: {lift_start}
Mug final Z offset: {mug_lift[-1]*100:.2f}cm

VERDICT: {verdict_str}

Keyframes saved: {kf_dir}/
"""
    print(summary)

    with open(os.path.join(out_dir, "audit_summary.txt"), "w") as f:
        f.write(summary)

    # Save plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        t = np.arange(n)

        axes[0].plot(t, ee_mug_dist * 100, 'b-', linewidth=1.5)
        axes[0].set_ylabel("EE-Mug dist (cm)")
        axes[0].set_title(f"Audit: {args.label or os.path.basename(args.traj_json)}")
        axes[0].axhline(y=3, color='g', linestyle='--', alpha=0.5, label='grasp range')
        axes[0].legend()

        axes[1].plot(t, ee_pos_log[:, 2] * 100, 'r-', linewidth=1.5, label='EE z')
        axes[1].plot(t, ee_targets[:, 2] * 100, 'r--', alpha=0.4, label='target z')
        axes[1].axhline(y=25.0, color='brown', linestyle='--', alpha=0.5, label='table top')
        axes[1].set_ylabel("Z (cm)")
        axes[1].legend()

        axes[2].plot(t, mug_lift * 100, 'g-', linewidth=1.5)
        axes[2].set_ylabel("Mug lift (cm)")

        axes[3].plot(t, grip_log, 'm-', linewidth=1.5)
        axes[3].set_ylabel("Gripper (1=open)")
        axes[3].set_xlabel("Frame")

        if close_frame > 0:
            for ax in axes:
                ax.axvline(x=close_frame, color='orange', linestyle=':', alpha=0.7)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "audit_plots.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"Plots saved: {plot_path}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # Save raw data
    np.savez(os.path.join(out_dir, "audit_data.npz"),
             ee_pos=ee_pos_log, mug_pos=mug_pos_log,
             ee_mug_dist=ee_mug_dist, grip=grip_log,
             mug_lift=mug_lift, ee_targets=ee_targets)
    print("Done.")


if __name__ == "__main__":
    main()
