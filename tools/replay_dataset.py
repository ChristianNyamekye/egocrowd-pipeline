#!/usr/bin/env python3
"""
Action Replay — render MuJoCo Panda tracking the exported EE trajectories.
Produces a proper video with smooth motion.
"""
import json, numpy as np, os, sys

def main():
    import mujoco, cv2
    
    # Load dataset
    ds = json.load(open("pipeline/lerobot_dataset_v2/dataset.json"))
    episodes = ds["episodes"]
    print(f"Loaded {len(episodes)} episodes, {sum(e['num_steps'] for e in episodes)} total steps")
    
    # Load model
    xml = "mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml"
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    
    # Find EE site
    ee_site = None
    for i in range(model.nsite):
        name = model.site(i).name
        if 'grip' in name.lower() or 'ee' in name.lower() or 'hand' in name.lower():
            ee_site = i
            break
    if ee_site is None:
        ee_site = model.nsite - 1
    print(f"EE site: {model.site(ee_site).name} (idx {ee_site})")
    
    frames = []
    fps = 30
    substeps = 50  # sim steps per control step — smooth motion
    
    for ei, ep in enumerate(episodes):
        steps = ep["steps"]
        print(f"\nEpisode {ei}: {len(steps)} steps")
        
        # Reset
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        
        # Render initial state for 0.5s
        renderer.update_scene(data)
        init_frame = renderer.render().copy()
        for _ in range(int(fps * 0.5)):
            frames.append(init_frame)
        
        for t, step in enumerate(steps):
            target = np.array(step["observation"]["ee_pos"])
            grip = step["observation"]["gripper_state"]
            
            for sub in range(substeps):
                # Get current EE
                current_ee = data.site_xpos[ee_site].copy()
                ee_error = target - current_ee
                
                # Jacobian IK
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, None, ee_site)
                J = jacp[:, :7]
                lam = 0.01
                dq = np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ ee_error)
                
                for j in range(min(7, model.nu)):
                    data.ctrl[j] = data.qpos[j] + dq[j] * 1.5
                
                # Gripper
                if model.nu > 7:
                    gv = grip * 0.04
                    for j in range(7, model.nu):
                        data.ctrl[j] = gv
                
                mujoco.mj_step(model, data)
                
                # Record frame every ~1/30s of sim time
                if sub % (substeps // 3) == 0:
                    renderer.update_scene(data)
                    frames.append(renderer.render().copy())
            
            # Also record at end of each control step
            renderer.update_scene(data)
            frames.append(renderer.render().copy())
        
        # Hold final pose for 1s
        renderer.update_scene(data)
        final_frame = renderer.render().copy()
        for _ in range(fps):
            frames.append(final_frame)
        
        final_ee = data.site_xpos[ee_site]
        last_target = np.array(steps[-1]["observation"]["ee_pos"])
        print(f"  Final error: {np.linalg.norm(final_ee - last_target):.4f}m")
    
    # Write video
    out_path = "pipeline/lerobot_dataset_v2/replay_v2.mp4"
    print(f"\nWriting {len(frames)} frames at {fps}fps...")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    
    # Re-encode with ffmpeg for Discord compatibility
    final_path = "pipeline/lerobot_dataset_v2/replay_final.mp4"
    os.system(f'ffmpeg -y -i "{out_path}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{final_path}" 2>nul')
    
    if os.path.exists(final_path) and os.path.getsize(final_path) > 1000:
        print(f"Saved: {final_path} ({os.path.getsize(final_path)/1024:.0f} KB)")
    else:
        print(f"Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
