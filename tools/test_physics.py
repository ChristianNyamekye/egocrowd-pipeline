import numpy as np, sys
sys.path.insert(0,"/tmp")
from sim_v3 import *
import mujoco
m,d = make_env()
mujoco.mj_resetData(m,d); mujoco.mj_forward(m,d)
for s in range(200):
    step_sim(m,d,scripted_action(s,200))
    if s % 40 == 0:
        w=d.qpos[:3]
        print(f"Step {s}: wrist={w}, dist={np.linalg.norm(w-MUG_POS):.4f}")
w=d.qpos[:3]
print(f"Final: wrist={w}, dist={np.linalg.norm(w-MUG_POS):.4f}")
