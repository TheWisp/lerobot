"""Generate held-out TEST goals for the reaching experiment.
Each goal = a STATIC equilibrium pose (settle under a random command from a held-out
seed), captured as (goal_image@256, goal_proprio[14], z_goal = base JEPA(goal_image)).
The goal image is what the policy sees; goal_proprio is for the success metric (+ oracle).
Seeds 10000+ -> disjoint from world_buffer collection. -> goals.npz
"""

import numpy as np
from sp_lib import Encoder, Vec

N_GOALS = 32
SETTLE = 45
OUT = "/tmp/selfplay_probe/goals.npz"

vec = Vec(N_GOALS)
rng = np.random.RandomState(123)
# diverse static commands; settle under each to get a reachable equilibrium goal
c = rng.uniform(-0.85, 0.85, (N_GOALS, 14)).astype(np.float32)
vec.reset(range(10000, 10000 + N_GOALS))
for t in range(SETTLE):
    (img, proprio), _, _ = vec.step(c)
goal_img, goal_proprio, goal_gxyz = img.copy(), proprio.copy(), vec.gripper_xyz()

enc = Encoder()
z_goal = enc.encode(goal_img)
print(
    f"goals: {N_GOALS} | proprio |.|range [{np.abs(goal_proprio).min():.2f},{np.abs(goal_proprio).max():.2f}]"
    f" | settle residual vs cmd {np.linalg.norm(goal_proprio - c, axis=1).mean():.3f}"
)
print(
    f"gripper-xyz goal spread (diversity): mean pairwise {np.mean([np.linalg.norm(goal_gxyz[i] - goal_gxyz[j]) for i in range(N_GOALS) for j in range(i + 1, N_GOALS)]):.3f} m"
)
np.savez_compressed(OUT, goal_img=goal_img, goal_proprio=goal_proprio, z_goal=z_goal, goal_gxyz=goal_gxyz)
print(f"[ok] saved {OUT}")
