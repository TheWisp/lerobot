"""Harness validation (soundness-first, no learning yet):
  1. reset(seed=list) -> per-env arm pose: fixed home or varied?
  2. HOLD (no-op delta=0): proprio should stay ~constant -> the no-op SR baseline.
  3. REACH (both-proprio oracle: delta=goal-proprio): proprio should converge to goal
     -> validates delta control + the SR metric ceiling.
Goals = snapshots of agent_pos after a short random walk from held-out seeds.
"""

import numpy as np
from sp_lib import Vec, delta_command, reach_err

N = 6
vec = Vec(N)

# --- 1. reset poses ---
p0 = vec.reset(range(100, 100 + N))[1]
p1 = vec.reset(range(200, 200 + N))[1]
print(f"reset arm pose spread across seeds: per-joint std = {p0.std(0).mean():.4f}")
print(f"  seed-set A mean |pose| = {np.abs(p0).mean():.3f}, B = {np.abs(p1).mean():.3f}")
print(f"  pose range over joints: [{p0.min():.2f}, {p0.max():.2f}]")

# --- generate goals: random walk for variety, THEN SETTLE so the goal is a static
#     equilibrium reachable by position control (else oracle can't hold against gravity) ---
rng = np.random.RandomState(0)
_, proprio = vec.reset(range(500, 500 + N))
c = rng.uniform(-0.8, 0.8, (N, 14)).astype(np.float32)  # random target command per env
for t in range(40):  # walk toward + settle at command c
    (_, proprio), _, _ = vec.step(c)
goal_proprio = proprio.copy()  # equilibrium under command c
print(f"\ngoal poses: per-env |pose| = {np.abs(goal_proprio).mean(1).round(2)}")
print(f"  settle residual (proprio vs command c): {reach_err(goal_proprio, c).mean():.3f}")

# --- 2. HOLD (no-op) from home ---
_, proprio = vec.reset(range(700, 700 + N))
start = proprio.copy()
d_start = reach_err(start, goal_proprio)
for t in range(50):
    cmd = delta_command(proprio, np.zeros((N, 14), np.float32))
    (_, proprio), _, _ = vec.step(cmd)
d_hold = reach_err(proprio, goal_proprio)
print(
    f"\n[HOLD/no-op]  start->goal dist {d_start.mean():.3f} | after 50 steps {d_hold.mean():.3f}  (drift from start: {reach_err(proprio, start).mean():.3f})"
)

# --- 3. REACH (both-proprio oracle) ---
_, proprio = vec.reset(range(700, 700 + N))
traj = []
for t in range(60):
    delta = goal_proprio - proprio  # privileged: uses both current + goal proprio
    cmd = delta_command(proprio, delta, dmax=0.5)
    (_, proprio), _, _ = vec.step(cmd)
    traj.append(reach_err(proprio, goal_proprio).mean())
d_reach = reach_err(proprio, goal_proprio)
print(f"[REACH/oracle] start->goal {d_start.mean():.3f} | final {d_reach.mean():.3f}")
print(f"  err trajectory (every 10 steps): {[round(traj[i], 3) for i in range(0, 60, 10)]}")
for eps in [0.1, 0.2, 0.3, 0.5]:
    print(f"  SR@eps={eps}: hold {(d_hold < eps).mean():.2f}  oracle {(d_reach < eps).mean():.2f}")
