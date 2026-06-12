"""Test goals = held-out PLAY frames (distribution-matched to HER training goals, which are
also play frames). Cartesian reach only needs a gripper POSITION target, so transient play
poses are fine (the settled-pose requirement was for the old joint oracle). All from cache
-> no sim. Held-out episodes (last 4) are disjoint from HER training (eid<ne-4)."""

import numpy as np

OUT = "/tmp/selfplay_probe"  # nosec B108
M = np.load(OUT + "/feat_cache.npz")["M"].astype(np.float32)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
GX = wb["world"][:, 6:12].astype(np.float32)
PR = wb["states"].astype(np.float32)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
# candidate frames: held-out episodes, skip first 40 (home region) so reach is non-trivial
cand = []
for eid in range(ne - 4, ne):
    s, e = ep[eid]
    cand += list(range(s + 40, e - 1))
cand = np.array(cand)
# pick 32 spread by gripper position (farthest-point-ish: just even stride after shuffle-by-pos)
rng = np.random.RandomState(7)
idx = rng.choice(cand, 32, replace=False)
home = GX[ep[0][0]]  # gripper at home (first frame)
d = np.linalg.norm(GX[idx] - home, axis=1)
print(f"{len(cand)} candidate held-out play frames -> 32 goals")
print(f"  start(home)->goal gripper dist: mean {d.mean():.3f}m range [{d.min():.3f},{d.max():.3f}]")
print(
    f"  goal gripper-xyz pairwise spread: {np.mean([np.linalg.norm(GX[idx[i]] - GX[idx[j]]) for i in range(32) for j in range(i + 1, 32)]):.3f}m"
)
np.savez_compressed(OUT + "/goals.npz", z_goal=M[idx], goal_gxyz=GX[idx], goal_proprio=PR[idx], goal_idx=idx)
print("[ok] goals.npz <- held-out play frames")
