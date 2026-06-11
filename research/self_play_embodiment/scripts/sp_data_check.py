# ruff: noqa
import sys, numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/selfplay_probe/wide_data.npz"  # nosec B108
print("data:", path)
d = np.load(path)
states = d["states"].astype(np.float64)
actions = d["actions"].astype(np.float64)
epid = d["epid"]
peg = d["peg0"].astype(np.float64)


def ep_arr(X):
    return {int(e): X[np.where(epid == e)[0]] for e in np.unique(epid)}


S = ep_arr(states)
A = ep_arr(actions)
P = ep_arr(peg)
print(
    "ALOHA layout: idx 0:6=LEFT arm, 6=left grip, 7:13=RIGHT arm, 13=right grip. (only RIGHT arm should be active)"
)
larm = []
rarm = []
lgrip = []
pegmove = []
pegtravel = []
oob = []
contact_ep = []
sat = []
for e in S:
    s = S[e]
    a = A[e]
    p = P[e]
    ds = np.abs(np.diff(s, axis=0))
    larm.append(ds[:, 0:6].mean())
    rarm.append(ds[:, 7:13].mean())
    lgrip.append(ds[:, 6].mean())
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    pegmove.append(dp.mean())
    pegtravel.append(dp.sum())
    oob.append(((np.abs(p[:, 0]) > 0.4) | (np.abs(p[:, 1] - 0.5) > 0.45)).mean())
    contact_ep.append(float(dp.max() > 0.005))
    sat.append((np.abs(a) > 0.9).mean())
f = np.mean
print(f"  |Δleft-arm|/step {f(larm):.4f} | |Δright-arm|/step {f(rarm):.4f} | |Δleft-grip| {f(lgrip):.4f}")
print(
    f"  peg move/step {f(pegmove) * 1000:.1f}mm | peg travel/ep {f(pegtravel) * 100:.1f}cm | eps-with-contact {f(contact_ep) * 100:.0f}% | peg-flung-OOB {f(oob) * 100:.0f}% | action-sat {f(sat) * 100:.0f}%"
)
