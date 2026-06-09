"""Render sp_results2.npz (Cartesian gripper reach). Headline: does e_ac beat e_free beat
noise~=a (on current OR goal), with oracle_xyz/bothprop as ceilings."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/selfplay_probe"
r = np.load(OUT + "/sp_results2.npz")
demos = r["demos"]
conds = ["oracle_xyz", "bothprop", "a", "noise", "b_cur", "c_cur", "b_goal", "c_goal"]
lab = {
    "oracle_xyz": "oracle(xyz)",
    "bothprop": "bothprop",
    "a": "(a) z",
    "noise": "z+rand",
    "b_cur": "z+e_free(cur)",
    "c_cur": "z+e_ac(cur)",
    "b_goal": "z+e_free(goal)",
    "c_goal": "z+e_ac(goal)",
}
col = {
    "oracle_xyz": "green",
    "bothprop": "lime",
    "a": "0.6",
    "noise": "0.8",
    "b_cur": "tab:orange",
    "c_cur": "tab:red",
    "b_goal": "gold",
    "c_goal": "tab:purple",
}
EPS_H = 0.1
print(f"{'#demos':>7s} | " + " ".join(f"{lab[c]:>15s}" for c in conds) + f"   (SR@{EPS_H}m / finalErr)")
for d in demos:
    print(
        f"{d:>7d} | "
        + " ".join(
            f"{(r[f'{c}|{d}'] < EPS_H).mean():.2f}/{r[f'{c}|{d}'].mean():.2f}".rjust(15) for c in conds
        )
    )
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for c in conds:
    sr = [(r[f"{c}|{d}"] < EPS_H).mean() for d in demos]
    fe = [r[f"{c}|{d}"].mean() for d in demos]
    st = "o--" if c in ("oracle_xyz", "bothprop") else "o-"
    ax[0].plot(demos, sr, st, color=col[c], label=lab[c])
    ax[1].plot(demos, fe, st, color=col[c], label=lab[c])
ax[0].set(
    xscale="log",
    xlabel="# HER demos",
    ylabel=f"success @ {EPS_H}m",
    title="Cartesian reach SR",
    ylim=(-0.02, 1.02),
)
ax[1].set(xscale="log", xlabel="# HER demos", ylabel="final gripper err (m)", title="Final reach error")
ax[0].legend(fontsize=7, ncol=2)
ax[0].grid(alpha=0.3)
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "/sp_curve2.png", dpi=120)
print(f"\n[ok] {OUT}/sp_curve2.png")
