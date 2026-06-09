"""Aggregate sp_results2_s*.npz across seeds -> mean+-std SR/finalErr per condition.
Headline: is c (action-conditioned e) > b (action-free) > noise ~= a in the low-demo
regime, robustly across seeds? Reports the c-b and c-noise deltas with std."""

import glob

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/selfplay_probe"
files = sorted(glob.glob(OUT + "/sp_results2_s*.npz"))
R = [np.load(f) for f in files]
demos = R[0]["demos"]
print(f"seeds: {len(R)} | demos {list(demos)}")
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
    "a": "0.55",
    "noise": "0.78",
    "b_cur": "tab:orange",
    "c_cur": "tab:red",
    "b_goal": "gold",
    "c_goal": "tab:purple",
}


def sr(cond, d, eps):  # per-seed SR (mean over goals) -> array over seeds
    return np.array([(r[f"{cond}|{d}"] < eps).mean() for r in R])


def fe(cond, d):
    return np.array([r[f"{cond}|{d}"].mean() for r in R])


for EPS_H in [0.1, 0.2]:
    print(f"\n=== SR@{EPS_H}m  (mean+-std over {len(R)} seeds) ===")
    print(f"{'#demos':>7s} | " + " ".join(f"{lab[c]:>14s}" for c in conds))
    for d in demos:
        print(
            f"{d:>7d} | "
            + " ".join(f"{sr(c, d, EPS_H).mean():.2f}±{sr(c, d, EPS_H).std():.2f}".rjust(14) for c in conds)
        )

print("\n=== INJECTION DELTAS (mean+-std over seeds), the thesis test ===")
print(
    f"{'#demos':>7s} | {'c_cur-b_cur':>16s} {'c_cur-noise':>16s} {'c_goal-b_goal':>16s} {'c_goal-a':>16s}   (SR@0.2)"
)
for d in demos:

    def dl(x, y):
        v = sr(x, d, 0.2) - sr(y, d, 0.2)
        return f"{v.mean():+.2f}±{v.std():.2f}"

    print(
        f"{d:>7d} | {dl('c_cur', 'b_cur'):>16s} {dl('c_cur', 'noise'):>16s} {dl('c_goal', 'b_goal'):>16s} {dl('c_goal', 'a'):>16s}"
    )
print(
    f"\n{'#demos':>7s} | finalErr(m): "
    + " ".join(f"{lab[c]:>13s}" for c in ["a", "noise", "b_cur", "c_cur", "b_goal", "c_goal"])
)
for d in demos:
    print(
        f"{d:>7d} |              "
        + " ".join(
            f"{fe(c, d).mean():.3f}±{fe(c, d).std():.2f}".rjust(13)
            for c in ["a", "noise", "b_cur", "c_cur", "b_goal", "c_goal"]
        )
    )

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for c in conds:
    for j, EPS_H in enumerate([0.1, 0.2]):
        m = np.array([sr(c, d, EPS_H).mean() for d in demos])
        s = np.array([sr(c, d, EPS_H).std() for d in demos])
        st = "o--" if c in ("oracle_xyz", "bothprop") else "o-"
        ax[j].plot(demos, m, st, color=col[c], label=lab[c])
        ax[j].fill_between(demos, m - s, m + s, color=col[c], alpha=0.15)
for j, EPS_H in enumerate([0.1, 0.2]):
    ax[j].set(
        xscale="log",
        xlabel="# HER demos",
        ylabel=f"success @ {EPS_H}m",
        title=f"Cartesian reach SR@{EPS_H}m ({len(R)} seeds)",
        ylim=(-0.02, 1.02),
    )
    ax[j].grid(alpha=0.3)
ax[0].legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT + "/sp_curve2.png", dpi=120)
print(f"\n[ok] {OUT}/sp_curve2.png")
