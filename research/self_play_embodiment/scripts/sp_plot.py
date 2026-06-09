"""Render sp_results.npz -> SR-vs-#demos table + curve. Headline = does c_ac beat b_free
beat noise~=a in the low-demo regime, with oracle as the ceiling."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/selfplay_probe"
r = np.load(OUT + "/sp_results.npz")
demos = r["demos"]
eps = r["eps"]
conds = ["a", "noise", "b_free", "c_ac", "oracle"]
lab = {
    "a": "(a) z only",
    "noise": "(noise) z+rand",
    "b_free": "(b) z+e_free",
    "c_ac": "(c) z+e_ac",
    "oracle": "oracle z+goalprop",
}
col = {"a": "0.6", "noise": "0.8", "b_free": "tab:orange", "c_ac": "tab:red", "oracle": "tab:green"}

EPS_H = 0.5
print(f"{'#demos':>8s} | " + " ".join(f"{lab[c]:>18s}" for c in conds) + "    (SR@%.1f / finalErr)" % EPS_H)
for d in demos:
    cells = []
    for c in conds:
        f = r[f"{c}|{d}"]
        cells.append(f"{(f < EPS_H).mean():.2f}/{f.mean():.2f}")
    print(f"{d:>8d} | " + " ".join(f"{x:>18s}" for x in cells))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for c in conds:
    sr = [(r[f"{c}|{d}"] < EPS_H).mean() for d in demos]
    fe = [r[f"{c}|{d}"].mean() for d in demos]
    ax[0].plot(demos, sr, "o-", color=col[c], label=lab[c])
    ax[1].plot(demos, fe, "o-", color=col[c], label=lab[c])
ax[0].set(
    xscale="log",
    xlabel="# HER demos",
    ylabel=f"success rate @ {EPS_H}",
    title="Reaching SR vs demos",
    ylim=(-0.02, 1.02),
)
ax[1].set(xscale="log", xlabel="# HER demos", ylabel="final joint-L2 err", title="Final reach error vs demos")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "/sp_curve.png", dpi=120)
print(f"\n[ok] {OUT}/sp_curve.png")
