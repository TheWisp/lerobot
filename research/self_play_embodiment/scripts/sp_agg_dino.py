# ruff: noqa
import glob

import numpy as np

R = [
    np.load(f)
    for f in sorted(
        glob.glob(
            "/tmp/selfplay_probe/sp_%s_results_s*.npz" % __import__("os").environ.get("SUBSTRATE", "dino")  # nosec B108
        )
    )
]
demos = R[0]["demos"]
print(f"DINOv2 | seeds {len(R)} | demos {list(demos)}")
conds = ["oracle_xyz", "bothprop", "a", "noise", "b_free", "c_ac", "c_invdyn"]
lab = {
    "oracle_xyz": "oracle",
    "bothprop": "bothprop",
    "a": "a(z)",
    "noise": "z+rand",
    "b_free": "z+e_free",
    "c_ac": "z+e_ac",
    "c_invdyn": "z+e_invdyn",
}


def sr(c, d, e):
    return np.array([(r[f"{c}|{d}"] < e).mean() for r in R])


def fe(c, d):
    return np.array([r[f"{c}|{d}"].mean() for r in R])


for E in [0.1, 0.2]:
    print(f"\n=== SR@{E}m (mean+-std, {len(R)} seeds) ===")
    print(f"{'#d':>5s} | " + " ".join(f"{lab[c]:>12s}" for c in conds))
    for d in demos:
        print(
            f"{d:>5d} | "
            + " ".join(f"{sr(c, d, E).mean():.2f}±{sr(c, d, E).std():.2f}".rjust(12) for c in conds)
        )
print("\n=== INJECTION DELTAS SR@0.2 (mean+-std) ===")
print(f"{'#d':>5s} | {'c_ac-noise':>14s} {'c_invdyn-noise':>15s} {'c_ac-a':>12s} {'c_invdyn-a':>13s}")
for d in demos:

    def dl(x, y):
        v = sr(x, d, 0.2) - sr(y, d, 0.2)
        return f"{v.mean():+.2f}±{v.std():.2f}"

    print(
        f"{d:>5d} | {dl('c_ac', 'noise'):>14s} {dl('c_invdyn', 'noise'):>15s} {dl('c_ac', 'a'):>12s} {dl('c_invdyn', 'a'):>13s}"
    )
print("\nfinalErr(m):")
print(f"{'#d':>5s} | " + " ".join(f"{lab[c]:>11s}" for c in ["a", "noise", "b_free", "c_ac", "c_invdyn"]))
for d in demos:
    print(
        f"{d:>5d} | "
        + " ".join(
            f"{fe(c, d).mean():.3f}±{fe(c, d).std():.2f}".rjust(11)
            for c in ["a", "noise", "b_free", "c_ac", "c_invdyn"]
        )
    )
