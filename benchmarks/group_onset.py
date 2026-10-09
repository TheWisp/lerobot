"""Which membership test sees a move sooner at the same false-alarm rate: a point's residual to its group's fitted
motion (the tracker's leave test), or how well it keeps its distances to the group's members (the pairwise rigidity
test of ClusterSLAM and multimotion visual odometry). Offline, on the tracks group_live.py --dump saved, so both
tests see the same tracks, the same groups and the same anchors.

    python benchmarks/group_onset.py TRACKS.npz RECORDING_DIR --event slide=400:540 [--event ...] [--out DIR]

An event is a frame window holding one move, still at both ends. Its truth is taken with hindsight: a point moved
when it ended up more than 15 mm from where it started, stayed when under 3 mm, and the move began at the first
frame from which the moved points' median displacement stays above 2 mm. In the group they shared before the move,
the side with fewer points is the one a membership test must flag (the group's fit follows the majority). Still
stretches, where the picture does not change for a second either side, give the false alarms: any flag there is
one. Each test flags a point when its statistic stays above a threshold for P frames in a row; thresholds are swept,
and the tests are compared at equal false-alarm rate."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings

import cv2
import numpy as np

from lerobot.showservo.groups import GroupTracker

FULL_TENURE = 30  # the tracker's: members this settled define a group's frame, and are the pairwise partners


def run_tracker(xyz: np.ndarray, seen: np.ndarray) -> dict:
    """The tracker over the dumped frames, as the live view ran it (tracks appear when they were seeded); per frame
    each track's group, anchor and tenure, and each group's motion."""
    f_n, n, _ = xyz.shape
    exists = np.isfinite(xyz).all(axis=2) | seen
    count = np.maximum.accumulate(np.array([(np.flatnonzero(e).max() + 1) if e.any() else 0 for e in exists]))
    tracker = GroupTracker()
    rec = {
        "group_of": np.full((f_n, n), -1, np.int32),
        "anchor": np.full((f_n, n, 3), np.nan, np.float32),
        "tenure": np.zeros((f_n, n), np.int32),
        "motions": [],
        "ms": np.zeros(f_n),
    }
    for f in range(f_n):
        k = int(count[f])
        t0 = time.perf_counter()
        tracker.update(xyz[f, :k], seen[f, :k])
        rec["ms"][f] = (time.perf_counter() - t0) * 1000
        m = len(tracker.group_of)
        rec["group_of"][f, :m] = tracker.group_of
        rec["anchor"][f, :m] = tracker.anchor
        rec["tenure"][f, :m] = tracker.tenure
        rec["motions"].append(
            {g: (grp.motion.rot.copy(), grp.motion.trans.copy()) for g, grp in tracker.groups.items()}
        )
    return rec


def image_still(
    recording: pathlib.Path, n_frames: int, margin: int = 15, changed: float = 0.005
) -> np.ndarray:
    """Frames with no change in the picture within ``margin`` frames either side: the share of pixels whose
    blurred quarter-size green differs by more than 12 grey levels from the frame before stays under ``changed``."""
    prev, moving = None, np.zeros(n_frames, bool)
    for i in range(n_frames):
        g = cv2.imread(str(recording / "rgb" / f"{i:06d}.jpg"))[:, :, 1]
        g = cv2.GaussianBlur(cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4)), (5, 5), 0).astype(np.float32)
        moving[i] = prev is not None and float(np.mean(np.abs(g - prev) > 12)) > changed
        prev = g
    near = np.convolve(moving.astype(int), np.ones(2 * margin + 1, int), mode="same") > 0
    return ~near


def ground_truth(xyz, seen, w0, w1, moved_m=0.015, still_m=0.003, onset_m=0.002):
    """Hindsight labels and onset for one event window."""

    def at(frames):
        x = np.where(seen[frames, :, None], xyz[frames], np.nan)
        ok = np.isfinite(x).all(axis=2).sum(axis=0) >= 3
        return np.nanmedian(x, axis=0), ok

    x_pre, ok_pre = at(slice(w0, w0 + 5))
    x_post, ok_post = at(slice(w1 - 4, w1 + 1))
    ok = ok_pre & ok_post
    d = np.where(ok, np.linalg.norm(x_post - x_pre, axis=1), np.nan)
    moved, still = ok & (d > moved_m), ok & (d < still_m)
    disp = np.full(w1 - w0 + 1, np.nan)
    for k, f in enumerate(range(w0, w1 + 1)):
        v = moved & seen[f] & np.isfinite(xyz[f]).all(axis=1)
        if v.sum() >= 5:
            disp[k] = np.median(np.linalg.norm(xyz[f, v] - x_pre[v], axis=1))
    above = disp > onset_m
    onset = next((w0 + k for k in range(len(above) - 3) if above[k : k + 4].all()), None)
    return moved, still, onset, disp


def statistics(xyz, seen, rec, r, members, frames, partners_m=16, rng=None):
    """For ``members`` of one group at frame ``r``: per frame in ``frames``, the residual to the group's fitted
    motion (A) and the median change of its distances to ``partners_m`` established members (B), both against the
    anchors of frame ``r``. Also the time each takes per frame. NaN where unseen or the group is gone."""
    g = int(rec["group_of"][r, members[0]])
    anchors = rec["anchor"][r].astype(np.float64)
    pool = np.flatnonzero((rec["group_of"][r] == g) & (rec["tenure"][r] >= FULL_TENURE))
    rng = rng or np.random.default_rng(0)
    partners = np.stack(
        [rng.choice(pool[pool != i], size=partners_m, replace=len(pool) <= partners_m) for i in members]
    )
    ref = np.linalg.norm(anchors[members][:, None] - anchors[partners], axis=2)  # in the group's own frame
    a = np.full((len(frames), len(members)), np.nan)
    b = np.full((len(frames), len(members)), np.nan)
    ms_a, ms_b = [], []
    for k, f in enumerate(frames):
        mot = rec["motions"][f].get(g)
        x = xyz[f].astype(np.float64)
        ok = seen[f, members] & np.isfinite(x[members]).all(axis=1)
        if mot is not None:
            t0 = time.perf_counter()
            pred = anchors[members] @ mot[0].T + mot[1]
            a[k] = np.where(ok, np.linalg.norm(pred - x[members], axis=1), np.nan)
            ms_a.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        now = np.linalg.norm(x[members][:, None] - x[partners], axis=2)
        pok = seen[f, partners] & np.isfinite(now)
        strain = np.where(pok, np.abs(now - ref), np.nan)
        enough = pok.sum(axis=1) >= partners_m // 2
        with np.errstate(all="ignore"):
            b[k] = np.where(ok & enough, np.nanmedian(strain, axis=1), np.nan)
        ms_b.append((time.perf_counter() - t0) * 1000)
    return a, b, float(np.mean(ms_a)) if ms_a else 0.0, float(np.mean(ms_b))


def first_fire(stat: np.ndarray, tau: float, p: int) -> np.ndarray:
    """Per column, the first row index at which ``stat`` has been above ``tau`` for ``p`` rows in a row; -1 if
    never. NaN (unseen) breaks a run."""
    above = np.nan_to_num(stat, nan=-1.0) > tau
    run = np.zeros(stat.shape[1], int)
    fire = np.full(stat.shape[1], -1)
    for k in range(stat.shape[0]):
        run = np.where(above[k], run + 1, 0)
        fire = np.where((fire < 0) & (run >= p), k, fire)
    return fire


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks")
    ap.add_argument("recording")
    ap.add_argument("--event", action="append", default=[], help="name=first:last frame")
    ap.add_argument("--out", default=None)
    ap.add_argument("--partners", type=int, default=16)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="All-NaN slice")  # unseen points: NaN by design
    d = np.load(args.tracks)
    xyz, seen, t = d["xyz"], d["seen"], d["t"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    print(f"{len(t)} frames x {xyz.shape[1]} tracks, {fps:.1f} fps")
    rec = run_tracker(xyz, seen)
    print(f"tracker: {np.median(rec['ms']):.1f} ms a frame (median)")
    still = image_still(pathlib.Path(args.recording), len(t))
    taus_a = [0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012, 0.015]
    taus_b = [0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
    persist = [1, 2, 3]
    rng = np.random.default_rng(0)

    # False alarms: still windows of 3 s, every grouped point, any flag is false.
    h = int(round(3 * fps))
    fa_points_s, fa = 0.0, {}
    starts = [r for r in range(0, len(t) - h, h) if still[r : r + h + 1].all()]
    timing = []
    for r in starts:
        for g in np.unique(rec["group_of"][r][rec["group_of"][r] >= 0]):
            members = np.flatnonzero(rec["group_of"][r] == g)
            if len(members) < 16:
                continue
            a, b, ms_a, ms_b = statistics(
                xyz, seen, rec, r, members, range(r + 1, r + h + 1), args.partners, rng
            )
            timing.append((len(members), ms_a, ms_b))
            fa_points_s += len(members) * h / fps
            for which, stat, taus in (("A", a, taus_a), ("B", b, taus_b)):
                for tau in taus:
                    for p in persist:
                        fa[(which, tau, p)] = fa.get((which, tau, p), 0) + int(
                            (first_fire(stat, tau, p) >= 0).sum()
                        )
    fa_rate = {k: v / fa_points_s * 60 for k, v in fa.items()}  # false flags per point per minute
    print(f"still windows: {len(starts)} of {h} frames, {fa_points_s / 60:.0f} point-minutes")

    # Events: latency of the flag on the side that must be flagged, and how far the move had gone by then.
    results = {
        "events": {},
        "fa_per_point_minute": {f"{k[0]} {k[1] * 1000:g}mm P{k[2]}": v for k, v in fa_rate.items()},
    }
    for spec in args.event:
        name, rng_s = spec.split("=")
        w0, w1 = (int(x) for x in rng_s.split(":"))
        moved, stayed, onset, disp = ground_truth(xyz, seen, w0, w1)
        if onset is None:
            print(f"{name}: no onset found")
            continue
        r = onset - 1
        g_of = rec["group_of"][r]
        shared = g_of[(moved | stayed) & (g_of >= 0)]
        g = int(np.bincount(shared).argmax())
        in_g = g_of == g
        n_moved, n_still = int((moved & in_g).sum()), int((stayed & in_g).sum())
        signal = np.flatnonzero((moved if n_moved < n_still else stayed) & in_g)
        frames = range(r + 1, w1 + 1)
        a, b, _, _ = statistics(xyz, seen, rec, r, signal, frames, args.partners, rng)
        # the tracker as it ran: when a new group first held 16 of the flagged side's points
        birth = next(
            (
                f
                for f in frames
                if any(
                    gg != g and int(((rec["group_of"][f] == gg)[signal]).sum()) >= 16
                    for gg in np.unique(rec["group_of"][f][signal])
                    if gg >= 0
                )
            ),
            None,
        )
        ev = {
            "onset": onset,
            "flagged_side": "moved" if n_moved < n_still else "stayed",
            "n_signal": len(signal),
            "group_birth_frames_after_onset": None if birth is None else birth - onset,
            "rows": {},
        }
        for which, stat, taus in (("A", a, taus_a), ("B", b, taus_b)):
            for tau in taus:
                for p in persist:
                    fire = first_fire(stat, tau, p)
                    lat = np.where(fire >= 0, fire + 1 + r - onset, 10**6)
                    med = float(np.median(lat))
                    at = (
                        float(disp[int(np.median(lat)) + onset - w0]) * 1000
                        if med < 10**6 and int(med) + onset - w0 < len(disp)
                        else None
                    )
                    ev["rows"][f"{which} {tau * 1000:g}mm P{p}"] = {
                        "median_frames": None if med >= 10**6 else med,
                        "detected": float((fire >= 0).mean()),
                        "moved_mm_at_detection": at,
                    }
        results["events"][name] = ev
        print(
            f"{name}: onset {onset}, flag the {ev['flagged_side']} side ({len(signal)} pts), "
            f"the tracker's new group {ev['group_birth_frames_after_onset']} frames after onset"
        )

    sizes, ms_a, ms_b = (np.array(x) for x in zip(*timing, strict=True)) if timing else ([], [], [])
    results["timing"] = {
        "members_median": float(np.median(sizes)),
        "A_ms": float(np.median(ms_a)),
        "B_ms": float(np.median(ms_b)),
        "partners": args.partners,
        "tracker_ms": float(np.median(rec["ms"])),
    }
    print(json.dumps(results["timing"]))
    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "onset.json").write_text(json.dumps(results, indent=1))
        print(f"wrote {out / 'onset.json'}")
        plot(results, out / "onset.png", fps)


def plot(results: dict, path: pathlib.Path, fps: float) -> None:
    """Per event, how far the move had gone when each setting flags it, against that setting's false alarms on
    still stretches: lower-left is better. The tracker's own setting is ringed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = results["events"]
    fig, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 4.2), squeeze=False)
    colours = {"A": "#1f77b4", "B": "#d62728"}
    labels = {"A": "residual to the group's fit", "B": "pairwise distances"}
    for ax, (name, ev) in zip(axes[0], events.items(), strict=True):
        for which in ("A", "B"):
            xs, ys, ps = [], [], []
            for key, row in ev["rows"].items():
                if not key.startswith(which) or row["moved_mm_at_detection"] is None or row["detected"] < 0.5:
                    continue
                xs.append(max(results["fa_per_point_minute"][key], 1e-3))
                ys.append(row["moved_mm_at_detection"])
                ps.append(int(key[-1]))
            for p, marker in ((1, "o"), (2, "s"), (3, "^")):
                sel = [i for i, q in enumerate(ps) if q == p]
                ax.scatter(
                    [xs[i] for i in sel],
                    [ys[i] for i in sel],
                    c=colours[which],
                    marker=marker,
                    s=28,
                    label=f"{labels[which]}, {p} frame{'s' if p > 1 else ''}",
                )
        own = ev["rows"].get("A 10mm P3")
        if own and own["moved_mm_at_detection"] is not None:
            ax.scatter(
                [max(results["fa_per_point_minute"]["A 10mm P3"], 1e-3)],
                [own["moved_mm_at_detection"]],
                s=180,
                facecolors="none",
                edgecolors="k",
                linewidths=1.5,
                label="the tracker today",
            )
        ax.set_xscale("log")
        ax.set_xlabel("false flags per point per minute, still scene")
        ax.set_ylabel("mm moved when flagged (median point)")
        ax.set_title(f"{name}: groups split {ev['group_birth_frames_after_onset']} frames after onset")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
