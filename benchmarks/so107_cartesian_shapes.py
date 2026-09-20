#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cartesian shapes on the real SO-107: heart / circle / square at constant commanded height.

The May 2026 hardware-trajectory protocol (``cartesian_ik_hardware_traj.py`` on
``feat/quest-vr-teleop``) on one arm: 256 waypoints at 30 Hz through the production
Cartesian-IK controller, ramped from a pinned seed pose to an anchor 5 cm above it. Same
figure as then — 3D commanded / FK(action) / FK(state) traces and a Δz-per-waypoint row —
so runs at different gains, or a different year, compare panel to panel. No lag guard: the
residual is the measurement; a joint 35 deg behind its command aborts the shape and the
arm recovers to the seed before the next one.

Results are in ``src/lerobot/robots/so107_description/docs/gravity_sag.md``.

Usage::

    PYTHONPATH=src python benchmarks/so107_cartesian_shapes.py --tag baseline
    PYTHONPATH=src python benchmarks/so107_cartesian_shapes.py --p 32 --ff-alpha 3.2 --tag P32_FF
    PYTHONPATH=src python benchmarks/so107_cartesian_shapes.py --replot-old may/run.npz
    PYTHONPATH=src python benchmarks/so107_cartesian_shapes.py --planview outputs/gravity_sag/shapes_* \\
        --old-npz may/run.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import so107_gravity_sag  # noqa: E402
from so107_gravity_sag import HZ, Run, connect_arm  # noqa: E402

from lerobot.robots.so107_description.cartesian_ik import (  # noqa: E402
    SO107_WORKSPACE_MAX,
    SO107_WORKSPACE_MIN,
    CartesianIKController,
    make_so107_arm_kinematics,
)
from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES  # noqa: E402

# May protocol: no lag guard, only a wide runaway guard (the commanded path is bounded, and a
# 35 deg drop at these poses still leaves the tip well above the tray).
so107_gravity_sag.DIVERGE_DEG = 35.0

N_WAYPOINTS, RAMP_TICKS, SETTLE_TICKS, WARMUP_TICKS = 256, 30, 15, 20
# One seed for every A/B run (motor deg): gripper down over the tray, reach ~165 mm, tip
# ~110 mm above the base.
SEED_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 74.0,
    "forearm_roll": 2.0,
    "wrist_flex": -41.0,
    "wrist_roll": -12.0,
}
SHAPES = [
    ("heart 50 mm wide (45 mm tall)", "heart", 0.050),
    ("circle 60 mm radius", "circle", 0.060),
    ("square 50 mm side", "square", 0.050),
]
OLD_KEYS = (
    ("heart 50 mm wide (45 mm tall)", "heart_50_mm"),
    ("circle 60 mm radius", "circle_60_mm_radius"),
    ("square 50 mm side", "square_50_mm_side"),
)
LOG_KEYS = ("ref_l", "shape_start", "shape_end", "cmd_ee_l", "ach_ee_l", "cmd_j_l", "ach_j_l")


def _heart_unit(n: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, y], axis=1)
    pts -= pts[0]
    return pts / float((pts.max(axis=0) - pts.min(axis=0)).max())


def plane_basis(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(seed EE, forward = away from the base, lateral) — the May bench's convention."""
    p = ref[:3, 3]
    flat = np.array([p[0], p[1], 0.0])
    norm = float(np.linalg.norm(flat))
    forward = flat / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    lateral = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    lateral /= np.linalg.norm(lateral)
    return p, forward, lateral


def shape_deltas(ref: np.ndarray, shape: str, size_m: float, n: int) -> list[np.ndarray]:
    p, forward, lateral = plane_basis(ref)
    if shape == "circle":
        r = size_m
        center = p + r * forward
        return [
            (center + r * (np.cos(2 * np.pi * i / n) * -forward + np.sin(2 * np.pi * i / n) * lateral)) - p
            for i in range(n)
        ]
    if shape == "heart":
        return [size_m * (u[0] * forward + u[1] * lateral) for u in _heart_unit(n)]
    s = size_m
    corners = [p, p + s * forward, p + s * forward + s * lateral, p + s * lateral]
    out = []
    for e in range(4):
        a, b = corners[e], corners[(e + 1) % 4]
        for k in range(n // 4):
            out.append((a + (b - a) * (k / (n // 4))) - p)
    return out


def run_shapes(run: Run, offset_up: float, offset_fwd: float) -> dict:
    results = {}
    ncol = len(MOTOR_NAMES)
    for label, shape, size in SHAPES:
        i0 = len(run.rows)
        p_seed = run.pos_cmd.copy()
        ref = np.eye(4)
        ref[:3, 3] = p_seed
        _, forward, _ = plane_basis(ref)
        anchor = p_seed + np.array([0, 0, offset_up]) + forward * offset_fwd
        ref_anchor = np.eye(4)
        ref_anchor[:3, 3] = anchor
        deltas = shape_deltas(ref_anchor, shape, size, N_WAYPOINTS)
        plan = [p_seed + (anchor - p_seed) * k / RAMP_TICKS for k in range(1, RAMP_TICKS + 1)]
        plan += [anchor + d for d in deltas]
        plan += [anchor + (p_seed - anchor) * k / RAMP_TICKS for k in range(1, RAMP_TICKS + 1)]
        plan += [p_seed] * SETTLE_TICKS
        aborted = None
        for k, pos in enumerate(plan):
            t_tick = time.perf_counter()
            try:
                run.tick(pos)  # May protocol: no lag-wait — the residual is the measurement
            except RuntimeError as e:
                aborted = k
                print(f"  {label}: ABORTED at plan tick {k} ({e}); recovering to the seed")
                break
            run.pos_cmd = pos.copy()
            time.sleep(max(0.0, 1.0 / HZ - (time.perf_counter() - t_tick)))
        rows = np.array(run.rows[i0:])
        if aborted is not None:
            # Joint-space recovery to the seed, then a fresh controller latched on the real pose.
            so107_gravity_sag.stream_joints(run.robot, run.present(), run.q_seed)
            q0 = run.present()
            run.ctrl = CartesianIKController(
                kinematics=run.kin,
                motor_names=list(MOTOR_NAMES),
                q_init=q0,
                workspace_min=SO107_WORKSPACE_MIN,
                workspace_max=SO107_WORKSPACE_MAX,
                label="left",
            )
            run.p0 = run.kin.forward_kinematics(q0)[:3, 3].copy()
            run.pos_cmd = run.p0.copy()
            run.q_now = None
        cmd_j = rows[:, 1 : 1 + ncol]
        ach_j = rows[:, 1 + ncol : 1 + 2 * ncol]
        cmd_ee = rows[:, 1 + 2 * ncol : 4 + 2 * ncol]
        ach_ee = np.array([run.kin.forward_kinematics(q)[:3, 3] for q in ach_j])
        end = min(RAMP_TICKS + N_WAYPOINTS, len(rows))
        results[label] = {
            "ref_l": ref_anchor,
            "shape_start": RAMP_TICKS,
            "shape_end": end,
            "aborted_tick": -1 if aborted is None else aborted,
            "cmd_ee_l": cmd_ee,
            "ach_ee_l": ach_ee,
            "cmd_j_l": cmd_j,
            "ach_j_l": ach_j,
        }
        s = RAMP_TICKS + WARMUP_TICKS
        if end <= s:
            print(f"  {label:32s} aborted before the shape started")
            continue
        fk_cmd = np.array([run.kin.forward_kinematics(q)[:3, 3] for q in cmd_j[s:end]])
        dz = 1000 * (ach_ee[s:end, 2] - cmd_ee[s:end, 2])
        print(
            f"  {label:32s} IK floor max {1000 * np.linalg.norm(fk_cmd - cmd_ee[s:end], axis=1).max():.2f} mm | "
            f"motor residual max {1000 * np.linalg.norm(ach_ee[s:end] - fk_cmd, axis=1).max():.1f} mm | "
            f"dz FK(state) mean {dz.mean():+.1f} / max |{np.abs(dz).max():.1f}| mm"
        )
    return results


def load_old(npz_path: pathlib.Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {label: {k: d[f"{key}__{k}"] for k in LOG_KEYS} for label, key in OLD_KEYS}


def load_run(run_dir: pathlib.Path) -> dict:
    d = np.load(run_dir / "run.npz", allow_pickle=True)
    out = {}
    for label, _ in OLD_KEYS:
        if f"{label}__ref_l" not in d.files:
            continue
        log = {k: d[f"{label}__{k}"] for k in LOG_KEYS}
        log["aborted_tick"] = (
            d[f"{label}__aborted_tick"] if f"{label}__aborted_tick" in d.files else np.int32(-1)
        )
        out[label] = log
    return out


def _canonical(log: dict, kin) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(commanded, achieved, FK(action)) in (forward, lateral, z) mm relative to the anchor."""
    ref = log["ref_l"]
    _, fwd, lat = plane_basis(ref)
    s, e = int(log["shape_start"]), int(log["shape_end"])
    to = lambda d: np.stack([d @ fwd, d @ lat, d[:, 2]], axis=1) * 1000.0  # noqa: E731
    fk = np.array([kin.forward_kinematics(q)[:3, 3] for q in log["cmd_j_l"][s:e]])
    return to(log["cmd_ee_l"][s:e] - ref[:3, 3]), to(log["ach_ee_l"][s:e] - ref[:3, 3]), to(fk - ref[:3, 3])


def plot_results(results: dict, path: pathlib.Path, kin, suptitle: str) -> None:
    """The May figure: per shape, 3D traces above a Δz-per-waypoint row, shared axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for label, log in results.items():
        if int(log["shape_end"]) - int(log["shape_start"]) < WARMUP_TICKS + 2:
            continue
        cmd, ach, act = _canonical(log, kin)
        ab = int(log.get("aborted_tick", -1))
        title = label + (
            f"  [ABORTED at waypoint {ab - int(log['shape_start'])}: joint 35 deg behind]" if ab >= 0 else ""
        )
        panels.append((title, cmd, ach, act))
    n = len(panels)
    fig = plt.figure(figsize=(5 * n, 9))
    gs = fig.add_gridspec(2, n, height_ratios=(2.2, 1.0))
    allv = np.concatenate([np.concatenate([c, a, f]) for _, c, a, f in panels])
    alldz = np.concatenate([np.concatenate([a[:, 2] - c[:, 2], f[:, 2] - c[:, 2]]) for _, c, a, f in panels])
    pad = lambda v: (v.min() - 0.05 * (np.ptp(v) + 1e-6), v.max() + 0.05 * (np.ptp(v) + 1e-6))  # noqa: E731
    fl, ll, zl, dzl = pad(allv[:, 0]), pad(allv[:, 1]), pad(allv[:, 2]), pad(alldz)
    w = WARMUP_TICKS
    for col, (title, cmd, ach, act) in enumerate(panels):
        err3d = np.linalg.norm(cmd - ach, axis=1)
        errxy = np.linalg.norm(cmd[:, :2] - ach[:, :2], axis=1)
        dz = ach[:, 2] - cmd[:, 2]
        ik_err = np.linalg.norm(act - cmd, axis=1)
        motor_err = np.linalg.norm(ach - act, axis=1)
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.plot(cmd[:, 0], cmd[:, 1], cmd[:, 2], color="C0", linewidth=2.2, label="commanded EE target")
        ax.plot(
            act[:, 0],
            act[:, 1],
            act[:, 2],
            color="C2",
            linewidth=1.2,
            linestyle=":",
            label="FK(action) — what IK asked the motors to do",
        )
        ax.plot(
            ach[:, 0],
            ach[:, 1],
            ach[:, 2],
            color="C3",
            linewidth=1.0,
            linestyle="--",
            label="FK(state) — where the motors landed",
        )
        ax.set_xlabel("forward (mm)")
        ax.set_ylabel("lateral (mm)")
        ax.set_zlabel("z (mm)")
        ax.set_xlim(*fl)
        ax.set_ylim(*ll)
        ax.set_zlim(*zl)
        ax.view_init(elev=18, azim=-65)
        ax.set_title(
            f"{title}\nFK(state) − target: max 3D {err3d[w:].max():.1f} mm (in-plane {errxy[w:].max():.1f}, |z| {np.abs(dz[w:]).max():.1f})\n"
            f"IK math floor: max {ik_err[w:].max():.2f} mm  |  motor tracking: max {motor_err[w:].max():.1f} mm",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8)
        zax = fig.add_subplot(gs[1, col])
        x = np.arange(len(cmd))
        zax.axhline(0.0, color="C0", linewidth=2.0, label="commanded z target (constant)")
        zax.plot(
            x,
            act[:, 2] - cmd[:, 2],
            color="C2",
            linewidth=1.5,
            linestyle=":",
            label="FK(action) z − commanded z (IK floor)",
        )
        zax.plot(x, dz, color="C3", linewidth=1.5, label="FK(state) z − commanded z")
        zax.set_xlim(0, len(cmd))
        zax.set_ylim(*dzl)
        zax.grid(alpha=0.3)
        zax.set_xlabel("waypoint (30 Hz tick; 1 waypoint ≈ 33 ms)")
        if col == 0:
            zax.set_ylabel("Δz vs commanded (mm)")
        zax.legend(loc="lower right", fontsize=7)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plan_view(run_dirs: list[str], old_npz: str | None, out: pathlib.Path) -> None:
    """Top view (equal aspect) and side view of the circle, one column per run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    runs = []
    if old_npz:
        runs.append(("May 2026 (PR #10 data)", load_old(pathlib.Path(old_npz))["circle 60 mm radius"]))
    for r in run_dirs:
        d = pathlib.Path(r)
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        runs.append((meta.get("tag", d.name), load_run(d)["circle 60 mm radius"]))
    fig, axes = plt.subplots(2, len(runs), figsize=(5 * len(runs), 9), squeeze=False)
    for col, (title, log) in enumerate(runs):
        cmd, ach, _ = _canonical(log, kin)
        origin = cmd[0].copy()
        cmd, ach = cmd - origin, ach - origin
        w = WARMUP_TICKS
        exy = ach[w:, :2] - cmd[w:, :2]
        ez = ach[w:, 2] - cmd[w:, 2]
        ax = axes[0, col]
        ax.plot(cmd[:, 0], cmd[:, 1], color="C0", lw=2.5, label="commanded")
        ax.plot(ach[:, 0], ach[:, 1], color="C3", lw=1.2, ls="--", label="achieved (FK of encoders)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("forward (mm)")
        ax.set_ylabel("lateral (mm)")
        ax.set_title(
            f"{title}\nTOP — in-plane rms {np.sqrt((exy**2).sum(1).mean()):.1f} mm, max {np.linalg.norm(exy, axis=1).max():.1f} mm",
            fontsize=10,
        )
        ax.set_xlim(-30, 150)
        ax.set_ylim(-90, 90)
        ax.legend(loc="lower right", fontsize=8)
        ax = axes[1, col]
        ax.plot(cmd[:, 0], cmd[:, 2], color="C0", lw=2.5, label="commanded (constant height)")
        ax.plot(ach[:, 0], ach[:, 2], color="C3", lw=1.2, ls="--", label="achieved")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("forward (mm)")
        ax.set_ylabel("height z (mm)")
        ax.set_title(
            f"SIDE — height error mean {ez.mean():+.1f} mm, max {np.abs(ez).max():.1f} mm", fontsize=10
        )
        ax.set_xlim(-30, 150)
        ax.set_ylim(-100, 20)
        ax.legend(loc="lower left", fontsize=8)
        print(
            f"{title:40s} in-plane rms {np.sqrt((exy**2).sum(1).mean()):5.1f} max {np.linalg.norm(exy, axis=1).max():5.1f} | "
            f"z mean {ez.mean():+6.1f} max |{np.abs(ez).max():5.1f}| mm ({len(cmd)} waypoints)"
        )
    fig.suptitle(
        "60 mm horizontal circle — commanded vs achieved, equal-aspect views, same code for every run",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="white_left")
    ap.add_argument(
        "--no-seed-move", action="store_true", help="start from wherever the arm is instead of SEED_DEG"
    )
    ap.add_argument("--offset-up", type=float, default=0.05)
    ap.add_argument(
        "--offset-fwd",
        type=float,
        default=-0.02,
        help="anchor offset along forward; -0.02 spans May's 145-265 mm reach band",
    )
    ap.add_argument(
        "--n-waypoints",
        type=int,
        default=256,
        help="256 = May protocol (~4-5 cm/s); 1024 = the plain follower's own protocol",
    )
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument(
        "--ff-alpha",
        type=float,
        default=0.0,
        help="gravity feed-forward in the follower, deg per N*m (0 = off)",
    )
    ap.add_argument("--max-rel", type=float, default=5.0)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--replot-old", default=None, help="May run.npz to replot with this code")
    ap.add_argument(
        "--planview",
        nargs="*",
        default=None,
        metavar="RUN_DIR",
        help="top/side circle views for these run dirs",
    )
    ap.add_argument("--old-npz", default=None, help="with --planview: the May run.npz as the first panel")
    args = ap.parse_args()
    global N_WAYPOINTS
    N_WAYPOINTS = args.n_waypoints
    out = pathlib.Path(args.out or f"outputs/gravity_sag/shapes_{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    if args.planview is not None:
        plan_view(args.planview, args.old_npz, out / "circle_planview.png")
        return
    if args.replot_old:
        plot_results(
            load_old(pathlib.Path(args.replot_old)),
            out / "old_replot.png",
            kin,
            "May 2026 run (PR #10 data, replotted) — left arm, P=16 I=0",
        )
        print("replotted old run ->", out / "old_replot.png")
        return

    args.start_joints = (
        None
        if args.no_seed_move
        else (SEED_DEG["shoulder_lift"], SEED_DEG["elbow_flex"], SEED_DEG["wrist_flex"])
    )
    robot, q0 = connect_arm(args)
    try:
        if not args.no_seed_move:
            # connect_arm moved lift/elbow/wrist_flex; pin the rest of the seed too so A/B runs start identically.
            q_to = q0.copy()
            for j, m in enumerate(MOTOR_NAMES):
                if m in SEED_DEG:
                    q_to[j] = SEED_DEG[m]
            q0 = so107_gravity_sag.stream_joints(robot, q0, q_to)
        t0 = kin.forward_kinematics(q0)
        p0 = t0[:3, 3].copy()
        print("seed joints (deg):", dict(zip(MOTOR_NAMES, np.round(q0, 1), strict=True)))
        print("seed EE (mm):", np.round(p0 * 1000, 1))
        ctrl = CartesianIKController(
            kinematics=kin,
            motor_names=list(MOTOR_NAMES),
            q_init=q0,
            workspace_min=SO107_WORKSPACE_MIN,
            workspace_max=SO107_WORKSPACE_MAX,
            label="left",
        )
        run = Run(robot, ctrl, kin, p0, float(q0[MOTOR_NAMES.index("gripper")]), out)
        run.q_seed = q0.copy()
        if args.ff_alpha > 0:
            offs = robot._gravity_ff.offset_deg(dict(zip(MOTOR_NAMES, q0, strict=True)))
            print(
                f"gravity feed-forward ON in the follower: alpha {args.ff_alpha} deg/(N*m); offset at seed (deg):",
                {m: round(v, 2) for m, v in offs.items()},
            )
        results = run_shapes(run, args.offset_up, args.offset_fwd)
        np.savez_compressed(
            out / "run.npz", **{f"{k}__{kk}": vv for k, v in results.items() for kk, vv in v.items()}
        )
        gains = f"P={args.p} I={args.i}" + (
            f" + gravity FF alpha={args.ff_alpha}" if args.ff_alpha > 0 else ""
        )
        plot_results(
            results,
            out / "trajectory_traces.png",
            kin,
            f"SO-107 left arm — hardware trajectory traces, {time.strftime('%Y-%m-%d')} ({args.tag}, {gains})",
        )
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "q0_deg": q0.tolist(),
                    "p0_m": p0.tolist(),
                    "tag": args.tag,
                    "p": args.p,
                    "i": args.i,
                    "d": args.d,
                    "max_rel": args.max_rel,
                    "ff_alpha": args.ff_alpha,
                    "n_waypoints": args.n_waypoints,
                    "offset_up": args.offset_up,
                    "offset_fwd": args.offset_fwd,
                },
                indent=1,
            )
        )
        print("saved", out)
    finally:
        robot.disconnect()
        print("arm disconnected (torque on, holding)")


if __name__ == "__main__":
    main()
