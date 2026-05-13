#!/usr/bin/env python
"""Teleop counterpart of backtest.py.

Sends single-frame intent dicts (NOT chunks) to the predictive follower at
the trajectory's source rate, simulating a leader-arm teleop stream. The
controller's internal velocity-LSQ extrapolation does the lookahead
(``motor_cmd = intent[t] + v_LSQ_over_70ms · L``) since there's no future
chunk to read from.

Purpose: apples-to-apples comparison of teleop-style command jitter vs.
chunked-action jitter on the same trajectory, robot, and lookahead value.

Trace files use the same .npz schema as backtest.py so analyze.py and
summarize_*.py work unchanged. The motor_cmd_lookahead value is computed
post-hoc here using the same LSQ velocity estimator the controller uses
internally — so the logged motor_cmd matches what the controller commanded.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.robots.bi_so107_follower_predictive import (
    BiSO107FollowerPredictive,
    BiSO107FollowerPredictiveConfig,
)
from lerobot.robots.rest_position import move_to_rest_position
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


# ── Trajectory loader (shared with backtest.py shape) ───────────────────────


def _load_safe_trajectory(path: Path):
    d = json.loads(path.read_text())
    joints = list(d["joints"])
    fps = float(d["fps"])
    positions = np.asarray(d["positions"], dtype=np.float64)
    return positions, joints, fps


def _load_dataset_episode(repo_id: str, episode_idx: int):
    import pyarrow.parquet as pq

    home = Path.home() / ".cache/huggingface/lerobot" / repo_id
    info = json.loads((home / "meta/info.json").read_text())
    fps = float(info["fps"])
    action_names = info["features"]["action"]["names"]
    for shard in sorted((home / "data").rglob("*.parquet")):
        t = pq.read_table(shard, columns=["episode_index", "action"])
        ep_col = t["episode_index"].to_numpy()
        if episode_idx not in ep_col:
            continue
        mask = ep_col == episode_idx
        actions = t["action"].to_numpy()[mask]
        positions = np.stack([np.asarray(a, dtype=np.float64) for a in actions])
        return positions, action_names, fps
    raise ValueError(f"Episode {episode_idx} not found in {repo_id}")


def load_source(source: str):
    if "@" in source and not Path(source).exists():
        repo_id, ep_str = source.rsplit("@", 1)
        return _load_dataset_episode(repo_id, int(ep_str))
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(source)
    return _load_safe_trajectory(path)


# ── LSQ velocity estimator (matches SO107FollowerPredictive's quad_end) ────


def _velocity_lsq_quad_end(ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
    """v(now) ≈ slope of quadratic fit at t = ts[-1]. Mirror of
    ``_PredictiveLookaheadController._velocity_lsq_quad_end`` so post-hoc
    motor_cmd matches what the controller actually computed."""
    if ts.shape[0] < 3:
        return _velocity_lsq_linear(ts, ps)
    t_rel = ts - ts[-1]
    design = np.stack([np.ones_like(t_rel), t_rel, t_rel * t_rel], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(design, ps, rcond=None)
    except np.linalg.LinAlgError:
        return _velocity_lsq_linear(ts, ps)
    return coef[1]


def _velocity_lsq_linear(ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
    ts_c = ts - ts.mean()
    denom = float((ts_c * ts_c).sum())
    if denom < 1e-12:
        return None
    return (ts_c @ ps) / denom


# ── Robot construction (same as backtest.py) ────────────────────────────────


def build_robot(profile_path: Path, lookahead_ms_override: float | None):
    profile = json.loads(profile_path.read_text())
    fields = dict(profile.get("fields", {}))
    if lookahead_ms_override is not None:
        fields["lookahead_ms"] = lookahead_ms_override
    fields["adaptive"] = False
    fields["max_lookahead_ms"] = fields.get("lookahead_ms", 0.0)
    fields["cameras"] = {}
    if "calibration_dir" in fields and fields["calibration_dir"] is not None:
        fields["calibration_dir"] = Path(fields["calibration_dir"]).expanduser()
    cfg = BiSO107FollowerPredictiveConfig(**{k: v for k, v in fields.items() if k != "type"})
    rest_position = profile.get("rest_position") or None
    return BiSO107FollowerPredictive(cfg), rest_position


# ── Main loop ───────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    gt, source_joints, source_fps = load_source(args.source)
    logger.info(
        "Source: %s — %d frames at %.0f fps (%.1f s), %d joints",
        args.source,
        len(gt),
        source_fps,
        len(gt) / source_fps,
        len(source_joints),
    )
    chunk_fps = source_fps
    velocity_window_s = args.velocity_window_ms / 1000.0
    lookahead_s = (args.lookahead_ms or 0.0) / 1000.0
    rng = np.random.default_rng(args.intent_noise_seed)
    n_joints = gt.shape[1]

    robot, rest_position = build_robot(Path(args.robot_profile).expanduser(), args.lookahead_ms)
    robot.connect()
    try:
        if rest_position is not None:
            logger.info("Moving to rest position before run")
            move_to_rest_position(robot, rest_position, duration_s=args.rest_duration_s)

        robot_joint_names = list(robot.action_features.keys())
        if source_joints != robot_joint_names:
            permutation = [source_joints.index(j) for j in robot_joint_names]
            gt = gt[:, permutation]
            joint_names = robot_joint_names
            logger.info("Permuted source columns to match robot.action_features order")
        else:
            joint_names = robot_joint_names

        outer_dt = 1.0 / chunk_fps

        # Ramp to trajectory start.
        logger.info("Ramping to trajectory start over %.1fs", args.ramp_to_start_s)
        start_pose = robot.get_observation()
        target_pose = {j: float(gt[0, joint_names.index(j)]) for j in joint_names}
        ramp_steps = max(1, int(args.ramp_to_start_s * chunk_fps))
        for step in range(1, ramp_steps + 1):
            alpha = step / ramp_steps
            action = {j: (1.0 - alpha) * start_pose[j] + alpha * target_pose[j] for j in joint_names}
            robot.send_action(action)
            time.sleep(outer_dt)

        ticks: list[dict] = []
        intent_log: list[tuple[float, np.ndarray]] = []  # (t, intent_array)
        run_start_t = time.perf_counter()
        max_ticks = int(len(gt) - 1)
        logger.info(
            "Running %d ticks (teleop @ %.0fHz, lookahead=%.0fms, intent_noise=%.3f deg)",
            max_ticks,
            chunk_fps,
            args.lookahead_ms or 0.0,
            args.intent_noise_deg,
        )

        while True:
            tick_start_t = time.perf_counter()
            playback_t = tick_start_t - run_start_t

            current_frame_idx = int(playback_t * chunk_fps)
            if current_frame_idx >= max_ticks:
                break

            # Build the intent for this tick. Real teleop = leader pose at
            # this wall time. We simulate this as ground_truth(t) + optional
            # Gaussian noise representing leader-encoder/mechanical jitter.
            gt_now = gt[current_frame_idx]
            if args.intent_noise_deg > 0:
                noise = rng.normal(0.0, args.intent_noise_deg, size=n_joints)
                intent_arr = gt_now + noise
            else:
                intent_arr = gt_now.copy()

            intent_dict = dict(zip(joint_names, intent_arr.tolist(), strict=True))
            robot.send_action(intent_dict)

            # Maintain rolling window for LSQ velocity.
            intent_log.append((tick_start_t, intent_arr))
            window_cutoff = tick_start_t - velocity_window_s
            while intent_log and intent_log[0][0] < window_cutoff:
                intent_log.pop(0)

            # Compute the post-hoc motor_cmd the controller's LSQ path
            # produces. Mirror of the controller's _tick math: extrapolate
            # by velocity_window_ms over intent samples in the window, then
            # motor_cmd = current_intent + v_LSQ * L.
            if len(intent_log) >= 3 and args.lookahead_ms:
                ts = np.array([t for t, _ in intent_log], dtype=np.float64)
                ps = np.stack([p for _, p in intent_log])
                v_leader = _velocity_lsq_quad_end(ts, ps)
                motor_cmd_lookahead = (
                    intent_arr + v_leader * lookahead_s if v_leader is not None else intent_arr
                )
            else:
                motor_cmd_lookahead = intent_arr

            motor_cmd_no_lookahead = intent_arr  # = chunk[0] equivalent

            obs = robot.get_observation()
            state_vec = np.asarray([obs[j] for j in joint_names], dtype=np.float64)

            ticks.append(
                {
                    "t": playback_t,
                    "emission_idx": current_frame_idx,
                    "gt": gt_now.copy(),
                    "motor_cmd_lookahead": motor_cmd_lookahead,
                    "motor_cmd_no_lookahead": motor_cmd_no_lookahead,
                    "state": state_vec,
                    "bias": np.zeros(n_joints),  # placeholder for schema compat
                }
            )

            elapsed = time.perf_counter() - tick_start_t
            if elapsed < outer_dt:
                time.sleep(outer_dt - elapsed)

        if rest_position is not None:
            logger.info("Moving to rest position after run")
            move_to_rest_position(robot, rest_position, duration_s=args.rest_duration_s)
    finally:
        robot.disconnect()

    if not ticks:
        logger.warning("No ticks recorded — nothing to save")
        return

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"teleop_L{int(args.lookahead_ms or 0)}_noise{args.intent_noise_deg}_seed{args.intent_noise_seed}"
    np.savez(
        out_dir / f"trace_{stem}.npz",
        t=np.asarray([t["t"] for t in ticks]),
        emission_idx=np.asarray([t["emission_idx"] for t in ticks]),
        gt=np.stack([t["gt"] for t in ticks]),
        motor_cmd_lookahead=np.stack([t["motor_cmd_lookahead"] for t in ticks]),
        motor_cmd_no_lookahead=np.stack([t["motor_cmd_no_lookahead"] for t in ticks]),
        state=np.stack([t["state"] for t in ticks]),
        bias=np.stack([t["bias"] for t in ticks]),
        joint_names=np.asarray(joint_names),
    )
    meta = {
        "mode": "teleop",
        "source": args.source,
        "chunk_fps": chunk_fps,
        "lookahead_ms": args.lookahead_ms or 0.0,
        "velocity_window_ms": args.velocity_window_ms,
        "intent_noise_deg": args.intent_noise_deg,
        "intent_noise_seed": args.intent_noise_seed,
        "n_ticks": len(ticks),
        "duration_s": ticks[-1]["t"],
    }
    (out_dir / f"meta_{stem}.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s (%d ticks)", out_dir / f"trace_{stem}.npz", len(ticks))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True)
    p.add_argument("--robot-profile", required=True)
    p.add_argument(
        "--lookahead-ms",
        type=float,
        default=None,
        help="Override the profile's lookahead_ms. Pass 0 for the no-lookahead baseline.",
    )
    p.add_argument(
        "--velocity-window-ms",
        type=float,
        default=70.0,
        help="Window for the LSQ velocity estimator. Matches the controller's default.",
    )
    p.add_argument(
        "--intent-noise-deg",
        type=float,
        default=0.0,
        help="Per-sample Gaussian noise (deg) added to the intent stream. "
        "Simulates leader-encoder / mechanical jitter. 0 = clean trajectory.",
    )
    p.add_argument("--intent-noise-seed", type=int, default=42)
    p.add_argument("--ramp-to-start-s", type=float, default=2.0)
    p.add_argument("--rest-duration-s", type=float, default=2.5)
    p.add_argument("--output-dir", default="outputs/chunk_cadence/teleop")
    return p.parse_args()


if __name__ == "__main__":
    init_logging()
    run(parse_args())
