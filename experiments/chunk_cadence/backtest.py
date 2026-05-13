#!/usr/bin/env python
"""Hardware backtest: chunked action delivery to the predictive follower.

Sends action chunks generated from a recorded trajectory (safe-trajectory
JSON or dataset episode) to a predictive follower, at a configurable
emission cadence and with a small bias on the overlap region of each
chunk to simulate the per-emission drift a real chunked policy would
exhibit.

What this script answers:
  * Does chunk-based control track ground truth under realistic update
    cadences (2-16 control frames between emissions)?
  * Does the predictive follower's lookahead deliver a useful action-to-
    state lead under chunked input?
  * How much jitter does the lookahead add (or remove) compared to
    a no-lookahead config? Measured as deviation from each config's own
    Savitzky-Golay smoothed trajectory.

Run twice per cell, same RNG seed:
  --lookahead-ms 0    (no-lookahead baseline)
  --lookahead-ms 80   (or your tuned value)

Then post-process the two trace files together.

Pure-experiment script — NOT for landing in core. Bypasses the teleop
framework: connects to the robot directly, runs its own outer loop at
``chunk_fps`` (default 30 Hz), the predictive follower's internal 200 Hz
control thread does the lookahead/extrapolation work.
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
from lerobot.types import ActionChunk
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


# ── Trajectory source loaders ───────────────────────────────────────────────


def _load_safe_trajectory(path: Path) -> tuple[np.ndarray, list[str], float]:
    """Load a safe-trajectory JSON. Returns (positions, joint_names, fps)."""
    d = json.loads(path.read_text())
    joints = list(d["joints"])
    fps = float(d["fps"])
    positions = np.asarray(d["positions"], dtype=np.float64)  # (n_frames, n_joints)
    return positions, joints, fps


def _load_dataset_episode(repo_id: str, episode_idx: int) -> tuple[np.ndarray, list[str], float]:
    """Load a single episode's action stream from a local dataset.

    Returns (positions, joint_names, fps).
    """
    import pyarrow.parquet as pq

    home = Path.home() / ".cache/huggingface/lerobot" / repo_id
    info = json.loads((home / "meta/info.json").read_text())
    fps = float(info["fps"])
    action_names = info["features"]["action"]["names"]

    # Scan parquet shards for the episode.
    data_dir = home / "data"
    for shard in sorted(data_dir.rglob("*.parquet")):
        t = pq.read_table(shard, columns=["episode_index", "action"])
        ep_col = t["episode_index"].to_numpy()
        if episode_idx not in ep_col:
            continue
        mask = ep_col == episode_idx
        actions = t["action"].to_numpy()[mask]
        # `action` column is list-of-floats per row → stack
        positions = np.stack([np.asarray(a, dtype=np.float64) for a in actions])
        return positions, action_names, fps
    raise ValueError(f"Episode {episode_idx} not found in {repo_id}")


def load_source(source: str) -> tuple[np.ndarray, list[str], float]:
    """Dispatch to the right loader based on source spec.

    Accepted shapes:
      * path to a safe-trajectory JSON
      * "<repo_id>@<episode_idx>" — e.g. "thewisp/cylinder_ring_assembly@287"
    """
    if "@" in source and not Path(source).exists():
        repo_id, ep_str = source.rsplit("@", 1)
        return _load_dataset_episode(repo_id, int(ep_str))
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(source)
    return _load_safe_trajectory(path)


# ── Chunk generator ─────────────────────────────────────────────────────────


def make_chunk(
    gt: np.ndarray,
    base_frame: int,
    chunk_size: int,
    update_every: int,
    bias: np.ndarray,
    joint_names: list[str],
    chunk_fps: float,
) -> ActionChunk | None:
    """Build one ActionChunk with overlap-region bias applied.

    Bias is applied only to ``frames[update_every:]`` — the part of this
    chunk that overlaps in wall time with the NEXT chunk (which has not
    arrived yet). The first ``update_every`` frames are clean ground truth;
    those are the frames this chunk uniquely owns until the next one lands.

    Returns ``None`` when the source runs out of frames.
    """
    end = base_frame + chunk_size
    if end > len(gt):
        return None
    frames = gt[base_frame:end].copy()  # (chunk_size, n_joints)
    # Apply bias to overlap region only.
    frames[update_every:] += bias[None, :]
    return ActionChunk(
        fps=chunk_fps,
        frames=tuple(dict(zip(joint_names, row.tolist(), strict=True)) for row in frames),
    )


# ── Robot construction ─────────────────────────────────────────────────────


def build_robot(
    profile_path: Path, lookahead_ms_override: float | None
) -> tuple[BiSO107FollowerPredictive, dict[str, float] | None]:
    """Construct a predictive follower from a GUI profile JSON.

    Returns (robot, rest_position) where rest_position is the optional
    ``rest_position`` dict from the profile (None if absent).

    Always forces ``adaptive=False`` and pins ``max_lookahead_ms`` to
    ``lookahead_ms`` so the effective L stays fixed at exactly the value
    we configured — adaptive xcorr drifting during the run breaks the
    "motor command at L matches what was sent" assumption and
    contaminates the lag measurement.
    """
    profile = json.loads(profile_path.read_text())
    fields = dict(profile.get("fields", {}))
    if lookahead_ms_override is not None:
        fields["lookahead_ms"] = lookahead_ms_override
    fields["adaptive"] = False
    # Pin the cap to the configured L. Even though adaptive is off, this
    # makes it impossible for any code path to grow L beyond what we
    # asked for.
    fields["max_lookahead_ms"] = fields.get("lookahead_ms", 0.0)
    # Cameras are explicitly skipped — this script doesn't need image obs and
    # opening cameras would just slow the loop and risk holding /dev/video* FDs.
    fields["cameras"] = {}
    # JSON gives strings for Path-typed fields. The launcher (draccus) does
    # this conversion automatically; we're bypassing it so we coerce here.
    if "calibration_dir" in fields and fields["calibration_dir"] is not None:
        fields["calibration_dir"] = Path(fields["calibration_dir"]).expanduser()
    # Drop fields the dataclass doesn't accept (forward-compat).
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

    # Match the chunk fps to the source fps. If we wanted a different chunk
    # cadence, we'd interpolate the source here; keeping them equal removes
    # one knob from the experiment.
    chunk_fps = source_fps

    rng = np.random.default_rng(args.bias_seed)
    n_joints = gt.shape[1]
    bias = np.zeros(n_joints)
    bias_threshold = args.bias_threshold_deg
    bias_sigma = args.bias_sigma_deg

    robot, rest_position = build_robot(Path(args.robot_profile).expanduser(), args.lookahead_ms)
    # Robot's action_features dict gives us the joint name → order the robot
    # expects. Source joints are aligned to it below.
    robot.connect()
    try:
        # Always start from rest position so back-to-back runs are
        # initial-condition-identical (no thermal / friction history from a
        # previous run's end pose biasing the comparison).
        if rest_position is not None:
            logger.info("Moving to rest position before run")
            move_to_rest_position(robot, rest_position, duration_s=args.rest_duration_s)
        else:
            logger.warning(
                "Profile has no rest_position — back-to-back runs may have differing initial state"
            )
        robot_joint_names = list(robot.action_features.keys())
        # Permute source columns to match robot order if needed.
        if source_joints != robot_joint_names:
            permutation = [source_joints.index(j) for j in robot_joint_names]
            gt = gt[:, permutation]
            joint_names = robot_joint_names
            logger.info("Permuted source columns to match robot.action_features order")
        else:
            joint_names = robot_joint_names

        outer_dt = 1.0 / chunk_fps
        emission_period_frames = args.update_every  # one emission every N source frames

        # Time we ramp to the first frame, smoothly, so we don't whiplash
        # whatever pose the robot was holding at start. Done with a brief
        # send_action loop instead of the full safe_trajectory.replay ramp
        # to keep this script self-contained.
        logger.info("Ramping to trajectory start over %.1fs", args.ramp_to_start_s)
        start_pose = robot.get_observation()
        target_pose = {j: float(gt[0, joint_names.index(j)]) for j in joint_names}
        ramp_steps = max(1, int(args.ramp_to_start_s * chunk_fps))
        for step in range(1, ramp_steps + 1):
            alpha = step / ramp_steps
            action = {j: (1.0 - alpha) * start_pose[j] + alpha * target_pose[j] for j in joint_names}
            robot.send_action(action)
            time.sleep(outer_dt)

        # Run the chunk loop.
        ticks: list[dict] = []
        active_chunk_frames: np.ndarray | None = None  # (chunk_size, n_joints)
        active_chunk_t0: float | None = None
        last_emission_idx = -1

        run_start_t = time.perf_counter()
        max_emissions = (len(gt) - args.chunk_size) // emission_period_frames
        logger.info(
            "Running %d chunk emissions (update_every=%d, chunk_size=%d, lookahead=%.0fms)",
            max_emissions,
            args.update_every,
            args.chunk_size,
            args.lookahead_ms or 0.0,
        )

        while True:
            tick_start_t = time.perf_counter()
            playback_t = tick_start_t - run_start_t

            # Decide if a new emission is due. We emit a fresh chunk every
            # `emission_period_frames` of source time.
            current_emission_idx = int(playback_t / (emission_period_frames / chunk_fps))
            if current_emission_idx > last_emission_idx and current_emission_idx <= max_emissions:
                base_frame = current_emission_idx * emission_period_frames
                # Update bias: random walk capped at threshold (only the
                # overlap part of the chunk sees this).
                bias = np.clip(
                    bias + rng.normal(0.0, bias_sigma, size=n_joints),
                    -bias_threshold,
                    bias_threshold,
                )
                chunk = make_chunk(
                    gt,
                    base_frame,
                    args.chunk_size,
                    args.update_every,
                    bias,
                    joint_names,
                    chunk_fps,
                )
                if chunk is None:
                    break
                robot.send_action(chunk)
                # Cache the resolved (post-bias) frames for the action-to-state
                # log so we can reconstruct what was sent without re-running the
                # RNG in post-processing.
                active_chunk_frames = np.stack(
                    [np.asarray([f[j] for j in joint_names], dtype=np.float64) for f in chunk.frames]
                )
                active_chunk_t0 = tick_start_t
                last_emission_idx = current_emission_idx
            elif current_emission_idx > max_emissions:
                break

            # Read robot state.
            obs = robot.get_observation()
            state_vec = np.asarray([obs[j] for j in joint_names], dtype=np.float64)

            # Compute what the controller's read-position into the chunk is at
            # this tick. The robot's internal 200 Hz controller does its own
            # version of this; we record the same math so post-processing can
            # cross-reference state vs intended motor target.
            if active_chunk_frames is not None and active_chunk_t0 is not None:
                t_in_chunk = tick_start_t - active_chunk_t0
                read_idx_f = (t_in_chunk + (args.lookahead_ms or 0.0) / 1000.0) * chunk_fps
                lo = int(np.clip(np.floor(read_idx_f), 0, args.chunk_size - 1))
                hi = int(np.clip(np.ceil(read_idx_f), 0, args.chunk_size - 1))
                frac = read_idx_f - lo if hi > lo else 0.0
                motor_cmd_lookahead = (1.0 - frac) * active_chunk_frames[lo] + frac * active_chunk_frames[hi]
                # Same math at lookahead=0 for the no-lookahead reference.
                read_idx_f0 = t_in_chunk * chunk_fps
                lo0 = int(np.clip(np.floor(read_idx_f0), 0, args.chunk_size - 1))
                hi0 = int(np.clip(np.ceil(read_idx_f0), 0, args.chunk_size - 1))
                frac0 = read_idx_f0 - lo0 if hi0 > lo0 else 0.0
                motor_cmd_no_lookahead = (1.0 - frac0) * active_chunk_frames[
                    lo0
                ] + frac0 * active_chunk_frames[hi0]
            else:
                motor_cmd_lookahead = state_vec
                motor_cmd_no_lookahead = state_vec

            # Ground truth at this wall time (for cross-config comparison).
            gt_frame_idx_f = playback_t * chunk_fps
            gt_lo = int(np.clip(np.floor(gt_frame_idx_f), 0, len(gt) - 1))
            gt_hi = int(np.clip(np.ceil(gt_frame_idx_f), 0, len(gt) - 1))
            gt_frac = gt_frame_idx_f - gt_lo if gt_hi > gt_lo else 0.0
            gt_at_now = (1.0 - gt_frac) * gt[gt_lo] + gt_frac * gt[gt_hi]

            ticks.append(
                {
                    "t": playback_t,
                    "emission_idx": last_emission_idx,
                    "gt": gt_at_now,
                    "motor_cmd_lookahead": motor_cmd_lookahead,
                    "motor_cmd_no_lookahead": motor_cmd_no_lookahead,
                    "state": state_vec,
                    "bias": bias.copy(),
                }
            )

            elapsed = time.perf_counter() - tick_start_t
            if elapsed < outer_dt:
                time.sleep(outer_dt - elapsed)

        # Return to rest before disconnecting. Same reason as the start-of-
        # run move: leaves the robot in a clean state for the next run, and
        # makes the workspace safe to approach.
        if rest_position is not None:
            logger.info("Moving to rest position after run")
            move_to_rest_position(robot, rest_position, duration_s=args.rest_duration_s)
    finally:
        robot.disconnect()

    # Save traces.
    if not ticks:
        logger.warning("No ticks recorded — nothing to save")
        return

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"N{args.update_every}_L{int(args.lookahead_ms or 0)}_bias{args.bias_threshold_deg}_seed{args.bias_seed}"
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
        "source": args.source,
        "update_every": args.update_every,
        "chunk_size": args.chunk_size,
        "chunk_fps": chunk_fps,
        "lookahead_ms": args.lookahead_ms or 0.0,
        "bias_threshold_deg": args.bias_threshold_deg,
        "bias_sigma_deg": args.bias_sigma_deg,
        "bias_seed": args.bias_seed,
        "n_ticks": len(ticks),
        "n_emissions": last_emission_idx + 1,
        "duration_s": ticks[-1]["t"],
    }
    (out_dir / f"meta_{stem}.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s (%d ticks)", out_dir / f"trace_{stem}.npz", len(ticks))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source",
        required=True,
        help="Path to safe-trajectory JSON, or 'repo_id@episode_idx' for a dataset",
    )
    p.add_argument("--robot-profile", required=True, help="Path to GUI robot profile JSON")
    p.add_argument("--update-every", type=int, default=4, help="Frames between chunk emissions")
    p.add_argument("--chunk-size", type=int, default=50, help="Frames per chunk")
    p.add_argument(
        "--lookahead-ms",
        type=float,
        default=None,
        help="Override the profile's lookahead_ms. Pass 0 for the no-lookahead baseline.",
    )
    p.add_argument("--bias-threshold-deg", type=float, default=0.5)
    p.add_argument("--bias-sigma-deg", type=float, default=0.1)
    p.add_argument("--bias-seed", type=int, default=42, help="Same seed → identical chunk stream")
    p.add_argument("--ramp-to-start-s", type=float, default=2.0)
    p.add_argument(
        "--rest-duration-s",
        type=float,
        default=2.5,
        help="Time to spend moving to rest position (before AND after the run)",
    )
    p.add_argument("--output-dir", default="outputs/chunk_cadence")
    return p.parse_args()


if __name__ == "__main__":
    init_logging()
    run(parse_args())
