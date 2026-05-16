"""
Extract (joints, EE_pose, next_joints, next_EE_pose) training tuples from
existing lerobot bi_so107_follower datasets.

The schema produced is intentionally tiny (numpy arrays in a single .npz)
because the model is tiny too. Run once, train many times.

Training-pair construction:
    For each consecutive (frame_t, frame_{t+1}) within an episode and per arm:
        joints_t       (7 motor degrees)
        joints_{t+1}   (7 motor degrees)
        ee_t           (3 mm: FK tip in mm, base frame)
        ee_{t+1}       (3 mm)
    Frame pairs spanning episode boundaries are dropped.
    Both arms (left + right) yield separate samples — same kinematic chain
    so a single network can learn both.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.learned_ik.dataset_extractor \\
        --dataset thewisp/cylinder_ring_assembly_merged_raw \\
        --out /tmp/so107_ik_train.npz \\
        --max-episodes 100  # optional cap
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..kinematics import (
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    So107Kinematics,
)

# 14-dim state from bi_so107_follower datasets is (left 7 joints, right 7 joints)
# in degrees, ordered: shoulder_pan, shoulder_lift, elbow_flex, forearm_roll,
# wrist_flex, wrist_roll, gripper.
LEFT_INDICES = list(range(0, 7))
RIGHT_INDICES = list(range(7, 14))


def extract(
    dataset_repo_id: str,
    out_path: Path,
    max_episodes: int | None = None,
    fps_subsample: int = 1,
    include_arms: tuple[str, ...] = ("left", "right"),
    longest_n: int | None = None,
    skip_longest_n: int = 0,
) -> None:
    """Load dataset; iterate frames; emit (motor7_t, ee3_t, motor7_t+1, ee3_t+1) pairs."""
    print(f"Loading {dataset_repo_id} ...")
    ds = LeRobotDataset(dataset_repo_id)
    n_total_ep = ds.num_episodes
    print(f"  fps={ds.fps}, episodes={n_total_ep}, total frames={ds.num_frames}")

    # Use the underlying HF dataset with no image columns — image decoding is
    # extremely slow and we only need motor state for IK training.
    hf = ds.hf_dataset.select_columns(["observation.state", "episode_index"])
    # Group rows by episode using a single pass.
    print("  scanning parquet for episode boundaries...")
    all_states = np.array(hf["observation.state"], dtype=np.float32)  # (N, 14)
    all_episodes = np.array(hf["episode_index"], dtype=np.int64)  # (N,)
    print(f"  loaded {len(all_states)} frames in memory")

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)  # for FK
    motor_names = list(MOTOR_NAMES)
    motor_names_no_grip = motor_names  # keep all 7 for input; gripper is also a motor

    in_joints: list[np.ndarray] = []
    out_joints: list[np.ndarray] = []
    in_ee: list[np.ndarray] = []
    out_ee: list[np.ndarray] = []
    arms_label: list[int] = []

    t0 = time.monotonic()

    # Precompute episode boundaries: indices where episode_index changes.
    boundaries = np.where(np.diff(all_episodes) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(all_episodes)]])
    lengths = ends - starts

    # Select which episodes to process based on longest_n / skip_longest_n.
    # Sort episode indices by length, descending.
    order_by_len_desc = np.argsort(-lengths)
    if longest_n is not None:
        selected = sorted(order_by_len_desc[:longest_n].tolist())
        print(
            f"  selecting {len(selected)} longest episodes "
            f"(lengths: {sorted(lengths[selected].tolist(), reverse=True)[:10]}{'...' if len(selected) > 10 else ''})"
        )
    elif skip_longest_n > 0:
        skipped = set(order_by_len_desc[:skip_longest_n].tolist())
        selected = [i for i in range(n_total_ep) if i not in skipped]
        print(f"  skipping {len(skipped)} longest episodes; using {len(selected)} others")
    else:
        selected = list(range(n_total_ep))
    if max_episodes is not None:
        selected = selected[:max_episodes]

    for ep_idx in selected:
        f_start = int(starts[ep_idx])
        f_end = int(ends[ep_idx])
        frame_obs = all_states[f_start:f_end:fps_subsample].astype(np.float64)
        if len(frame_obs) < 2:
            continue

        # Compute FK per arm per frame.
        for arm in include_arms:
            idx = LEFT_INDICES if arm == "left" else RIGHT_INDICES
            arm_joints = frame_obs[:, idx]  # (T, 7) in degrees
            ee_xyz = np.zeros((len(frame_obs), 3))
            for ti, q in enumerate(arm_joints):
                motor_dict = {n: float(q[i]) for i, n in enumerate(motor_names_no_grip)}
                T = kin.fk_from_motors(motor_dict)
                ee_xyz[ti] = T[:3, 3]

            # Build (t, t+1) pairs.
            in_joints.append(arm_joints[:-1])
            out_joints.append(arm_joints[1:])
            in_ee.append(ee_xyz[:-1])
            out_ee.append(ee_xyz[1:])
            arms_label.extend([0 if arm == "left" else 1] * (len(arm_joints) - 1))

        if len(in_joints) % 20 == 0 or ep_idx == selected[-1]:
            elapsed = time.monotonic() - t0
            n_pairs_so_far = sum(a.shape[0] for a in in_joints)
            print(
                f"  processed {len(in_joints) // (2 if len(include_arms) == 2 else 1)}/{len(selected)}  "
                f"pairs={n_pairs_so_far}  elapsed={elapsed:.1f}s"
            )

    in_joints_arr = np.concatenate(in_joints, axis=0).astype(np.float32)
    out_joints_arr = np.concatenate(out_joints, axis=0).astype(np.float32)
    in_ee_arr = np.concatenate(in_ee, axis=0).astype(np.float32)
    out_ee_arr = np.concatenate(out_ee, axis=0).astype(np.float32)
    arms_arr = np.array(arms_label, dtype=np.int8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        in_joints=in_joints_arr,  # (N, 7) motor degrees at time t
        out_joints=out_joints_arr,  # (N, 7) motor degrees at t+1
        in_ee=in_ee_arr,  # (N, 3) FK xyz (meters) at t
        out_ee=out_ee_arr,  # (N, 3) FK xyz at t+1
        arm=arms_arr,  # 0=left, 1=right
        motor_names=np.array(motor_names_no_grip),
    )
    print(f"\nSaved {in_joints_arr.shape[0]} training pairs to {out_path}")
    print(
        f"  in_joints: {in_joints_arr.shape}, range [{in_joints_arr.min():.1f}, {in_joints_arr.max():.1f}]°"
    )
    print(
        f"  in_ee:     {in_ee_arr.shape}, x[{in_ee_arr[:, 0].min():+.3f},{in_ee_arr[:, 0].max():+.3f}] "
        f"y[{in_ee_arr[:, 1].min():+.3f},{in_ee_arr[:, 1].max():+.3f}] "
        f"z[{in_ee_arr[:, 2].min():+.3f},{in_ee_arr[:, 2].max():+.3f}] m"
    )
    # Per-step motion stats
    dq = out_joints_arr - in_joints_arr
    dee = out_ee_arr - in_ee_arr
    print(f"  joint step: |dq|max={np.abs(dq).max():.2f}°, median={np.median(np.abs(dq)):.3f}°")
    print(
        f"  ee step:    |dee|max={np.abs(dee).max() * 1000:.1f}mm, median={np.median(np.linalg.norm(dee, axis=1)) * 1000:.2f}mm"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="thewisp/cylinder_ring_assembly_merged_raw",
        help="LeRobotDataset repo_id (must be cached locally OR will be downloaded)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "so107_ik_train.npz",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="cap episode count for quick iterations"
    )
    parser.add_argument(
        "--longest-n",
        type=int,
        default=None,
        help="extract only the N longest episodes (useful for eval — typical complete trajectories)",
    )
    parser.add_argument(
        "--skip-longest-n",
        type=int,
        default=0,
        help="exclude the N longest episodes (use for train data when those are reserved for eval)",
    )
    parser.add_argument(
        "--fps-subsample", type=int, default=1, help="take every Nth frame within episodes (1 = all)"
    )
    parser.add_argument(
        "--arms",
        default="left,right",
        help="comma-separated: which arms to extract (left,right or just right)",
    )
    args = parser.parse_args()
    arms = tuple(a.strip() for a in args.arms.split(","))
    extract(
        args.dataset,
        args.out,
        args.max_episodes,
        args.fps_subsample,
        arms,
        longest_n=args.longest_n,
        skip_longest_n=args.skip_longest_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
