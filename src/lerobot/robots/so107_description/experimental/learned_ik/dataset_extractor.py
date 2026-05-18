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
    .venv/bin/python -m lerobot.robots.so107_description.experimental.learned_ik.dataset_extractor \\
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

from ...kinematics import (
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    So107Kinematics,
)

# 14-dim state from bi_so107_follower datasets is (left 7 joints, right 7 joints)
# in degrees, ordered: shoulder_pan, shoulder_lift, elbow_flex, forearm_roll,
# wrist_flex, wrist_roll, gripper.
LEFT_INDICES = list(range(0, 7))
RIGHT_INDICES = list(range(7, 14))


def extract(dataset_repo_id: str, out_path: Path) -> None:
    """Load dataset; iterate frames; emit (motor7_t, ee3_t, motor7_t+1, ee3_t+1) pairs
    for both arms. Source column is `action` (commanded joints) — `observation.state`
    lags by motor tracking error and is the wrong supervision signal for IK."""
    print(f"Loading {dataset_repo_id} ...")
    ds = LeRobotDataset(dataset_repo_id)
    print(f"  fps={ds.fps}, episodes={ds.num_episodes}, total frames={ds.num_frames}")

    hf = ds.hf_dataset.select_columns(["action", "episode_index"])
    all_states = np.array(hf["action"], dtype=np.float32)  # (N, 14)
    all_episodes = np.array(hf["episode_index"], dtype=np.int64)  # (N,)
    print(f"  loaded {len(all_states)} frames in memory")

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)  # for FK
    motor_names = list(MOTOR_NAMES)
    motor_names_no_grip = motor_names  # keep all 7 for input; gripper is also a motor

    in_joints: list[np.ndarray] = []
    out_joints: list[np.ndarray] = []
    in_ee: list[np.ndarray] = []
    out_ee: list[np.ndarray] = []
    in_rot: list[np.ndarray] = []
    out_rot: list[np.ndarray] = []
    arms_label: list[int] = []

    t0 = time.monotonic()

    boundaries = np.where(np.diff(all_episodes) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(all_episodes)]])

    for ep_idx in range(ds.num_episodes):
        f_start = int(starts[ep_idx])
        f_end = int(ends[ep_idx])
        frame_obs = all_states[f_start:f_end].astype(np.float64)
        if len(frame_obs) < 2:
            continue

        for arm in ("left", "right"):
            idx = LEFT_INDICES if arm == "left" else RIGHT_INDICES
            arm_joints = frame_obs[:, idx]
            ee_xyz = np.zeros((len(frame_obs), 3))
            ee_rot = np.zeros((len(frame_obs), 9))
            for ti, q in enumerate(arm_joints):
                motor_dict = {n: float(q[i]) for i, n in enumerate(motor_names_no_grip)}
                T = kin.fk_from_motors(motor_dict)
                ee_xyz[ti] = T[:3, 3]
                ee_rot[ti] = T[:3, :3].flatten()

            in_joints.append(arm_joints[:-1])
            out_joints.append(arm_joints[1:])
            in_ee.append(ee_xyz[:-1])
            out_ee.append(ee_xyz[1:])
            in_rot.append(ee_rot[:-1])
            out_rot.append(ee_rot[1:])
            arms_label.extend([0 if arm == "left" else 1] * (len(arm_joints) - 1))

        if ep_idx % 50 == 0 or ep_idx == ds.num_episodes - 1:
            n_pairs_so_far = sum(a.shape[0] for a in in_joints)
            print(
                f"  ep {ep_idx + 1}/{ds.num_episodes}  pairs={n_pairs_so_far}  elapsed={time.monotonic() - t0:.1f}s"
            )

    in_joints_arr = np.concatenate(in_joints, axis=0).astype(np.float32)
    out_joints_arr = np.concatenate(out_joints, axis=0).astype(np.float32)
    in_ee_arr = np.concatenate(in_ee, axis=0).astype(np.float32)
    out_ee_arr = np.concatenate(out_ee, axis=0).astype(np.float32)
    in_rot_arr = np.concatenate(in_rot, axis=0).astype(np.float32)
    out_rot_arr = np.concatenate(out_rot, axis=0).astype(np.float32)
    arms_arr = np.array(arms_label, dtype=np.int8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        in_joints=in_joints_arr,  # (N, 7) motor degrees at time t
        out_joints=out_joints_arr,  # (N, 7) motor degrees at t+1
        in_ee=in_ee_arr,  # (N, 3) FK xyz (meters) at t
        out_ee=out_ee_arr,  # (N, 3) FK xyz at t+1
        in_rot=in_rot_arr,  # (N, 9) flattened 3x3 rotation matrix at t
        out_rot=out_rot_arr,  # (N, 9) at t+1
        arm=arms_arr,
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
    parser.add_argument("--dataset", default="thewisp/cylinder_ring_assembly_merged_raw")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "so107_ik_train.npz")
    args = parser.parse_args()
    extract(args.dataset, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
