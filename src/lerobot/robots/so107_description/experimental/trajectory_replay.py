"""
Offline IK evaluation and optional physical replay using a recorded teleop episode.

Given a recorded lerobot episode of human teleop:
    For each frame:
        target_ee_t = FK(human_joints_t)           # the EE trajectory the human created
        predicted_joints_t = IK(prev_joints, target_T_t)
    Compare predicted joints vs human joints (joint-MAE), and predicted EE
    accuracy via FK round-trip. Optionally replay predicted joints on the
    physical arm and compare physical motion to the human's recording.

IK methods:
    dls    - DLS minimum-motion Jacobian
    nn     - pure NN inference
    nn_dls - NN + DLS refinement (recommended default)

Usage (offline / dry-run):
    .venv/bin/python -m lerobot.robots.so107_description.experimental.trajectory_replay \\
        --dataset thewisp/cylinder_ring_assembly_merged_raw \\
        --longest --arm right \\
        --ik-method nn_dls --model /tmp/so107_ik_model_big.pt \\
        --dry-run

Usage (physical replay):
    .venv/bin/python -m lerobot.robots.so107_description.experimental.trajectory_replay \\
        --dataset thewisp/cylinder_ring_assembly_merged_raw \\
        --longest --arm right \\
        --ik-method nn_dls --model /tmp/so107_ik_model_big.pt \\
        --port /dev/ttyACM2 --id right_white \\
        --rate-hz 30
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pinocchio as pin

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from . import get_urdf_path  # noqa: E402
from .kinematics import (  # noqa: E402
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    So107Kinematics,
    motor_pos_to_urdf_q,
    urdf_q_to_motor_pos,
)

LEFT_INDICES = list(range(0, 7))
RIGHT_INDICES = list(range(7, 14))


def dls_step(model_pin, data_pin, frame_id, motor_pos, target_ee, damping=0.05):
    """One DLS Newton step toward target_ee from current motors."""
    q = motor_pos_to_urdf_q(motor_pos, RIGHT_ARM_MAP)
    pin.computeJointJacobians(model_pin, data_pin, q)
    pin.updateFramePlacements(model_pin, data_pin)
    J = pin.getFrameJacobian(model_pin, data_pin, frame_id, pin.LOCAL_WORLD_ALIGNED)
    Jp = J[:3, :]
    err = target_ee - data_pin.oMf[frame_id].translation
    A = Jp @ Jp.T + (damping**2) * np.eye(3)
    q_delta = Jp.T @ np.linalg.solve(A, err)
    return urdf_q_to_motor_pos(q + q_delta, RIGHT_ARM_MAP)


def load_episode(
    dataset_id: str,
    episode_idx: int | None,
    longest: bool,
    arm: str,
    source_column: str = "observation.state",
) -> np.ndarray:
    """Return (T, 7) joint trajectory in degrees for one arm of one episode.

    source_column:
        "observation.state" — what the follower's motors actually reached
                              (after gravity sag).
        "action"            — what the leader was commanding (anti-sag aware).
                              For ground-truth replay use "action".
    """
    print(f"Loading {dataset_id} ... (source: {source_column})")
    ds = LeRobotDataset(dataset_id)
    hf = ds.hf_dataset.select_columns([source_column, "episode_index"])
    all_states = np.array(hf[source_column], dtype=np.float32)
    all_episodes = np.array(hf["episode_index"], dtype=np.int64)
    boundaries = np.where(np.diff(all_episodes) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(all_episodes)]])
    lengths = ends - starts

    if longest:
        ep_idx = int(np.argmax(lengths))
        print(f"  selected longest episode {ep_idx} ({lengths[ep_idx]} frames)")
    else:
        ep_idx = episode_idx or 0
        print(f"  selected episode {ep_idx} ({lengths[ep_idx]} frames)")

    s, e = int(starts[ep_idx]), int(ends[ep_idx])
    states = all_states[s:e]
    idx = LEFT_INDICES if arm == "left" else RIGHT_INDICES
    return states[:, idx].astype(np.float64)  # (T, 7) degrees


def make_ik_step_fn(
    method: str,
    model_path: Path | None,
    model_pin,
    data_pin,
    frame_id,
    ground_truth_joints: np.ndarray | None = None,
):
    """Return a function (motor_pos_dict, target_T_4x4, frame_idx) -> new_motor_pos_dict.

    For ground_truth method, returns the recorded human_joints[frame_idx] directly.
    """
    if method == "ground_truth":
        if ground_truth_joints is None:
            raise ValueError("--ik-method ground_truth requires human_joints array")

        def step_fn(mp, target_T, frame_idx=0):  # noqa: N803  (target_T is SE(3))
            return {n: float(ground_truth_joints[frame_idx, i]) for i, n in enumerate(MOTOR_NAMES)}

        return step_fn

    if method == "dls":

        def step_fn(mp, target_T, frame_idx=0):  # noqa: N803  (target_T is SE(3))
            return dls_step(model_pin, data_pin, frame_id, mp, target_T[:3, 3])

        return step_fn

    if model_path is None:
        raise ValueError(f"--model is required for ik_method={method}")
    from .learned_ik.kinematics_nn import So107NNKinematics

    refine = method == "nn_dls"
    kin_nn = So107NNKinematics(model_path=model_path, refine_with_dls=refine)

    def step_fn(mp, target_T, frame_idx=0):  # noqa: N803  (target_T is SE(3))
        new_mp, _ = kin_nn.ik_to_motors(mp, target_T)
        return new_mp

    return step_fn


def offline_analysis(
    human_joints: np.ndarray,
    ik_step,
    kin: So107Kinematics,
    mode: str = "teacher_forced",
) -> dict:
    """Run IK across the human trajectory. Two modes:

    teacher_forced: each tick's IK uses HUMAN previous joints as seed (mimics
        physical replay where we read real motor positions each tick).
    open_loop: chains predicted joints (errors compound; bad eval methodology
        for replay, kept only for comparison).
    """
    T = len(human_joints)
    # Full FK trajectory: position AND rotation.
    target_T_seq = np.zeros((T, 4, 4))
    target_ee = np.zeros((T, 3))
    for t in range(T):
        mp = {n: float(human_joints[t, i]) for i, n in enumerate(MOTOR_NAMES)}
        target_T_seq[t] = kin.fk_from_motors(mp)
        target_ee[t] = target_T_seq[t][:3, 3]

    predicted = np.zeros_like(human_joints)
    predicted[0] = human_joints[0]
    for t in range(1, T):
        seed = human_joints[t - 1] if mode == "teacher_forced" else predicted[t - 1]
        prev_mp = {n: float(seed[i]) for i, n in enumerate(MOTOR_NAMES)}
        new_mp = ik_step(prev_mp, target_T_seq[t], t)
        for i, n in enumerate(MOTOR_NAMES):
            predicted[t, i] = new_mp[n]

    achieved_ee = np.zeros((T, 3))
    for t in range(T):
        mp = {n: float(predicted[t, i]) for i, n in enumerate(MOTOR_NAMES)}
        achieved_ee[t] = kin.fk_from_motors(mp)[:3, 3]

    joint_err = predicted - human_joints
    ee_err = achieved_ee - target_ee
    return {
        "joint_mae_per_joint": np.abs(joint_err).mean(axis=0),
        "joint_mae_overall": float(np.abs(joint_err).mean()),
        "joint_max_drift_per_joint": np.abs(joint_err).max(axis=0),
        "ee_err_mm": np.linalg.norm(ee_err, axis=1) * 1000,
        "human_joints": human_joints,
        "predicted_joints": predicted,
        "target_ee": target_ee,
        "achieved_ee": achieved_ee,
        "mode": mode,
    }


def physical_replay(
    human_joints: np.ndarray,
    ik_step,
    port: str,
    robot_id: str,
    rate_hz: float,
    max_relative_target: float,
    log_path: Path,
) -> None:
    """ONLINE replay: each tick reads real motors, computes IK against the
    human's recorded EE pose for that frame, sends result. This avoids the
    compounding-error problem of pre-computed offline trajectories — the IK
    is always seeded from the actual physical state."""
    from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

    config = SO107FollowerConfig(
        port=port,
        id=robot_id,
        use_degrees=True,
        cameras={},
        max_relative_target=max_relative_target,
    )
    robot = SO107Follower(config)
    robot.connect(calibrate=False)
    print(f"\nConnected to {robot}.")

    joint_limits = {}
    for name, c in robot.bus.calibration.items():
        mid = (c.range_min + c.range_max) / 2
        deg_min = (c.range_min - mid) * 360.0 / 4095
        deg_max = (c.range_max - mid) * 360.0 / 4095
        joint_limits[name] = (deg_min + 1.0, deg_max - 1.0)

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)

    # Pre-compute target_T sequence (FK on the human's joints).
    n = len(human_joints)
    target_T_seq = np.zeros((n, 4, 4))
    for t in range(n):
        mp = {nm: float(human_joints[t, i]) for i, nm in enumerate(MOTOR_NAMES)}
        target_T_seq[t] = kin.fk_from_motors(mp)

    # Move arm to the human's starting pose safely before replay.
    print("Moving to start pose (human's frame 0 joints) ...")
    start_q = {n2: float(human_joints[0, i]) for i, n2 in enumerate(MOTOR_NAMES)}
    period = 1.0 / rate_hz
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        obs = robot.bus.sync_read("Present_Position")
        mp = {n2: float(obs[n2]) for n2 in MOTOR_NAMES}
        gap = max(abs(start_q[n2] - mp[n2]) for n2 in MOTOR_NAMES if n2 != "gripper")
        if gap < 1.0:
            break
        action = {f"{n2}.pos": start_q[n2] for n2 in MOTOR_NAMES}
        robot.send_action(action)
        time.sleep(period)
    print("  reached start (or 10s timeout)")
    input("Press Enter to begin replay (Ctrl-C to abort during)...")

    log_f = log_path.open("w", newline="")
    w = csv.writer(log_f)
    w.writerow(
        [
            "t",
            "frame",
            *(f"motor_{n2}" for n2 in MOTOR_NAMES),
            *(f"cmd_{n2}" for n2 in MOTOR_NAMES),
            *(f"human_{n2}" for n2 in MOTOR_NAMES),
            "ee_cur_x",
            "ee_cur_y",
            "ee_cur_z",
            "ee_tgt_x",
            "ee_tgt_y",
            "ee_tgt_z",
        ]
    )

    t_start = time.monotonic()
    try:
        for t in range(n):
            tick_t0 = time.monotonic()
            # 1. Read real motors (this is the IK seed — analogous to teacher-forced).
            obs = robot.bus.sync_read("Present_Position")
            mp = {nm: float(obs[nm]) for nm in MOTOR_NAMES}
            # 2. Run IK targeting the human's full EE pose for this frame.
            ik_motors = ik_step(mp, target_T_seq[t], t)
            # 3. Joint-limit clamp + send.
            new_q = {}
            for nm in MOTOR_NAMES:
                v = ik_motors[nm]
                lo, hi = joint_limits[nm]
                if v > hi:
                    v = hi
                elif v < lo:
                    v = lo
                new_q[nm] = v
            robot.send_action({f"{nm}.pos": new_q[nm] for nm in MOTOR_NAMES})

            cur_T = kin.fk_from_motors(mp)
            w.writerow(
                [
                    f"{time.monotonic() - t_start:.3f}",
                    t,
                    *(f"{mp[nm]:.3f}" for nm in MOTOR_NAMES),
                    *(f"{new_q[nm]:.3f}" for nm in MOTOR_NAMES),
                    *(f"{human_joints[t, i]:.3f}" for i, nm in enumerate(MOTOR_NAMES)),
                    f"{cur_T[0, 3]:.4f}",
                    f"{cur_T[1, 3]:.4f}",
                    f"{cur_T[2, 3]:.4f}",
                    f"{target_T_seq[t, 0, 3]:.4f}",
                    f"{target_T_seq[t, 1, 3]:.4f}",
                    f"{target_T_seq[t, 2, 3]:.4f}",
                ]
            )

            elapsed = time.monotonic() - tick_t0
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        log_f.close()
        robot.disconnect()
        print(f"\nReplay log: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="thewisp/cylinder_ring_assembly_merged_raw")
    parser.add_argument("--ik-method", choices=["dls", "nn", "nn_dls", "ground_truth"], default="nn_dls")
    parser.add_argument("--model", type=Path, default=None, help="required for nn / nn_dls")
    parser.add_argument("--dry-run", action="store_true", help="offline analysis only, no physical replay")
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=None,
        help="dataset episode to use. Default = longest episode.",
    )
    args = parser.parse_args()

    if args.ik_method in ("nn", "nn_dls") and args.model is None:
        parser.error("--model required when ik_method is nn or nn_dls")

    # Locked-in defaults: right arm, action source, 30 Hz, mrt=30, id=white_right.
    human_joints = load_episode(
        args.dataset,
        episode_idx=args.episode_idx,
        longest=(args.episode_idx is None),
        arm="right",
        source_column="action",
    )
    print(f"  trajectory: {len(human_joints)} frames @ 30 fps -> {len(human_joints) / 30:.1f}s")

    # FK and IK
    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)
    model_pin = pin.buildModelFromUrdf(str(get_urdf_path()))
    data_pin = model_pin.createData()
    frame_id = model_pin.getFrameId("L7_1")
    ik_step = make_ik_step_fn(
        args.ik_method, args.model, model_pin, data_pin, frame_id, ground_truth_joints=human_joints
    )

    print(f"Running offline analysis: ik_method={args.ik_method}")
    t0 = time.monotonic()
    res = offline_analysis(human_joints, ik_step, kin)
    print(f"  done in {time.monotonic() - t0:.1f}s\n")

    print(f"=== Offline IK Evaluation ({args.ik_method}) ===")
    print(f"Trajectory: {len(human_joints)} frames\n")
    print("[1] Joint prediction accuracy (per-joint MAE deg):")
    for i, n in enumerate(MOTOR_NAMES):
        print(
            f"    {n:14s}: MAE {res['joint_mae_per_joint'][i]:6.2f}°  max drift {res['joint_max_drift_per_joint'][i]:6.2f}°"
        )
    print(f"    overall MAE: {res['joint_mae_overall']:.3f}°")
    print("\n[2] EE accuracy (predicted joints -> FK -> vs human EE):")
    err = res["ee_err_mm"]
    print(f"    median: {np.median(err):.2f}mm  p95: {np.percentile(err, 95):.2f}mm  max: {err.max():.2f}mm")

    # Drift tracking — does predicted trajectory diverge over time?
    drift_first_quarter = np.median(err[: len(err) // 4])
    drift_last_quarter = np.median(err[-len(err) // 4 :])
    print("\n[3] EE drift over time:")
    print(f"    median first 25%: {drift_first_quarter:.2f}mm")
    print(f"    median last  25%: {drift_last_quarter:.2f}mm")
    if drift_last_quarter > 2 * drift_first_quarter:
        print("    ⚠ DIVERGING — errors compound across the trajectory")
    elif drift_last_quarter < 1.5 * drift_first_quarter:
        print("    ✓ stable across trajectory")

    # Save dry-run results
    out_dir = Path(tempfile.gettempdir())
    dry_path = out_dir / f"so107_replay_{args.ik_method}_{dt.datetime.now():%H%M%S}_dry.csv"
    with dry_path.open("w", newline="") as f:
        ww = csv.writer(f)
        ww.writerow(
            [
                "t",
                *(f"human_{n}" for n in MOTOR_NAMES),
                *(f"pred_{n}" for n in MOTOR_NAMES),
                "tgt_x",
                "tgt_y",
                "tgt_z",
                "ach_x",
                "ach_y",
                "ach_z",
                "ee_err_mm",
            ]
        )
        for t in range(len(human_joints)):
            ww.writerow(
                [
                    t,
                    *(f"{res['human_joints'][t, i]:.3f}" for i in range(7)),
                    *(f"{res['predicted_joints'][t, i]:.3f}" for i in range(7)),
                    *(f"{res['target_ee'][t, i]:.4f}" for i in range(3)),
                    *(f"{res['achieved_ee'][t, i]:.4f}" for i in range(3)),
                    f"{err[t]:.3f}",
                ]
            )
    print(f"\nDry-run CSV: {dry_path}")

    if args.dry_run:
        return 0

    # Physical replay
    replay_log = out_dir / f"so107_replay_{args.ik_method}_{dt.datetime.now():%H%M%S}_real.csv"
    print("\n*** PHYSICAL REPLAY MODE ***")
    print(f"Will replay {len(human_joints)} predicted joint commands at 30 Hz")
    print(f"({len(human_joints) / 30:.1f}s of motion)")
    input("Press Enter to confirm and connect to the robot (Ctrl-C to cancel)...")
    physical_replay(
        human_joints,
        ik_step,
        port=args.port,
        robot_id="white_right",
        rate_hz=30.0,
        max_relative_target=30.0,
        log_path=replay_log,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
