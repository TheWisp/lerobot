"""
Evaluate a trained learned-IK model on held-out data.

Reports four metrics:
    1. per-joint MAE in degrees — does the NN predict joints that match
       what the human actually did?
    2. EE accuracy without DLS — joint config -> FK -> compare to target.
       Captures "is the NN alone a functional IK?"
    3. EE accuracy with DLS refinement — what teleop actually uses.
    4. Posture distance — how close are the NN's predicted joints to the
       nearest training-set example? Out-of-distribution proxy.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.learned_ik.eval \\
        --model /tmp/so107_ik_model.pt \\
        --train-data /tmp/so107_ik_train.npz \\
        --eval-data /tmp/so107_ik_eval.npz   # different dataset for OOD test
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from ..kinematics import MOTOR_NAMES, RIGHT_ARM_MAP, So107Kinematics  # noqa: E402
from .kinematics_nn import So107NNKinematics  # noqa: E402
from .model import IKMLP, IKModelConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--eval-data", type=Path, required=True, help="held-out data (same format as dataset_extractor .npz)"
    )
    parser.add_argument(
        "--train-data", type=Path, default=None, help="optional: training data for posture-distance reference"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--n-samples", type=int, default=5000, help="random subsample of eval data (for speed)"
    )
    args = parser.parse_args()

    print(f"Loading eval data {args.eval_data}")
    z = np.load(args.eval_data)
    in_joints = z["in_joints"].astype(np.float32)
    out_joints = z["out_joints"].astype(np.float32)
    in_ee = z["in_ee"].astype(np.float32)
    out_ee = z["out_ee"].astype(np.float32)

    # Same filter as train.py: drop no-motion / large-jump samples.
    motion = np.linalg.norm(out_ee - in_ee, axis=1)
    djmax = np.abs(out_joints - in_joints).max(axis=1)
    mask = (motion > 1e-5) & (djmax < 30.0)
    in_joints = in_joints[mask]
    out_joints = out_joints[mask]
    in_ee = in_ee[mask]
    out_ee = out_ee[mask]
    n = len(in_joints)
    print(f"  {n} valid eval pairs")

    if n > args.n_samples:
        idx = np.random.RandomState(0).choice(n, args.n_samples, replace=False)
        in_joints = in_joints[idx]
        out_joints = out_joints[idx]
        in_ee = in_ee[idx]
        out_ee = out_ee[idx]
        n = args.n_samples
        print(f"  subsampled to {n}")

    # Load NN with DLS off for raw inference.
    print(f"Loading model {args.model}")
    ckpt = torch.load(args.model, map_location=args.device, weights_only=False)
    cfg = IKModelConfig(**ckpt["config"])
    raw_model = IKMLP(cfg).to(args.device)
    raw_model.load_state_dict(ckpt["state_dict"])
    raw_model.eval()

    # Batch inference for raw NN.
    ee_delta = out_ee - in_ee  # (N, 3)
    x = np.concatenate([in_joints, ee_delta], axis=1)
    with torch.no_grad():
        x_t = torch.from_numpy(x).to(args.device)
        pred_delta = raw_model(x_t).cpu().numpy()
    pred_joints_nn = in_joints + pred_delta

    # --- Metric 1: per-joint MAE (joint prediction vs ground truth) ---
    joint_err = pred_joints_nn - out_joints
    per_joint_mae = np.abs(joint_err).mean(axis=0)
    overall_joint_mae = np.abs(joint_err).mean()
    joint_err_norm = np.linalg.norm(joint_err, axis=1)
    print("\n[1] Joint-prediction accuracy (NN vs human's actual joints):")
    print(f"    per-joint MAE (deg): [{', '.join(f'{m:.2f}' for m in per_joint_mae)}]")
    print(
        f"    overall MAE: {overall_joint_mae:.3f}°  norm: median={np.median(joint_err_norm):.2f}° p95={np.percentile(joint_err_norm, 95):.2f}°"
    )

    # --- Metric 2: EE accuracy with NN alone (no DLS) ---
    fk = So107Kinematics(joint_map=RIGHT_ARM_MAP)
    achieved_ee_nn = np.zeros_like(out_ee)
    for i in range(n):
        mp = {nm: float(pred_joints_nn[i, j]) for j, nm in enumerate(MOTOR_NAMES)}
        T = fk.fk_from_motors(mp)
        achieved_ee_nn[i] = T[:3, 3]
    ee_err_nn = np.linalg.norm(achieved_ee_nn - out_ee, axis=1) * 1000  # mm
    print("\n[2] EE accuracy, NN-only (predicted joints -> FK -> vs target EE):")
    print(
        f"    median: {np.median(ee_err_nn):.2f}mm  p95: {np.percentile(ee_err_nn, 95):.2f}mm  max: {ee_err_nn.max():.2f}mm"
    )

    # --- Metric 3: EE accuracy with DLS refinement ---
    nn_with_dls = So107NNKinematics(
        model_path=args.model,
        joint_map=RIGHT_ARM_MAP,
        device=args.device,
        refine_with_dls=True,
    )
    ee_err_dls = np.zeros(n)
    for i in range(n):
        current_mp = {nm: float(in_joints[i, j]) for j, nm in enumerate(MOTOR_NAMES)}
        target_T = np.eye(4)
        target_T[:3, 3] = out_ee[i]
        _, err_mm = nn_with_dls.ik_to_motors(current_mp, target_T)
        ee_err_dls[i] = err_mm
    print("\n[3] EE accuracy, NN + 2-step DLS refinement (what teleop uses):")
    print(
        f"    median: {np.median(ee_err_dls):.2f}mm  p95: {np.percentile(ee_err_dls, 95):.2f}mm  max: {ee_err_dls.max():.2f}mm"
    )

    # --- Metric 4: Posture distance — how close are NN outputs to training set? ---
    if args.train_data is not None:
        print(f"\n[4] Loading training data for posture-distance check: {args.train_data}")
        z_train = np.load(args.train_data)
        train_joints = z_train["out_joints"].astype(np.float32)
        train_ee = z_train["out_ee"].astype(np.float32)
        # For each eval prediction, find nearest training point by EE, measure joint distance.
        print(f"    {len(train_joints)} training poses available")
        rng = np.random.RandomState(0)
        sample_idx = rng.choice(n, min(500, n), replace=False)
        nearest_joint_dist = []
        for i in sample_idx:
            d2 = np.sum((train_ee - achieved_ee_nn[i]) ** 2, axis=1)
            j = np.argmin(d2)
            jd = np.linalg.norm(train_joints[j] - pred_joints_nn[i])
            nearest_joint_dist.append(jd)
        njd = np.array(nearest_joint_dist)
        print("    NN-predicted joints vs nearest training joints (by EE):")
        print(f"      median={np.median(njd):.2f}°  p95={np.percentile(njd, 95):.2f}°  max={njd.max():.2f}°")
        print("    Low = NN picks poses similar to humans at the same EE.")
        print("    High = NN extrapolating into regions humans didn't visit.")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("  Joint MAE <1° = NN matches human's exact choice well")
    print("  EE err (NN-only) <5mm = NN alone is functional as IK")
    print("  EE err (with DLS) <0.5mm = teleop is geometrically accurate")
    print("  Posture distance — informative, no fixed threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
