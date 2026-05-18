"""
Train the SO-107 learned-IK MLP.

Reads the .npz produced by dataset_extractor.py; constructs
    x = [joints_t (7), ee_delta = ee_{t+1} - ee_t (3)]
    y = delta_joints = joints_{t+1} - joints_t
Both quantities are normalized using their own mean/std computed on the
training split.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.experimental.learned_ik.train \\
        --data /tmp/so107_ik_train.npz \\
        --out /tmp/so107_ik_model.pt
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import IKMLP, IKModelConfig


def load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, Y) where X is the full-pose input (22-D) and Y is joint delta (7-D)."""
    z = np.load(path)
    in_joints = z["in_joints"].astype(np.float32)
    out_joints = z["out_joints"].astype(np.float32)
    in_ee = z["in_ee"].astype(np.float32)
    out_ee = z["out_ee"].astype(np.float32)
    has_rot = "in_rot" in z.files
    if has_rot:
        in_rot = z["in_rot"].astype(np.float32)
        out_rot = z["out_rot"].astype(np.float32)
    else:
        # Backwards compat: position-only dataset. Synthesize identity rotation.
        in_rot = np.tile(np.eye(3, dtype=np.float32).flatten(), (len(in_joints), 1))
        out_rot = in_rot.copy()
        print("WARNING: dataset is position-only; rotations set to identity (will train as 22-D anyway)")

    ee_delta = out_ee - in_ee  # (N, 3)
    delta_joints = out_joints - in_joints  # (N, 7)

    # Rotation delta in axis-angle form (computed once in numpy to keep training simple).
    in_R = in_rot.reshape(-1, 3, 3)
    out_R = out_rot.reshape(-1, 3, 3)
    R_delta = np.einsum("nij,nkj->nik", out_R, in_R)  # out * in.T

    def rotmat_to_axis_angle(R):  # noqa: N803  (R is the rotation matrix)
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos = np.clip((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
        theta = np.arccos(cos)
        sin = np.where(np.abs(theta) < 1e-6, 1.0, np.sin(theta))
        axis = np.stack(
            [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=-1
        ) / (2 * sin[:, None])
        return axis * theta[:, None]

    rot_delta_aa = rotmat_to_axis_angle(R_delta)  # (N, 3) radians axis-angle

    # Filter: keep motion AND not huge joint jumps. Also keep some no-motion
    # pairs as "hold" examples (1-in-10 sampled) so model learns to predict
    # near-zero deltas when commanded zero motion. Otherwise it might never
    # see "stay still" in training.
    motion = np.linalg.norm(ee_delta, axis=1)
    rot_motion = np.linalg.norm(rot_delta_aa, axis=1)
    djmax = np.abs(delta_joints).max(axis=1)
    moving = ((motion > 1e-5) | (rot_motion > 1e-3)) & (djmax < 30.0)
    rng = np.random.RandomState(0)
    keep_idle = (~moving) & (rng.rand(len(moving)) < 0.1) & (djmax < 5.0)
    mask = moving | keep_idle
    print(
        f"Filtering: kept {mask.sum()}/{len(mask)} pairs ({moving.sum()} moving + {keep_idle.sum()} idle samples)"
    )

    X = np.concatenate(
        [
            in_joints[mask],  # 7
            in_rot[mask],  # 9
            ee_delta[mask],  # 3
            rot_delta_aa[mask],  # 3
        ],
        axis=1,
    )  # (N, 22)
    Y = delta_joints[mask]  # (N, 7)
    return X.astype(np.float32), Y.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="path to dataset_extractor .npz")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "so107_ik_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", default="128,128", help="comma-separated hidden layer widths")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.data}")
    X, Y = load_pairs(args.data)
    n = len(X)
    print(f"  X: {X.shape}  Y: {Y.shape}")

    perm = np.random.permutation(n)
    n_val = int(n * args.val_split)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xva, Yva = X[val_idx], Y[val_idx]
    print(f"  train: {len(Xtr)}, val: {len(Xva)}")

    # Per-feature mean/std (computed on train split only).
    in_mean = torch.tensor(Xtr.mean(0))
    in_std = torch.tensor(Xtr.std(0))
    out_mean = torch.tensor(Ytr.mean(0))
    out_std = torch.tensor(Ytr.std(0))

    hidden = tuple(int(h) for h in args.hidden.split(","))
    cfg = IKModelConfig(in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=hidden)
    model = IKMLP(cfg).to(args.device)
    model.set_normalization(
        in_mean.to(args.device),
        in_std.to(args.device),
        out_mean.to(args.device),
        out_std.to(args.device),
    )

    Xtr_t = torch.from_numpy(Xtr).to(args.device)
    Ytr_t = torch.from_numpy(Ytr).to(args.device)
    Xva_t = torch.from_numpy(Xva).to(args.device)
    Yva_t = torch.from_numpy(Yva).to(args.device)
    train_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=args.batch, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    print(f"Training {sum(p.numel() for p in model.parameters())} params on {args.device}")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.monotonic()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.shape[0]
            n_train += xb.shape[0]
        sched.step()
        train_loss /= n_train

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t)
            val_loss = loss_fn(pred_va, Yva_t).item()
            # MAE per joint in degrees
            mae = (pred_va - Yva_t).abs().mean(0).cpu().numpy()
        improved = "*" if val_loss < best_val else " "
        if val_loss < best_val:
            best_val = val_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                },
                args.out,
            )
        elapsed = time.monotonic() - t0
        print(
            f"epoch {epoch + 1:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"per-joint MAE (deg): [{', '.join(f'{m:.2f}' for m in mae)}]  "
            f"lr={opt.param_groups[0]['lr']:.2e}  {elapsed:.1f}s  {improved}"
        )

    print(f"\nBest val loss {best_val:.4f}. Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
