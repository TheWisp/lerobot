# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Diagnose a checkpoint that misbehaves on the robot, offline.

Motivated by an OpenArm 2.0 checkpoint that idled and shook in place. From
across the room, a policy that learned nothing, one whose actions are
denormalized wrongly, and one whose feature order disagrees with training all
look the same.

**This is a pathology detector, not an evaluation.** The distinction matters
because the obvious metric is wrong here: with multimodal demonstrations there
is more than one correct action, and the minimizer of a pointwise L1/L2 distance
to the recorded action is the *average of the modes*, which is frequently itself
invalid — go left or right around an obstacle are both fine, their mean drives
through it. A policy can therefore score well by being wrong and badly by being
decisively right. Flow S1 is a flow-matching model chosen precisely to represent
that multimodality, so scoring it with a unimodal metric would be inconsistent
with the model class.

What survives that objection, in the order worth reading:

1. **Training loss on training batches.** The model's own objective, defined on
   the multimodal data, needing no metric philosophy. High here means it did not
   fit, and nothing else matters until that is fixed.
2. **Between-frame spread.** If predictions barely change across frames whose
   recorded actions differ a lot, the policy is ignoring its input. No appeal to
   "a different action may be valid" rescues that.
3. **Same-frame sampling spread.** Sample the same observation repeatedly. A
   wide conditional distribution plus a fresh sample every query interval means
   the arm switches modes between chunks — a mechanism for "shakes in place"
   that is a *deployment* problem, not a training one.
4. **Best-of-K distance to the recorded action.** The multimodality-aware
   replacement for MAE: if the policy covers the mode the operator used, the
   closest of K samples is near even when their mean is far. Normalized per
   joint by the dataset's action std, because a shoulder and a gripper are not
   comparable in raw units.

Mean distance is still reported, as a descriptive number. It is never a verdict.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)


@dataclass
class FrameProbe:
    """One frame, sampled K times."""

    episode_index: int
    frame_index: int
    action_dim: int
    horizon: int
    samples: int
    # Closest of K samples to what the operator recorded, per joint, in units of
    # that joint's dataset std. Multimodality-tolerant: covering the mode counts.
    best_of_k_per_joint: list[float]
    best_of_k: float
    # Mean over samples, for reference only — see the module docstring.
    mean_distance: float
    # Disagreement between samples of the SAME observation. High means the
    # policy's conditional distribution is wide; resampling per query interval
    # then produces mode switching.
    sample_spread: float
    predicted_first: list[float]
    recorded_first: list[float]
    ground_truth_frames: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeReport:
    checkpoint: str
    dataset: str
    frames: list[FrameProbe] = field(default_factory=list)
    # The model's own objective on the frames probed; None if it does not expose one.
    training_loss: float | None = None

    @property
    def best_of_k(self) -> float:
        return float(np.mean([f.best_of_k for f in self.frames])) if self.frames else float("nan")

    @property
    def sample_spread(self) -> float:
        """Mean disagreement between samples of one observation."""
        return float(np.mean([f.sample_spread for f in self.frames])) if self.frames else float("nan")

    @property
    def between_frame_spread(self) -> float:
        """Variation of the first predicted action across different frames."""
        if len(self.frames) < 2:
            return float("nan")
        firsts = np.array([f.predicted_first for f in self.frames])
        return float(np.mean(firsts.max(axis=0) - firsts.min(axis=0)))

    @property
    def recorded_spread(self) -> float:
        """The same quantity for the recorded actions — the scale to judge against."""
        if len(self.frames) < 2:
            return float("nan")
        rec = np.array([f.recorded_first for f in self.frames])
        return float(np.mean(rec.max(axis=0) - rec.min(axis=0)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "dataset": self.dataset,
            "n_frames": len(self.frames),
            "training_loss": self.training_loss,
            "best_of_k": self.best_of_k,
            "sample_spread": self.sample_spread,
            "between_frame_spread": self.between_frame_spread,
            "recorded_spread": self.recorded_spread,
            "findings": self.findings(),
            "frames": [f.as_dict() for f in self.frames],
        }

    def findings(self) -> list[str]:
        """Independent observations, strongest first. Not a score.

        Each is phrased as what to go and look at, because the point is to
        bisect the search space rather than to grade the checkpoint.
        """
        out: list[str] = []
        if not self.frames:
            return ["no frames could be compared — see the warnings above"]

        if self.training_loss is not None and self.training_loss > 1.0:
            out.append(
                f"DID NOT FIT: the model's own training loss is {self.training_loss:.3f} on frames it "
                "trained on. Nothing downstream matters until that is explained — check that loss "
                "actually fell during training, and that normalization stats came from this dataset."
            )

        scale = self.recorded_spread
        if np.isfinite(scale) and scale > 0 and self.between_frame_spread < 0.05 * scale:
            out.append(
                f"IGNORES ITS INPUT: predictions vary by {self.between_frame_spread:.4f} across frames "
                f"whose recorded actions vary by {scale:.4f}. Multimodality cannot explain this — the "
                "observation is not reaching the policy, or it collapsed to a mean pose."
            )

        if np.isfinite(scale) and scale > 0 and self.sample_spread > 0.5 * scale:
            out.append(
                f"WIDE CONDITIONAL: sampling the same frame twice moves the action by "
                f"{self.sample_spread:.4f}, against a between-frame spread of {scale:.4f}. Resampling "
                "each query interval will switch modes — this is a plausible mechanism for shaking in "
                "place, and it is a deployment fix (chunk blending, RTC, fewer resamples), not a "
                "training one."
            )

        if not out:
            out.append(
                "NO PATHOLOGY DETECTED offline: the model fits its training data, responds to input, "
                "and samples consistently. If the robot still misbehaves, suspect the deployment path "
                "— inference-time normalization, action ordering, chunk blending, latency."
            )
        return out


def _sample_chunks(policy, batch: dict[str, torch.Tensor], k: int) -> torch.Tensor:
    """Return ``[k, horizon, action_dim]`` on CPU.

    Resets between samples so a chunk-queue policy re-runs inference instead of
    serving the previous sample from its queue.
    """
    out = []
    for _ in range(k):
        if hasattr(policy, "reset"):
            policy.reset()
        with torch.no_grad():
            if hasattr(policy, "predict_action_chunk"):
                chunk = policy.predict_action_chunk(batch)
            else:
                chunk = policy.select_action(batch).unsqueeze(1)
        chunk = chunk.detach().to("cpu").float()
        if chunk.ndim == 3:
            chunk = chunk[0]
        elif chunk.ndim == 2:
            chunk = chunk[:1].reshape(1, -1)
        out.append(chunk)
    return torch.stack(out)


def _action_scale(dataset, action_dim: int) -> np.ndarray:
    """Per-joint std from the dataset, so joints are comparable.

    Falls back to ones when stats are unavailable; a scale of 1 is raw units,
    which is wrong but visible, rather than a silent divide-by-zero.
    """
    try:
        std = np.asarray(dataset.meta.stats[ACTION]["std"], dtype=np.float64).reshape(-1)
        if std.shape[0] == action_dim and np.all(np.isfinite(std)):
            return np.where(std > 1e-6, std, 1.0)
    except Exception:  # noqa: BLE001 — absent stats are a fallback, not a failure
        logger.warning("no action std in dataset stats; reporting distances in raw units")
    return np.ones(action_dim)


def training_loss_on(policy, dataset, frame_indices, *, preprocessor=None, device="cpu") -> float | None:
    """Run the model's own training objective over the given frames.

    The most direct answer to "did it learn": it is the quantity that was
    optimized, it is defined on multimodal data, and it needs no assumption
    about which action is correct. Returns None when the policy does not expose
    a training forward.
    """
    losses = []
    for idx in frame_indices:
        batch = _to_batch(dataset[idx], device)
        if preprocessor is not None:
            batch = preprocessor(batch)
        try:
            with torch.no_grad():
                out = policy.forward(batch)
        except Exception as exc:  # noqa: BLE001 — a policy without a train forward is not an error
            logger.warning("policy has no usable training forward (%s); skipping loss", type(exc).__name__)
            return None
        loss = out[0] if isinstance(out, tuple) else out
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else None


def _to_batch(item: dict, device: str) -> dict:
    return {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}


def probe_frames(
    policy,
    dataset,
    frame_indices,
    *,
    preprocessor=None,
    device: str | None = None,
    samples: int = 4,
    checkpoint_name: str = "<policy>",
) -> ProbeReport:
    """Sample the policy on recorded frames and measure the four signals.

    Preconditions: ``policy`` is in eval mode; ``dataset`` items carry ``ACTION``
    (shape ``[dim]`` or ``[horizon, dim]``); indices are valid. ``samples`` > 1
    is required for the sampling-spread signal to mean anything.

    Postcondition: one :class:`FrameProbe` per index that produced a comparable
    prediction. Frames whose ground truth cannot be aligned are skipped with a
    warning rather than contributing a wrong number.
    """
    frame_indices = list(frame_indices)
    report = ProbeReport(checkpoint=checkpoint_name, dataset=getattr(dataset, "repo_id", "<dataset>"))
    device = device or getattr(getattr(policy, "config", None), "device", None) or "cpu"

    for idx in frame_indices:
        item = dataset[idx]
        recorded = item.get(ACTION)
        if recorded is None:
            logger.warning("frame %d has no %s; skipping", idx, ACTION)
            continue

        batch = _to_batch(item, device)
        if preprocessor is not None:
            batch = preprocessor(batch)
        chunks = _sample_chunks(policy, batch, samples)

        truth = recorded.detach().to("cpu").float()
        if truth.ndim == 1:
            truth = truth.unsqueeze(0)

        n = min(chunks.shape[1], truth.shape[0])
        if n == 0 or chunks.shape[-1] != truth.shape[-1]:
            logger.warning(
                "frame %d: prediction %s and ground truth %s do not line up; skipping",
                idx,
                tuple(chunks.shape[1:]),
                tuple(truth.shape),
            )
            continue

        scale = torch.tensor(_action_scale(dataset, int(chunks.shape[-1])), dtype=torch.float32)
        err = ((chunks[:, :n] - truth[:n]) / scale).abs()  # [k, n, dim]
        per_sample_joint = err.mean(dim=1)  # [k, dim]
        best_idx = int(per_sample_joint.mean(dim=1).argmin())

        report.frames.append(
            FrameProbe(
                episode_index=int(item.get("episode_index", -1)),
                frame_index=int(item.get("frame_index", idx)),
                action_dim=int(chunks.shape[-1]),
                horizon=int(chunks.shape[1]),
                samples=samples,
                best_of_k_per_joint=[round(v, 6) for v in per_sample_joint[best_idx].tolist()],
                best_of_k=float(per_sample_joint[best_idx].mean()),
                mean_distance=float(per_sample_joint.mean()),
                sample_spread=float((chunks[:, 0].max(dim=0).values - chunks[:, 0].min(dim=0).values).mean())
                if samples > 1
                else 0.0,
                predicted_first=[round(v, 6) for v in chunks[best_idx, 0].tolist()],
                recorded_first=[round(v, 6) for v in truth[0].tolist()],
                ground_truth_frames=int(n),
            )
        )

    report.training_loss = training_loss_on(
        policy, dataset, frame_indices, preprocessor=preprocessor, device=device
    )
    return report
