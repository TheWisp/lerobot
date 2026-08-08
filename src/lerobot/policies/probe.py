# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Run a checkpoint on recorded frames and compare it to what the human did.

Motivated by a checkpoint that idled and shook in place on the robot. Watching
the robot cannot tell you whether the policy learned nothing, whether its
outputs are being denormalized wrongly, or whether the feature order at
inference disagrees with training — all three look like "it moves badly".

The question this answers is deliberately narrower and objective: **on a frame
the model was trained on, how far is its predicted chunk from the action the
operator actually recorded?** That bisects the problem in one measurement:

- Large error on training frames -> training, normalization, or feature
  contract. The robot is not involved and no hardware session will help.
- Small error on training frames but bad behaviour on the robot -> deployment:
  inference-time normalization, action ordering, chunk blending, latency.

Everything here is offline and deterministic: dataset in, numbers out, no robot
and no control loop. That is why it is a library function rather than a GUI
endpoint — a sweep, a regression test and a UI panel can all sit on top, and
none of them can sit on top of an HTTP handler.
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
    """One frame's prediction measured against its recorded action."""

    episode_index: int
    frame_index: int
    action_dim: int
    horizon: int
    # Per-joint mean absolute error over the horizon the ground truth covers.
    mae_per_joint: list[float]
    mae: float
    # Predicted and recorded first actions, the ones that actually get executed.
    predicted_first: list[float]
    recorded_first: list[float]
    # Spread of the predicted chunk. A policy that has collapsed to a constant
    # emits a near-zero range regardless of what it was shown.
    predicted_range_per_joint: list[float]
    ground_truth_frames: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeReport:
    """Aggregate over probed frames, plus the diagnosis that follows from it."""

    checkpoint: str
    dataset: str
    frames: list[FrameProbe] = field(default_factory=list)

    @property
    def mae(self) -> float:
        return float(np.mean([f.mae for f in self.frames])) if self.frames else float("nan")

    @property
    def action_spread(self) -> float:
        """Mean per-joint range of predictions across all probed frames.

        Near zero means the policy emits the same chunk whatever it is shown —
        the signature of a collapsed model, and what "idles in place" looks like
        in numbers.
        """
        if not self.frames:
            return float("nan")
        return float(np.mean([np.mean(f.predicted_range_per_joint) for f in self.frames]))

    @property
    def between_frame_spread(self) -> float:
        """How much the *first* predicted action varies across probed frames.

        Distinguishes the two collapse modes: a policy can emit a flat chunk per
        frame yet still respond to input (low action_spread, high
        between_frame_spread), or emit the same chunk everywhere (both low).
        """
        if len(self.frames) < 2:
            return float("nan")
        firsts = np.array([f.predicted_first for f in self.frames])
        return float(np.mean(firsts.max(axis=0) - firsts.min(axis=0)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "dataset": self.dataset,
            "n_frames": len(self.frames),
            "mae": self.mae,
            "action_spread": self.action_spread,
            "between_frame_spread": self.between_frame_spread,
            "verdict": self.verdict(),
            "frames": [f.as_dict() for f in self.frames],
        }

    def verdict(self) -> str:
        """A one-line reading of the numbers.

        Deliberately coarse. It exists so the output is actionable without
        knowing what a good MAE looks like for this robot, not to be a
        classifier — the per-joint numbers are what you act on.
        """
        if not self.frames:
            return "no frames probed"
        recorded = np.array([f.recorded_first for f in self.frames])
        recorded_spread = (
            float(np.mean(recorded.max(axis=0) - recorded.min(axis=0))) if len(recorded) > 1 else 0.0
        )

        if self.between_frame_spread < 0.01 * max(recorded_spread, 1e-6):
            return (
                "COLLAPSED: predictions barely change across frames whose recorded actions do. "
                "The policy is ignoring its input — look at training (did loss actually fall?) "
                "and at normalization stats, not at the robot."
            )
        if self.action_spread < 1e-3:
            return (
                "FLAT CHUNKS: each prediction is near-constant over the horizon. Consistent with "
                "a policy that learned a mean pose, or with action denormalization collapsing."
            )
        return (
            "RESPONSIVE: predictions vary with input and across the horizon. If the robot still "
            "misbehaves, suspect deployment — inference-time normalization, action ordering, "
            "chunk blending, or latency — rather than the weights."
        )


def _chunk_from_policy(policy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Get an action chunk, falling back to a single action for policies without one.

    Postcondition: returns ``[horizon, action_dim]`` on CPU.
    """
    with torch.no_grad():
        if hasattr(policy, "predict_action_chunk"):
            chunk = policy.predict_action_chunk(batch)
        else:
            chunk = policy.select_action(batch).unsqueeze(1)
    chunk = chunk.detach().to("cpu").float()
    if chunk.ndim == 3:  # [batch, horizon, dim]
        chunk = chunk[0]
    elif chunk.ndim == 2:  # [batch, dim] -> one-step horizon
        chunk = chunk[:1].reshape(1, -1)
    return chunk


def probe_frames(
    policy,
    dataset,
    frame_indices,
    *,
    preprocessor=None,
    device: str | None = None,
    checkpoint_name: str = "<policy>",
) -> ProbeReport:
    """Run ``policy`` on dataset frames and measure it against recorded actions.

    Preconditions: ``policy`` is in eval mode; ``dataset`` yields items whose
    ``ACTION`` entry is the recorded action (shape ``[dim]`` or
    ``[horizon, dim]`` when delta timestamps are configured); every index in
    ``frame_indices`` is valid for ``dataset``.

    Postcondition: the report holds one :class:`FrameProbe` per index that
    produced a comparable prediction. Frames whose ground truth cannot be
    lined up with the prediction are skipped with a warning rather than
    silently contributing a wrong number.
    """
    report = ProbeReport(checkpoint=checkpoint_name, dataset=getattr(dataset, "repo_id", "<dataset>"))
    device = device or getattr(getattr(policy, "config", None), "device", None) or "cpu"

    for idx in frame_indices:
        item = dataset[idx]
        recorded = item.get(ACTION)
        if recorded is None:
            logger.warning("frame %d has no %s; skipping", idx, ACTION)
            continue

        batch = {}
        for key, value in item.items():
            batch[key] = value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        if preprocessor is not None:
            batch = preprocessor(batch)

        if hasattr(policy, "reset"):
            policy.reset()  # chunk-queue policies would otherwise serve a stale chunk
        chunk = _chunk_from_policy(policy, batch)

        truth = recorded.detach().to("cpu").float()
        if truth.ndim == 1:
            truth = truth.unsqueeze(0)

        # Compare only the overlap: the dataset's delta-timestamp window and the
        # policy's horizon are configured independently and rarely match.
        n = min(chunk.shape[0], truth.shape[0])
        if n == 0 or chunk.shape[-1] != truth.shape[-1]:
            logger.warning(
                "frame %d: prediction %s and ground truth %s do not line up; skipping",
                idx,
                tuple(chunk.shape),
                tuple(truth.shape),
            )
            continue

        err = (chunk[:n] - truth[:n]).abs()
        report.frames.append(
            FrameProbe(
                episode_index=int(item.get("episode_index", -1)),
                frame_index=int(item.get("frame_index", idx)),
                action_dim=int(chunk.shape[-1]),
                horizon=int(chunk.shape[0]),
                mae_per_joint=[round(v, 6) for v in err.mean(dim=0).tolist()],
                mae=float(err.mean()),
                predicted_first=[round(v, 6) for v in chunk[0].tolist()],
                recorded_first=[round(v, 6) for v in truth[0].tolist()],
                predicted_range_per_joint=[
                    round(v, 6) for v in (chunk.max(dim=0).values - chunk.min(dim=0).values).tolist()
                ],
                ground_truth_frames=int(n),
            )
        )

    return report
