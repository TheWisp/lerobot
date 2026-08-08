#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Ask a checkpoint what it would do on frames it was trained on.

    lerobot-policy-probe --policy.path=outputs/train/.../pretrained_model \\
                         --dataset.repo_id=thewisp/my_dataset

Prints a per-joint comparison against the recorded actions and a one-line
verdict; ``--json out.json`` writes the full structure for a sweep or a test.

Use it when a checkpoint misbehaves on the robot and you cannot tell whether
the weights are bad or the deployment is. See :mod:`lerobot.policies.probe`
for why that distinction is the whole point.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _frame_indices(dataset, args) -> list[int]:
    """Pick which frames to probe.

    Spread across the dataset by default rather than taking the first N: the
    opening frames of every episode look alike (the robot at rest), and a
    policy that has collapsed to a rest pose scores *well* on exactly those.
    """
    total = len(dataset)
    if args.frames:
        return [int(f) for f in args.frames.split(",")]
    n = min(args.n_frames, total)
    if n <= 1:
        return [0]
    step = max(1, total // n)
    return list(range(0, total, step))[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="checkpoint directory")
    parser.add_argument("--dataset.repo_id", dest="repo_id", required=True, help="dataset to probe against")
    parser.add_argument("--dataset.root", dest="root", default=None, help="local dataset root")
    parser.add_argument("--n-frames", type=int, default=8, help="frames to probe, spread across the dataset")
    parser.add_argument("--frames", default=None, help="explicit comma-separated frame indices")
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", dest="json_out", default=None, help="write the full report here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.probe import probe_frames

    dataset = LeRobotDataset(args.repo_id, root=args.root)

    # Loaded the same way inference does, so the probe exercises the path that
    # actually runs on the robot rather than a bespoke one that might differ.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    policy = get_policy_class(cfg.type).from_pretrained(
        args.policy_path, config=cfg, dataset_meta=dataset.meta
    )
    policy.eval()

    preprocessor = None
    try:
        preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.policy_path)
    except Exception as exc:  # noqa: BLE001 — a missing processor is informative, not fatal
        logger.warning("no processor pipeline (%s); probing the policy directly", type(exc).__name__)

    report = probe_frames(
        policy,
        dataset,
        _frame_indices(dataset, args),
        preprocessor=preprocessor,
        device=args.device,
        checkpoint_name=str(args.policy_path),
    )

    if not report.frames:
        logger.error("No frames could be compared — see the warnings above.")
        return 1

    print(f"\ncheckpoint : {report.checkpoint}")
    print(f"dataset    : {report.dataset}  ({len(report.frames)} frames probed)")
    print(f"\nMAE vs recorded actions : {report.mae:.4f}")
    print(f"chunk spread (per frame) : {report.action_spread:.4f}")
    print(f"prediction spread (across frames) : {report.between_frame_spread:.4f}")

    worst = max(report.frames, key=lambda f: f.mae)
    print(f"\nworst frame: episode {worst.episode_index} frame {worst.frame_index} (MAE {worst.mae:.4f})")
    print("  per-joint MAE:", ", ".join(f"{v:.3f}" for v in worst.mae_per_joint))
    print("  predicted[0] :", ", ".join(f"{v:.3f}" for v in worst.predicted_first))
    print("  recorded[0]  :", ", ".join(f"{v:.3f}" for v in worst.recorded_first))

    print(f"\n{report.verdict()}\n")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.as_dict(), indent=2))
        print(f"full report -> {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
