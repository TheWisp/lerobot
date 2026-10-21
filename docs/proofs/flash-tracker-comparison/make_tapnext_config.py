"""Replace only the point frontend in a pinned Point2Pose YAML for ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-resolution", type=int, choices=(256, 512), default=256)
    parser.add_argument("--iteration-sample-period", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("TAPNext++ checkpoint was not found")
    config = OmegaConf.load(args.base)
    config.tracker = OmegaConf.create(
        {
            "type": "tapnext",
            "params": {
                "device": "cuda",
                "input_resolution": args.input_resolution,
                "visible_threshold": 0.5,
                "half_precision": True,
                "checkpoint_path": str(args.checkpoint.absolute()),
            },
        }
    )
    if args.iteration_sample_period is not None:
        if args.iteration_sample_period < 1:
            parser.error("iteration sample period must be positive")
        config.criterion.type = "iteration"
        config.criterion.params.update_per_iter = args.iteration_sample_period
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
