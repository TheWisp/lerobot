#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prepare a 224x224 H.264 derivative of a LeRobot dataset for HVLA training.

The output is a plain standard LeRobot Dataset; train with it exactly as
before, only pointing ``--dataset-repo-id`` (or root) at the derivative.
TRAIN's ``--resize-images 224x224`` stays as-is and becomes a same-size no-op.

Example:

```shell
lerobot-prepare-hvla-dataset \
  --source-repo-id thewisp/dddd1 \
  --output-repo-id thewisp/dddd1_hvla224
```
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-repo-id", required=True, help="Repo id of the source dataset.")
    parser.add_argument(
        "--source-root",
        default=None,
        help="Local root of the source dataset (default: standard LeRobot cache).",
    )
    parser.add_argument("--output-repo-id", required=True, help="Repo id of the derivative dataset.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Local root of the derivative dataset (default: $HF_LEROBOT_HOME/output-repo-id).",
    )
    args = parser.parse_args()

    # Deferred imports keep `--help` fast and free of heavy dependencies.
    from lerobot.datasets.hvla_preparation import prepare_hvla_dataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    output_root = Path(args.output_root) if args.output_root else HF_LEROBOT_HOME / args.output_repo_id

    def print_progress(done: int, total: int, current: str) -> None:
        print(f"[{done}/{total}] {current}", flush=True)

    try:
        out = prepare_hvla_dataset(
            source_repo_id=args.source_repo_id,
            source_root=args.source_root,
            output_repo_id=args.output_repo_id,
            output_root=output_root,
            progress=print_progress,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Prepared dataset: {args.output_repo_id} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
