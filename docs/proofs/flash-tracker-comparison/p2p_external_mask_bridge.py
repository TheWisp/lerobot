"""Replay-only adapter: replace Point2Pose's current SAM2 mask with a supplied mask.

The pinned sealed bridge and Point2Pose checkout remain untouched. The bridge
still initializes its SAM2 session from the usual first mask; every subsequent
frame uses the caller's current causal SAM3 mask. This ablates segmentation,
not pose fitting, and is not a production integration.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-bridge", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--honor-initial-mask", action="store_true")
    args = parser.parse_args()

    specification = importlib.util.spec_from_file_location("sealed_p2p_bridge", args.sealed_bridge)
    if specification is None or specification.loader is None:
        raise ValueError("could not load the sealed Point2Pose bridge")
    bridge = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(bridge)
    original_read = bridge._read
    original_step = bridge.Session.step
    current_mask: np.ndarray | None = None

    def read_with_mask(stream):
        nonlocal current_mask
        request = original_read(stream)
        current_mask = None if request is None or "mask" not in request else np.asarray(request["mask"])
        return request

    def step_with_mask(session, rgb: np.ndarray, depth_m: np.ndarray):
        if current_mask is None:
            raise ValueError("this ablation requires a current mask on every step")
        import torch

        segmenter = session.pipe.frontend.segmenter
        original_segment = segmenter.segment
        masks = np.asarray(current_mask, dtype=bool)
        if masks.ndim == 2:
            masks = masks[None]
        if masks.shape[0] != len(session.pipe.objects) or masks.shape[1:] != rgb.shape[:2]:
            raise ValueError("external masks must align with every tracked object")
        logits = torch.from_numpy(np.where(masks[:, None], 1.0, -1.0).astype(np.float32)).to(segmenter.device)
        segmenter.segment = lambda _rgb: (list(range(len(masks))), logits)
        try:
            return original_step(session, rgb, depth_m)
        finally:
            segmenter.segment = original_segment

    bridge._read = read_with_mask
    bridge.Session.step = step_with_mask
    if args.honor_initial_mask:
        # The ordinary initializer feeds a supplied mask to SAM2, then
        # immediately replaces it with SAM2's first output before seeding the
        # Point2Pose point map. Keep that first output equal to the supplied
        # mask in this ablation, without editing the sealed bridge or P2P.
        sys.path.insert(0, str(args.repo))
        from point2pose.pipeline.components.front_end import FrontEnd

        original_initialize = FrontEnd.initialize

        def initialize_with_exact_mask(frontend, frame):
            import torch

            supplied = np.asarray(frame.mask, dtype=bool)
            if supplied.ndim != 4 or supplied.shape[1] != 1:
                raise ValueError("initial masks must have shape (N, 1, H, W)")
            logits = torch.from_numpy(np.where(supplied, 1.0, -1.0).astype(np.float32)).to(
                frontend.segmenter.device
            )
            original_segment = frontend.segmenter.segment
            frontend.segmenter.segment = lambda _rgb: (list(range(len(supplied))), logits)
            try:
                return original_initialize(frontend, frame)
            finally:
                frontend.segmenter.segment = original_segment

        FrontEnd.initialize = initialize_with_exact_mask
    sys.argv = [str(args.sealed_bridge), "--repo", str(args.repo), "--config", str(args.config)]
    bridge.main()


if __name__ == "__main__":
    main()
