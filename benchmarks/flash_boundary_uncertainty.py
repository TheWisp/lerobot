"""Isolated boundary-depth ablation; does not change production defaults.

Withhold readings in a symmetric mask band, then run the unchanged consumer.
The existing nearest-depth neighborhood remains in effect. The width below is
the raw pixel band, not the total footprint of that conservative neighborhood.
"""

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial import cKDTree

from lerobot.policies.flash.surface_grounding import PreparedTargetSurface, project
from lerobot.policies.flash.surface_observation import reliable_observation_depth, spatial_sample
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession


def boundary_unknown(mask, width, side="both"):
    if width <= 0:
        return np.zeros_like(mask, dtype=bool)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    unknown = (~mask) & (outside <= width)
    if side == "both":
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        unknown |= mask & (inside <= width)
    return unknown


@contextmanager
def boundary_band(width, side="both", *, source_margin=0):
    original = PreparedTargetSurface.build.__func__

    def build(cls, observation, config):
        prepared = original(cls, observation, config)
        if width <= 0 and source_margin <= 0:
            return prepared
        unknown = boundary_unknown(observation["mask"], width, side)
        if source_margin:
            interior = (
                cv2.erode(
                    observation["mask"].astype(np.uint8), np.ones((2 * source_margin + 1,) * 2, np.uint8)
                )
                > 0
            )
            unknown |= observation["mask"] & ~interior
        depth = prepared.depth.copy()
        depth[unknown] = 0
        scene = reliable_observation_depth(observation, observation.get("scene_depth"))
        scene[unknown] = 0
        radius = int(np.ceil(config.mask_boundary_tolerance_px))
        nearest = cv2.erode(
            scene, np.ones((2 * radius + 1,) * 2, np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        result = replace(prepared, depth=depth, scene_nearest_depth=nearest)
        if source_margin:
            # A reverse check transports actual current depth samples. Their
            # reliability belongs to their original pixels, before any pose.
            k = observation["intrinsics"]
            xy = np.rint(project(prepared.points, k)).astype(int)
            points = prepared.points[interior[xy[:, 1], xy[:, 0]]]
            result = replace(
                result,
                points=points,
                tree=cKDTree(points),
                queries=points[spatial_sample(points, config.max_fit_points)],
            )
        return result

    with patch.object(PreparedTargetSurface, "build", classmethod(build)):
        yield


@contextmanager
def measurement_band(inside, outside):
    """Full replay: clean the measurement once, before fitting/growth/retention."""
    original = SurfaceTrackingSession.prepare

    def prepare(self, *args, **kwargs):
        config = self.observation_config
        self.observation_config = replace(config, mask_margin_px=inside)
        try:
            result = original(self, *args, **kwargs)
        finally:
            self.observation_config = config
        current = result["current"]
        unknown = boundary_unknown(current["mask"], outside, "outside")
        current["obs"]["depth_reliable"] = current["obs"]["depth_reliable"] & ~unknown
        depth = current["depth"].copy()
        depth[unknown] = 0
        current["depth"] = current["scene_depth"] = depth
        return result

    with patch.object(SurfaceTrackingSession, "prepare", prepare):
        yield


if __name__ == "__main__":
    import argparse
    import hashlib
    import json
    from pathlib import Path

    from benchmarks.flash_surface_refresh import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--inside", type=int, required=True)
    parser.add_argument("--outside", type=int, required=True)
    parser.add_argument("--arm", choices=("off", "replace"), default="replace")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "boundary-policy.json").write_text(
        json.dumps(
            {
                "inside_square_margin_px": args.inside,
                "outside_euclidean_band_px": args.outside,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Measurement cleanup before fitter, retained views, growth and refresh. All other rules unchanged.",
            },
            indent=2,
        )
    )
    with measurement_band(args.inside, args.outside):
        run(args.clip, args.arm, args.out, [2013, 2021, 2054, 2078, 2086, 2882])
