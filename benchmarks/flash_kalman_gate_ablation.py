"""Paired clip replay with only Kalman's innovation veto changed.

Uses recorded RGB, selected depth and masks; shared LK, grouping, fitting,
Kalman and growth run normally. Both arms initialize at the first retained
frame, which may differ from the original live initialization. This is a paired
comparison, not a reconstruction of pruned live state or labelled accuracy.
Every measured publication is rechecked against the PRE-growth geometry.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.flash_clip_replay import replay
from benchmarks.flash_kalman_recovery_control import StrictPoseKalmanFilter
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter


def paired_replay(clip, output, overrides=()):
    output.mkdir(parents=True, exist_ok=True)
    original = GrowingSurfaceSession.update
    summaries = {}
    for enabled in (True, False):
        name = "gated" if enabled else "ungated"
        audit = []
        clouds = {}

        def checked_update(self, prepared, time_s, *, _name=name, _audit=audit, _clouds=clouds, **kwargs):
            geometry = kwargs.get("geometry")
            surface = self.store.points.copy()
            result = original(self, prepared, time_s, **kwargs)
            details = result["diagnostics"]
            kalman = details.get("kalman", {})
            check = None
            if result["status"] == "observed_candidate":
                assert geometry is not None
                check = geometry.check_pose(
                    np.asarray(result["camera_from_reference"]),
                    max_outlier_fraction=self.config.uncertainty.max_visible_outlier_fraction,
                    agreement_m=self.config.agreement_m,
                    min_overlap=self.config.min_overlap,
                )
                assert check["accepted"], (_name, prepared["frame_id"], check)
            _audit.append(
                {
                    "frame": prepared["frame_id"],
                    "status": result["status"],
                    "reason": details.get("reason"),
                    "kalman_reason": kalman.get("reason"),
                    "filter_reason": kalman.get("filter_reason"),
                    "score": kalman.get("innovation_squared"),
                    "limit": kalman.get("innovation_limit"),
                    "geometry": check,
                    "surface_points": len(surface),
                    "fit_pose": kalman.get("observation_pose"),
                    "prior_pose": kalman.get("prior_pose"),
                }
            )
            _clouds[str(prepared["frame_id"])] = surface
            return result

        with (
            patch.object(GrowingSurfaceSession, "update", checked_update),
            patch(
                "lerobot.policies.flash.experimental_surface_session.PoseKalmanFilter",
                StrictPoseKalmanFilter if enabled else PoseKalmanFilter,
            ),
        ):
            rows, summary = replay(
                clip,
                overrides=[
                    *overrides,
                    "surface_experiment.uncertainty.reset_on_geometry_conflict=false",
                ],
                quiet=True,
            )
        summary["reasons"] = dict(Counter(r["reason"] for r in audit if r["reason"]))
        summary["checked_publications"] = sum(r["geometry"] is not None for r in audit)
        summary["scope"] = "Same first-retained-frame initialization in both arms; recorded depth/masks."
        (output / f"{name}.json").write_text(json.dumps({"summary": summary, "rows": rows, "audit": audit}))
        np.savez_compressed(output / f"{name}-surfaces.npz", **clouds)
        summaries[name] = summary
        print(json.dumps({name: summary}), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    paired_replay(args.clip, args.out, args.set)


if __name__ == "__main__":
    main()
