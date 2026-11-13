"""One recorded frame, same seed and acceptance map, two solver source clouds.

The extra solves are detached diagnostics. The real update receives the original
result. This is not an outer recovery loop or a proposed production change.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.flash_clip_replay import replay
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession, ImageConditionedFit
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
from lerobot.policies.flash.surface_observation import move


def compare(clip, output, frame, weight):
    output.mkdir(parents=True, exist_ok=True)
    real_update = GrowingSurfaceSession.update
    real_fit = ImageConditionedFit.correct_surface_first
    context, report = {}, {}

    def update(self, prepared, time_s, **kwargs):
        context.update(session=self, frame=prepared["frame_id"])
        return real_update(self, prepared, time_s, **kwargs)

    def fit(self, proposal, previous, *, uncertainty=None, check=None):
        baseline = real_fit(self, proposal, previous, uncertainty=uncertainty, check=check)
        if context["frame"] != frame:
            return baseline
        session = context["session"]
        before = session.store.points.copy()
        acceptance = self.geometry
        recent = move(session.accepted_observation["obs"]["points"], np.linalg.inv(session.accepted_pose))
        report.update(
            frame=frame,
            accepted_frame=session.accepted_frame,
            weight=weight,
            map_sha256=hashlib.sha256(before.tobytes()).hexdigest(),
            map_points=len(before),
            recent_points=len(recent),
            seed=proposal.tolist(),
            registration=vars(self.config.registration),
            cases={},
        )
        for iterations in (self.config.registration.iterations_per_level, 100):
            shape = replace(
                self.config, registration=replace(self.config.registration, iterations_per_level=iterations)
            )
            for name, points in (("accumulated", before), ("recent", recent)):
                geometry = WholeSurfaceFit(
                    points,
                    self.observation,
                    self.k,
                    shape,
                    reference_views=self.reference_views,
                    appearance_views=self.appearance_views,
                )
                objective = ImageConditionedFit(
                    points,
                    self.observation,
                    self.k,
                    shape,
                    geometry=geometry,
                    image_reference=self.image_reference,
                    image_current=self.image_current,
                    image_scale_m=self.image_scale_m,
                    image_weight=self.image_weight,
                    point_evidence=self.point_evidence,
                    reference_views=self.reference_views,
                    appearance_views=self.appearance_views,
                )
                pose, detail = real_fit(
                    objective, proposal.copy(), previous.copy(), uncertainty=uncertainty, check=check
                )
                candidate = pose if pose is not None else np.asarray(detail["rejected_camera_from_reference"])
                limits = {
                    "max_outlier_fraction": session.config.uncertainty.max_visible_outlier_fraction,
                    "agreement_m": session.config.agreement_m,
                    "min_overlap": session.config.min_overlap,
                }
                report["cases"][f"{name}/{iterations}"] = {
                    "accepted": pose is not None,
                    "pose": candidate.tolist(),
                    "untouched_map_check": acceptance.check_pose(candidate, **limits),
                    "proposer_source_check": geometry.check_pose(candidate, **limits),
                    "fit_detail": detail,
                }
        np.testing.assert_array_equal(session.store.points, before)
        arrays = {
            "map": before,
            "recent": recent,
            "seed": proposal,
            "intrinsics": self.k,
            "depth": self.observation["depth"],
            "mask": self.observation["mask"],
            "points": self.observation["obs"]["points"],
            "image_reference": self.image_reference,
            "image_current": self.image_current,
        }
        np.savez_compressed(output / f"frame-{frame}.npz", **arrays)
        return baseline

    with (
        patch.object(GrowingSurfaceSession, "update", update),
        patch.object(ImageConditionedFit, "correct_surface_first", fit),
    ):
        rows, summary = replay(
            clip, stop=frame, overrides=[f"surface_experiment.refinement_max_weight={weight}"], quiet=True
        )
    if not report:
        raise ValueError("Requested frame did not reach fitting")
    report.update(summary=summary, replay_rows=rows)
    (output / "comparison.json").write_text(json.dumps(report, default=lambda x: x.tolist(), indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {
                    "accepted": v["accepted"],
                    "map_check": v["untouched_map_check"],
                    "source_check": v["proposer_source_check"],
                }
                for k, v in report["cases"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=2882)
    parser.add_argument("--weight", type=int, default=8)
    args = parser.parse_args()
    compare(args.clip, args.out, args.frame, args.weight)
