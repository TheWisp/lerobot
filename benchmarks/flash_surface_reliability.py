"""Trial: full-map ICP, but only freshly measured map points may reject.

Runs inside the shared tracker; changes neither production defaults nor solver
inputs. Rejection evidence is frozen from the previous accepted refresh, before
the current candidate exists. Old hidden/boundary samples remain in the map.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lerobot.policies.flash import experimental_surface_session as ess
from lerobot.policies.flash.surface_grounding import SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_refresh import (
    LatestObservationFit,
    attach_rejection_geometry,
    refreshed_membership,
)


@contextmanager
def latest_observed_rejection():
    original_prepare = ess.GrowingSurfaceSession.prepare_geometry
    original_check = WholeSurfaceFit.check_pose
    original_update = ess.GrowingSurfaceSession.update
    audit, masks = [], {}

    def prepare(self, prepared):
        full = original_prepare(self, prepared)
        if full is not None:
            if self.accepted_observation is None:
                raise AssertionError("Initialized session has no accepted measurement")
            attach_rejection_geometry(
                full, SurfaceFitView(self.accepted_observation, self.accepted_intrinsics, self.accepted_pose)
            )
        return full

    def check(self, pose, *, max_outlier_fraction=None, agreement_m=None, min_overlap=0.0):
        # ImageConditionedFit delegates the check to the same frozen geometry.
        full = self.geometry if isinstance(self, ess.ImageConditionedFit) else self
        if not hasattr(full, "_latest_rejection"):
            return original_check(
                self,
                pose,
                max_outlier_fraction=max_outlier_fraction,
                agreement_m=agreement_m,
                min_overlap=min_overlap,
            )
        return LatestObservationFit.check_pose(
            full,
            pose,
            max_outlier_fraction=max_outlier_fraction,
            agreement_m=agreement_m,
            min_overlap=min_overlap,
        )

    def update(self, prepared, time_s, **kwargs):
        before = self.store.points.copy()
        previous = self.accepted_frame
        previous_observation, previous_pose = self.accepted_observation, self.accepted_pose
        before_flags = (
            refreshed_membership(before, previous_observation, previous_pose)
            if previous_observation is not None
            else np.zeros(len(before), bool)
        )
        result = original_update(self, prepared, time_s, **kwargs)
        after_flags = (
            refreshed_membership(self.store.points, self.accepted_observation, self.accepted_pose)
            if self.accepted_observation is not None
            else np.zeros(len(self.store.points), bool)
        )
        if result["status"] not in ("initialized", "observed_candidate"):
            np.testing.assert_array_equal(self.store.points, before)
            np.testing.assert_array_equal(after_flags, before_flags)
            assert self.accepted_frame == previous
        frame = prepared["frame_id"]
        detail = {
            "frame": frame,
            "accepted_frame_before": previous,
            "accepted_frame_after": self.accepted_frame,
            "full_before": len(before),
            "reliable_before": int(before_flags.sum()),
            "full_after": len(self.store.points),
            "reliable_after": int(after_flags.sum()),
        }
        masks[f"{frame}_before"], masks[f"{frame}_after"] = before_flags, after_flags
        audit.append(detail)
        result["diagnostics"]["latest_surface_reliability"] = detail
        return result

    with (
        patch.object(ess.GrowingSurfaceSession, "prepare_geometry", prepare),
        patch.object(WholeSurfaceFit, "check_pose", check),
        patch.object(ess.GrowingSurfaceSession, "update", update),
    ):
        yield audit, masks


def run(clip, output):
    import json

    from benchmarks.flash_boundary_uncertainty import measurement_band
    from benchmarks.flash_surface_refresh import run as refresh_run

    output.mkdir(parents=True, exist_ok=False)
    with measurement_band(4, 2), latest_observed_rejection() as (audit, masks):
        refresh_run(
            clip, "replace", output, [2013, 2021, 2078, 2086, 2701, 2723, 2873, 2882, 2958, 3057, 3305]
        )
    data = json.loads((output / "replace.json").read_text())
    # The refresh harness's after-update check deliberately constructs a plain
    # full-map validator. It is a counterfactual, not this trial's acceptance.
    data["summary"]["full_map_counterfactual_failures"] = data["summary"].pop("invalid_after_update")
    data["summary"]["rule"] = "full-map fit; last-accepted measured map samples reject; accepted-only refresh"
    data["summary"]["boundary_inside_px"] = 4
    data["summary"]["boundary_outside_px"] = 2
    for row in data["audit"]:
        if "after_update" in row["checks"]:
            row["checks"]["full_map_after_update_counterfactual"] = row["checks"].pop("after_update")
    data["reliability"] = audit
    (output / "replace.json").write_text(json.dumps(data))
    np.savez_compressed(output / "reliability.npz", **masks)
    print(json.dumps(data["summary"]), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.clip, args.out)
