"""Audit the production append-only and accepted-observation refresh policies.

One arm per fresh process. Both use the same small_gicp/LK settings and 4/2
pixel boundary cleanup. Checks use the frozen pre-update validator; map
snapshots are diagnostics, never acceptance evidence.
"""

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.flash_clip_replay import replay
from lerobot.policies.flash import experimental_surface_session as ess
from lerobot.policies.flash.experimental_surface_session import SurfaceStore
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
from lerobot.policies.flash.surface_refresh import RefreshingSurfaceStore


def run(clip, arm, output, capture_frames=()):
    output.mkdir(parents=True, exist_ok=True)
    update = ess.GrowingSurfaceSession.update
    store_class = RefreshingSurfaceStore if arm == "replace" else SurfaceStore
    audits, clouds, maintenance_ms = [], {}, []
    extend = store_class.extend

    def timed_extend(self, *args, **kwargs):
        start = time.perf_counter()
        result = extend(self, *args, **kwargs)
        maintenance_ms.append((time.perf_counter() - start) * 1000)
        return result

    def audited_update(self, prepared, time_s, **kwargs):
        before = self.store.points.copy()
        geometry = kwargs.get("geometry")
        frame = prepared["frame_id"]
        frozen = None
        if frame in capture_frames:
            frozen = {
                "map": before,
                "observation": prepared["current"],
                "intrinsics": prepared["intrinsics"],
                "shape_config": self.shape_config,
                "session_config": self.config,
                "reference_views": geometry.reference_views,
                "accepted_pose": self.accepted_pose.copy(),
                "accepted_frame": self.accepted_frame,
            }
        result = update(self, prepared, time_s, **kwargs)
        checks = {}
        if result["status"] == "observed_candidate":
            limits = {
                "max_outlier_fraction": self.config.uncertainty.max_visible_outlier_fraction,
                "agreement_m": self.config.agreement_m,
                "min_overlap": self.config.min_overlap,
            }
            checks["before_update"] = geometry.check_pose(
                np.asarray(result["camera_from_reference"]), **limits
            )
            assert checks["before_update"]["accepted"], f"Published pose failed unchanged-map check: {frame}"
            after = WholeSurfaceFit(
                self.store.points,
                prepared["current"],
                prepared["intrinsics"],
                self.shape_config,
                reference_views=geometry.reference_views,
            )
            checks["after_update"] = after.check_pose(np.asarray(result["camera_from_reference"]), **limits)
        elif result["status"] != "initialized":
            np.testing.assert_array_equal(self.store.points, before, err_msg=f"Rejected map changed: {frame}")
        details = result["diagnostics"]
        if frozen is not None:
            frozen["result"] = result
            with (output / f"{arm}-{frame}.pkl").open("wb") as stream:
                pickle.dump(frozen, stream)
        audits.append(
            {
                "frame": frame,
                "status": result["status"],
                "reason": details.get("reason"),
                "growth": details.get("growth"),
                "checks": checks,
                "fit": details.get("fit"),
                "prediction_geometry": details.get("prediction_geometry"),
                "kalman": details.get("kalman"),
                "rejected_pose": (details.get("fit") or {}).get("rejected_camera_from_reference"),
                "map_points_before": len(before),
                "map_points_after": len(self.store.points),
            }
        )
        clouds[f"{frame}_before"], clouds[f"{frame}_after"] = before, self.store.points.copy()
        return result

    with (
        patch.object(ess.GrowingSurfaceSession, "update", audited_update),
        patch.object(store_class, "extend", timed_extend),
    ):
        rows, summary = replay(
            clip,
            overrides=[
                f"surface_experiment.refresh_observed_surface={str(arm == 'replace').lower()}",
                'config.shape.registration.backend="small_gicp"',
                'shared_matching.backend="lk"',
                "config.observation.mask_margin_px=4",
                "config.observation.outer_mask_margin_px=2",
            ],
            quiet=True,
        )
    second = [r for r in rows if 2873 <= r["frame"] <= 3057]
    summary.update(
        arm=arm,
        second_loss_observed=sum(r["status"] == "observed_candidate" for r in second),
        second_loss_frames=len(second),
        refreshed_frames=sum(bool((r["growth"] or {}).get("refined_points")) for r in audits),
        retired_points=sum((r["growth"] or {}).get("retired_points", 0) for r in audits),
        invalid_after_update=[
            r["frame"] for r in audits if r["checks"] and not r["checks"]["after_update"]["accepted"]
        ],
        maintenance_ms_mean=float(np.mean(maintenance_ms)),
        maintenance_ms_p99=float(np.percentile(maintenance_ms, 99)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pose_sha256=hashlib.sha256(json.dumps([(r["status"], r["pose"]) for r in rows]).encode()).hexdigest(),
        scope="Full retained clip, recorded RGB/depth/masks; production surface policies; no pose labels.",
    )
    (output / f"{arm}.json").write_text(json.dumps({"summary": summary, "rows": rows, "audit": audits}))
    np.savez_compressed(output / f"{arm}-surfaces.npz", **clouds)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--arm", choices=("off", "replace"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-frame", type=int, action="append", default=[])
    args = parser.parse_args()
    run(args.clip, args.arm, args.out, args.capture_frame)
