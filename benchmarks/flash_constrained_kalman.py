"""Shadow experiments for geometry-constrained Kalman corrections.

No tracking state, covariance, production setting or registration solver changes.
Compare proposals on IDENTICAL incoming fits and unconstrained Kalman results.
The local QP uses sampled, fit-relative bounds; it is not an exact translation
of the spatial/outlier acceptance policy. Every answer needs the full validator.
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from benchmarks.flash_motion_recovery import full_recording, load_fixture, replay
from benchmarks.flash_partial_surface_control import pose_error
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession, ImageConditionedFit
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_motion import pose_with_delta
from lerobot.policies.flash.surface_observation import move, spatial_sample


def pose_difference(origin, target, center_object):
    """Camera-axis rotation radians, then fixed object point movement in metres."""
    return np.r_[
        Rotation.from_matrix(target[:3, :3] @ origin[:3, :3].T).as_rotvec(),
        move(center_object[None], target)[0] - move(center_object[None], origin)[0],
    ]


class GeometryBounds:
    """Sampled local surrogate; exact visibility decisions remain external.

    Nearest-surface and outline errors get the existing metric tolerance. Empty
    space gets zero additional violation. Existing errors at the valid fit are
    allowed but may not worsen. This preserves an initially feasible point without
    pretending the original validator requires every sampled point to match.
    """

    def __init__(self, geometry, fitted, samples=256):
        self.full = geometry
        self.validator = getattr(geometry, "_latest_rejection", (None, geometry))[1]
        self.directions = []
        if self.validator is not None:
            # First direction is map to current; others current to retained.
            self.directions = [(self.validator, None)] + [
                (fit, camera) for camera, fit in self.validator.reverse_fits
            ]
        self.samples = [f.reference[spatial_sample(f.reference, samples)] for f, _ in self.directions]
        points = geometry.observation["obs"]["points"]
        self.current = points[spatial_sample(points, samples)]
        self.calls = 0
        self.elapsed_ms = 0.0
        initial, floors = self.features(fitted, floors=True)
        self.bounds = np.maximum(initial, floors)

    def features(self, pose, *, floors=False):
        start = perf_counter()
        inverse = np.linalg.inv(pose)
        values, tolerances = [], []
        for (fit, camera), samples in zip(self.directions, self.samples, strict=True):
            predicted = move(samples, pose if camera is None else camera @ inverse)
            visible, empty, silhouette = fit._projection_evidence(predicted)
            distance = fit.target_tree.query(predicted)[0]
            free, _, _ = fit._scene_depth_evidence(predicted)
            if camera is not None:
                # Same retained-view exception as the actual scene-depth veto.
                local = move(samples, inverse)
                retained_distance = np.minimum(self.validator.reference_tree.query(local)[0], distance)
                free[retained_distance <= self.validator.config.depth_tolerance_m] = 0
            tolerance = fit.config.depth_tolerance_m
            values.extend(
                (
                    np.sqrt(visible) * distance / tolerance,
                    np.sqrt(empty) * silhouette / tolerance,
                    free / tolerance,
                )
            )
            tolerances.extend((np.ones(len(samples)), np.ones(len(samples)), np.zeros(len(samples))))
        # Current points must retain positive support in the whole stored map.
        distance = self.full.reference_tree.query(move(self.current, inverse))[0]
        values.append(distance / self.full.config.depth_tolerance_m)
        tolerances.append(np.ones(len(distance)))
        result = np.concatenate(values)
        self.calls += 1
        self.elapsed_ms += (perf_counter() - start) * 1000
        return (result, np.concatenate(tolerances)) if floors else result


def constrained_proposal(fitted, fused, covariance, center_object, geometry, check, *, rounds=1, samples=256):
    """At most `rounds` linearizations and 20 QP iterations per linearization.

    Returns a shadow pose or None. Does not supply a calibrated constrained
    covariance and must not be inserted into the live estimator as-is.
    """
    begin = perf_counter()
    exact_checks = 1
    fused_check = check(fused)
    common_ms = (perf_counter() - begin) * 1000
    detail = {
        "fused_accepted": bool(fused_check["accepted"]),
        "common_check_ms": common_ms,
        "rounds": 0,
        "qp_iterations": 0,
        "feature_calls": 0,
        "feature_ms": 0.0,
        "qp_ms": 0.0,
    }

    def finish(pose, reason):
        detail.update(reason=reason, exact_checks=exact_checks, total_ms=(perf_counter() - begin) * 1000)
        detail["additional_ms"] = detail["total_ms"] - common_ms
        return pose, detail

    if fused_check["accepted"]:
        return finish(fused.copy(), "already_compatible")
    exact_checks += 1
    if not check(fitted)["accepted"]:
        return finish(None, "no_compatible_fit")
    bounds = GeometryBounds(geometry, fitted, samples)
    pivot = move(center_object[None], fitted)[0]
    target = pose_difference(fitted, fused, center_object)
    # Local covariance weighting in camera axes; no covariance update is claimed.
    scale = np.linalg.cholesky(covariance)
    target_y = np.linalg.solve(scale, target)
    x = np.zeros(6)
    epsilon = np.r_[[1e-4] * 3, [1e-5] * 3]
    limits = np.r_[[np.deg2rad(45)] * 3, [0.05] * 3]
    for iteration in range(rounds):
        base = bounds.features(pose_with_delta(fitted, pivot, x))
        jac = np.column_stack(
            [
                (bounds.features(pose_with_delta(fitted, pivot, x + np.eye(6)[axis] * epsilon[axis])) - base)
                / epsilon[axis]
                for axis in range(6)
            ]
        )
        a = jac @ scale
        b = bounds.bounds - base + jac @ x
        # Keep constraints nearest their boundary. Zero-gradient violations
        # cannot be repaired by this local approximation.
        active = np.linalg.norm(a, axis=1) > 1e-10
        if np.any(~active & (b < -1e-7)):
            detail["local_constant_violation"] = True
            break
        a, b = a[active], b[active]
        a = np.vstack((a, scale, -scale))
        b = np.r_[b, limits, limits]
        started = perf_counter()
        solution = minimize(
            lambda y: 0.5 * float((y - target_y) @ (y - target_y)),
            np.linalg.solve(scale, x),
            jac=lambda y: y - target_y,
            method="SLSQP",
            constraints={"type": "ineq", "fun": lambda y, a=a, b=b: b - a @ y, "jac": lambda y, a=a: -a},
            options={"maxiter": 20, "ftol": 1e-8},
        )
        detail["qp_ms"] += (perf_counter() - started) * 1000
        detail["qp_iterations"] += int(solution.nit)
        detail["rounds"] = iteration + 1
        detail["qp_success"] = bool(solution.success)
        detail["qp_message"] = str(solution.message)
        x = scale @ solution.x
        candidate = pose_with_delta(fitted, pivot, x)
        exact_checks += 1
        final_check = check(candidate)
        accepted = final_check["accepted"]
        detail["last_geometry"] = final_check
        detail["linear_constraint_max_violation"] = float(np.max(a @ solution.x - b))
        detail.update(
            feature_calls=bounds.calls,
            feature_ms=bounds.elapsed_ms,
            constraints=len(b),
            sampled_points=sum(map(len, bounds.samples)) + len(bounds.current),
        )
        if accepted and solution.success and detail["linear_constraint_max_violation"] <= 1e-6:
            detail["weighted_distance_squared"] = float((solution.x - target_y) @ (solution.x - target_y))
            detail["fitted_distance_squared"] = float(target_y @ target_y)
            return finish(candidate, "projected_compatible")
    detail.update(feature_calls=bounds.calls, feature_ms=bounds.elapsed_ms)
    return finish(None, "bounded_projection_failed")


def segment_control(fitted, fused, center_object, check, *, steps=6):
    """Feasibility control on one path only; not a six-dimensional projection.

    Nonconvex geometry means bisection need not find the closest valid interval.
    This intentionally weak control exposes what accepting a nearby fit buys.
    """
    begin = perf_counter()
    common = check(fused)
    common_ms = (perf_counter() - begin) * 1000
    if common["accepted"]:
        return fused.copy(), {
            "reason": "already_compatible",
            "additional_ms": (perf_counter() - begin) * 1000 - common_ms,
            "exact_checks": 1,
        }
    if not check(fitted)["accepted"]:
        return None, {
            "reason": "no_compatible_fit",
            "additional_ms": (perf_counter() - begin) * 1000 - common_ms,
            "exact_checks": 2,
        }
    delta = pose_difference(fitted, fused, center_object)
    pivot = move(center_object[None], fitted)[0]
    low, high, best = 0.0, 1.0, fitted.copy()
    for _ in range(steps):
        fraction = (low + high) / 2
        candidate = pose_with_delta(fitted, pivot, fraction * delta)
        if check(candidate)["accepted"]:
            low, best = fraction, candidate
        else:
            high = fraction
    return best, {
        "reason": "segment_compatible",
        "fraction": low,
        "additional_ms": (perf_counter() - begin) * 1000 - common_ms,
        "exact_checks": steps + 2,
    }


@contextmanager
def shadow_experiment(*, repeats=1, methods=("qp1", "qp3", "segment"), truth=None, capture=None):
    """Observe actual pipeline boundaries, restore patches even on exceptions."""
    original_fit = ImageConditionedFit.correct_surface_first
    original_correct = PoseKalmanFilter.correct
    original_update = GrowingSurfaceSession.update
    current = {}
    records = []

    def fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        current["geometry"] = self.geometry
        return result

    def update(self, prepared, *args, **kwargs):
        current.clear()
        current.update(session=self, frame=prepared["frame_id"])
        return original_update(self, prepared, *args, **kwargs)

    def correct(self, fitted, *args, **kwargs):
        result = original_correct(self, fitted, *args, **kwargs)
        if not result["accepted"]:
            return result
        geometry, session = current["geometry"], current["session"]
        fused = self.pose.copy()
        covariance = self.covariance.copy()

        def check(pose):
            return geometry.check_pose(
                pose,
                max_outlier_fraction=session.config.uncertainty.max_visible_outlier_fraction,
                agreement_m=session.config.agreement_m,
                min_overlap=session.config.min_overlap,
            )

        row = {
            "frame": current["frame"],
            "fitted": fitted.tolist(),
            "fused": fused.tolist(),
            "center_object": self.center_object.tolist(),
            "fitted_check": check(fitted),
            "fused_check": check(fused),
            "methods": {},
        }
        if truth is not None and current["frame"] in truth:
            row["fitted_error"] = pose_error(session.reference, fitted, truth[current["frame"]])
            row["fused_error"] = pose_error(session.reference, fused, truth[current["frame"]])
        for repeat in range(repeats):
            for name in methods[:: 1 if repeat % 2 == 0 else -1]:
                if name == "segment":
                    proposed, detail = segment_control(fitted, fused, self.center_object, check)
                else:
                    proposed, detail = constrained_proposal(
                        fitted, fused, covariance, self.center_object, geometry, check, rounds=int(name[-1])
                    )
                detail["pose"] = None if proposed is None else proposed.tolist()
                detail["geometry"] = None if proposed is None else check(proposed)
                if proposed is not None:
                    assert detail["geometry"]["accepted"]
                    if truth is not None and current["frame"] in truth:
                        detail["error"] = pose_error(session.reference, proposed, truth[current["frame"]])
                row["methods"].setdefault(name, []).append(detail)
        np.testing.assert_array_equal(self.pose, fused)
        np.testing.assert_array_equal(self.covariance, covariance)
        if capture is not None:
            capture(geometry, fitted, fused, covariance, self.center_object, session.config, current["frame"])
        records.append(row)
        return result

    with (
        patch.object(ImageConditionedFit, "correct_surface_first", fit),
        patch.object(GrowingSurfaceSession, "update", update),
        patch.object(PoseKalmanFilter, "correct", correct),
    ):
        yield records


def distribution(values):
    values = list(values)
    return {
        "count": len(values),
        **(
            {
                k: float(v)
                for k, v in zip(
                    ("median", "p95", "max"),
                    (np.median(values), np.percentile(values, 95), max(values)),
                    strict=True,
                )
            }
            if values
            else {}
        ),
    }


def summary(records):
    rejected = [r for r in records if not r["fused_check"]["accepted"]]
    result = {"corrections": len(records), "contradictory_fused": len(rejected), "methods": {}}
    for name in ("qp1", "qp3", "segment"):
        trials = [t for r in rejected for t in r["methods"].get(name, [])]
        fast = [t for r in records if r["fused_check"]["accepted"] for t in r["methods"].get(name, [])]
        result["methods"][name] = {
            "recovered": sum(r["methods"][name][0]["pose"] is not None for r in rejected),
            "conflict_additional_ms": distribution(t["additional_ms"] for t in trials),
            "compatible_additional_ms": distribution(t["additional_ms"] for t in fast),
            "feature_ms": distribution(t.get("feature_ms", 0) for t in trials),
            "qp_ms": distribution(t.get("qp_ms", 0) for t in trials),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--full-recording", type=Path)
    parser.add_argument("--capture-frame", type=int)
    parser.add_argument("--capture-path", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    fixture = full_recording(args.full_recording) if args.full_recording else load_fixture()
    output = {}

    def capture(*inputs):
        if inputs[-1] == args.capture_frame:
            from benchmarks.flash_constraint_checkpoint import save

            save(args.capture_path, *inputs)

    for case in fixture[0]["cases"]:
        with shadow_experiment(
            repeats=args.repeats, capture=capture if args.capture_frame else None
        ) as records:
            rows = replay(case, fixture=fixture)
        output[case] = {
            "summary": summary(records),
            "records": records,
            "baseline": [{k: r[k] for k in ("frame", "status", "elapsed_ms")} for r in rows],
        }
        print(case, json.dumps(output[case]["summary"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
