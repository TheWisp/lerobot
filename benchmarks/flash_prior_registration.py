"""Opt-in native pose-prior experiment; never installed into the live pipeline.

All patches are process-local and undone on exit. The recorded consumer still
checks geometry and owns map publication. Native controls separate the prior
from the change from upstream LM to bounded Gauss-Newton. Covariance remains
the existing depth-normal approximation; these are NOT calibrated estimates.
"""

import argparse
import gc
import importlib
import importlib.util
import json
import subprocess
import sys
import sysconfig
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from benchmarks.flash_motion_recovery import full_recording, load_fixture, replay
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter, skew
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_registration import LibraryMatcher

UPSTREAM_COMMIT = "57c1106daf83c2c79ee0c58a9c7ed0032298ff4e"


def build(upstream, eigen, output):
    """Explicit offline build against pinned headers; no package mutation."""
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Expected small_gicp {UPSTREAM_COMMIT}, got {revision}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Build requires clean upstream headers")
    spec = importlib.util.find_spec("torch")
    pybind = Path(spec.origin).parent / "include"
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / ("flash_prior_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(
        [
            "c++",
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-fopenmp",
            f"-I{pybind}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{eigen}",
            f"-I{upstream / 'include'}",
            str(Path(__file__).with_suffix(".cpp")),
            "-o",
            str(artifact),
        ],
        check=True,
    )
    return artifact


def native_module(directory):
    sys.path.insert(0, str(directory))
    return importlib.import_module("flash_prior_native")


def cloud_arrays(cloud):
    return (
        np.asarray(cloud.points())[:, :3],
        np.asarray(cloud.normals())[:, :3],
        np.asarray(cloud.covs())[:, :3, :3].reshape(-1, 9),
    )


@contextmanager
def experiment(native, mode="integrated", strength=1.0, effective_points=32, diagnostics=None, captured=None):
    """Prior in both warm start and final fit, with unchanged per-level budget.

    GICP's regularized normal variance is 0.001 per endpoint. Scale its
    objective into an assumed metric noise model and cap independent evidence
    at effective_points. This scalar is an explicit ablation, not calibration.
    """
    active = [None]
    update_original = GrowingSurfaceSession.update
    refine_original = LibraryMatcher.refine
    correct_original = PoseKalmanFilter.correct

    def update(self, *args, **kwargs):
        old = active[0]
        active[0] = self
        try:
            return update_original(self, *args, **kwargs)
        finally:
            active[0] = old

    def refine(self, fitter, initial, *, admissible=None, max_evaluations=None):
        session = active[0]
        if session is None or session.kalman is None:
            return refine_original(
                self, fitter, initial, admissible=admissible, max_evaluations=max_evaluations
            )
        begin = time.perf_counter()
        state = session.kalman
        levels = self.prepare(fitter)
        c = self.config
        prior_info = np.linalg.inv(state.covariance)
        precision = prior_info * (strength if mode == "integrated" else 0.0)
        sigma_sq = state.config.independent_error_m**2 + state.config.correlated_error_m**2
        poses, calls, used = [initial.copy()], [], 0
        pose = initial.copy()
        for voxel, source, target, _, source_center, target_center in levels:
            limit = min(c.iterations_per_level, (max_evaluations or 50) - used)
            if limit <= 0:
                break
            prepared = native.Prepared(*cloud_arrays(target), *cloud_arrays(source))
            prediction = state.pose.copy()
            prediction[:3, 3] += prediction[:3, :3] @ source_center - target_center
            center = state.center_object - source_center
            # Effective sample cap avoids treating denser sampling as unlimited
            # independent evidence. Its effect is measured, not hidden.
            evidence_scale = min(1.0, effective_points / source.size()) / sigma_sq
            allocations = (("point", max(1, limit // 2)), ("gicp", limit - max(1, limit // 2)))
            for factor, budget in allocations:
                if budget <= 0:
                    continue
                local = pose.copy()
                local[:3, 3] += local[:3, :3] @ source_center - target_center
                result = prepared.align(
                    local,
                    prediction,
                    precision,
                    center,
                    evidence_scale * (0.002 if factor == "gicp" else 1.0),
                    voxel * c.correspondence_voxels,
                    budget,
                    c.threads,
                    c.translation_epsilon_m,
                    c.rotation_epsilon_rad,
                    factor,
                )
                if captured is not None and factor == "gicp":
                    captured.append(
                        (
                            prepared,
                            local.copy(),
                            prediction.copy(),
                            prior_info.copy(),
                            center.copy(),
                            evidence_scale * 0.002,
                            voxel * c.correspondence_voxels,
                            budget,
                            c.threads,
                        )
                    )
                used += result["iterations"]
                pose = np.asarray(result["pose"]).copy()
                pose[:3, 3] += target_center - pose[:3, :3] @ source_center
                poses.append(pose.copy())
                calls.append(
                    {
                        "factor": factor,
                        "budget": budget,
                        "iterations": result["iterations"],
                        "inliers": result["inliers"],
                        "converged": result["converged"],
                    }
                )
        self._local.proposals = poses
        fitter.batch_residuals(poses)
        detail = {
            "evaluations": used,
            "library": "prior_native_gn",
            "converged": bool(calls and calls[-1]["converged"]),
            "prior_mode": mode,
            "prior_strength": strength,
            "effective_points": effective_points,
            "native_calls": calls,
            "registration_ms": (time.perf_counter() - begin) * 1000,
        }
        if diagnostics is not None:
            diagnostics.append(detail)
        return pose, detail

    def correct(self, observation_pose, information, observation_pivot, *, contradicted=False):
        if mode == "post":
            return correct_original(
                self, observation_pose, information, observation_pivot, contradicted=contradicted
            )
        # The prior was already used in registration: no second mean correction.
        # Measurement-only information excludes the prior, preventing repeated
        # assimilation. Retain the established depth-normal covariance model;
        # do NOT promote the GICP Hessian to calibrated confidence.
        previous = self.snapshot()
        shift = self.center(observation_pose) - observation_pivot
        change = np.eye(6)
        change[3:, :3] = skew(shift)
        info = change.T @ information @ change
        prior_scale = strength if mode == "integrated" else 1.0
        self.covariance = np.linalg.inv(prior_scale * np.linalg.inv(self.covariance) + info)
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.pose = observation_pose.copy()
        return {
            **self.last_prediction,
            "accepted": True,
            "reason": "experimental_prior_in_fit",
            "filter_reason": "experimental_prior_in_fit",
            "observation_pose": observation_pose.tolist(),
            "previous_pose": previous["pose"].tolist(),
            "prior_covariance": previous["covariance"].tolist(),
            "posterior_covariance": self.covariance.tolist(),
            "calibrated": False,
            "prediction_failed_geometry": bool(contradicted),
            "prior_mode": mode,
        }

    with (
        patch.object(GrowingSurfaceSession, "update", update),
        patch.object(LibraryMatcher, "refine", refine),
        patch.object(PoseKalmanFilter, "correct", correct),
    ):
        yield


@contextmanager
def collection_timing():
    """Attribute occasional Python collection stalls; never disable collection."""
    original = LiveSurfaceSession.step
    events, started = [], [0.0]

    def callback(phase, info):
        if phase == "start":
            started[0] = time.perf_counter()
        else:
            events.append((info["generation"], 1000 * (time.perf_counter() - started[0])))

    def step(self, *args, **kwargs):
        begin = len(events)
        result = original(self, *args, **kwargs)
        details = result[0]["objects"]["object"]["group_growing_details"]
        details["benchmark_gc_ms"] = sum(elapsed for _, elapsed in events[begin:])
        details["benchmark_gc_generation2_ms"] = sum(
            elapsed for generation, elapsed in events[begin:] if generation == 2
        )
        return result

    gc.callbacks.append(callback)
    try:
        with patch.object(LiveSurfaceSession, "step", step):
            yield
    finally:
        gc.callbacks.remove(callback)


@collection_timing()
def compact(native, rounds=3):
    fixture = load_fixture()
    methods = [
        ("baseline", None, 1.0),
        ("native_post", "post", 1.0),
        ("native_fit_only", "none", 0.0),
        ("prior_0.1", "integrated", 0.1),
        ("prior_1", "integrated", 1.0),
        ("prior_10", "integrated", 10.0),
    ]
    rows = []
    for repeat in range(rounds + 1):
        for label, mode, strength in methods[:: 1 if repeat % 2 == 0 else -1]:
            for case in ("cube", "pointer", "ring"):
                diagnostics = []
                if mode is None:
                    results = replay(case, fixture=fixture)
                else:
                    with experiment(native, mode, strength, diagnostics=diagnostics):
                        results = replay(case, fixture=fixture)
                if repeat == 0:
                    continue
                for row in results[1:]:
                    fit = row["details"].get("fit", {})
                    center = (
                        None
                        if row["pose"] is None
                        else move(np.asarray(row["reference"]), np.asarray(row["pose"])).mean(0)
                    )
                    rows.append(
                        {
                            "repeat": repeat,
                            "method": label,
                            "case": case,
                            "frame": row["frame"],
                            "status": row["status"],
                            "reason": row["details"].get("reason"),
                            "elapsed_ms": row["elapsed_ms"],
                            "gc_ms": row["details"]["benchmark_gc_ms"],
                            "gc_generation2_ms": row["details"]["benchmark_gc_generation2_ms"],
                            "accepted_frame": row["accepted_frame"],
                            "map_version": row["map_version"],
                            "center_m": None if center is None else center.tolist(),
                            "pose": row["pose"],
                            "evaluations": fit.get("evaluations"),
                            "numerical_solves": fit.get("numerical_solves"),
                        }
                    )
    return rows


def controlled(native, count=10):
    """Known-pose plane and narrow strip; no ground-truth pairs enter the solver."""
    rows = []
    for seed in range(count):
        rng = np.random.default_rng(seed)
        for scenario, width, angle in (
            ("strip", 0.002, 0),
            ("broad", 0.06, 0),
            ("moved", 0.06, 20),
            ("occluded_strip", 0.002, 0),
        ):
            x, y = np.meshgrid(np.linspace(-0.05, 0.05, 51), np.linspace(-width / 2, width / 2, 9))
            points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
            truth = np.eye(4)
            truth[:3, :3] = Rotation.from_euler("x", angle, degrees=True).as_matrix()
            truth[2, 3] = 0.004
            target = move(points, truth) + rng.normal(0, 0.0005, points.shape)
            normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
            cov = np.tile(np.diag([1.0, 1.0, 0.001]).reshape(1, 9), (len(points), 1))
            # Noise in observed normals can invent weak rotation information.
            observed_normals = normals @ truth[:3, :3].T + rng.normal(0, 0.03, normals.shape)
            observed_normals /= np.linalg.norm(observed_normals, axis=1)[:, None]
            observed_cov = np.asarray([np.eye(3) - 0.999 * np.outer(n, n) for n in observed_normals]).reshape(
                -1, 9
            )
            if scenario == "occluded_strip":
                sx, sy = np.meshgrid(np.linspace(-0.05, 0.05, 51), np.linspace(-0.03, 0.03, 31))
                points = np.column_stack((sx.ravel(), sy.ravel(), np.zeros(sx.size)))
                normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
                cov = np.tile(np.diag([1.0, 1.0, 0.001]).reshape(1, 9), (len(points), 1))
            prepared = native.Prepared(target, observed_normals, observed_cov, points, normals, cov)
            prior = np.eye(4)
            initial = np.eye(4)
            initial[:3, :3] = Rotation.from_euler("x", angle + 12, degrees=True).as_matrix()
            covariance = np.diag([np.deg2rad(3) ** 2] * 3 + [0.01**2] * 3)
            for factor in ("plane", "gicp"):
                scale = (0.002 if factor == "gicp" else 1.0) * 32 / len(points) / 0.002**2
                for strength in (0.0, 0.1, 1.0, 10.0):
                    start = time.perf_counter()
                    result = prepared.align(
                        initial,
                        prior,
                        np.linalg.inv(covariance) * strength,
                        np.zeros(3),
                        scale,
                        0.03,
                        20,
                        2,
                        0.0001,
                        0.001,
                        factor,
                    )
                    elapsed = (time.perf_counter() - start) * 1000
                    pose = np.asarray(result["pose"])
                    rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "width_m": width,
                            "factor": factor,
                            "strength": strength,
                            "rotation_error_deg": np.rad2deg(
                                Rotation.from_matrix(pose[:3, :3] @ truth[:3, :3].T).magnitude()
                            ),
                            "position_error_mm": 1000 * np.linalg.norm(pose[:3, 3] - truth[:3, 3]),
                            "depth_correction_mm": 1000 * pose[2, 3],
                            "elapsed_ms": elapsed,
                            "iterations": result["iterations"],
                        }
                    )
    return rows


def fixed_input_latency(native, rounds=20):
    """Alternate prior on/off using identical captured clouds, poses and budgets."""
    captured = []
    with experiment(native, captured=captured):
        replay("ring")
    rows = []
    for repeat in range(rounds + 2):
        for index, (
            prepared,
            initial,
            prediction,
            precision,
            center,
            scale,
            distance,
            budget,
            threads,
        ) in enumerate(captured):
            for enabled in (False, True) if repeat % 2 else (True, False):
                start = time.perf_counter()
                # Zero stopping tolerances force exactly the same iteration cap.
                result = prepared.align(
                    initial,
                    prediction,
                    precision if enabled else np.zeros((6, 6)),
                    center,
                    scale,
                    distance,
                    budget,
                    threads,
                    0.0,
                    0.0,
                    "gicp",
                )
                elapsed = (time.perf_counter() - start) * 1000
                if repeat >= 2:
                    rows.append(
                        {
                            "repeat": repeat,
                            "input": index,
                            "prior": enabled,
                            "iterations": result["iterations"],
                            "budget": budget,
                            "elapsed_ms": elapsed,
                        }
                    )
    return rows


def negative_and_missing(native):
    from copy import deepcopy

    meta, frames = load_fixture()
    results = {}
    for strength in (0.1, 1.0, 10.0):
        changed = deepcopy(meta)
        changed["cases"]["wrong_object"] = [2735, 4759, 4765]
        with experiment(native, strength=strength):
            runs = {
                "wrong_object": replay("wrong_object", fixture=(changed, frames)),
                **{
                    case: replay(case, fixture=(meta, frames), missing=True)
                    for case in ("cube", "pointer", "ring")
                },
            }
        results[str(strength)] = {
            name: [{k: r[k] for k in ("frame", "status", "accepted_frame", "map_version")} for r in rows]
            for name, rows in runs.items()
        }
    return results


def full_history(native, recording, methods):
    fixture = full_recording(recording)
    rows = []
    options = {
        "baseline": (None, 1.0),
        "native_post": ("post", 1.0),
        "prior_1": ("integrated", 1.0),
        "prior_0.1": ("integrated", 0.1),
    }
    for label in methods:
        mode, strength = options[label]
        for case in fixture[0]["cases"]:
            if mode is None:
                results = replay(case, fixture=fixture)
            else:
                with experiment(native, mode=mode, strength=strength):
                    results = replay(case, fixture=fixture)
            for row in results[1:]:
                fit = row["details"].get("fit", {})
                rows.append(
                    {
                        "method": label,
                        "case": case,
                        "frame": row["frame"],
                        "status": row["status"],
                        "reason": row["details"].get("reason"),
                        "elapsed_ms": row["elapsed_ms"],
                        "accepted_frame": row["accepted_frame"],
                        "map_version": row["map_version"],
                        "pose": row["pose"],
                        "evaluations": fit.get("evaluations"),
                        "numerical_solves": fit.get("numerical_solves"),
                    }
                )
            print(
                json.dumps(
                    {
                        "method": label,
                        "case": case,
                        "updates": len(results) - 1,
                        "observed": sum(r["status"] == "observed_candidate" for r in results[1:]),
                    }
                ),
                flush=True,
            )
    return rows


def write_history(path, report):
    """One JSON row per update keeps pose traces compact and individually diffable."""
    header = json.dumps({key: value for key, value in report.items() if key != "recorded"})[:-1]
    rows = ",\n".join(json.dumps(row, separators=(",", ":")) for row in report["recorded"])
    path.write_text(header + ', "recorded": [\n' + rows + "\n]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--eigen", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--full-recording", type=Path)
    parser.add_argument(
        "--full-methods", nargs="+", default=["baseline", "native_post", "prior_1", "prior_0.1"]
    )
    args = parser.parse_args()
    if args.build:
        print(build(args.upstream, args.eigen, args.native_dir), flush=True)
        return
    cv2.setNumThreads(1)
    native = native_module(args.native_dir)
    if args.full_recording is not None:
        report = {
            "upstream_commit": UPSTREAM_COMMIT,
            "scope": "Full-history pose availability, not ground-truth accuracy.",
            "recorded": full_history(native, args.full_recording, args.full_methods),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_history(args.output, report)
        return
    report = {
        "upstream_commit": UPSTREAM_COMMIT,
        "scope": "Experimental GN registration; unchanged geometry rejection. Recorded availability is not pose accuracy.",
        "controlled": controlled(native, args.controls),
        "recorded": compact(native, args.rounds),
        "negative_and_missing": negative_and_missing(native),
        "fixed_input_latency": fixed_input_latency(native),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
