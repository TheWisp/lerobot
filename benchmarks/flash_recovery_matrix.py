"""Compare recovery rules before choosing one; no production defaults change.

Use --repeats for fresh timing, not cached results. Measurements cover CPU RGB-D
preparation/fitting/filtering, excluding segmentation and image tracking. The
measured-marker control supplies exact group points at the public interface.
"""

import json
import os
import platform
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from benchmarks.flash_kalman_recovery_control import HIDDEN, MODES, WARMUP, run_scene, scene
from benchmarks.flash_recovery_fixture import run_mug
from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    SurfaceStore,
)
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import SurfacePoseUncertainty


def runtime_environment():
    import cv2
    import scipy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "thread_environment": {
            k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def scenarios():
    corrupted = scene(0, occluded=False)
    # A depth artifact while the true rigid object remains stationary. The
    # following clear frames test recovery and preservation of the stored shape.
    for frame in corrupted[WARMUP : WARMUP + HIDDEN]:
        offset = 0.04 * np.sin(np.linspace(0, 8 * np.pi, frame["depth"].shape[1]))
        frame["depth"][frame["mask"]] += np.broadcast_to(offset, frame["depth"].shape)[frame["mask"]]
    return {
        "hidden_motion": scene(),
        "hidden_stationary": scene(0),
        "visible_motion": scene(occluded=False),
        "partial_stationary": scene(0, strip_px=12),
        "group_and_camera": scene(camera_displacement_m=-0.02, group_motion=True),
        "corrupted_then_clear": corrupted,
    }


class WorkProbe:
    """Measure real method calls, without replacing their computation or result.

    Stage times are inclusive and must not be summed: uncertainty calculation
    may call the geometry residual too. Use prepare_ms + update_ms for total.
    """

    def __enter__(self):
        self.calls, self.ms, self.max_refine_evaluations = defaultdict(int), defaultdict(float), 0
        self.stack = ExitStack()
        for cls, method, name in [
            (WholeSurfaceFit, "refine", "fit_refinement"),
            (SurfacePoseUncertainty, "__init__", "observation_uncertainty"),
            (PoseKalmanFilter, "predict", "kalman_predict"),
            (PoseKalmanFilter, "correct", "kalman_correct"),
            (SurfaceStore, "extend", "surface_growth"),
        ]:
            self.stack.enter_context(patch.object(cls, method, self._wrap(getattr(cls, method), name)))
        return self

    def _wrap(self, original, name):
        @wraps(original)
        def measured(*args, **kwargs):
            start = perf_counter()
            self.calls[name] += 1
            try:
                result = original(*args, **kwargs)
                if name == "fit_refinement":
                    self.max_refine_evaluations = max(self.max_refine_evaluations, result[1]["evaluations"])
                return result
            finally:
                self.ms[name] += (perf_counter() - start) * 1000

        return measured

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def result(self):
        return {
            "calls": dict(self.calls),
            "inclusive_ms": dict(self.ms),
            "max_refine_evaluations": self.max_refine_evaluations,
        }


def summary(result):
    rows = result["rows"]
    visible = [r for r in rows if not r["hidden"]]
    available = [r for r in visible if r["available"]]
    good = [
        r
        for r in available
        if r["error"]["median_point_error_mm"] < 5 and r["error"]["rotation_error_deg"] < 1
    ]
    after = rows[WARMUP + HIDDEN :]
    recovered = next(
        (
            r
            for r in after
            if r["available"]
            and r["error"]["median_point_error_mm"] < 5
            and r["error"]["rotation_error_deg"] < 1
        ),
        None,
    )
    times = [r["prepare_ms"] + r["update_ms"] for r in rows[1:]]
    return {
        "visible_frames": len(visible),
        "visible_available": len(available),
        "visible_accurate": len(good),
        "visible_wrong_accepted": len(available) - len(good),
        "recovery_delay_s": recovered["time_s"] - after[0]["time_s"] if recovered else None,
        "final_error": rows[-1]["error"],
        "final_available": rows[-1]["available"],
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "maximum_ms": float(max(times)),
    }


def render_visible_motion(output):
    """Show real rendered input and published estimates, not just a score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = scenarios()["visible_motion"]
    first = frames[0]
    prepared = SurfaceTrackingSession(None).prepare(
        first["rgb"], first["depth"], first["mask"], first["intrinsics"], first["index"]
    )
    center = prepared["current"]["obs"]["points"].mean(0)

    def displacement(pose):
        # Pose translation alone moves the reference camera origin, not the
        # object. Evaluate the same fixed point on its initial observed surface.
        pose = np.asarray(pose)
        return (pose[:3, :3] @ center + pose[:3, 3] - center)[0] * 1000

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])
    for col, index in enumerate((5, 11, 25)):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(frames[index]["rgb"])
        ax.set_title(f"Frame {index}: object remains visible", fontsize=11)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    ax.plot(
        [f["time_s"] for f in frames],
        [displacement(f["truth"]) for f in frames],
        color="#333333",
        linestyle="--",
        label="Actual movement",
    )
    for mode, color, label in (
        ("current_reset", "#df8b22", "Current reset"),
        ("strict", "#3677b2", "No reset, current uncertainty"),
        ("strict_broader_motion", "#16876d", "No reset, broader uncertainty"),
    ):
        result = json.loads((output / f"visible_motion-{mode}-0.json").read_text())
        ax.plot(
            [r["time_s"] for r in result["rows"]],
            [displacement(r["pose"]) if r["available"] else np.nan for r in result["rows"]],
            color=color,
            label=label,
            linewidth=2.2,
        )
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Movement to the right (mm)",
        title="No reset + broader uncertainty follows motion; the tight filter stops returning a pose",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(output / "visible-motion.png", dpi=160)
    plt.close(fig)


def run_matrix(output, repeats=1):
    output.mkdir(parents=True, exist_ok=True)
    # Warm the shared numeric paths once; do not count startup/import time.
    run_scene(scene(0, occluded=False)[:2], "current_reset")
    reports = []
    for repeat in range(repeats):
        # Rotate order to avoid always timing a particular rule first.
        for mode in MODES[repeat % len(MODES) :] + MODES[: repeat % len(MODES)]:
            for name, frames in scenarios().items():
                with WorkProbe() as probe:
                    # Match the original recovery experiment; separately enable
                    # growth on the corrupt-observation control and real mug.
                    result = run_scene(frames, mode, grow=name == "corrupted_then_clear")
                (output / f"{name}-{mode}-{repeat}.json").write_text(json.dumps(result) + "\n")
                report = {
                    "case": name,
                    "mode": mode,
                    "repeat": repeat,
                    **summary(result),
                    "work": probe.result(),
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
            with WorkProbe() as probe:
                result = run_mug(mode)
            (output / f"mug-{mode}-{repeat}.json").write_text(json.dumps(result) + "\n")
            report = {
                "case": "mug_checkpoint",
                "mode": mode,
                "repeat": repeat,
                **{k: v for k, v in result.items() if k not in {"diagnostics", "pose"}},
                "work": probe.result(),
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "results": reports,
                "environment": runtime_environment(),
                "rendered_accuracy": {"median_point_error_mm": 5, "rotation_error_deg": 1},
                "limitations": [
                    "No SAM or UFM runtime included",
                    "Exact synthetic supporting marker inputs",
                    "Real mug uses recorded pre-update checkpoint",
                    "Inclusive stage times overlap",
                    "BLAS/OpenMP thread settings recorded; fitter uses configured search workers; sequential repeats",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    render_visible_motion(output)


def measure_hidden_scaling(object_count, frames=600):
    """Prediction-only latency/RAM control in a fresh process, without a robot."""
    import resource

    import psutil

    from lerobot.policies.flash.surface_grounding import ShapeConfig

    process = psutil.Process(os.getpid())
    inputs = scene(0)
    first, hidden = inputs[0], inputs[WARMUP]

    def prepare(frame):
        return SurfaceTrackingSession(None).prepare(
            frame["rgb"], frame["depth"], frame["mask"], frame["intrinsics"], frame["index"]
        )

    initial, missing = prepare(first), prepare(hidden)
    sessions = [
        GrowingSurfaceSession(
            ShapeConfig(), GrowingSurfaceConfig(partial_surface_matching=True, retain_unobserved_pose=True)
        )
        for _ in range(object_count)
    ]
    for session in sessions:
        session.update(initial, 0.0)
    rss_before = process.memory_info().rss
    times = []
    with WorkProbe() as probe:
        for i in range(1, frames + 1):
            start = perf_counter()
            for session in sessions:
                session.update({**missing, "frame_id": i}, i / 30)
            times.append((perf_counter() - start) * 1000)
    return {
        "objects": object_count,
        "environment": runtime_environment(),
        "frames": frames,
        "scope": "Preprocessed empty masks -> pose prediction; excludes segmentation, matching and fitting",
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": process.memory_info().rss,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "map_points": [len(s.store.points) for s in sessions],
        "patches": [len(s.store.patches) for s in sessions],
        "work": probe.result(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--hidden-objects", type=int, choices=(1, 3))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.hidden_objects:
        args.output.mkdir(parents=True, exist_ok=True)
        result = measure_hidden_scaling(args.hidden_objects)
        (args.output / f"hidden-scaling-{args.hidden_objects}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result))
    else:
        run_matrix(args.output, args.repeats)
