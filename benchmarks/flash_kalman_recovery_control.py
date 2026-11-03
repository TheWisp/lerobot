"""Rendered RGB-D recovery control, using the actual surface session and Kalman.

No robot, SAM model or neural matcher is needed. An exact visible object mask
isolates fitting/filtering from segmentation. The fitter receives only measured
depth, mask and camera intrinsics, never box dimensions or its true pose.
Three fixed settings are compared; the broader process noise is an ablation,
not a new default. No tracking configuration is selected by the scene label.
"""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from benchmarks.flash_partial_surface_control import box_observation, pose_error
from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    PoseHint,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_kalman import MeasuredGroupMotion, PoseKalmanFilter
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig

FPS = 30
WARMUP = 6
HIDDEN = 12
AFTER = 8
MODES = ("current_reset", "strict", "strict_broader_motion")


class StrictPoseKalmanFilter(PoseKalmanFilter):
    """Experiment-only removal of the reset; all Kalman calculations unchanged."""

    def correct(self, observation_pose, information, observation_pivot, *, contradicted=False):
        result = super().correct(observation_pose, information, observation_pivot, contradicted=False)
        # The session's later geometry check can overwrite `reason`; preserve
        # the filter decision separately so rejection is not called fusion.
        result["filter_reason"] = result["reason"]
        result["prediction_failed_geometry"] = bool(contradicted)
        return result


def scene(
    displacement_m=0.04, *, occluded=True, camera_displacement_m=0.0, group_motion=False, strip_px=None
):
    initial = box_observation(20)[2]
    frames = []
    for i in range(WARMUP + HIDDEN + AFTER):
        progress = float(np.clip((i - WARMUP + 1) / HIDDEN, 0, 1))
        # Smooth physical movement during the hidden interval, not teleportation.
        progress = progress * progress * (3 - 2 * progress)
        current, k, pose = box_observation(
            20, displacement_m=[(displacement_m - camera_displacement_m) * progress, 0, 0]
        )
        depth, mask = current["depth"].copy(), current["mask"].copy()
        rgb = np.full((*mask.shape, 3), (37, 45, 56), dtype=np.uint8)
        shade = np.clip((0.77 - depth) / 0.15, 0, 1)
        rgb[mask] = np.column_stack(
            (90 + 90 * shade[mask], 120 + 85 * shade[mask], 140 + 85 * shade[mask])
        ).astype(np.uint8)
        hidden = bool(occluded and WARMUP <= i < WARMUP + HIDDEN)
        if hidden:
            # A nearer rectangular screen hides the whole object. No hidden
            # object points or motion are provided to the tracking pipeline.
            screen = np.zeros_like(mask)
            screen[20:160, 5:235] = True
            if strip_px is not None:
                left = np.where(mask)[1].min()
                screen[:, : left + strip_px] = False
            depth[screen] = 0.5
            rgb[screen] = (155, 122, 73)
            mask[screen] = False
        camera = np.eye(4)
        camera[0, 3] = camera_displacement_m * progress
        frames.append(
            {
                "index": i,
                "time_s": i / FPS,
                "rgb": rgb,
                "depth": depth,
                "mask": mask,
                "intrinsics": k,
                "truth": pose @ np.linalg.inv(initial),
                "hidden": hidden and not mask.any(),
                "base_from_camera": camera,
            }
        )
        if group_motion:
            # Ideal measured supporting markers, supplied at the public motion
            # interface. This tests fusion, not a neural 2D tracker or membership.
            support = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.06, 0.06) for z in (0.6, 0.7)])
            frames[-1]["support_points"] = support + [displacement_m * progress, 0, 0]
    return frames


def run_scene(frames, mode, *, grow=False, config=None):
    if mode not in MODES:
        raise ValueError(mode)
    uncertainty = PoseUncertaintyConfig(enabled=True, mode="kalman", max_visible_outlier_fraction=0.25)
    if mode == "strict_broader_motion":
        uncertainty = replace(uncertainty, position_process_std_m_sqrt_s=0.03)
    shape = ShapeConfig()
    config = config or GrowingSurfaceConfig(
        grow=grow, partial_surface_matching=True, retain_unobserved_pose=True
    )
    config = replace(config, uncertainty=uncertainty)
    session = GrowingSurfaceSession(shape, config)
    rows = []
    for frame in frames:
        started = perf_counter()
        # A fresh observation builder deliberately supplies no image matches.
        # The GrowingSurfaceSession still retains the real temporal pose/map.
        preparation = SurfaceTrackingSession(None, shape_config=shape)
        prepared = preparation.prepare(
            frame["rgb"], frame["depth"], frame["mask"], frame["intrinsics"], frame["index"]
        )
        prepare_ms = (perf_counter() - started) * 1000
        hint = None
        if frame["index"] and "support_points" in frame:
            previous = frames[frame["index"] - 1]
            hint = PoseHint(
                session.pose,
                "synthetic measured markers",
                frame["time_s"],
                frame["time_s"],
                session.reference_frame,
                motion=MeasuredGroupMotion(
                    previous["support_points"],
                    frame["support_points"],
                    previous["time_s"],
                    frame["time_s"],
                    "synthetic measured markers",
                ),
            )
        started = perf_counter()
        result = session.update(
            prepared, frame["time_s"], hint=hint, base_from_camera=frame.get("base_from_camera")
        )
        update_ms = (perf_counter() - started) * 1000
        if frame["index"] == 0 and mode != "current_reset":
            old = session.kalman
            strict = StrictPoseKalmanFilter(old.pose, old.center_object, old.time_s, old.config)
            strict.covariance = old.covariance.copy()
            session.kalman = strict
        details = result["diagnostics"]
        candidate = (
            details.get("fit", {}).get("uncertainty", {}).get("camera_from_reference_at_linearization")
        )
        row = {
            "index": frame["index"],
            "time_s": frame["time_s"],
            "hidden": frame["hidden"],
            "visible_pixels": int(frame["mask"].sum()),
            "status": result["status"],
            "available": result["camera_from_reference"] is not None,
            "reason": details.get("reason"),
            "pose": session.pose.tolist(),
            "map_points": len(session.store.points),
            "map_version": session.store.version,
            "retained_patches": len(session.store.patches),
            "prepare_ms": prepare_ms,
            "update_ms": update_ms,
            "covariance": session.kalman.covariance.tolist(),
            "growth": details.get("growth"),
            "measurement_count": details.get("fit", {}).get("uncertainty", {}).get("measurement_count"),
            "kalman": {
                key: details.get("kalman", {}).get(key)
                for key in [
                    "accepted",
                    "reason",
                    "innovation_squared",
                    "innovation_limit",
                    "prior_rejected_by_depth_mask",
                    "prediction_failed_geometry",
                    "filter_reason",
                ]
            },
            "error": pose_error(session.reference, session.pose, frame["truth"]),
        }
        if candidate is not None:
            row["candidate_error"] = pose_error(session.reference, np.array(candidate), frame["truth"])
        rows.append(row)
    return {"mode": mode, "config": asdict(config), "rows": rows}


def summarize(result):
    first = result["rows"][WARMUP + HIDDEN]
    final = result["rows"][-1]
    return {
        "mode": result["mode"],
        "first_visible": {
            "status": first["status"],
            "available": first["available"],
            "error": first["error"],
            "candidate_error": first.get("candidate_error"),
            "kalman": first["kalman"],
        },
        "final": {"status": final["status"], "available": final["available"], "error": final["error"]},
    }


def render(results, frames, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])
    for col, (index, title) in enumerate(
        [
            (WARMUP - 1, "Before"),
            (WARMUP + HIDDEN // 2, "Object moves behind screen"),
            (WARMUP + HIDDEN, "Visible again: 4 cm to the right"),
        ]
    ):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(frames[index]["rgb"])
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    t = [f["time_s"] for f in frames]
    ax.plot(
        t, [f["truth"][0, 3] * 1000 for f in frames], color="#20252b", linewidth=3, label="Actual position"
    )
    names = {
        "current_reset": "Current automatic reset",
        "strict": "No reset, current uncertainty (internal prediction)",
        "strict_broader_motion": "No reset, broader motion uncertainty",
    }
    for result, color, style in zip(
        results, ["#c63c3c", "#3263b7", "#15946b"], ["--", "-", ":"], strict=True
    ):
        ax.plot(
            t,
            [r["pose"][0][3] * 1000 for r in result["rows"]],
            color=color,
            linestyle=style,
            linewidth=2,
            label=names[result["mode"]],
        )
    ax.axvspan(WARMUP / FPS, (WARMUP + HIDDEN) / FPS, color="#c2a06c", alpha=0.18)
    ax.annotate(
        "Blue: no pose returned after reappearance",
        xy=(0.72, 0),
        xytext=(0.49, 9),
        fontsize=10,
        color="#3263b7",
        arrowprops={"arrowstyle": "->", "color": "#3263b7"},
    )
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Horizontal movement (mm)",
        title="Same rendered observations; only the filter setting changes",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = scene()
    results = []
    for mode in MODES:
        result = run_scene(frames, mode)
        results.append(result)
        (args.output / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(summarize(result)), flush=True)
    render(results, frames, args.output / "comparison.png")
    controls = []
    for name, control in [
        ("small_hidden_motion", scene(0.003)),
        ("visible_motion", scene(0.04, occluded=False)),
    ]:
        result = run_scene(control, "strict")
        controls.append(dict(case=name, **summarize(result)))
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(controls[-1]), flush=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "rendered_scene": {
                    "fps": FPS,
                    "hidden_frames": HIDDEN,
                    "motion_m": 0.04,
                    "exact_mask": True,
                    "depth_noise": False,
                    "image_or_group_motion": False,
                    "surface_growth": False,
                },
                "comparisons": [summarize(r) for r in results],
                "controls": controls,
            },
            indent=2,
        )
        + "\n"
    )
