"""Real ring images and exact acceptance checks from a common pre-frame state."""

import argparse
import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from benchmarks.flash_motion_recovery import load_fixture, replay
from benchmarks.flash_prior_registration import experiment, native_module
from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.scene_depth import project
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_observation import move


def depth_evidence(geometry, pose):
    """Observe the actual veto after its exceptions; do not reimplement it."""
    _, validator, frame = geometry._latest_rejection
    original = WholeSurfaceFit._coherent_protrusion
    captured = {}

    def inspect(fit, points, bad):
        verdict = original(fit, points, bad)
        if fit.observation.get("frame_id") == frame:
            captured.update(
                old_xyz=points.copy(),
                old_uv=project(points, fit.k),
                bad=bad.copy(),
                coherent=verdict,
            )
        return verdict

    with patch.object(WholeSurfaceFit, "_coherent_protrusion", inspect):
        result = validator.scene_depth_check(pose)
    view = next(v for v in result["scene_depth_views"] if v["frame_id"] == frame)
    assert captured["coherent"] == view["coherent_protrusion"]
    return captured


def capture(native, fixture):
    records = {}
    step_original = LiveSurfaceSession.step
    for method in ("baseline", "native_post", "prior_0.1"):
        data = {}

        def step(live, frame, *args, method=method, data=data, **kwargs):
            if frame.frame_id != 4759:
                return step_original(live, frame, *args, **kwargs)
            context = (
                nullcontext()
                if method == "baseline"
                else experiment(
                    native, mode="post" if method == "native_post" else "integrated", strength=0.1
                )
            )
            with context:
                fit_original = ImageConditionedFit.correct_surface_first
                correct_original = PoseKalmanFilter.correct

                def fit(fitter, proposal, previous, **kw):
                    data.update(
                        fitter=fitter,
                        check=kw["check"],
                        reference=fitter.reference.copy(),
                        seed=proposal.copy(),
                    )
                    result, details = fit_original(fitter, proposal, previous, **kw)
                    assert result is not None
                    data["fit"] = result.copy()
                    return result, details

                def correct(state, *a, **kw):
                    data["prior"] = state.pose.copy()
                    data["prior_covariance"] = state.covariance.copy()
                    result = correct_original(state, *a, **kw)
                    data["after"] = state.pose.copy()
                    # Capture while the real search pool and session are alive.
                    for kind in ("prior", "fit", "after"):
                        data[kind + "_check"] = deepcopy(data["check"](data[kind]))
                    for kind in ("fit", "after"):
                        evidence = depth_evidence(data["fitter"].geometry, data[kind])
                        view = next(
                            v for v in data[kind + "_check"]["scene_depth_views"] if v["frame_id"] == 4750
                        )
                        assert evidence["coherent"] == view["coherent_protrusion"]
                        data[kind + "_depth"] = evidence
                    return result

                with (
                    patch.object(ImageConditionedFit, "correct_surface_first", fit),
                    patch.object(PoseKalmanFilter, "correct", correct),
                ):
                    return step_original(live, frame, *args, **kwargs)

        with patch.object(LiveSurfaceSession, "step", step):
            rows = replay("ring", fixture=fixture)
        data["previous_accepted_pose"] = np.asarray(rows[1]["pose"])
        data["row"] = next(r for r in rows if r["frame"] == 4759)
        assert data["row"]["details"]["accepted_frame_before"] == 4750
        assert data["row"]["details"]["kalman"]["motion_source"] is None
        np.testing.assert_array_equal(data["prior"], data["previous_accepted_pose"])
        records[method] = data
    np.testing.assert_array_equal(fixture[1][4750][0].base_from_camera, fixture[1][4759][0].base_from_camera)
    np.testing.assert_array_equal(fixture[1][4750][0].intrinsics, fixture[1][4759][0].intrinsics)
    for data in records.values():
        for name in ("prior", "prior_covariance", "reference", "seed"):
            np.testing.assert_allclose(data[name], records["baseline"][name], rtol=0, atol=1e-12)
    assert records["baseline"]["row"]["status"] == "lost"
    assert records["native_post"]["row"]["status"] == "lost"
    assert records["prior_0.1"]["row"]["status"] == "observed_candidate"
    # This frame keeps the initial geometry-compatible candidate. Do not present
    # its recovery as evidence that the prior improved the numerical solution.
    np.testing.assert_array_equal(records["baseline"]["fit"], records["baseline"]["seed"])
    np.testing.assert_array_equal(records["prior_0.1"]["after"], records["baseline"]["fit"])
    return records


def render(records, fixture, output):
    _, frames = fixture
    rgb, mask = frames[4759][0].rgb, frames[4759][1]["object"]
    k = frames[4759][0].intrinsics
    baseline = records["baseline"]
    center = baseline["reference"].mean(0)
    axis_points = np.vstack((center, center + np.eye(3) * 0.016))
    before, after = baseline["fit"], baseline["after"]
    angle = float(np.rad2deg(Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).magnitude()))
    position = float(1000 * np.linalg.norm(move(center[None], after) - move(center[None], before)))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "figure.facecolor": "#101922",
            "text.color": "#eaf0f6",
            "axes.titlecolor": "#eaf0f6",
            "savefig.facecolor": "#101922",
        }
    )

    def panel(ax, image, outline, crop):
        ax.imshow(image, interpolation="nearest")
        ax.contour(outline, levels=[0.5], colors=["#ffe259"], linewidths=1.2)
        ax.set(xlim=crop[::2], ylim=crop[3:0:-2])
        ax.axis("off")

    elapsed = frames[4759][0].source_time_s - frames[4750][0].source_time_s
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.7))
    fig.suptitle("Here, prediction means: assume the ring stayed where it was", fontsize=16, y=0.96)
    fig.text(
        0.5,
        0.91,
        f"Two actual camera frames, {elapsed:.2f} seconds apart. The camera did not move.",
        ha="center",
        fontsize=12,
    )
    uv = project(move(baseline["reference"], baseline["prior"]), k)
    for ax, number, title, caption in zip(
        axes,
        (4750, 4759),
        ("Earlier frame 4750", "Current frame 4759"),
        ("Orange: previously accepted placement", "Same orange placement; the real ring has moved"),
        strict=True,
    ):
        frame, masks = frames[number]
        panel(ax, frame.rgb, masks["object"], (265, 280, 385, 410))
        ax.scatter(*uv.T, s=9, color="#ffb04c", alpha=0.8)
        ax.set_title(title, pad=12, fontsize=14)
        ax.text(0.5, -0.075, caption, transform=ax.transAxes, ha="center", fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.2, wspace=0.12)
    fig.text(
        0.055,
        0.095,
        "The orange dots are stored 3D surface points, drawn at the same position and rotation in both photos.",
        fontsize=11,
    )
    fig.text(
        0.055,
        0.05,
        "Yellow: the ring found in each image. No motion update was applied to this prediction.",
        fontsize=11,
    )
    fig.savefig(output / "ring-prediction-origin.png", dpi=145)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10.6))
    fig.suptitle("Actual ring frame 4759: where the result changes", fontsize=18, y=0.97)
    fig.text(0.5, 0.933, "Same camera image, depth and incoming tracker state in every panel", ha="center")
    titles = [
        "1. Prediction from earlier frames",
        "2. Placement selected by fitting",
        "3A. Current: Kalman changes it",
        "3B. Experiment: no later pose averaging",
    ]
    entries = [
        (baseline, "prior", "#ffb04c", "Prediction already disagrees with geometry"),
        (baseline, "fit", "#3de1f0", "Geometry check: PASS"),
        (baseline, "after", "#ff629e", f"Rotated {angle:.1f}° after fitting → REJECTED"),
        (records["prior_0.1"], "after", "#3de1f0", "Same placement as panel 2 → ACCEPTED"),
    ]
    for ax, title, (data, kind, color, caption) in zip(axes.flat, titles, entries, strict=True):
        panel(ax, rgb, mask, (265, 275, 385, 400))
        uv = project(move(baseline["reference"], data[kind]), k)
        ax.scatter(*uv.T, s=9, c=color, alpha=0.75)
        uv_axes = project(move(axis_points, data[kind]), k)
        for endpoint, axis_color in zip(uv_axes[1:], ("#ff6969", "#6fe08a", "#75a8ff"), strict=True):
            ax.plot(*np.stack((uv_axes[0], endpoint)).T, color=axis_color, lw=2.2)
        ax.set_title(title, pad=11, fontsize=13)
        ax.text(0.5, -0.055, caption, transform=ax.transAxes, ha="center", color=color, fontsize=11)
    fig.subplots_adjust(left=0.045, right=0.955, top=0.88, bottom=0.145, hspace=0.26, wspace=0.13)
    fig.text(
        0.055,
        0.09,
        "Dots: the same stored 3D surface projected onto this photo. Yellow: SAM outline.",
        fontsize=11,
    )
    fig.text(
        0.055,
        0.065,
        "Short red/green/blue lines: the same object coordinate axes, showing rotation.",
        fontsize=11,
    )
    fig.text(
        0.055,
        0.04,
        f"Kalman change: {angle:.2f}° rotation; {position:.2f} mm surface-centre movement. Not ground-truth errors.",
        fontsize=11,
    )
    fig.savefig(output / "ring-real-placements.png", dpi=145)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    old = frames[4750]
    fig.suptitle("The depth contradiction is checked in earlier frame 4750", fontsize=17, y=0.96)
    counts = {}
    for ax, kind, title in zip(
        axes,
        ("fit", "after"),
        ("Before Kalman: no broad contradiction", "After Kalman: contradiction → reject"),
        strict=True,
    ):
        evidence = baseline[kind + "_depth"]
        panel(ax, old[0].rgb, old[1]["object"], (270, 305, 380, 415))
        uv, bad = evidence["old_uv"], evidence["bad"] >= 1 - 1e-6
        ax.scatter(*uv.T, s=3, c="#3de1f0", alpha=0.16)
        ax.scatter(*uv[bad].T, s=30, c="#ff629e", marker="x", linewidths=1.3)
        counts[kind] = int(bad.sum())
        ax.set_title(title, fontsize=12, pad=12)
        ax.text(
            0.5,
            -0.08,
            f"{bad.sum()} fully depth-contradicted samples",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
        )
    fig.subplots_adjust(left=0.035, right=0.965, top=0.86, bottom=0.24, wspace=0.11)
    fig.text(
        0.04,
        0.14,
        "Current measured 3D points are mapped back into the earlier camera view using each placement.",
        fontsize=11,
    )
    fig.text(
        0.04,
        0.095,
        "Pink ×: the proposed point lies in space that the earlier depth measurement says was empty.",
        fontsize=11,
    )
    fig.text(
        0.04,
        0.05,
        "The actual check requires spatially supported contradictions, not merely a nonzero sample count.",
        fontsize=11,
    )
    fig.savefig(output / "ring-real-depth-rejection.png", dpi=145)
    plt.close(fig)
    return {
        "kalman_rotation_change_deg": angle,
        "kalman_center_movement_mm": position,
        "fully_contradicted_samples": counts,
    }


def main(native_dir, output):
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    fixture = load_fixture()
    records = capture(native_module(native_dir), fixture)
    summary = render(records, fixture, output)
    info = {
        "scope": "All methods use baseline history through 4750 and switch only at 4759.",
        **summary,
        "methods": {},
    }
    arrays = {}
    for method, data in records.items():
        info["methods"][method] = {name: data[name] for name in ("prior_check", "fit_check", "after_check")}
        info["methods"][method].update(status=data["row"]["status"], details=data["row"]["details"])
        for name in (
            "reference",
            "previous_accepted_pose",
            "prior",
            "prior_covariance",
            "seed",
            "fit",
            "after",
        ):
            arrays[f"{method}_{name}"] = data[name]
        for kind in ("fit", "after"):
            for name in ("old_xyz", "old_uv", "bad"):
                arrays[f"{method}_{kind}_{name}"] = data[kind + "_depth"][name]
    (output / "ring-real-example.json").write_text(json.dumps(info, indent=2) + "\n")
    np.savez_compressed(output / "ring-real-example.npz", **arrays)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.native_dir, args.output)
