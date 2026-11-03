"""Noiseless RGB-D control for fitting two partial views of a rigid object.

The box is used ONLY to render measurements and evaluate pose error. Neither
fitter receives its dimensions, hidden faces, or ground-truth pose as evidence.
Run: python -m benchmarks.flash_partial_surface_control --output /tmp/partial-fit
"""

from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, observation
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


def box_observation(angle_deg, *, k=None, displacement_m=None):
    height, width = 180, 240
    k = np.array([[600.0, 0, width / 2], [0, 600.0, height / 2], [0, 0, 1]]) if k is None else k
    center, half_size = np.array([0.0, 0.0, 0.7]), np.array([0.06, 0.04, 0.05])
    displacement = np.zeros(3) if displacement_m is None else np.asarray(displacement_m, dtype=float)
    y, x = np.indices((height, width))
    rays = np.stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)), axis=-1)
    rotation = Rotation.from_euler("y", angle_deg, degrees=True).as_matrix()
    direction, origin = rays @ rotation, -(center + displacement) @ rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        first, second = (-half_size - origin) / direction, (half_size - origin) / direction
    near, far = np.minimum(first, second).max(-1), np.maximum(first, second).min(-1)
    mask = (far >= near) & (near > 0)
    depth = np.where(mask, near, 1.5).astype(np.float32)
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig(component_gap_m=None)))
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, center + displacement - rotation @ center
    return {"obs": obs, "depth": depth, "mask": mask}, k, pose


def perturbed_pose(pose):
    center = np.array([0.0, 0.0, 0.7])
    result = pose.copy()
    result[:3, :3] = Rotation.from_euler("y", 5, degrees=True).as_matrix() @ pose[:3, :3]
    result[:3, 3] = center - result[:3, :3] @ center + [0.003, -0.002, 0]
    return result


def pose_error(reference, result, expected):
    return {
        "median_point_error_mm": float(
            1000 * np.median(np.linalg.norm(move(reference, result) - move(reference, expected), axis=1))
        ),
        "rotation_error_deg": float(
            np.rad2deg(Rotation.from_matrix(result[:3, :3] @ expected[:3, :3].T).magnitude())
        ),
    }


def run_control():
    source, k, _ = box_observation(0)
    reference = source["obs"]["points"]
    config = PoseUncertaintyConfig(max_visible_outlier_fraction=0.25)
    results = []
    for angle in (20, 40, 60):
        current, _, expected = box_observation(angle)
        for name, source_view in (("legacy", None), ("two_measured_views", source)):
            fitter = WholeSurfaceFit(reference, current, k, ShapeConfig(), reference_observation=source_view)
            for seed_name, seed in (("exact", expected), ("perturbed", perturbed_pose(expected))):
                result, detail = fitter.correct(seed)
                wrong = expected.copy()
                wrong[0, 3] += 0.04
                results.append(
                    {
                        "turn_deg": angle,
                        "method": name,
                        "seed": seed_name,
                        **pose_error(reference, result, expected),
                        "exact_pose_check": check_surface_contradiction(fitter, expected, config),
                        "result_check": check_surface_contradiction(fitter, result, config),
                        "wrong_pose_check": check_surface_contradiction(fitter, wrong, config),
                        "pose": result.tolist(),
                        "fit": detail,
                    }
                )
    return results


def render_comparison(results, path):
    """Draw evaluated poses on the same synthetic depth silhouette."""
    import cv2

    from lerobot.policies.flash.surface_grounding import project

    current, k, truth = box_observation(40)
    # Renderer geometry is for illustration only, never provided to fitting.
    corners = np.array([[-0.06, -0.04, 0.65], [0.06, -0.04, 0.65], [0.06, 0.04, 0.65], [-0.06, 0.04, 0.65]])
    base = np.full((*current["mask"].shape, 3), 35, np.uint8)
    base[current["mask"]] = [225, 210, 180]
    cv2.fillConvexPoly(base, np.rint(project(move(corners, truth), k)).astype(np.int32), (155, 170, 185))
    selected = {r["method"]: r for r in results if r["turn_deg"] == 40 and r["seed"] == "exact"}
    poses = [("Correct alignment", truth)]
    for method, title in (("legacy", "Before"), ("two_measured_views", "Fixed")):
        row = selected[method]
        poses.append((f"{title}: {row['rotation_error_deg']:.2f} deg error", np.array(row["pose"])))
    panels = []
    for title, pose in poses:
        panel = cv2.resize(base, (480, 360), interpolation=cv2.INTER_NEAREST)
        polygon = np.rint(project(move(corners, pose), k) * 2).astype(np.int32)
        cv2.polylines(panel, [polygon], True, (50, 220, 245), 3, cv2.LINE_AA)
        panel = np.pad(panel, ((48, 0), (0, 0), (0, 0)), constant_values=24)
        cv2.putText(panel, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(panel)
    image = np.pad(np.concatenate(panels, axis=1), ((0, 47), (0, 0), (0, 0)), constant_values=24)
    cv2.putText(
        image,
        "Grey: earlier face. Tan: newly visible side. Cyan: where the fitter places the earlier face.",
        (16, image.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = run_control()
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    render_comparison(results, args.output / "comparison.jpg")
    for row in results:
        print(
            f"{row['turn_deg']:2}° {row['method']:20} {row['seed']:9} "
            f"{row['median_point_error_mm']:.2f} mm / {row['rotation_error_deg']:.2f}° "
            f"accepted={row['result_check']['accepted']}"
        )
