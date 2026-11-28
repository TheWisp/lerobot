"""Audit a local, prior-conditioned partial pose update on held-out trials.

This does not alter the FLASH estimator. The measured Jacobian is conditional
on the visual candidate's correspondences, so its spectrum is not proof that
the candidate identifies the object or that its proposed motion is correct.
The future RGB-D direction is a visible-axis check, not a full-pose label.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def partial_pose(
    prior: np.ndarray, candidate: np.ndarray, information: dict, *, prior_rms_m: float, depth_error_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project a candidate's twist through its local surface information.

    ``prior_rms_m`` is an assumed whole-object motion standard deviation;
    ``depth_error_m`` is an assumed visible-surface measurement deviation.
    The resulting weights are an ablation, not a calibrated posterior.
    """
    if prior_rms_m <= 0 or depth_error_m <= 0:
        raise ValueError("uncertainty scales must be positive")
    basis = np.asarray(information["twist_basis_base"], dtype=float)
    strength = np.asarray(information["strength"], dtype=float)
    if basis.shape == (6, 0) and strength.shape == (6,) and not np.any(strength):
        return prior.copy(), np.zeros(6)
    if (
        basis.shape != (6, 6)
        or strength.shape != (6,)
        or not np.isfinite(basis).all()
        or not np.isfinite(strength).all()
        or np.any(strength < 0)
    ):
        raise ValueError("need a full finite six-dimensional conditional basis")
    pivot_candidate = np.asarray(information["pivot_base_m"], dtype=float)
    pivot_object = candidate[:3, :3].T @ (pivot_candidate - candidate[:3, 3])
    pivot_prior = prior[:3, :3] @ pivot_object + prior[:3, 3]
    delta = np.concatenate(
        (
            Rotation.from_matrix(candidate[:3, :3] @ prior[:3, :3].T).as_rotvec(),
            pivot_candidate - pivot_prior,
        )
    )
    coordinates = np.linalg.solve(basis, delta)
    weight = strength * prior_rms_m**2 / (strength * prior_rms_m**2 + depth_error_m**2)
    projected = basis @ (weight * coordinates)
    turn = Rotation.from_rotvec(projected[:3]).as_matrix()
    output = prior.copy()
    output[:3, :3] = turn @ prior[:3, :3]
    output[:3, 3] = pivot_prior + projected[3:] - output[:3, :3] @ pivot_object
    return output, weight


def axis_gap_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(first @ second), -1, 1))))


def audit(
    result: dict,
    axis_audit: dict,
    future_axis: dict,
    *,
    prior_rms_m: float,
    depth_error_m: float,
    horizon_s: float,
) -> dict:
    frames = {item["row"]: item for item in result["frames"]}
    measured = {item["row"]: item for item in axis_audit["measurements"]}
    horizon = next(
        (item for item in future_axis["horizons"] if np.isclose(item["requested_horizon_s"], horizon_s)),
        None,
    )
    if horizon is None:
        raise ValueError("future-axis audit lacks the requested horizon")
    rows = []
    for sample in horizon["samples"]:
        source = frames[sample["row"]]
        source_axis = measured[sample["row"]]
        future = measured[sample["future_row"]]
        info = source["conditional_surface_information"]
        if info is None or source["candidate_pose"] is None or source["attachment"] is None:
            continue
        attachment = np.asarray(source["attachment"], dtype=float)
        base_from_gripper = np.asarray(source_axis["base_from_gripper"], dtype=float)
        prior = base_from_gripper @ attachment
        candidate = np.asarray(source["candidate_pose"], dtype=float)
        projected, weight = partial_pose(
            prior, candidate, info, prior_rms_m=prior_rms_m, depth_error_m=depth_error_m
        )
        gripper_axis = np.asarray(source_axis["proposals"]["gripper_fallback"]["axis_gripper"])
        axis_object = attachment[:3, :3].T @ gripper_axis
        projected_axis = base_from_gripper[:3, :3].T @ projected[:3, :3] @ axis_object
        observed_axis = np.asarray(future["tracked_surface_axis_gripper"])
        rows.append(
            {
                "row": sample["row"],
                "projected_gap_deg": axis_gap_deg(projected_axis, observed_axis),
                "vision_gap_deg": sample["vision_gap_deg"],
                "gripper_gap_deg": sample["gripper_gap_deg"],
                "mean_weight": float(np.mean(weight)),
                "max_weight": float(np.max(weight)),
                "matched_fraction": info["matched_reference_points"] / info["reference_points"],
            }
        )
    if not rows:
        raise ValueError("no source frames have a visual candidate, gripper pose and future-axis check")
    return {
        "trial": result["trial"],
        "horizon_s": horizon_s,
        "prior_rms_m": prior_rms_m,
        "depth_error_m": depth_error_m,
        "compared_frames": len(rows),
        "median_projected_gap_deg": float(np.median([row["projected_gap_deg"] for row in rows])),
        "median_vision_gap_deg": float(np.median([row["vision_gap_deg"] for row in rows])),
        "median_gripper_gap_deg": float(np.median([row["gripper_gap_deg"] for row in rows])),
        "projected_better_than_gripper": sum(
            row["projected_gap_deg"] < row["gripper_gap_deg"] for row in rows
        ),
        "projected_better_than_vision": sum(row["projected_gap_deg"] < row["vision_gap_deg"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--axis-audit", type=Path, required=True)
    parser.add_argument("--future-axis", type=Path, required=True)
    parser.add_argument("--prior-rms-m", type=float, default=0.01)
    parser.add_argument("--depth-error-m", type=float, default=0.005)
    parser.add_argument("--horizon-s", type=float, default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        read_json(args.result),
        read_json(args.axis_audit),
        read_json(args.future_axis),
        prior_rms_m=args.prior_rms_m,
        depth_error_m=args.depth_error_m,
        horizon_s=args.horizon_s,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
