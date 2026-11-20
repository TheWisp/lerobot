"""Lossless geometry-only checkpoints at the Kalman correction boundary."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lerobot.policies.flash.surface_grounding import ShapeConfig, SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_refresh import LatestObservationFit


def save(path, geometry, fitted, fused, covariance, center, config, frame):
    arrays = {"fitted": fitted, "fused": fused, "covariance": covariance, "center": center}
    observations, seen = [], {}

    def encode(value):
        if isinstance(value, np.ndarray):
            name = f"array_{len(arrays)}"
            arrays[name] = value
            return {"array": name}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        return value

    def observation(value):
        if id(value) not in seen:
            seen[id(value)] = len(observations)
            kept = {
                key: value[key]
                for key in ("depth", "mask", "scene_depth", "all_surface_points", "frame_id")
                if key in value
            }
            kept["obs"] = {
                key: value["obs"][key]
                for key in ("points", "depth_reliable", "valid_image", "depth_range_rejected")
                if key in value["obs"]
            }
            observations.append(encode(kept))
        return seen[id(value)]

    def node(fit, *, views=True):
        return {
            "reference": encode(fit.reference),
            "k": encode(fit.k),
            "config": asdict(fit.config),
            "observation": observation(fit.observation),
            "views": [
                {
                    "camera": encode(camera),
                    "k": encode(reverse.k),
                    "observation": observation(reverse.observation),
                }
                for camera, reverse in fit.reverse_fits
            ]
            if views
            else [],
        }

    eligible, validator, accepted_frame = geometry._latest_rejection
    meta = {
        "frame": frame,
        "source": "saved-747dd29b27a745418524e2744ac29cb5",
        "scope": "Frozen post-Kalman geometry; no RGB, video, labels or chronological state replay.",
        "full": node(geometry, views=False),
        "validator": node(validator) if validator is not None else None,
        "eligible": encode(eligible),
        "accepted_frame": accepted_frame,
        "checks": {
            "max_outlier_fraction": config.uncertainty.max_visible_outlier_fraction,
            "agreement_m": config.agreement_m,
            "min_overlap": config.min_overlap,
        },
        "observations": observations,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    meta["sha256"] = hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    restored, inputs, check = load(path)
    for pose in (fitted, fused):
        assert check(pose) == geometry.check_pose(pose, **meta["checks"])
    return restored, inputs, check


def load(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Constrained-correction checkpoint checksum mismatch")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"array"}:
                return arrays[value["array"]]
            return {key: decode(item) for key, item in value.items()}
        return value

    observations = [decode(value) for value in meta["observations"]]

    def build(node, *, latest=False):
        args = (
            decode(node["reference"]),
            observations[node["observation"]],
            decode(node["k"]),
            ShapeConfig(**node["config"]),
        )
        views = tuple(
            SurfaceFitView(observations[v["observation"]], decode(v["k"]), decode(v["camera"]))
            for v in node["views"]
        )
        if latest:
            fit = LatestObservationFit.__new__(LatestObservationFit)
            WholeSurfaceFit.__init__(fit, *args, reference_views=views)
            return fit
        return WholeSurfaceFit(*args, reference_views=views)

    geometry = build(meta["full"], latest=True)
    validator = build(meta["validator"]) if meta["validator"] else None
    geometry._latest_rejection = (decode(meta["eligible"]), validator, meta["accepted_frame"])
    return (
        geometry,
        {key: arrays[key] for key in ("fitted", "fused", "covariance", "center")},
        lambda pose: geometry.check_pose(pose, **meta["checks"]),
    )
