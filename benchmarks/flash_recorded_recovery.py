"""Recorded RGB-D recovery checkpoints with frozen upstream predictions.

These tests run observation preparation, surface fitting, Kalman and map updates.
They do not replay SAM/UFM or claim that frozen group predictions are correct.
Dataset poses and image-review annotations are consumed only by the evaluator.
The original recording paths are needed only to export the small fixtures.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmarks.flash_experiment_inputs import sample_at
from benchmarks.flash_experiment_labels import load_pose_labels
from benchmarks.flash_full_pose_score import score_transforms
from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    PoseHint,
    SurfacePatch,
    SurfaceView,
)
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_observation import voxel_sample
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

FIXTURES = Path(__file__).resolve().parents[1] / "tests/policies/flash/fixtures/recorded_recovery"


def export(run, frame, name, output=FIXTURES):
    run = Path(run)
    case = json.loads((run / "case.json").read_text())
    role = "gamepad" if "gamepad" in case["roles"] else case["roles"][0]
    provenance = json.loads((run.parent / "provenance.json").read_text())
    rows, retained = [], {}
    for line in (run / "frames.jsonl").open():
        row = json.loads(line)
        rows.append(row)
        if row["frame"] >= frame:
            continue
        for event in row["objects"][role]["group_growing_details"].get("map_events", []):
            f = event["frame"]
            if event["event"] in {"added", "pending"}:
                retained[f] = {"active": event["event"] == "added", "support_frames": [f]}
            elif event["event"] == "confirmed":
                retained[f]["active"] = True
                retained[f]["support_frames"].append(event["by_frame"])
            elif event["event"] in {"excluded", "expired", "evicted"}:
                retained.pop(f, None)
            else:
                raise ValueError(event)
    row = rows[frame]
    detail = row["objects"][role]["group_growing_details"]
    ref_frame = detail["reference_frame"]
    connected = next(
        r
        for r in reversed(rows[:frame])
        if r["objects"][role]["group_growing_status"] in {"initialized", "observed_candidate"}
    )
    connected_frame = connected["frame"]
    data = {}
    # Retain measured source views for partial-surface experiments without
    # substituting a rendered surface for a real depth observation.
    for f in sorted(set(retained) | {frame, ref_frame, connected_frame}):
        rgbd, masks = sample_at(case, f)
        data[f"depth_{f}"] = rgbd.metric_depth().astype(np.float32)
        data[f"mask_{f}"] = masks[role]
        data[f"intrinsics_{f}"] = rgbd.intrinsics
        if f in {frame, ref_frame}:
            data[f"rgb_{f}"] = rgbd.rgb
        if f in retained:
            with np.load(run / f"surface_patches/group_growing/{role}/{f:06d}.npz") as patch:
                data.update({f"patch_{f}_{k}": patch[k] for k in patch.files})
    with np.load(run / "reference_points.npz") as archive:
        data["reference"] = archive[role]
    with np.load(
        run / f"surface_models/group_growing/{role}/{detail['map_version_used']:06d}.npz"
    ) as archive:
        data["map_points"] = archive["points"]
    data["prior_pose"] = np.asarray(detail["kalman"]["prior_pose"])
    data["connected_pose"] = np.asarray(connected["objects"][role]["group_growing"])
    data["prior_covariance"] = np.asarray(detail["kalman"]["prior_covariance"])
    data["process_covariance"] = np.asarray(detail["kalman"]["process_covariance"])
    data["recorded_pose"] = np.asarray(row["objects"][role]["group_growing"] or np.full((4, 4), np.nan))
    if name == "gamepad_271":
        # Evaluation-only visible patch, frozen from the reviewed RGB-D frame.
        # An experiment must not change its own expected surface direction by
        # changing the observation filter. This is not a full pose annotation.
        vision = VisionPipelineConfig.from_dict(provenance["config"])
        prepared = SurfaceTrackingSession(
            None,
            observation_config=vision.observation,
            shape_config=vision.shape,
            motion_config=vision.motion,
        ).prepare(
            data[f"rgb_{frame}"],
            data[f"depth_{frame}"],
            data[f"mask_{frame}"],
            data[f"intrinsics_{frame}"],
            frame,
        )
        data["evaluation_visible_patch"] = prepared["current"]["obs"]["points"]
    labels, label_sources = load_pose_labels(case, rows)
    if role in labels:
        data["label_pose"] = labels[role][frame] @ np.linalg.inv(labels[role][ref_frame])
    group = detail.get("group", {})
    if "camera_from_reference" in group:
        data["group_seed"] = np.asarray(group["camera_from_reference"])
    metadata = {
        "name": name,
        "case": case["id"],
        "frame": frame,
        "reference_frame": ref_frame,
        "connected_frame": connected_frame,
        "time_s": row["time_s"],
        "previous_time_s": rows[frame - 1]["time_s"],
        "last_observed_time_s": row["time_s"] - detail["observation_age_s"],
        "map_version": detail["map_version_used"],
        "patches": retained,
        "vision": provenance["config"],
        "surface": provenance["surface_experiment"],
        "label_source": label_sources.get(role),
        "source_run": str(run),
        "recorded_status": row["objects"][role]["group_growing_status"],
        "scope": "Frozen pre-fitting pose/covariance and optional group seed; no image pairs or group residuals. Real RGB-D preparation, fitting, Kalman and growth follow. Labels are evaluation only.",
    }
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "data.npz", **data)
    metadata["sha256"] = hashlib.sha256((destination / "data.npz").read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination


def load(name):
    root = FIXTURES / name
    metadata = json.loads((root / "manifest.json").read_text())
    assert hashlib.sha256((root / "data.npz").read_bytes()).hexdigest() == metadata["sha256"]
    with np.load(root / "data.npz") as archive:
        data = {k: archive[k] for k in archive.files}
    return data, metadata


def replay(
    name,
    *,
    surface_changes=None,
    observation_changes=None,
    shape_changes=None,
    initial_surface_only=False,
    filter_factory=PoseKalmanFilter,
):
    data, metadata = load(name)
    vision = VisionPipelineConfig.from_dict(metadata["vision"])
    vision = replace(
        vision,
        observation=replace(vision.observation, **(observation_changes or {})),
        shape=replace(vision.shape, **(shape_changes or {})),
    )
    config = replace(GrowingSurfaceConfig(**metadata["surface"]), **(surface_changes or {}))
    session = GrowingSurfaceSession(vision.shape, config)

    def prepare(f):
        builder = SurfaceTrackingSession(
            None,
            observation_config=vision.observation,
            shape_config=vision.shape,
            motion_config=vision.motion,
        )
        rgb = data.get(f"rgb_{f}", np.zeros((*data[f"depth_{f}"].shape, 3), np.uint8))
        return builder.prepare(rgb, data[f"depth_{f}"], data[f"mask_{f}"], data[f"intrinsics_{f}"], f)

    for key, patch in metadata["patches"].items():
        f = int(key)
        old = prepare(f)
        session.store.patches[f] = SurfacePatch(
            f,
            data[f"patch_{f}_measured"],
            data[f"patch_{f}_camera_from_reference"],
            int(data[f"patch_{f}_fitted_map_version"]),
            set(patch["support_frames"]),
            patch["active"],
            SurfaceView.from_observation(
                old["current"], old["intrinsics"], session.growth_shape_config, retain_measurement=True
            ),
        )
    session.store.points = data["map_points"].copy()
    if initial_surface_only:
        session.store.patches = {
            f: p for f, p in session.store.patches.items() if f == metadata["reference_frame"]
        }
        session.store.points = data["reference"][voxel_sample(data["reference"], config.spacing_m)]
    session.store.version = metadata["map_version"]
    session.reference = data["reference"].copy()
    session.reference_frame = metadata["reference_frame"]
    session.pose = data["prior_pose"].copy()
    session.accepted_frame = metadata["connected_frame"]
    connected = prepare(session.accepted_frame)
    session.accepted_pose = data["connected_pose"].copy()
    session.accepted_observation = connected["current"]
    session.accepted_intrinsics = connected["intrinsics"]
    session.last_frame = metadata["frame"] - 1
    session.last_time = metadata["previous_time_s"]
    session.last_observed_time = session.last_supported_time = metadata["last_observed_time_s"]
    # Snapshot is AFTER group/camera prediction. Retain that mean and motion
    # covariance, subtract this frame's process noise, then add it exactly once
    # through the ordinary predict call. No group delta is applied again.
    session.kalman = filter_factory(
        session.pose, session.reference.mean(0), metadata["previous_time_s"], config.uncertainty
    )
    session.kalman.covariance = data["prior_covariance"] - data["process_covariance"]
    prepared = prepare(metadata["frame"])
    hint = None
    if "group_seed" in data:
        hint = PoseHint(
            data["group_seed"],
            "recorded_group_seed",
            metadata["time_s"],
            metadata["time_s"],
            session.reference_frame,
        )
    before = session.store.points.copy()
    result = session.update(prepared, metadata["time_s"], hint=hint)
    error = None
    if "label_pose" in data and result["camera_from_reference"] is not None:
        pose = np.asarray(result["camera_from_reference"])
        cloud, angle, _ = score_transforms(
            pose[None], np.array([True]), data["label_pose"][None], data["reference"]
        )
        error = {"cloud_mm": float(cloud[0]), "angle_deg": float(angle[0])}
    return {
        "result": result,
        "error": error,
        "prepared": prepared,
        "data": data,
        "metadata": metadata,
        "surface_changed": not np.array_equal(before, session.store.points),
    }
