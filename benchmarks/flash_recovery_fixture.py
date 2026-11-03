"""Portable real-recording checkpoint for the false mug recovery.

Export measured RGB-D, retained patches and the pre-update filter state. Replay
runs the actual observation builder, candidate search, filter and growth gate;
the recorded output and dataset pose are evaluation-only. No external dataset,
network or model is needed after export. This is a checkpoint replay, not a
fresh end-to-end segmentation or group-tracker test.
"""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.flash_kalman_recovery_control import MODES, StrictPoseKalmanFilter
from benchmarks.flash_partial_surface_control import pose_error
from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    SurfacePatch,
    SurfaceView,
)
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

FIXTURE = Path(__file__).resolve().parents[1] / "tests/policies/flash/fixtures/mug_recovery"


def write_fixture(output, arrays, metadata):
    """One lossless archive per source frame; keep recorded views inspectable."""
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for name, value in arrays.items():
        match = re.search(r"_(\d+)(?:_|$)", name)
        file = f"frame_{int(match[1]):06d}.npz" if match else "state.npz"
        groups.setdefault(file, {})[name] = value
    files = {}
    for name, values in groups.items():
        path = output / name
        np.savez_compressed(path, **values)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {k: v for k, v in metadata.items() if k != "fixture_sha256"}
    metadata["files_sha256"] = files
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def export(run, output=FIXTURE, frame=49):
    from benchmarks.flash_experiment_inputs import sample_at
    from benchmarks.flash_experiment_labels import load_pose_labels

    rows = [json.loads(s) for s in (run / "frames.jsonl").read_text().splitlines()]
    case = json.loads((run / "case.json").read_text())
    provenance = json.loads((run.parent / "provenance.json").read_text())
    role, method = "object", "group_growing"
    details = rows[frame]["objects"][role][method + "_details"]
    kalman = details["kalman"]
    retained = {}
    for row in rows[:frame]:
        for event in row["objects"][role][method + "_details"].get("map_events", []):
            f = event["frame"]
            if event["event"] in {"added", "pending"}:
                retained[f] = {"active": event["event"] == "added", "support_frames": [f]}
            elif event["event"] == "confirmed":
                retained[f]["active"] = True
                retained[f]["support_frames"].append(event["by_frame"])
            elif event["event"] in {"excluded", "evicted", "expired"}:
                retained.pop(f, None)
            else:
                raise ValueError(f"unhandled patch event: {event}")
    connected = details["image_source_frame"]
    arrays = {}
    for f in sorted(set(retained) | {connected, frame}):
        rgbd, masks = sample_at(case, f)
        arrays[f"depth_{f}"] = rgbd.metric_depth().astype(np.float32)
        arrays[f"mask_{f}"] = masks[role]
        arrays[f"intrinsics_{f}"] = rgbd.intrinsics
        if f in {connected, frame}:
            arrays[f"rgb_{f}"] = rgbd.rgb
        if f in retained:
            with np.load(run / f"surface_patches/{method}/{role}/{f:06d}.npz") as patch:
                for key in patch.files:
                    arrays[f"patch_{f}_{key}"] = patch[key]
    with np.load(run / "reference_points.npz") as reference:
        arrays["reference"] = reference[role]
    with np.load(run / f"surface_models/{method}/{role}/{details['map_version_used']:06d}.npz") as model:
        arrays["map_points"] = model["points"]
    labels, _ = load_pose_labels(case, rows)
    arrays.update(
        prior_pose=np.array(kalman["prior_pose"]),
        prior_covariance=np.array(kalman["prior_covariance"]),
        process_covariance=np.array(kalman["process_covariance"]),
        connected_pose=np.array(rows[connected]["objects"][role][method]),
        expected_recorded_pose=np.array(rows[frame]["objects"][role][method]),
        label_pose=labels[role][frame] @ np.linalg.inv(labels[role][0]),
    )
    # A checkpoint starts before this frame's predict/update, not after rejection.
    metadata = {
        "case": case["id"],
        "dataset": {
            "name": "DexYCB",
            "url": "https://dex-ycb.github.io/",
            "sequence": "20201002-subject-08/20201002_111710/841412060263",
            "input": "Recorded RGB, depth aligned to color, visible-object masks and calibrated intrinsics",
            "labels": "Evaluation only; no labelled pose is used by the tracking update",
        },
        "frame": frame,
        "connected_frame": connected,
        "time_s": rows[frame]["time_s"],
        "previous_time_s": rows[frame - 1]["time_s"],
        "observed_time_s": rows[connected]["time_s"],
        "patches": retained,
        "map_version": details["map_version_used"],
        "vision_config": provenance["config"],
        "surface_config": provenance["surface_experiment"],
        "recorded_cost": details["fit"]["geometry_costs"],
        "recorded_innovation_squared": kalman["innovation_squared"],
        "scope": "One real frame from a recorded pre-update checkpoint; labels never enter tracking.",
        "broader_variant": "Additional position process variance since last accepted observation, same checkpoint mean and history.",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (run / "frames.jsonl", run / "case.json", run.parent / "provenance.json")
        },
    }
    return write_fixture(output, arrays, metadata)


def restore(mode, path=FIXTURE):
    if mode not in MODES:
        raise ValueError(mode)
    metadata = json.loads((path / "manifest.json").read_text())
    data = {}
    for name, digest in metadata["files_sha256"].items():
        file = path / name
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError("fixture digest changed")
        with np.load(file) as archive:
            data.update({key: archive[key] for key in archive.files})
    vision = VisionPipelineConfig.from_dict(metadata["vision_config"])
    config = GrowingSurfaceConfig(**metadata["surface_config"])
    old_noise = config.uncertainty.position_process_std_m_sqrt_s
    if mode == "strict_broader_motion":
        config = replace(config, uncertainty=replace(config.uncertainty, position_process_std_m_sqrt_s=0.03))
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

    # Recover exact retained measurements, not the mesh rendered at its estimate.
    for f_text, patch in metadata["patches"].items():
        f = int(f_text)
        old = prepare(f)
        view = SurfaceView.from_observation(
            old["current"], old["intrinsics"], session.growth_shape_config, retain_measurement=True
        )
        session.store.patches[f] = SurfacePatch(
            f,
            data[f"patch_{f}_measured"],
            data[f"patch_{f}_camera_from_reference"],
            int(data[f"patch_{f}_fitted_map_version"]),
            set(patch["support_frames"]),
            patch["active"],
            view,
        )
    session.store.points = data["map_points"].copy()
    session.store.version = metadata["map_version"]
    session.reference = data["reference"].copy()
    session.reference_frame = 0
    session.pose = data["prior_pose"].copy()
    session.connected_frame = metadata["connected_frame"]
    session.connected_pose = data["connected_pose"].copy()
    connected = prepare(session.connected_frame)
    session.connected_observation = connected["current"]
    session.connected_intrinsics = connected["intrinsics"]
    session.last_frame = metadata["frame"] - 1
    session.last_time = metadata["previous_time_s"]
    session.last_observed_time = session.last_supported_time = metadata["observed_time_s"]
    filter_type = PoseKalmanFilter if mode == "current_reset" else StrictPoseKalmanFilter
    session.kalman = filter_type(
        session.pose, session.reference.mean(0), session.last_time, config.uncertainty
    )
    session.kalman.covariance = data["prior_covariance"] - data["process_covariance"]
    if mode == "strict_broader_motion":
        dt = session.last_time - session.last_observed_time
        session.kalman.covariance[3:, 3:] += np.eye(3) * (0.03**2 - old_noise**2) * dt
    return session, prepare(metadata["frame"]), metadata, data


def run_mug(mode, path=FIXTURE):
    start = perf_counter()
    session, prepared, metadata, data = restore(mode, path)
    prepared_ms = (perf_counter() - start) * 1000
    before = session.store.points.copy()
    start = perf_counter()
    result = session.update(prepared, metadata["time_s"])
    elapsed_ms = (perf_counter() - start) * 1000
    d = result["diagnostics"]
    return {
        "mode": mode,
        "available": result["camera_from_reference"] is not None,
        "pose": session.pose.tolist(),
        "error": pose_error(session.reference, session.pose, data["label_pose"]),
        "recorded_pose_difference": pose_error(
            session.reference, session.pose, data["expected_recorded_pose"]
        ),
        "surface_unchanged": np.array_equal(before, session.store.points),
        "prepare_ms": prepared_ms,
        "update_ms": elapsed_ms,
        "diagnostics": d,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path, default=FIXTURE)
    args = parser.parse_args()
    if args.export:
        print(export(args.export, args.output))
    else:
        for mode in MODES:
            print(json.dumps(run_mug(mode, args.output)), flush=True)
