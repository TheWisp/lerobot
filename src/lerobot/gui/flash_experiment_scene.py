"""Read-only 3D inspection of saved experiments. Labels never enter tracking.

One worker owns the two-entry metadata cache. RGB-D is read at the requested
frame, not retained for the entire clip. The wire format is uint32 JSON length,
JSON, float32 little-endian depth in metres, then a lossless RGB PNG.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from benchmarks.flash_experiment_inputs import sample_at
from benchmarks.flash_experiment_labels import load_pose_labels
from benchmarks.flash_full_pose_score import score_transforms
from benchmarks.flash_surface_score import LabelledSurface, distance_statistics
from lerobot.gui.flash_live_scene import frame_evidence, object_evidence
from lerobot.gui.flash_pose_overlay import reference_axes
from lerobot.policies.flash.surface_observation import spatial_sample


@lru_cache(maxsize=2)
def _recording(run: Path, case_stamp: int, rows_stamp: int, summary_stamp: int):
    case = json.loads((run / "case.json").read_text())
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    summary = json.loads((run / "summary.json").read_text())
    with np.load(run / "reference_points.npz", allow_pickle=False) as saved:
        references = {role: saved[role].copy() for role in saved.files}
    # Some old exports have no label files. Missing labels must not disable RGB-D.
    try:
        truth, _ = load_pose_labels(case, rows)
        label_error = None
    except (OSError, ValueError) as exc:
        truth, label_error = {}, str(exc)
    return case, rows, summary, references, truth, label_error


def _points(values):
    points = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("Surface contains invalid coordinates")
    # Display-only upload budget. Never changes stored geometry or scoring.
    if len(points) > 12000:
        points = points[np.linspace(0, len(points) - 1, 12000, dtype=int)]
    return points.tolist()


@lru_cache(maxsize=2)
def _label_surface(file: Path, modified_ns: int):
    return np.asarray(np.load(file, mmap_mode="r", allow_pickle=False))


def scene_packet(manifest: dict, index: int, method: str) -> bytes:
    if method not in manifest["methods"]:
        raise ValueError("Unknown saved tracking method")
    run = Path(manifest["source_run"]).expanduser().resolve()
    stamps = [(run / name).stat().st_mtime_ns for name in ("case.json", "frames.jsonl", "summary.json")]
    case, rows, summary, references, truth, label_error = _recording(run, *stamps)
    if summary["input_sha256"] != manifest["input_sha256"] or len(rows) != manifest["frames"]:
        raise ValueError("Scene and playback belong to different experiment inputs")
    if not 0 <= index < len(rows):
        raise IndexError("Frame is outside this recording")
    frame, masks = sample_at(case, index)
    row = rows[index]
    if row["frame"] != frame.frame_id or not np.isclose(
        row["time_s"], frame.source_time_s, atol=1e-6, rtol=0
    ):
        raise ValueError("RGB-D and estimated poses belong to different frames")
    objects = {}
    for role, reference in references.items():
        result = row["objects"][role]
        model = reference
        initial_surface = reference
        model_used = reference
        version = result.get(method + "_details", {}).get("map_version_after", 0)
        if version and method in {"conditioned_fixed", "conditioned_growing", "group_fixed", "group_growing"}:
            model_path = (run / "surface_models" / method / role / f"{int(version):06d}.npz").resolve()
            if not model_path.is_relative_to(run):
                raise ValueError("Surface is outside its experiment")
            with np.load(model_path, allow_pickle=False) as saved:
                model = saved["points"]
            before = result[method + "_details"].get("map_version_used", version)
            if before:
                with np.load(model_path.with_name(f"{int(before):06d}.npz"), allow_pickle=False) as saved:
                    model_used = saved["points"]
            else:
                model_used = np.empty((0, 3))
            # Show the initial stored map at its actual sampling density, not
            # the denser observation retained separately for scoring and axes.
            with np.load(model_path.with_name("000001.npz"), allow_pickle=False) as saved:
                initial_surface = saved["points"]
        first = summary.get("roles", {}).get(role, {}).get("reference_frame")
        if first is None:
            first = next((i for i, r in enumerate(rows) if r["objects"][role].get(method) is not None), None)
        labelled_pose = None
        if role in truth and first is not None and index >= first:
            labelled_pose = (truth[role][index] @ np.linalg.inv(truth[role][first])).tolist()
        error = None
        if labelled_pose is not None and result.get(method) is not None:
            distance, angle, _ = score_transforms(
                np.asarray([result[method]]), np.asarray([True]), np.asarray([labelled_pose]), reference
            )
            error = {"point_mm": float(distance[0]), "angle_deg": float(angle[0])}
        label_surface = reference
        label_kind = "initial_observation_at_labelled_pose" if labelled_pose is not None else None
        surface_spec = case.get("evaluation_surfaces", {}).get(role)
        surface_error = None
        if labelled_pose is not None and surface_spec:
            file = Path(surface_spec["points_m"]).expanduser().resolve()
            model_points = _label_surface(file, file.stat().st_mtime_ns)
            # Supplied models use the label's object coordinates. Convert once
            # into the same initial camera frame as the tracker's reference.
            origin = truth[role][first]
            label_surface = model_points @ origin[:3, :3].T + origin[:3, 3]
            label_kind = "dataset_model"
            if result.get(method) is not None and len(model):
                surface_error = distance_statistics(
                    LabelledSurface(model_points).distances_mm(
                        model, np.asarray(result[method]), truth[role][index]
                    )
                )
        objects[role] = {
            **object_evidence(result, method),
            "surface": _points(model),
            "surface_used": _points(model_used),
            "initial_surface": _points(initial_surface),
            "pose": result.get(method),
            "reference": _points(reference),
            "axis_points": reference_axes(reference).tolist(),
            "labelled_pose": labelled_pose,
            "labelled_surface": _points(label_surface),
            "label_kind": label_kind,
            "model_version": version,
            "error": error,
            "surface_error": surface_error,
            "group_tracks": result.get(method + "_details", {}).get("group", {}),
        }
    header = {
        **frame_evidence(row, masks),
        "frame": index,
        "time_s": row["time_s"],
        "method": method,
        "input_sha256": manifest["input_sha256"],
        "objects": objects,
        "scene_group_tracks": row.get("scene_group_tracks", []),
        "label_error": label_error,
    }
    from lerobot.gui.flash_live_scene import encode_rgbd_scene

    return encode_rgbd_scene(frame, header)


def rejected_overlay(manifest: dict, index: int, method: str) -> dict:
    """Small, frame-aligned candidate geometry for the opt-in 2D video overlay."""
    if method not in manifest["methods"]:
        raise ValueError("Unknown saved tracking method")
    run = Path(manifest["source_run"]).expanduser().resolve()
    stamps = [(run / name).stat().st_mtime_ns for name in ("case.json", "frames.jsonl", "summary.json")]
    case, rows, summary, references, _, _ = _recording(run, *stamps)
    if summary["input_sha256"] != manifest["input_sha256"] or len(rows) != manifest["frames"]:
        raise ValueError("Candidate and playback belong to different experiment inputs")
    if not 0 <= index < len(rows):
        raise IndexError("Frame is outside this recording")
    row = rows[index]
    frame, _ = sample_at(case, index)
    if row["frame"] != frame.frame_id or not np.isclose(
        row["time_s"], frame.source_time_s, atol=1e-6, rtol=0
    ):
        raise ValueError("Candidate and RGB-D belong to different frames")
    objects = {}
    for role, reference in references.items():
        result = row["objects"][role]
        poses = object_evidence(result, method)["rejected_poses"]
        if not any(pose is not None for pose in poses.values()):
            continue
        details = result.get(method + "_details", {})
        version = details.get("map_version_used")
        surface = reference
        if version and method in {"conditioned_fixed", "conditioned_growing", "group_fixed", "group_growing"}:
            path = run / "surface_models" / method / role / f"{int(version):06d}.npz"
            with np.load(path, allow_pickle=False) as saved:
                surface = saved["points"]
        surface = surface[spatial_sample(surface, 1200)]
        objects[role] = {
            "status": result.get(method + "_status"),
            "surface_used": _points(surface),
            "axis_points": reference_axes(reference).tolist(),
            "rejected_poses": poses,
        }
    return {
        "frame": index,
        "method": method,
        "camera": {"intrinsics": frame.intrinsics.tolist()},
        "objects": objects,
    }
