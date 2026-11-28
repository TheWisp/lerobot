"""Export completed tracker experiments for read-only GUI playback; no inference.

The video retains source timestamps. Masks and projected poses use the exact same
source frames, so seeking or switching the displayed method cannot move the overlay
to another camera frame. Nothing is written into the source dataset.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_flow_paths import FlowPaths
from lerobot.gui.flash_group_overlay import GroupOverlay, SceneGroupOverlay
from lerobot.gui.flash_live_scene import object_evidence
from lerobot.gui.flash_pose_overlay import reference_axes
from lerobot.policies.flash.surface_grounding import project
from lerobot.policies.flash.surface_observation import move, spatial_sample

METHODS = {
    "ufm_chain": "Pose from UFM matches",
    "surface_corrected": "Pose after surface correction",
}
# Display budgets only. Full measured correspondence evidence stays in the run.
DISPLAY_SURFACE_POINTS = 1200
DISPLAY_MATCHES = 256
DEFAULT_OUTPUT = Path.home() / ".cache/flash-tracker-restart/browser"


def projected_geometry(reference, pose, intrinsics):
    if pose is None:
        return None
    points = move(reference, np.asarray(pose))
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        return None
    return np.round(project(points, intrinsics), 2).tolist()


def display_matches(pair, times):
    if pair is None:
        return None
    source, target = pair["source_frame"], pair["target_frame"]
    if source not in times or target not in times or source >= target:
        raise ValueError("correspondence source must precede its recorded target")
    a, b, stage = np.asarray(pair["source_uv"]), np.asarray(pair["target_uv"]), np.asarray(pair["stage"])
    if len(a) != len(b) or len(a) != len(stage):
        raise ValueError("correspondence pixel arrays disagree")

    # Separate raw and fit-used display samples so sparse accepted pairs cannot
    # disappear merely because the raw display budget selected other pixels.
    def selected(indices):
        return (
            indices[np.linspace(0, len(indices) - 1, min(len(indices), DISPLAY_MATCHES), dtype=int)]
            if len(indices)
            else indices
        )

    raw = selected(np.arange(len(a)))
    used = selected(np.flatnonzero(stage == 4))
    return {
        "source_frame": source,
        "target_frame": target,
        "source_t": times[source],
        "target_t": times[target],
        "raw": np.column_stack((a[raw], b[raw])).tolist() if len(raw) else [],
        "raw_stage": stage[raw].astype(int).tolist(),
        "used": np.column_stack((a[used], b[used])).tolist() if len(used) else [],
        "counts": np.bincount(stage.astype(int), minlength=5).tolist(),
        "fit_available": pair["fit_available"],
    }


def export(run: Path, output: Path = DEFAULT_OUTPUT, *, check_cancel=None):
    run = run.expanduser().resolve()
    case = json.loads((run / "case.json").read_text())
    summary = json.loads((run / "summary.json").read_text())  # Only completed results.
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    if len(rows) != summary["frames"] or len(rows) < 2:
        raise ValueError("completed result and trace length disagree")
    times = np.asarray([r["time_s"] for r in rows], dtype=float)
    times -= times[0]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("source timestamps must strictly increase")
    rate = Fraction(1 / float(np.median(np.diff(times)))).limit_denominator(10000)
    with np.load(run / "reference_points.npz", allow_pickle=False) as saved:
        references = {}
        for role in saved.files:
            ref = saved[role]
            references[role] = np.concatenate(
                (reference_axes(ref), ref[spatial_sample(ref, DISPLAY_SURFACE_POINTS)])
            )
    recorded_fields = {key for row in rows for obj in row["objects"].values() for key in obj}
    methods = {name: label for name, label in METHODS.items() if name in recorded_fields}
    if any("ufm_retained_reference" in obj for obj in rows[0]["objects"].values()):
        methods["ufm_retained_reference"] = "Pose with earlier-frame recovery"
    for name, label in (
        ("conditioned_fixed", "Image + surface: fixed model"),
        ("conditioned_growing", "Image + surface: growing model"),
        ("group_fixed", "Depth + mask fit, UFM + groups"),
        ("group_growing", "Depth + mask fit, UFM + groups — growing surface"),
    ):
        if any(name in obj for obj in rows[0]["objects"].values()):
            methods[name] = label
    shared = summary.get("shared_matching", {}).get("config")
    if shared:
        methods = {name: label.replace("UFM", shared["backend"].upper()) for name, label in methods.items()}
    output.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256((str(run) + ":schema9").encode()).hexdigest()[:20]
    destination = output / identity
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=output))
    frames = []
    times_by_frame = {row["frame"]: float(t) for row, t in zip(rows, times, strict=True)}
    evidence_path = run / "correspondences.jsonl.gz"
    digest = hashlib.sha256()
    flow_paths = {}
    group_overlays = defaultdict(GroupOverlay)
    scene_group_overlay = SceneGroupOverlay()
    model_cache = {}
    try:
        with (
            av.open(str(temporary / "video.mp4"), "w", options={"movflags": "+faststart"}) as container,
            gzip.open(evidence_path, "rt") if evidence_path.exists() else nullcontext() as evidence,
        ):
            stream = container.add_stream("libx264", rate=rate)
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 1_000_000)
            stream.codec_context.time_base = stream.time_base
            stream.options = {"crf": "20", "preset": "fast", "bf": "0"}
            width = height = None
            for (frame, masks), row, t in zip(samples(case), rows, times, strict=True):
                if check_cancel is not None:
                    check_cancel()
                if frame.frame_id != row["frame"] or not np.isclose(frame.source_time_s, row["time_s"]):
                    raise ValueError("result does not belong to this camera frame")
                for value in (
                    frame.rgb,
                    frame.depth,
                    frame.intrinsics,
                    np.asarray(frame.source_time_s),
                    *masks.values(),
                ):
                    a = np.asarray(value)
                    digest.update(str((a.shape, a.dtype)).encode() + a.tobytes())
                h, w = frame.rgb.shape[:2]
                if width is None:
                    width, height = w, h
                    # Padding only supports odd-sized sources; overlay coordinates remain unchanged.
                    stream.width, stream.height = w + w % 2, h + h % 2
                elif (w, h) != (width, height):
                    raise ValueError("camera dimensions changed")
                rgb = np.pad(frame.rgb, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                video_frame.pts = round(float(t) * 1_000_000)
                video_frame.time_base = stream.time_base
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                match_row = json.loads(next(evidence)) if evidence is not None else None
                if match_row is not None and match_row["frame"] != row["frame"]:
                    raise ValueError("correspondence evidence belongs to a different frame")
                if match_row is not None and any(
                    pair is not None and pair["target_frame"] != row["frame"]
                    for pairs in match_row["objects"].values()
                    for pair in pairs.values()
                ):
                    raise ValueError("correspondence target belongs to a different frame")
                objects = {}
                for role in case["roles"]:
                    matches = None
                    if match_row is not None:
                        matches = {}
                        for name, pair in match_row["objects"][role].items():
                            display = display_matches(pair, times_by_frame)
                            if display is not None:
                                paths = flow_paths.setdefault((role, name), FlowPaths())
                                display["paths"] = pair["paths"] if "paths" in pair else paths.update(pair)
                                display["path_sampling"] = pair.get("path_sampling", "saved_grid")
                            matches[name] = display
                    contours, _ = cv2.findContours(
                        masks[role].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                    )
                    posed = {}
                    rejected_projected = {}
                    for name in methods:
                        reference = references.get(role)
                        details = row["objects"][role].get(name + "_details", {})
                        version = details.get("map_version_after", 0)
                        if version and name in {
                            "conditioned_fixed",
                            "conditioned_growing",
                            "group_fixed",
                            "group_growing",
                        }:
                            key = (name, role)
                            if model_cache.get(key, (None,))[0] != version:
                                model_path = run / "surface_models" / name / role / f"{version:06d}.npz"
                                with np.load(model_path, allow_pickle=False) as model:
                                    points = model["points"]
                                    cloud = points[spatial_sample(points, DISPLAY_SURFACE_POINTS)]
                                    # Axes stay attached to the original coordinate frame,
                                    # while the displayed measured surface can grow.
                                    geometry = np.concatenate((references[role][:4], cloud))
                                model_cache[key] = (version, geometry)
                            reference = model_cache[key][1]
                        posed[name] = (
                            projected_geometry(reference, row["objects"][role].get(name), frame.intrinsics)
                            if reference is not None
                            else None
                        )
                        rejected = object_evidence(row["objects"][role], name)["rejected_poses"]
                        if any(pose is not None for pose in rejected.values()):
                            used = references[role]
                            before = details.get("map_version_used")
                            if before and name in {
                                "conditioned_fixed",
                                "conditioned_growing",
                                "group_fixed",
                                "group_growing",
                            }:
                                with np.load(
                                    run / "surface_models" / name / role / f"{int(before):06d}.npz",
                                    allow_pickle=False,
                                ) as saved:
                                    cloud = saved["points"]
                                used = np.concatenate((references[role][:4], cloud))
                            cloud = used[4:]
                            used = np.concatenate((used[:4], cloud[spatial_sample(cloud, 500)]))
                            rejected_projected[name] = {
                                kind: projected_geometry(used, pose, frame.intrinsics)
                                for kind, pose in rejected.items()
                            }
                    objects[role] = {
                        "matches": matches,
                        "match_sources": {
                            name: row["objects"][role][name + "_details"].get("image_source_frame")
                            for name in (
                                "conditioned_fixed",
                                "conditioned_growing",
                                "group_fixed",
                                "group_growing",
                            )
                            if name + "_details" in row["objects"][role]
                        },
                        "mask": [c[:, 0].tolist() for c in contours if len(c) >= 3],
                        "poses": posed,
                        "rejected_projected": rejected_projected,
                        "group_tracks": {
                            name: (
                                row["objects"][role].get(name + "_details", {}).get("group", {})
                                if "scene_group_tracks" in row
                                else group_overlays[role, name].update(
                                    row["objects"][role].get(name + "_details", {}).get("group", {})
                                )
                            )
                            for name in methods
                            if name.startswith("group_")
                        },
                        "pose_status": {name: row["objects"][role].get(name + "_status") for name in methods},
                    }
                frames.append(
                    {
                        "frame": frame.frame_id,
                        "t": float(t),
                        "objects": objects,
                        "scene_group_tracks": scene_group_overlay.update(
                            row.get("scene_group_tracks", []),
                            frame_size=(frame.rgb.shape[1], frame.rgb.shape[0]),
                        ),
                    }
                )
            if evidence is not None and next(evidence, None) is not None:
                raise ValueError("extra correspondence frames after video ended")
            for packet in stream.encode():
                container.mux(packet)
        if digest.hexdigest() != summary["input_sha256"]:
            raise ValueError("source RGB-D/masks changed since this experiment; refusing misaligned playback")
        data = {
            "schema_version": 9,
            "surface_experiment": summary.get("surface_experiment"),
            "shared_matching": shared,
            "has_correspondences": evidence_path.exists(),
            "width": stream.width,
            "height": stream.height,
            "methods": methods,
            "frames": frames,
        }
        (temporary / "overlays.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
        manifest = {
            "id": identity,
            "surface_registration": summary.get("surface_registration", {"backend": "scipy"}),
            "shared_matching": shared,
            "name": case.get("display_name", case["id"].replace("_", " "))
            + (" (deformable control)" if "deformable" in case.get("tags", []) else "")
            + (f" — shared {shared['backend'].upper()}" if shared else ""),
            "experiment": f"Shared {shared['backend'].upper()} matching" if shared else run.parent.name,
            "source_run": str(run),
            "frames": len(frames),
            "duration_s": float(times[-1]),
            "methods": methods,
            "input_sha256": digest.hexdigest(),
            "has_correspondences": evidence_path.exists(),
            "mask_source": case.get(
                "mask_source",
                "Dataset annotation" if case["kind"] == "ycb_annotated_pose" else "Recorded segmentation",
            ),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temporary / "summary.json").write_text(json.dumps(summary, allow_nan=False) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(destination, flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for run in args.runs:
        export(run, args.output.expanduser())
