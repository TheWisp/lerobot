"""Versioned pose proposals for reviewing offline trackers against a FLASH demo.

The raw RGB-D episode and the selected prepared skill are immutable. A proposal
uses the prepared object's reference frame, so projecting it with the saved
appearance points is meaningful. Applying one creates another prepared skill.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .prepared import PreparedFlashSkill
from .skill import Relation
from .visible_points import read_mask, surface_support

TRACKER_NAMES = {
    "sam3_dino_rgbd": "SAM3 + DINO + RGB-D",
    "point2pose": "Point2Pose",
    "bundlesdf": "BundleSDF",
    "tapnext": "TAPNext++",
    "cotracker3": "CoTracker3 online",
    "litetracker": "LiteTracker",
    "pany": "PANY",
    "trackeverything": "TrackEverything",
}


def tracker_runner_config() -> dict:
    """Read local offline runner installations; never store model paths in a demo."""
    import os

    path = Path(os.environ.get("LEROBOT_FLASH_TRACKERS_CONFIG", "~/.config/lerobot/flash_trackers.json"))
    path = path.expanduser()
    if not path.is_file():
        return {}
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"tracker runner config must be an object: {path}")
    return config


def runner_availability(config: dict | None = None) -> list[dict]:
    """Distinguish a trace we can import from a locally runnable model."""
    config = tracker_runner_config() if config is None else config
    result = []
    for tracker, name in TRACKER_NAMES.items():
        setup = config.get(tracker)
        runnable = isinstance(setup, dict) and (
            (
                tracker == "trackeverything"
                and Path(str(setup.get("python", ""))).is_file()
                and (Path(str(setup.get("repository", ""))) / "demo_hf.py").is_file()
                and Path(str(setup.get("checkpoint", ""))).is_file()
            )
            or (
                isinstance(setup.get("command"), list)
                and bool(setup["command"])
                and Path(str(setup["command"][0])).is_file()
            )
        )
        item = {"id": tracker, "name": name, "runnable": runnable}
        if tracker == "trackeverything" and isinstance(setup, dict) and "command" not in setup:
            item["initial_points_per_object"] = int(setup.get("query_count", 128))
            item["supports_point_growth"] = True
        result.append(item)
    return result


def process_demo(
    dataset_root: Path,
    bundle_path: Path,
    *,
    tracker: str,
    role: str | None = None,
    candidate_id: str,
    expected_episode: int | None = None,
    max_points_per_object: int | None = None,
    new_points_per_view: int | None = None,
    surface_tolerance_m: float | None = None,
) -> dict:
    """Run a configured offline tracker and import its trace for review.

    Other trackers may register an argv template in the local config. They must
    produce the same frames/poses/lost NPZ contract; no shell is involved.
    """
    root, bundle = _dataset_bundle(dataset_root, bundle_path)
    prepared = PreparedFlashSkill.load(bundle)
    if expected_episode is not None and prepared.skill.source_episode != expected_episode:
        raise ValueError("selected episode differs from the prepared demo")
    references = [reference for reference in prepared.references if role is None or reference.role == role]
    if not references:
        raise ValueError("tracker role is not in this demo")
    if any(len(reference.snippets) != 1 for reference in references):
        raise ValueError("this runner needs a single appearance reference view per object")
    if tracker not in TRACKER_NAMES:
        raise ValueError("unknown tracker")
    growth = max_points_per_object is not None
    if not growth and (new_points_per_view is not None or surface_tolerance_m is not None):
        raise ValueError("point growth settings need a total point cap")
    if growth and (
        tracker != "trackeverything" or new_points_per_view is None or surface_tolerance_m is None
    ):
        raise ValueError("point growth needs TrackEverything and complete growth settings")
    if growth and (
        max_points_per_object < 3
        or new_points_per_view < 3
        or not np.isfinite(surface_tolerance_m)
        or surface_tolerance_m <= 0
    ):
        raise ValueError("point growth limits and surface tolerance must be positive")
    _candidate_dir(root, candidate_id)
    config = tracker_runner_config()
    setup = config.get(tracker)
    if growth and isinstance(setup, dict) and "command" in setup:
        raise ValueError("the configured external TrackEverything adapter does not support point growth")
    if growth and isinstance(setup, dict) and max_points_per_object < int(setup.get("query_count", 128)):
        raise ValueError("total point cap must include the initial reference queries")
    if not isinstance(setup, dict) or not next(
        (entry for entry in runner_availability(config) if entry["id"] == tracker), {}
    ).get("runnable"):
        raise ValueError(f"{TRACKER_NAMES[tracker]} has no local runner configured")
    last_frame = len(np.atleast_1d(np.loadtxt(root / "recording" / "times.txt"))) - 1
    specs = []
    for reference in references:
        snippet = reference.snippets[0]
        mask = root / snippet.mask_artifact
        if not 0 <= snippet.frame_index <= last_frame or not mask.is_file():
            raise ValueError(f"missing reference mask or out-of-range frame for {reference.role}")
        specs.append({"role": reference.role, "frame_index": snippet.frame_index, "mask_path": str(mask)})
    annotation = json.loads(
        _input_path(root, bundle, "annotation.json", Path("flash/v1/import-annotations.json")).read_text(
            encoding="utf-8"
        )
    )
    run_dir = root / "flash" / "v1" / "tracker-runs" / candidate_id
    run_dir.mkdir(parents=True, exist_ok=False)
    source_hash = _fingerprint(bundle)
    sources = {}
    if tracker == "trackeverything" and "command" not in setup:
        # Track all objects together; each object's correspondences are fitted
        # separately. Different reference timestamps are supported by the model.
        specs_path = run_dir / "references.json"
        specs_path.write_text(json.dumps(specs), encoding="utf-8")
        if growth:
            segment_dir = run_dir / "segmentation"
            segment_dir.mkdir()
            prompts_path = run_dir / "prompts.json"
            prompts_path.write_text(json.dumps(prepared.prompts_by_role), encoding="utf-8")
            segment_script = (
                Path(__file__).resolve().parents[4] / "benchmarks" / "flash_segment_demo_masks.py"
            )
            segment_args = [
                sys.executable,
                str(segment_script),
                "--recording",
                str(root / "recording"),
                "--reference-specs",
                str(specs_path),
                "--prompts",
                str(prompts_path),
                "--output",
                str(segment_dir / "masks"),
            ]
            if annotation.get("sam_resolution") is not None:
                segment_args.extend(("--resolution", str(annotation["sam_resolution"])))
            _run_tracker(segment_args, segment_dir, setup)
        output = run_dir / "scene.npz"
        script = Path(__file__).resolve().parents[4] / "benchmarks" / "flash_trackeverything_pose.py"
        args = [
            str(setup["python"]),
            str(script),
            "--recording",
            str(root / "recording"),
            "--reference-specs",
            str(specs_path),
            "--repository",
            str(setup["repository"]),
            "--checkpoint",
            str(setup["checkpoint"]),
            "--first-frame",
            str(min(spec["frame_index"] for spec in specs)),
            "--last-frame",
            str(last_frame),
            "--output",
            str(output),
            "--query-count",
            str(setup.get("query_count", 128)),
            "--mode",
            str(setup.get("mode", "dense")),
            "--fit-threshold-m",
            str(annotation["registration_threshold_m"]),
            "--min-inliers",
            str(annotation["registration_min_inliers"]),
            "--fit-trials",
            str(annotation["registration_trials"]),
        ]
        if growth:
            args.extend(
                (
                    "--growth-masks",
                    str(segment_dir / "masks"),
                    "--max-points-per-role",
                    str(max_points_per_object),
                    "--points-per-view",
                    str(new_points_per_view),
                    "--support-tolerance-m",
                    str(surface_tolerance_m),
                )
            )
            for setting, flag in (
                ("surface_icp_iterations", "--surface-icp-iterations"),
                ("surface_trim_fraction", "--surface-trim-fraction"),
                ("surface_edge_pixels", "--surface-edge-pixels"),
                ("max_surface_points", "--max-surface-points"),
            ):
                if setting in setup:
                    args.extend((flag, str(setup[setting])))
        import os

        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, env.get("PYTHONPATH"))))
        _run_tracker(args, run_dir, setup, env)
        with np.load(output, allow_pickle=False) as data:
            if data["roles"].tolist() != [spec["role"] for spec in specs]:
                raise ValueError("tracker output object identities differ from the requested objects")
            metadata = json.loads(str(data["metadata"]))
            query_roles = (
                data["query_roles"]
                if "query_roles" in data
                else np.concatenate(
                    [np.repeat(spec["role"], int(metadata["query_counts"][spec["role"]])) for spec in specs]
                )
            )
            initial_count = (
                int(data["original_query_count"]) if "original_query_count" in data else len(query_roles)
            )
            surface_refined_frames_by_role = (
                {spec["role"]: int(data["surface_refined"][index].sum()) for index, spec in enumerate(specs)}
                if growth
                else {}
            )
            for index, spec in enumerate(specs):
                source = run_dir / f"object-{index}.npz"
                if growth:
                    visible = data["visible_surface_support"][:, query_roles == spec["role"]].sum(axis=1)
                else:
                    initial_role = query_roles[:initial_count] == spec["role"]
                    visible = (data["query_visibility_logits"][:, :initial_count][:, initial_role] > 0).sum(
                        axis=1
                    )
                np.savez_compressed(
                    source,
                    frames=data["frames"],
                    poses=data["poses"][index],
                    lost=data["lost"][index],
                    inliers=data["inliers"][index],
                    visible_queries=visible,
                    temporal_conflicts=data["temporal_conflicts"][index],
                )
                sources[spec["role"]] = source
    else:
        # External single-object adapters still receive every annotated object,
        # sequentially to avoid duplicate GPU allocations. One UI job and result.
        for index, spec in enumerate(specs):
            object_dir = run_dir / f"object-{index}"
            object_dir.mkdir()
            output = object_dir / "poses.npz"
            values = {
                "dataset_root": str(root),
                "recording": str(root / "recording"),
                "bundle_path": str(bundle),
                "reference_mask": spec["mask_path"],
                "role": spec["role"],
                "first_frame": str(spec["frame_index"]),
                "last_frame": str(last_frame),
                "output": str(output),
                "run_dir": str(object_dir),
            }
            try:
                args = [str(part).format_map(values) for part in setup["command"]]
            except KeyError as exc:
                raise ValueError(f"unknown runner command placeholder: {exc}") from exc
            _run_tracker(args, object_dir, setup)
            sources[spec["role"]] = output
    if _fingerprint(bundle) != source_hash:
        raise ValueError("prepared demo changed while the tracker was processing; rerun on the new version")
    result = import_relative_traces(
        root,
        bundle,
        tracker=tracker,
        sources=sources,
        candidate_id=candidate_id,
        expected_episode=expected_episode,
    )
    if tracker == "trackeverything" and "command" not in setup:
        destination = _candidate_dir(root, candidate_id) / "manifest.json"
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        manifest["inference_mode"] = str(setup.get("mode", "dense"))
        manifest["raw_query_trace"] = str(output)
        if growth:
            manifest["point_growth"] = {
                "max_points_per_object": max_points_per_object,
                "new_points_per_view": new_points_per_view,
                "surface_tolerance_m": surface_tolerance_m,
                "events": metadata["growth_events"],
                "pose_fit": metadata["fit_mode"],
                "surface_refined_frames_by_role": surface_refined_frames_by_role,
                "surface_refinement": metadata["surface_refinement"],
            }
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result.update(
            inference_mode=manifest["inference_mode"],
            raw_query_trace=manifest["raw_query_trace"],
            point_growth=manifest.get("point_growth"),
        )
    return result


def _run_tracker(args: list[str], run_dir: Path, setup: dict, env: dict | None = None) -> None:
    if not args or not Path(args[0]).is_file():
        raise ValueError("tracker runner executable is missing")
    timeout_s = float(setup.get("timeout_s", 3600))
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("tracker timeout_s must be positive")
    # Kill descendants too (some adapters launch a separate model environment).
    import os
    import signal

    with (run_dir / "runner.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("tracker runner exceeded its configured processing time limit") from exc
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if code:
        tail = (run_dir / "runner.log").read_text(encoding="utf-8")[-1500:]
        raise ValueError(f"tracker runner failed ({code}): {tail}")


def _dataset_bundle(dataset_root: Path, bundle_path: Path) -> tuple[Path, Path]:
    root = dataset_root.expanduser().resolve()
    bundle = bundle_path.expanduser().resolve()
    prepared_root = root / "flash" / "v1" / "prepared"
    if bundle.name != "bundle.json" or bundle.parent.parent != prepared_root or not bundle.is_file():
        raise ValueError("select a prepared FLASH skill in this dataset")
    if not (root / "meta" / "info.json").is_file():
        raise ValueError("the selected dataset has no metadata")
    return root, bundle


def _candidate_dir(root: Path, candidate_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", candidate_id) is None:
        raise ValueError("invalid tracker candidate ID")
    return root / "flash" / "v1" / "tracker-candidates" / candidate_id


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_path(root: Path, bundle: Path, filename: str, fallback: Path) -> Path:
    versioned = bundle.parent / filename
    source = versioned if versioned.is_file() else root / fallback
    if not source.is_file():
        raise ValueError(f"demo has no saved {filename}")
    return source


def _validate_poses(frames: np.ndarray, poses: np.ndarray, lost: np.ndarray, total_frames: int) -> None:
    if (
        frames.ndim != 1
        or not len(frames)
        or poses.shape != (len(frames), 4, 4)
        or lost.shape != (len(frames),)
        or lost.dtype != np.bool_
        or not np.issubdtype(frames.dtype, np.integer)
        or frames[0] < 0
        or frames[-1] >= total_frames
        or np.any(np.diff(frames) <= 0)
    ):
        raise ValueError("tracker output needs increasing in-range frames and one 4×4 pose per frame")
    valid = poses[~lost]
    if not np.isfinite(valid).all() or not np.allclose(valid[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("available tracker poses must be finite homogeneous transforms")
    rotations = valid[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-2) or not np.allclose(
        np.linalg.det(rotations), 1, atol=1e-2
    ):
        raise ValueError("available tracker rotations must be rigid and right-handed")


def import_relative_trace(
    dataset_root: Path,
    bundle_path: Path,
    *,
    tracker: str,
    role: str,
    source_path: Path,
    candidate_id: str,
    expected_episode: int | None = None,
) -> dict:
    """Import a camera-reference trace without touching the demo skill.

    The input NPZ contains ``frames``, ``poses`` and ``lost``. Each pose maps
    points in the selected appearance reference camera frame into the current
    camera frame. The source must include a valid self-pose at that reference
    frame; this is normalized away before mapping into the calibrated base.
    """
    return import_relative_traces(
        dataset_root,
        bundle_path,
        tracker=tracker,
        sources={role: source_path},
        candidate_id=candidate_id,
        expected_episode=expected_episode,
    )


def _relative_trace(root: Path, bundle: Path, prepared: PreparedFlashSkill, role: str, source_path: Path):
    if role not in {reference.role for reference in prepared.references}:
        raise ValueError("tracker role is not in this demo")
    reference = next(reference for reference in prepared.references if reference.role == role)
    if len(reference.snippets) != 1:
        raise ValueError("this importer needs a single appearance reference view")
    reference_frame = reference.snippets[0].frame_index
    source = source_path.expanduser().resolve()
    if not source.is_file() or source.suffix != ".npz" or source.stat().st_size > 1024**3:
        raise ValueError("select an existing tracker NPZ under 1 GiB")
    camera_times = np.atleast_1d(np.loadtxt(root / "recording" / "times.txt"))
    with np.load(source, allow_pickle=False) as data:
        frames = np.asarray(data["frames"])
        poses = np.asarray(data["poses"], dtype=np.float64)
        lost = np.asarray(data["lost"])
        support = (
            (
                np.asarray(data["inliers"]),
                np.asarray(data["visible_queries"]),
                np.asarray(data["temporal_conflicts"]) if "temporal_conflicts" in data else None,
            )
            if "inliers" in data and "visible_queries" in data
            else None
        )
    _validate_poses(frames, poses, lost, len(camera_times))
    matches = np.flatnonzero(frames == reference_frame)
    if len(matches) != 1 or lost[matches[0]]:
        raise ValueError(f"tracker output needs a valid pose at appearance frame {reference_frame}")
    annotation = json.loads(
        _input_path(root, bundle, "annotation.json", Path("flash/v1/import-annotations.json")).read_text(
            encoding="utf-8"
        )
    )
    calibration = json.loads(Path(annotation["calibration_path"]).expanduser().read_text(encoding="utf-8"))
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=np.float64)
    if base_from_camera.shape != (4, 4) or not np.isfinite(base_from_camera).all():
        raise ValueError("demo calibration has no valid base-from-camera transform")
    object_from_reference_camera = np.asarray(reference.snippets[0].object_from_camera, dtype=np.float64)
    base_poses = (
        base_from_camera
        @ poses
        @ np.linalg.inv(poses[matches[0]])
        @ np.linalg.inv(object_from_reference_camera)
    )
    base_poses[lost] = np.eye(4)
    if support is not None and (
        support[0].shape != lost.shape
        or support[1].shape != lost.shape
        or (support[2] is not None and support[2].shape != lost.shape)
    ):
        raise ValueError("tracker support diagnostics do not match pose frames")
    info = {
        "source_trace": str(source),
        "source_trace_sha256": _fingerprint(source),
        "reference_frame": reference_frame,
        "valid_frames": int((~lost).sum()),
        "total_frames": len(frames),
    }
    if support is not None and (~lost).any():
        info["median_inliers"] = float(np.median(support[0][~lost]))
        info["median_visible_queries"] = float(np.median(support[1][~lost]))
        if support[2] is not None:
            info["temporal_conflict_frames"] = int(np.count_nonzero(support[2]))
    return {"frames": frames, "poses": base_poses, "lost": lost}, info


def import_relative_traces(
    dataset_root: Path,
    bundle_path: Path,
    *,
    tracker: str,
    sources: dict[str, Path],
    candidate_id: str,
    expected_episode: int | None = None,
) -> dict:
    """Validate every object's trace before publishing one review candidate."""
    root, bundle = _dataset_bundle(dataset_root, bundle_path)
    if tracker not in TRACKER_NAMES or not sources:
        raise ValueError("unknown tracker or empty object selection")
    prepared = PreparedFlashSkill.load(bundle)
    if expected_episode is not None and prepared.skill.source_episode != expected_episode:
        raise ValueError("selected episode differs from the prepared demo")
    arrays, coverage = {}, {}
    for index, (role, source) in enumerate(sources.items()):
        trace, info = _relative_trace(root, bundle, prepared, role, source)
        arrays.update({f"{key}_{index}": value for key, value in trace.items()})
        coverage[role] = info
    manifest = {
        "schema_version": 2,
        "tracker": tracker,
        "roles": list(sources),
        "source_episode": prepared.skill.source_episode,
        "source_bundle": str(bundle),
        "source_bundle_sha256": _fingerprint(bundle),
        "pose_frame": "base_from_prepared_object",
        "coverage_by_role": coverage,
        "status": "preview",
    }
    # Keep single-object clients compatible, without inventing scene-level frame counts.
    if len(sources) == 1:
        role = next(iter(sources))
        manifest.update(role=role, **coverage[role])
    destination = _candidate_dir(root, candidate_id)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(destination / "poses.npz", **arrays)
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        # safe-destruct: remove only the new version created by this failed operation
        shutil.rmtree(destination)
        raise
    return {"candidate_id": candidate_id, **manifest}


def list_candidates(dataset_root: Path, bundle_path: Path) -> list[dict]:
    root, bundle = _dataset_bundle(dataset_root, bundle_path)
    found = []
    for path in (root / "flash" / "v1" / "tracker-candidates").glob("*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("source_bundle") == str(bundle) and manifest.get("schema_version") in (1, 2):
                found.append({"candidate_id": path.parent.name, **manifest})
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda item: item["candidate_id"])


def load_candidate(
    dataset_root: Path, bundle_path: Path, candidate_id: str
) -> tuple[dict, dict[int, np.ndarray]]:
    manifest, poses_by_role = load_candidate_set(dataset_root, bundle_path, candidate_id)
    if len(poses_by_role) != 1:
        raise ValueError("use load_candidate_set for a multi-object candidate")
    return manifest, next(iter(poses_by_role.values()))


def load_candidate_set(dataset_root: Path, bundle_path: Path, candidate_id: str):
    root, bundle = _dataset_bundle(dataset_root, bundle_path)
    destination = _candidate_dir(root, candidate_id)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_bundle") != str(bundle) or manifest.get("source_bundle_sha256") != _fingerprint(
        bundle
    ):
        raise ValueError("tracker candidate belongs to another or changed prepared demo")
    version = manifest.get("schema_version")
    if version not in (1, 2):
        raise ValueError("unknown tracker candidate schema")
    roles = manifest["roles"] if version == 2 else [manifest["role"]]
    total = len(np.atleast_1d(np.loadtxt(root / "recording" / "times.txt")))
    poses_by_role = {}
    with np.load(destination / "poses.npz", allow_pickle=False) as data:
        for index, role in enumerate(roles):
            suffix = f"_{index}" if version == 2 else ""
            frames, poses, lost = (data[key + suffix] for key in ("frames", "poses", "lost"))
            _validate_poses(frames, poses, lost, total)
            poses_by_role[role] = {
                int(frame): pose
                for frame, pose, unavailable in zip(frames, poses, lost, strict=True)
                if not unavailable
            }
    return manifest, poses_by_role


def load_candidate_point_tracks(dataset_root: Path, bundle_path: Path, candidate_id: str) -> dict:
    """Project the model's raw query tracks, without using the fitted poses.

    The residual is supplied separately so the replay can distinguish a point
    trajectory from the rigid fit imposed on those trajectories. TrackEverything
    currently runs with an identity camera extrinsic on fixed-camera demos.
    """
    root, bundle = _dataset_bundle(dataset_root, bundle_path)
    manifest, _ = load_candidate_set(root, bundle, candidate_id)
    if manifest.get("tracker") != "trackeverything" or not manifest.get("raw_query_trace"):
        return {"available": False, "reason": "This candidate did not retain raw point tracks."}
    trace_path = Path(manifest["raw_query_trace"]).resolve()
    if not trace_path.is_relative_to((root / "flash/v1/tracker-runs").resolve()):
        raise ValueError("raw point trace is outside this dataset's tracker runs")
    intrinsics = np.loadtxt(root / "recording/cam_K.txt").reshape(3, 3)
    with np.load(trace_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"], dtype=int)
        xyz = np.asarray(data["query_coords_m"], dtype=np.float64)
        visible = np.asarray(data["query_visibility_logits"]) > 0
        queries = np.asarray(data["query_points_2d"])
        roles = [str(role) for role in data["roles"]]
        poses = np.asarray(data["poses"], dtype=np.float64)
        lost = np.asarray(data["lost"], dtype=bool)
        metadata = json.loads(str(data["metadata"]))
        supported_trace = (
            np.asarray(data["visible_surface_support"], dtype=bool)
            if "visible_surface_support" in data
            else None
        )
        query_roles = (
            np.asarray(data["query_roles"])
            if "query_roles" in data
            else np.concatenate([np.repeat(role, int(metadata["query_counts"][role])) for role in roles])
        )
    if "fit_threshold_m" in metadata:
        threshold_mm = float(metadata["fit_threshold_m"]) * 1000
    else:
        # Older retained traces did not save their fit tolerance. Their source
        # preparation is immutable; use the annotation that generated it.
        annotation = json.loads(
            _input_path(root, bundle, "annotation.json", Path("flash/v1/import-annotations.json")).read_text(
                encoding="utf-8"
            )
        )
        threshold_mm = float(annotation["registration_threshold_m"]) * 1000
    counts = [int((query_roles == role).sum()) for role in roles]
    if roles != manifest["roles"] or xyz.shape != (len(frames), sum(counts), 3):
        raise ValueError("raw point trace does not match the candidate")
    if visible.shape != xyz.shape[:2] or queries.shape != (sum(counts), 3):
        raise ValueError("raw point visibility or origins do not match the trace")
    if supported_trace is not None and supported_trace.shape != visible.shape:
        raise ValueError("visible surface support does not match the point trace")
    if poses.shape != (len(roles), len(frames), 4, 4) or lost.shape != (len(roles), len(frames)):
        raise ValueError("raw pose diagnostics do not match the point trace")
    growth = manifest.get("point_growth")
    masks_root = trace_path.parent / "segmentation" / "masks" if growth else None
    output_roles = []
    for role_index, (role, count) in enumerate(zip(roles, counts, strict=True)):
        role_indices = np.flatnonzero(query_roles == role)
        births = queries[role_indices, 0].astype(int)
        if np.any((births < 0) | (births >= len(frames))):
            raise ValueError("raw point birth frame lies outside the trace")
        source_points = np.full((count, 3), np.nan)
        for point_index, (column, birth) in enumerate(zip(role_indices, births, strict=True)):
            if not lost[role_index, birth]:
                fit = poses[role_index, birth]
                source_points[point_index] = (xyz[birth, column] - fit[:3, 3]) @ fit[:3, :3]
        positions = []
        for index in range(len(frames)):
            current = xyz[index, role_indices]
            point_visible = visible[index, role_indices] & (births <= index)
            if masks_root is not None and supported_trace is not None:
                point_visible &= supported_trace[index, role_indices]
            elif masks_root is not None:
                raw_depth = cv2.imread(
                    str(root / "recording" / "depth" / f"{frames[index]:06d}.png"),
                    cv2.IMREAD_UNCHANGED,
                )
                if raw_depth is None:
                    raise ValueError(f"missing recorded depth frame {frames[index]}")
                depth = raw_depth.astype(np.float32) * 0.001
                mask = read_mask(masks_root / str(role_index), int(frames[index]), depth.shape)
                if mask is None:
                    point_visible[:] = False
                else:
                    point_visible, _, _ = surface_support(
                        current,
                        point_visible.astype(float),
                        mask,
                        depth,
                        intrinsics,
                        float(growth["surface_tolerance_m"]),
                    )
            if lost[role_index, index]:
                residual_mm = np.full(count, np.nan)
            else:
                fit = poses[role_index, index]
                expected = source_points @ fit[:3, :3].T + fit[:3, 3]
                residual_mm = np.linalg.norm(expected - current, axis=1) * 1000
            row = []
            for point_index, point in enumerate(current):
                if not point_visible[point_index] or not np.isfinite(point).all() or point[2] <= 0:
                    row.append(None)
                    continue
                u = intrinsics[0, 0] * point[0] / point[2] + intrinsics[0, 2]
                v = intrinsics[1, 1] * point[1] / point[2] + intrinsics[1, 2]
                if not np.isfinite([u, v]).all():
                    row.append(None)
                    continue
                residual = residual_mm[point_index]
                row.append(
                    [
                        round(float(u), 2),
                        round(float(v), 2),
                        round(float(residual), 2) if np.isfinite(residual) else None,
                    ]
                )
            positions.append(row)
        output_roles.append(
            {"role": role, "count": count, "positions": positions, "birth_frames": frames[births].tolist()}
        )
    return {
        "available": True,
        "candidate_id": candidate_id,
        "frames": frames.tolist(),
        "fit_threshold_mm": threshold_mm,
        "surface_filtered": masks_root is not None,
        "roles": output_roles,
    }


def apply_candidate(
    dataset_root: Path,
    bundle_path: Path,
    candidate_id: str,
    *,
    output_name: str,
    expected_episode: int | None = None,
) -> tuple[Path, dict]:
    """Create a new prepared version with candidate poses at exact camera frames.

    Missing candidate frames retain their former sample pose and evidence. The
    report counts both outcomes; sparse outputs never masquerade as 30 Hz data.
    """
    root, bundle = _dataset_bundle(dataset_root, bundle_path)
    manifest, poses_by_role = load_candidate_set(root, bundle, candidate_id)
    destination = root / "flash" / "v1" / "prepared" / output_name
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", output_name) is None or destination.exists():
        raise ValueError("choose an unused prepared version name")
    prepared = PreparedFlashSkill.load(bundle)
    if expected_episode is not None and prepared.skill.source_episode != expected_episode:
        raise ValueError("selected episode differs from the prepared demo")
    if manifest["source_episode"] != prepared.skill.source_episode:
        raise ValueError("tracker candidate belongs to another episode")
    with np.load(root / "showservo_demo.npz", allow_pickle=False) as arm:
        arm_times_s = np.asarray(arm["t"], dtype=float)
        origin_s = float(arm["t0"])
    camera_times_s = np.atleast_1d(np.loadtxt(root / "recording" / "times.txt")) - origin_s
    changed_reference = changed_held = preserved = 0
    intervals = []
    for interval in prepared.skill.intervals:
        samples = []
        for sample in interval.samples:
            frame = (
                int(np.argmin(np.abs(camera_times_s - arm_times_s[sample.source_frame_index])))
                if sample.source_frame_index is not None
                else -1
            )
            updates = {}
            for role, poses in poses_by_role.items():
                candidate_pose = poses.get(frame)
                is_reference = interval.reference_role == role and interval.relation != Relation.EE_TO_START
                is_held = interval.held_role == role and sample.demo_held is not None
                if candidate_pose is not None:
                    if is_reference:
                        updates.update(
                            demo_reference=candidate_pose,
                            reference_evidence=f"observed:{manifest['tracker']}",
                        )
                        changed_reference += 1
                    if is_held:
                        updates.update(
                            demo_held=candidate_pose, held_evidence=f"observed:{manifest['tracker']}"
                        )
                        changed_held += 1
                elif is_reference or is_held:
                    preserved += 1
            samples.append(replace(sample, **updates) if updates else sample)
        intervals.append(replace(interval, samples=tuple(samples)))
    if not changed_reference and not changed_held:
        raise ValueError("candidate has no valid pose at any affected skill sample")
    skill = replace(prepared.skill, intervals=tuple(intervals))
    revised = replace(prepared, skill=skill)
    revised.save(destination)
    try:
        for name, fallback in (
            ("annotation.json", Path("flash/v1/import-annotations.json")),
            ("keypoints.json", Path("keypoints.json")),
        ):
            shutil.copy2(_input_path(root, bundle, name, fallback), destination / name)
        original_report = bundle.parent / "preparation-report.json"
        report = json.loads(original_report.read_text(encoding="utf-8")) if original_report.is_file() else {}
        # Full-frame extraction counts and caches describe the old version's
        # pose trace. They cannot honestly describe this sample-level edit.
        for key in (
            "observed_pose_frames",
            "pose_evidence_frames",
            "pose_trace_cache",
            "pose_trace_cache_hit",
        ):
            report.pop(key, None)
        report.update(
            review_status="accepted",
            applied_tracker_candidate=candidate_id,
            applied_tracker=manifest["tracker"],
            changed_reference_samples=changed_reference,
            changed_held_samples=changed_held,
            preserved_samples=preserved,
            previous_bundle=str(bundle),
        )
        (destination / "preparation-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        # safe-destruct: remove only the new version created by this failed operation
        shutil.rmtree(destination)
        raise
    return destination / "bundle.json", report
