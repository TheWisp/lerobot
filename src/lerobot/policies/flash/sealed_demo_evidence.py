"""Rebuild review evidence from a sealed RGB-D demo with the shared FLASH pipeline.

The output is a sidecar to an existing prepared version. It does not replace
the skill, its poses, or its reference models. Each frame keeps current masks
and the scene-point identities actually retained by the causal group tracker.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .appearance import Sam3MaskSource
from .causal_service import CausalTrackingConfig
from .dino_features import DinoPatchExtractor
from .offline_perception import extract_pose_trace
from .prepared import PreparedFlashSkill
from .sealed_demo_import import _frame, _nearest_index
from .vision_pipeline import VisionPipelineConfig, build_vision_backend


def _mask_runs(mask: np.ndarray) -> list[list[int]]:
    flat = np.asarray(mask, dtype=np.uint8).ravel()
    edges = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
    return [[int(start), int(end - start)] for start, end in edges.reshape(-1, 2)]


def rebuild(root: Path, bundle: Path, *, through_frame: int | None = None, device: str = "cuda") -> Path:
    if through_frame is not None and through_frame < 0:
        raise ValueError("through_frame must be nonnegative")
    root = root.expanduser().resolve()
    bundle = bundle.expanduser().resolve()
    bundle.relative_to(root / "flash" / "v1" / "prepared")
    prepared = PreparedFlashSkill.load(bundle)
    annotation = json.loads((bundle.parent / "annotation.json").read_text(encoding="utf-8"))
    with np.load(root / "showservo_demo.npz", allow_pickle=False) as arm:
        origin_s = float(arm["t0"])
        arm_times = np.asarray(arm["t"], dtype=float)
    camera_times = np.atleast_1d(np.loadtxt(root / "recording" / "times.txt")) - origin_s
    total = len(camera_times) if through_frame is None else min(len(camera_times), through_frame + 1)
    intrinsics = np.loadtxt(root / "recording" / "cam_K.txt").reshape(3, 3)
    base_from_camera = np.asarray(
        json.loads(Path(annotation["calibration_path"]).expanduser().read_text(encoding="utf-8"))["camera"][
            "T_base_cam"
        ],
        dtype=float,
    )
    version = hashlib.sha256(
        intrinsics.tobytes() + base_from_camera.tobytes() + str(annotation["reference_frame"]).encode()
    ).hexdigest()[:12]
    config = prepared.vision_pipeline or VisionPipelineConfig.from_settings(annotation, backend="appearance")
    causal_config = (
        CausalTrackingConfig.from_config(annotation["causal_tracking"])
        if annotation.get("causal_tracking") is not None
        else None
    )
    backend = build_vision_backend(
        config,
        Sam3MaskSource(
            prepared.prompts_by_role,
            camera_id="flash-demo-evidence",
            device=device,
            resolution=config.sam_resolution,
        ),
        models=prepared.appearance_models,
        extractor=DinoPatchExtractor(model_id=prepared.descriptor_model_id, device=device),
        axial_symmetries=prepared.axial_symmetries,
    )
    grasp_camera = _nearest_index(
        camera_times, arm_times[_nearest_index(arm_times, annotation["grasp_time_s"])]
    )
    release_camera = _nearest_index(
        camera_times, arm_times[_nearest_index(arm_times, annotation["release_time_s"])]
    )
    destination = bundle.parent / "scene-evidence"
    destination.mkdir(exist_ok=True)
    identity = hashlib.sha256(
        bundle.read_bytes() + (bundle.parent / "annotation.json").read_bytes()
    ).hexdigest()
    manifest_path = destination / "manifest.json"
    manifest = {
        "schema_version": 1,
        "source_sha256": identity,
        "total_frames": total,
        "frames_completed": 0,
        "complete": False,
        "roles": list(prepared.prompts_by_role),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def record(frame, result, groups, estimates) -> None:
        data = {
            "frame_index": frame.frame_id,
            "image_size": list(frame.depth.shape),
            "masks_rle": {role: _mask_runs(mask) for role, mask in result.masks.items()},
            "groups": {
                role: {
                    "verified": bool(group["verified"]),
                    "measured_this_frame": bool(group["measured_this_frame"]),
                    "ids": group["ids"].tolist(),
                    "pixels": group["xy"].tolist(),
                    "base_xyz_m": np.round(group["base_xyz"], 6).tolist(),
                    "last_group_measurement_s": group["last_group_measurement_s"],
                }
                for role, group in groups.items()
            },
            "estimates": {
                role: {
                    "evidence": estimate.evidence.value,
                    "estimator": estimate.estimator,
                    "pose": estimate.pose.tolist() if estimate.pose is not None else None,
                }
                for role, estimate in estimates.items()
            },
            "proposals": {
                proposal.role: {
                    "evidence": proposal.estimate.evidence.value,
                    "estimator": proposal.estimate.estimator,
                    "raw_pose": proposal.raw_pose.tolist() if proposal.raw_pose is not None else None,
                    "failure": proposal.failure,
                    "visible_support": asdict(proposal.visible_support),
                }
                for proposal in result.proposals
            },
        }
        frame_path = destination / f"{frame.frame_id:06d}.json.gz"
        temporary = frame_path.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
        temporary.replace(frame_path)
        manifest["frames_completed"] = frame.frame_id + 1
        if (frame.frame_id + 1) % 30 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Rebuilt {frame.frame_id + 1}/{total} frames", flush=True)

    try:
        extract_pose_trace(
            (_frame(root, index, intrinsics, base_from_camera, version) for index in range(total)),
            backend,
            prepared.references,
            causal_tracking=causal_config,
            object_points_by_role={
                model.points.role: model.points.points_object_m for model in prepared.appearance_models
            },
            axial_symmetries=prepared.axial_symmetries,
            attached_role_at_frame=lambda frame_id: (
                annotation["held_role"] if grasp_camera <= frame_id < release_camera else None
            ),
            on_frame_evidence=record,
        )
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--through-frame", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(rebuild(args.dataset_root, args.bundle, through_frame=args.through_frame, device=args.device))


if __name__ == "__main__":
    main()
