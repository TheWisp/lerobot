"""Rebuild review evidence from a sealed RGB-D demo with the shared FLASH pipeline.

The output is a sidecar to an existing prepared version. It does not replace
the skill, its poses, or its reference models. Each frame keeps current masks
and the scene-point identities actually retained by the causal group tracker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .appearance import Sam3MaskSource
from .causal_service import CausalTrackingConfig
from .dino_features import DinoPatchExtractor
from .offline_perception import extract_pose_trace
from .prepared import PreparedFlashSkill
from .scene_evidence_writer import SceneEvidenceWriter
from .sealed_demo_import import _frame, _nearest_index
from .vision_pipeline import VisionPipelineConfig, build_vision_backend


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
    identity = hashlib.sha256(
        bundle.read_bytes() + (bundle.parent / "annotation.json").read_bytes()
    ).hexdigest()
    writer = SceneEvidenceWriter(
        destination, total_frames=total, roles=tuple(prepared.prompts_by_role), source_sha256=identity
    )

    def record(frame, result, groups, estimates) -> None:
        writer.record(frame, result, groups, estimates)
        if (frame.frame_id + 1) % 30 == 0:
            print(f"Rebuilt {frame.frame_id + 1}/{total} frames", flush=True)

    try:
        extract_pose_trace(
            (
                _frame(root, index, intrinsics, base_from_camera, version, float(camera_times[index]))
                for index in range(total)
            ),
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
    writer.finish()
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
