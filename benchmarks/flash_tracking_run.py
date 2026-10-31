"""Rerun frozen RGB-D benchmark cases through the shared FLASH perception path.

Writes new evidence beside the baseline; never uses pose labels for inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def _run_ycb(case: dict, output: Path, config_path: Path) -> None:
    from lerobot.policies.flash.appearance import Sam3MaskSource
    from lerobot.policies.flash.causal_service import CausalTrackingConfig
    from lerobot.policies.flash.offline_perception import extract_pose_trace
    from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
    from lerobot.policies.flash.scene_evidence_writer import SceneEvidenceWriter
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend

    source = Path(case["source_root"]).expanduser()
    names = sorted(path.name for path in (source / "rgb").glob("*.png"))
    start = case["start_frame"]
    names = names[start : start + case["expected_frames"]]
    if len(names) != case["expected_frames"]:
        raise ValueError(f"missing YCB frames: {case['id']}")
    times_ns = np.asarray([int(Path(name).stem) for name in names], dtype=np.int64)
    times = (times_ns - times_ns[0]) * 1e-9
    if np.any(np.diff(times) <= 0):
        raise ValueError("source timestamps are not increasing")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    vision_config = VisionPipelineConfig.from_dict(config["vision_pipeline"])
    segmentation = case.get("segmentation", {"source": "supplied"})
    if segmentation.get("source") == "supplied":
        mask_source = SuppliedMasks(source, names)
        mask_description = "supplied YCBInEOAT mask on every frame"
    elif segmentation.get("source") == "sam3":
        prompt = segmentation.get("prompts_by_role", {}).get("object", "").strip()
        if not prompt:
            raise ValueError(f"{case['id']}: SAM3 segmentation needs a prompt for object")
        seeds = segmentation.get("seed_points_by_role", {})
        if set(seeds) - {"object"}:
            raise ValueError(f"{case['id']}: SAM3 seed has no matching object role")
        mask_source = Sam3MaskSource(
            {"object": prompt},
            camera_id="rgb",
            resolution=vision_config.sam_resolution,
            seed_points_by_role=seeds,
            detection_correction=vision_config.detection_correction.control(offline=True),
        )
        mask_description = "SAM3 text prompt on RGB frames"
    else:
        raise ValueError(f"{case['id']}: unknown segmentation source {segmentation.get('source')!r}")
    intrinsics = np.loadtxt(source / "cam_K.txt").reshape(3, 3)
    digest = hashlib.sha256()
    for name in names:
        for kind in ("rgb", "depth", "gt_mask"):
            digest.update((source / kind / name).read_bytes())
    provenance = {
        "source": str(source),
        "start": start,
        "count": len(names),
        "names": names,
        "source_times_s": times.tolist(),
        "intrinsics": intrinsics.tolist(),
        "mask_source": mask_description,
        "segmentation": segmentation,
        "annotated_poses_used_by_tracker": False,
        "gripper_prediction": False,
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input_sha256": digest.hexdigest(),
        "checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    def frames():
        for index, name in enumerate(names):
            rgb = cv2.imread(str(source / "rgb" / name), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(source / "depth" / name), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None or depth.dtype != np.uint16:
                raise ValueError("invalid source RGB-D frame")
            yield RGBDFrame(
                index,
                cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                depth,
                intrinsics,
                0.001,
                np.eye(4),
                float(times[index]),
                float(times[index]),
                "fixed-camera",
            )

    writer = SceneEvidenceWriter(output / "evidence", total_frames=len(names), roles=("object",))
    backend = build_vision_backend(vision_config, mask_source, local_reference=True)
    reference = ObjectReference(
        "object",
        source.name,
        (
            ReferenceSnippet(
                "research/ycbineoat",
                0,
                0,
                "object",
                np.eye(4),
                segmentation.get("prompts_by_role", {}).get("object", "supplied-mask"),
                "fixed-camera",
            ),
        ),
    )
    try:
        extract_pose_trace(
            frames(),
            backend,
            (reference,),
            causal_tracking=CausalTrackingConfig.from_config(config["causal_tracking"]),
            on_frame_evidence=writer.record,
        )
        writer.finish()
    finally:
        backend.close()


class SuppliedMasks:
    """Use dataset masks as tracker inputs only in the supplied-mask baseline."""

    def __init__(self, source: Path, names: list[str]):
        self.source = source
        self.names = names

    def masks(self, frame) -> dict:
        mask = cv2.imread(str(self.source / "gt_mask" / self.names[frame.frame_id]), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != frame.depth.shape:
            raise ValueError("source mask is missing or misaligned")
        return {"object": mask > 0}


def run_case(case: dict, run_root: Path, config_path: Path | None = None) -> dict:
    output = run_root / case["id"]
    if output.exists():
        raise FileExistsError(f"rerun directory already exists: {output}")
    output.mkdir(parents=True)
    updated = dict(case)
    started = time.perf_counter()
    if case["kind"] == "ycb_annotated_pose":
        path = config_path or Path(__file__).with_name("flash_tracking_config.json")
        _run_ycb(case, output, path)
        updated["artifact_root"] = str(output)
    elif case["kind"] == "lerobot_rgbd":
        from lerobot.policies.flash.observation_processing import extract

        dataset = Path(case["artifact_root"]).expanduser()
        source = json.loads((dataset / "flash/v1/source.json").read_text())
        roles = (
            case.get("roles")
            or json.loads(
                (dataset / "flash/v1/observation-evidence/episode_000000/manifest.json").read_text()
            )["roles"]
        )
        segmentation = case.get("segmentation", {"source": "supplied"})
        if segmentation.get("source") == "supplied":
            masks = {role: dataset / "flash/v1/imported-masks" / role for role in roles}
            prompts = {}
            seeds = {}
        elif segmentation.get("source") == "sam3":
            masks = {}
            prompts = segmentation.get("prompts_by_role", {})
            seeds = segmentation.get("seed_points_by_role", {})
            if set(prompts) != set(roles) or any(not text.strip() for text in prompts.values()):
                raise ValueError(f"{case['id']}: SAM3 needs one prompt per role")
            if set(seeds) - set(prompts):
                raise ValueError(f"{case['id']}: SAM3 seed has no matching object role")
        else:
            raise ValueError(f"{case['id']}: unknown segmentation source {segmentation.get('source')!r}")
        path = config_path or dataset / "flash/v1/observation-config.json"
        destination = output / "evidence"
        extract(
            dataset,
            masks,
            path,
            episode=0,
            output=destination,
            prompts_by_role=prompts,
            seed_points_by_role=seeds,
        )
        updated["evidence_dir"] = str(destination)
        updated["source"] = source["source"]
    else:
        raise ValueError(f"unsupported case kind: {case['kind']}")
    (output / "run.json").write_text(
        json.dumps({"offline_wall_s_including_initialization": time.perf_counter() - started}) + "\n"
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path(__file__).with_name("flash_tracking_suite.json"))
    parser.add_argument("--case", action="append", required=True, help="Case ID; repeat for more cases")
    parser.add_argument("--run-root", type=Path, required=True, help="New directory for this candidate run")
    parser.add_argument(
        "--config-bundle", type=Path, help="Override tracker settings for a controlled comparison"
    )
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    selected = set(args.case)
    cases = [case for case in suite["cases"] if case["id"] in selected]
    if len(cases) != len(selected):
        parser.error("unknown or duplicate case ID")
    if args.run_root.exists():
        parser.error("run root already exists; use a new directory for each candidate")
    args.run_root.mkdir(parents=True)
    completed = [run_case(case, args.run_root, args.config_bundle) for case in cases]
    (args.run_root / "resolved_suite.json").write_text(
        json.dumps(
            {"schema": suite["schema"], "name": suite["name"] + " candidate", "cases": completed},
            indent=2,
        )
        + "\n"
    )
    print(args.run_root / "resolved_suite.json")


if __name__ == "__main__":
    main()
