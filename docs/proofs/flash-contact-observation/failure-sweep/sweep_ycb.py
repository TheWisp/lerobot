"""Frozen shared-pipeline sweep; annotated poses are never opened by this runner.

Use the central 181 consecutive frames of each locally cached YCBInEOAT
sequence. Selection depends on sequence length, not tracking results or labels.
Dataset masks isolate failures after segmentation; this is not a SAM benchmark.
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

from lerobot.policies.flash.appearance import Sam3MaskSource
from lerobot.policies.flash.causal_service import CausalTrackingConfig
from lerobot.policies.flash.offline_perception import extract_pose_trace
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.scene_evidence_writer import SceneEvidenceWriter
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend

ROOT = Path.home() / ".cache/ycbineoat"
OUT = Path.home() / ".cache/flash-failure-sweep-20261029"
CONFIG = Path(__file__).with_name("config.json")


class Masks:
    def __init__(self, root, names):
        self.root, self.names = root, names

    def masks(self, frame):
        mask = cv2.imread(str(self.root / "gt_mask" / self.names[frame.frame_id]), 0)
        if mask is None or mask.shape != frame.depth.shape:
            raise ValueError("missing or unaligned source mask")
        return {"object": mask > 0}


def run(root, count, prompt=None):
    names = sorted(p.name for p in (root / "rgb").glob("*.png"))
    start = max(0, (len(names) - count) // 2)
    names = names[start : start + count]
    destination = OUT / (root.name + ("-sam3" if prompt else ""))
    if (destination / "completed.json").exists():
        print("already complete", root.name, flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    k = np.loadtxt(root / "cam_K.txt").reshape(3, 3)
    times = np.asarray([int(Path(n).stem) for n in names], dtype=np.int64)
    times = (times - times[0]) * 1e-9
    if np.any(np.diff(times) <= 0):
        raise ValueError("non-increasing source times")
    cfg = json.loads(CONFIG.read_text())
    provenance = {
        "source": str(root),
        "start": start,
        "count": len(names),
        "names": names,
        "source_times_s": times.tolist(),
        "intrinsics": k.tolist(),
        "mask_source": "SAM3" if prompt else "supplied YCBInEOAT mask on every frame",
        "prompt": prompt,
        "annotated_poses_used_by_tracker": False,
        "gripper_prediction": False,
        "config": cfg,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    digest = hashlib.sha256()
    for name in names:
        for kind in ("rgb", "depth", "gt_mask"):
            digest.update((root / kind / name).read_bytes())
    provenance["input_sha256"] = digest.hexdigest()
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2))
    writer = SceneEvidenceWriter(destination / "evidence", total_frames=len(names), roles=("object",))
    mask_source = (
        Sam3MaskSource(
            {"object": prompt}, camera_id="rgb", resolution=cfg["vision_pipeline"]["sam_resolution"]
        )
        if prompt
        else Masks(root, names)
    )
    backend = build_vision_backend(
        VisionPipelineConfig.from_dict(cfg["vision_pipeline"]), mask_source, local_reference=True
    )
    reference = ObjectReference(
        "object",
        root.name,
        (ReferenceSnippet("research/ycbineoat", 0, 0, "object", np.eye(4), "supplied-mask", "fixed-camera"),),
    )

    def frames():
        for i, name in enumerate(names):
            rgb = cv2.imread(str(root / "rgb" / name), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(root / "depth" / name), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None or depth.dtype != np.uint16:
                raise ValueError("invalid source RGBD")
            yield RGBDFrame(
                i,
                cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                depth,
                k,
                0.001,
                np.eye(4),
                float(times[i]),
                float(times[i]),
                "fixed-camera",
            )

    began = time.perf_counter()

    def record(frame, result, groups, estimates):
        writer.record(frame, result, groups, estimates)
        if frame.frame_id % 30 == 0:
            print(
                root.name,
                frame.frame_id,
                estimates["object"].evidence.value,
                round(time.perf_counter() - began, 1),
                flush=True,
            )

    try:
        trace = extract_pose_trace(
            frames(),
            backend,
            (reference,),
            causal_tracking=CausalTrackingConfig.from_config(cfg["causal_tracking"]),
            on_frame_evidence=record,
        )
        writer.finish()
        (destination / "completed.json").write_text(
            json.dumps({"frames": len(trace), "wall_s": time.perf_counter() - began})
        )
    finally:
        backend.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=181)
    p.add_argument("--sequence", default=None)
    p.add_argument("--sam3-prompt", default=None)
    args = p.parse_args()
    for root in sorted(ROOT.iterdir()):
        if (root / "rgb").is_dir() and (args.sequence is None or root.name == args.sequence):
            run(root, args.count, args.sam3_prompt)
