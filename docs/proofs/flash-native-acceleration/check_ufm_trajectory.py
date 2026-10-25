"""Compare real batched UFM outputs through complete causal trajectories.

The cache only avoids rerunning the network: preprocessing, point rejection,
rigid motion, fitting, pose composition and acceptance use the real backend.
These passes measure correctness, not throughput (capture adds CPU transfers).
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from benchmark_fits import compare_poses
from benchmark_pipeline import inputs
from candidates import predict_pairs

from lerobot.policies.flash.perception_step import PerceptionPipeline, PerceptionRunner
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.tracking_debug_worker import debug_reference
from lerobot.policies.flash.vision_pipeline import build_vision_backend


class CapturedModel:
    def __init__(self, model):
        self.model, self.frame_id = model, None
        self.inputs, self.single, self.batched = {}, {}, {}
        self.mode = "capture"

    def predict_correspondences_batched(self, a, b):
        aa, bb = a.detach().cpu().numpy(), b.detach().cpu().numpy()
        key = (self.frame_id, hashlib.sha256(aa.tobytes() + bb.tobytes()).hexdigest())
        if self.mode == "capture":
            result = self.model.predict_correspondences_batched(a, b)
            flow = result.flow.flow_output.detach().float().cpu()
            vis = result.covisibility.mask.detach().float().cpu()
            self.inputs[key] = (aa, bb)
            self.single[key] = (flow, vis)
        else:
            flow, vis = (self.single if self.mode == "single" else self.batched)[key]
        return SimpleNamespace(flow=SimpleNamespace(flow_output=flow), covisibility=SimpleNamespace(mask=vis))


def run(args):
    cv2.setNumThreads(1)
    sequence, masks, config = inputs(args)
    roles = list(masks[sequence[0].frame_id])
    refs = tuple(debug_reference(role, sequence[0]) for role in roles)

    class Masks:
        def masks(self, frame):
            return masks[frame.frame_id]

    model = CapturedModel(load_ufm())

    def trajectory(mode):
        model.mode = mode
        backend = build_vision_backend(config, Masks(), matcher=model, local_reference=True)
        prepare = backend.prepare

        def tagged_prepare(frame):
            model.frame_id = frame.frame_id
            return prepare(frame)

        backend.prepare = tagged_prepare
        pipeline = PerceptionPipeline(backend, refs)
        source = iter(sequence)
        results = []
        with PerceptionRunner(pipeline) as runner:
            while (item := runner.process_next(lambda: next(source, None))) is not None:
                results.append(item[1])
        return results

    original = trajectory("capture")
    groups = defaultdict(list)
    for key in model.inputs:
        groups[key[0]].append(key)
    for keys in groups.values():
        aa, bb = zip(*(model.inputs[key] for key in keys), strict=True)
        outputs = predict_pairs(model.model, list(aa), list(bb))
        for key, (flow, vis) in zip(keys, outputs, strict=True):
            model.batched[key] = (torch.from_numpy(flow[None]), torch.from_numpy(vis[None]))
    report = {
        "trace": str(args.trace),
        "frames": [f.frame_id for f in sequence],
        "config": config.to_dict(),
        "model_pairs": len(model.inputs),
        "comparisons": {},
    }
    for mode in ("single", "batched"):
        results = trajectory(mode)
        errors, decisions = [], []
        for old, new in zip(original, results, strict=True):
            assert (old.frame_id, old.source_time_s) == (new.frame_id, new.source_time_s)
            for role in roles:
                np.testing.assert_array_equal(old.masks[role], new.masks[role])
                a, b = old.details[role], new.details[role]
                assert ("camera_from_reference" in a) == ("camera_from_reference" in b)
                if (a["accepted"], a["status"]) != (b["accepted"], b["status"]):
                    decisions.append(
                        {"frame": new.frame_id, "role": role, "old": a["accepted"], "new": b["accepted"]}
                    )
                if "camera_from_reference" in a and "camera_from_reference" in b:
                    point_delta = (
                        np.linalg.norm(np.array(a["reference_m"]) - np.array(b["reference_m"]), axis=1) * 1000
                    )
                    errors.append(
                        {
                            "frame": new.frame_id,
                            "role": role,
                            **compare_poses(
                                np.array(b["camera_from_reference"]), np.array(a["camera_from_reference"])
                            ),
                            "surface_max_mm": float(point_delta.max()),
                        }
                    )
        summary = {
            "accepted": {role: sum(r.details[role]["accepted"] for r in results) for role in roles},
            "decision_changes": decisions,
            "max_difference": {
                key: max(e[key] for e in errors)
                for key in ("rotation_deg", "translation_mm", "matrix_max_abs", "surface_max_mm")
            },
            "differences": errors,
        }
        report["comparisons"][mode] = summary
        print(mode, {key: value for key, value in summary.items() if key != "differences"}, flush=True)
        if mode == "single":
            assert not decisions and summary["max_difference"]["matrix_max_abs"] == 0
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
