"""Paired GPU registration and attachment-math regression on saved act find images.

Masks are segmented once and shared by both paths; this isolates registration
and caching changes, not scheduling or SAM nondeterminism. No ground truth is
assumed. A baseline checkout supplies the previous manipulation_worker.py.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import random

os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import manipulation_worker as new
import numpy as np
import torch
from act_replay_bench import _demo

from lerobot.gui.api.manipulation import held_views, pose_updates

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("demo", type=pathlib.Path)
parser.add_argument("baseline", type=pathlib.Path)
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("--acts", nargs="+", required=True)
parser.add_argument("--object", required=True)
args = parser.parse_args()
root = args.demo
spec = importlib.util.spec_from_file_location(
    "baseline_worker", str(args.baseline / "benchmarks/manipulation_worker.py")
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
models = new.Models("cuda", "facebook/dinov3-vits16-pretrain-lvd1689m", 1008)
sam, tier = models.ensure("object")
demo = _demo(root)
logs = {f["id"]: f for f in map(json.loads, (root / "finds/finds.jsonl").read_text().splitlines())}
rows = []
for name in args.acts:
    act = json.loads((root / "acts" / name / "act.json").read_text())
    intr = new.CameraIntrinsics(**{k: act["intr"][k] for k in ["fx", "fy", "cx", "cy"]})
    bc = np.array(act["t_bc"])
    refs = {r["frame"]: r for r in held_views.prepare_references(demo, args.object, bc)}
    refmasks = {}
    attempts = {
        a["find_id"]: a for batch in act["live_held_attempts"] for a in batch["attempts"] if a.get("find_id")
    }
    for id, a in attempts.items():
        f = logs[id]
        rf = a["reference_frame"]
        ref = refs[rf]
        if rf not in refmasks:
            rgb = cv2.cvtColor(
                cv2.imread(str(pathlib.Path(demo.recording) / "rgb" / f"{rf:06d}.jpg")), cv2.COLOR_BGR2RGB
            )
            mask = cv2.resize(
                np.array(ref["mask"], np.uint8), rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST
            )
            y, x = np.unravel_index(cv2.distanceTransform(mask, cv2.DIST_L2, 5).argmax(), mask.shape)
            refmasks[rf] = sam.mask_at(rgb, int(x), int(y))
        rr = {"recording": demo.recording, "frame": rf, "mask": refmasks[rf]}
        rgb = cv2.cvtColor(cv2.imread(str(root / "finds" / f"{id}.jpg")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(root / "finds" / f"{id}_depth.png"), -1).astype(np.float32) / 1000
        frame = new._Frame(rgb, depth, id)
        mask = sam.mask_at(rgb, *f["click"])
        row = {"act": name, "find_id": id, "recorded_accepted": a["accepted"]}
        if mask is None or rr["mask"] is None:
            row["missing_segmentation"] = True
            rows.append(row)
            continue
        outputs = []
        for module in [old, new, new]:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            out = module._find_reference(rr, frame, mask, tier, intr)
            outputs.append(out)
        x, y, z = outputs
        row.update(
            old_ok=x["ref_ok"],
            new_ok=y["ref_ok"],
            warm_ok=z["ref_ok"],
            old_inliers=x.get("ref_inliers"),
            new_inliers=y.get("ref_inliers"),
            warm_inliers=z.get("ref_inliers"),
        )
        if "ref_delta" in x and "ref_delta" in z:
            px, pz = np.array(x["ref_delta"]), np.array(z["ref_delta"])
            row["pose_translation_mm"] = float(np.linalg.norm(px[:3, 3] - pz[:3, 3]) * 1000)
            row["pose_rotation_deg"] = float(
                np.degrees(np.arccos(np.clip((np.trace(px[:3, :3].T @ pz[:3, :3]) - 1) / 2, -1, 1)))
            )
            row["matrix_max_abs"] = float(np.max(abs(px - pz)))
        if a["accepted"]:
            point = np.r_[a["reference_point"], 1]
            tip = np.array(a["tip"])
            D = np.array(f["delta"])
            H = np.linalg.inv(tip) @ bc @ D
            desired = (np.linalg.inv(ref["tip"]) @ bc @ point)[:3]
            observed = (H @ point)[:3]
            rotation = (np.linalg.inv(ref["tip"]) @ bc)[:3, :3] @ H[:3, :3].T
            sample = pose_updates.attachment_sample(
                pose_updates.PoseObservation(args.object, rf, 0, D),
                tip,
                bc,
                np.linalg.inv(ref["tip"]),
                (bc @ point)[:3],
            )
            row["adapter_max_abs"] = float(
                max(
                    np.max(abs(sample["desired"] - desired)),
                    np.max(abs(sample["observed"] - observed)),
                    np.max(abs(sample["rotation"] - rotation)),
                )
            )
        rows.append(row)
        args.out.write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
print("DONE", len(rows), flush=True)
