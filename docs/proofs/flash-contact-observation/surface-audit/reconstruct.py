import gzip
import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.surface_observation import initial_landmarks, move, observation, voxel_sample
from lerobot.policies.flash.surface_tracking import reference_depth_support

root = Path("/home/feit/.cache/huggingface/lerobot/demos")
jar = root / "flash_external_caryx_lid_jar_aligned"
for variant in ["sam3", "sam3-text", "sam3-text-body"]:
    entries = [
        json.loads(gzip.decompress(p.read_bytes()))
        for p in sorted((jar / f"flash/v1/observation-evidence-{variant}/episode_000000").glob("*.json.gz"))
    ]
    print("JAR", variant, "n", len(entries))
    for role in ["lid", "jar"]:
        vals = []
        for e in entries:
            m = np.zeros(e["image_size"], bool)
            for st, n in e["masks_rle"].get(role, []):
                m.flat[st : st + n] = 1
            gt = cv2.imread(str(jar / f"flash/v1/imported-masks/{role}/{e['frame_index']:06d}.png"), 0) > 0
            vals.append(float(np.sum(m & gt) / max(np.sum(m | gt), 1)))
        print(
            role,
            "state",
            dict(Counter(e["estimates"][role]["evidence"] for e in entries)),
            "mask frames",
            sum(role in e["masks_rle"] for e in entries),
            "iou",
            np.percentile(vals, [0, 10, 50, 90, 100]).round(3).tolist(),
        )
p = root / "flash_external_caryx_device_bowl_aligned"
s = json.loads((p / "flash/v1/source.json").read_text())
config = json.loads((p / "flash/v1/observation-config.json").read_text())["vision_pipeline"]
k = np.array(s["camera_intrinsics"])
ds = LeRobotDataset(s["repo_id"], root=p, video_backend="pyav")
out = Path(tempfile.gettempdir()) / "flash_dish_surface_audit"
out.mkdir(exist_ok=True)
metrics = []
ref = None
for i in range(ds.num_frames):
    evidence_path = p / f"flash/v1/observation-evidence-sam3/episode_000000/{i:06d}.json.gz"
    e = json.loads(gzip.decompress(evidence_path.read_bytes()))
    row = ds[i]
    rgb = np.ascontiguousarray(
        np.rint(row["observation.images.rgb"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
    )
    depth = row["observation.images.depth"].numpy().squeeze().astype(np.float32) * 0.001
    mask = np.zeros(depth.shape, bool)
    for st, n in e["masks_rle"]["bowl"]:
        mask.flat[st : st + n] = 1
    obs = observation(depth, mask, k, config["observation"])
    cam = np.array(e["base_from_camera"])
    raw = np.array(e["proposals"]["bowl"]["raw_pose"])
    support = e["proposals"]["bowl"]["visible_support"]
    if ref is None:
        ref = obs["points"]
        cam0 = cam
        raw0 = raw
        a, basis = initial_landmarks(ref)
        np.savez(out / "reference.npz", points=ref, basis=basis, anchors=a, intrinsics=k)
    pose = np.linalg.inv(cam) @ raw @ np.linalg.inv(raw0) @ cam0
    predicted = move(ref, pose)
    y, x = np.nonzero(obs["valid_image"])
    z = depth[y, x]
    allpts = np.column_stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z))
    allpts = allpts[voxel_sample(allpts, min(obs["spacing_m"], config["shape"]["surface_spacing_m"]))]
    err = cKDTree(predicted).query(allpts)[0] * 1000
    n, dep = reference_depth_support(predicted, obs["valid_image"], depth, k)
    static_pose = np.linalg.inv(cam) @ cam0
    static_pts = move(ref, static_pose)
    static_err = cKDTree(static_pts).query(allpts)[0] * 1000
    statn, statdep = reference_depth_support(static_pts, obs["valid_image"], depth, k)
    metric = {
        "frame": i,
        "time_s": e["source_time_s"],
        "state": e["estimates"]["bowl"]["evidence"],
        "proposal": e["proposals"]["bowl"]["evidence"],
        "points": len(allpts),
        "surface_p90_mm": float(np.percentile(err, 90)),
        "depth_p90_mm": dep,
        "stored_surface_p90_mm": support.get("depth_residual_p90_m", 0) * 1000,
        "static_surface_p90_mm": float(np.percentile(static_err, 90)),
        "static_depth_p90_mm": statdep,
        "group_points": len(e["groups"].get("bowl", {}).get("ids", [])),
    }
    metrics.append(metric)
    if i in [0, 1, 8, 9, 10, 14, 15, 20, 25, 28, 29, 30, 35, 50, 70, 100, 120, 142]:
        np.savez(
            out / f"{i:06d}.npz",
            rgb=rgb,
            depth=depth,
            mask=mask,
            valid=obs["valid_image"],
            reference=ref,
            predicted=predicted,
            observed=allpts,
            observed_pixels=obs["pixels"],
            clean_points=obs["points"],
            pose=pose,
            static_pose=static_pose,
            intrinsics=k,
            error_mm=err,
            cam=cam,
        )
print("DISH", json.dumps([m for m in metrics if m["frame"] < 40 or m["frame"] % 20 == 0]))
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(
    "RECONSTRUCTION_MAX_DELTA_MM", max(abs(m["surface_p90_mm"] - m["stored_surface_p90_mm"]) for m in metrics)
)
