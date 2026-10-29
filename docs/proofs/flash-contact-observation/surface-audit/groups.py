import gzip
import json
import tempfile
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.scene_groups import ScenePointTracker

p = Path("/home/feit/.cache/huggingface/lerobot/demos/flash_external_caryx_device_bowl_aligned")
s = json.loads((p / "flash/v1/source.json").read_text())
d = LeRobotDataset(s["repo_id"], root=p, video_backend="pyav")
k = np.array(s["camera_intrinsics"])
scene = ScenePointTracker(max_points=800, image_scale=1.0, depth_support_radius_px=1)
prev = None
result = []
for i in range(36):
    evidence_path = p / f"flash/v1/observation-evidence-sam3/episode_000000/{i:06d}.json.gz"
    e = json.loads(gzip.decompress(evidence_path.read_bytes()))
    r = d[i]
    rgb = (r["observation.images.rgb"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    depth = r["observation.images.depth"].numpy().squeeze().astype(np.float32)
    f = RGBDFrame(
        i,
        rgb,
        depth,
        k,
        0.001,
        np.array(e["base_from_camera"]),
        e["source_time_s"],
        e["source_time_s"],
        "source-world",
    )
    t = scene.update(f)
    g = e["groups"].get("bowl")
    lookup = {int(x): n for n, x in enumerate(t.ids)}
    v = {"frame": i, "scene_points": len(t.ids)}
    if g:
        pairs = [(j, lookup[x]) for j, x in enumerate(g["ids"]) if x in lookup]
        v["group_ids_present"] = len(pairs)
        v["saved_group_size"] = len(g["ids"])
        v["pixel_match"] = bool(all(np.array_equal(g["pixels"][j], t.xy[n]) for j, n in pairs))
        v["group_current_measurement"] = g["measured_this_frame"]
    if prev:
        old = set(prev["ids"])
        active = set(map(int, scene._ids))
        valid = set(map(int, t.ids))
        kept = set(g["ids"]) if g else set()
        v.update(
            prior_group=len(old),
            prior_ids_KLT_lost=len(old - active),
            prior_ids_no_depth=len((old & active) - valid),
            prior_ids_current_valid=len(old & valid),
            prior_ids_removed_despite_depth=len((old & valid) - kept),
        )
    result.append(v)
    print(json.dumps(v), flush=True)
    prev = g
output = Path(tempfile.gettempdir()) / "flash_dish_surface_audit"
output.mkdir(exist_ok=True)
(output / "group_diagnostics.json").write_text(json.dumps(result, indent=2))
