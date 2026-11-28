"""Run released BUFFER-X weights on saved clouds, including unit/motion controls."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import open3d as o3d
    import torch

    sys.path.insert(0, str(a.repository.resolve()))
    from config.indoor_config import IndoorBaseConfig
    from models.BUFFERX import BufferX
    from utils.tools import sphericity_based_voxel_analysis

    a.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(a.repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint_sha256": {},
        "normalization": "Both raw metres and centered/shared scale: largest cloud extent becomes 1. Undo normalization on returned pose.",
        "normalization_used_labels": False,
        "scope": "Pair matching, not sequential tracking. Cold first call is identified; saved mask/depth cleaning precedes these clouds.",
        "torch": torch.__version__,
        "open3d": o3d.__version__,
    }
    cfg = IndoorBaseConfig().get_cfg()
    cfg.stage = "test"
    cfg.match.enable_early_exit = True
    model = BufferX(cfg).cuda().eval()
    for stage in cfg.train.all_stage:
        checkpoint = a.repository / f"snapshot/threedmatch/{stage}/best.pth"
        provenance["checkpoint_sha256"][stage] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        loaded = torch.load(checkpoint, map_location="cuda", weights_only=True)
        state = model.state_dict()
        updates = {k: v for k, v in loaded.items() if k.startswith(stage + ".")}
        assert set(updates) == {k for k in state if k.startswith(stage + ".")}
        state.update(updates)
        model.load_state_dict(state)
    rows = []
    for entry in json.loads((a.pairs / "manifest.json").read_text()):
        data = np.load(a.pairs / f"{entry['name']}.npz")
        source, target = data["source"], data["target"]
        for normalize in [False, True]:
            np.random.seed(123)
            torch.manual_seed(123)
            o3d.utility.random.seed(123)
            src_center = source.mean(0) if normalize else np.zeros(3)
            tgt_center = target.mean(0) if normalize else np.zeros(3)
            scale = 1 / max(np.ptp(source, axis=0).max(), np.ptp(target, axis=0).max()) if normalize else 1.0
            torch.cuda.synchronize()
            start = time.perf_counter()
            clouds = [
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector((points - center) * scale))
                for points, center in [(source, src_center), (target, tgt_center)]
            ]
            voxel, _, _ = sphericity_based_voxel_analysis(*clouds)
            clouds = [
                np.asarray(cloud.voxel_down_sample(voxel).points).astype(np.float32) for cloud in clouds
            ]
            for points in clouds:
                np.random.shuffle(points)
            # FPS cannot choose more distinct points than exist; apply identically to every pair.
            cfg.patch.num_fps = min(1500, *map(len, clouds))
            cfg.patch.num_points_radius_estimate = min(2000, *map(len, clouds))
            inputs = {
                "src_fds_pcd": torch.from_numpy(clouds[0]).cuda(),
                "tgt_fds_pcd": torch.from_numpy(clouds[1]).cuda(),
                "is_aligned_to_global_z": False,
            }
            row = {
                **entry,
                "cold_first_call": not rows,
                "normalized": normalize,
                "scale": scale,
                "input_counts": list(map(len, clouds)),
                "voxel": voxel,
            }
            try:
                with torch.inference_mode():
                    result = model(inputs)
                torch.cuda.synchronize()
                row["total_ms"] = (time.perf_counter() - start) * 1000
                pose = np.asarray(result[0]).copy()
                pose[:3, 3] = tgt_center + pose[:3, 3] / scale - pose[:3, :3] @ src_center
                moved = source @ pose[:3, :3].T + pose[:3, 3]
                row.update(
                    pose=pose.tolist(),
                    num_inliers=result[2],
                    scales=result[-1],
                    surface_median_mm=float(np.median(cKDTree(target).query(moved)[0]) * 1000),
                )
                if "truth" in data:
                    truth = data["truth"]
                    expected = source @ truth[:3, :3].T + truth[:3, 3]
                    row.update(
                        point_error_mm=float(np.median(np.linalg.norm(moved - expected, axis=1)) * 1000),
                        angle_error_deg=float(
                            Rotation.from_matrix(pose[:3, :3] @ truth[:3, :3].T).magnitude() * 180 / np.pi
                        ),
                    )
            except (RuntimeError, ValueError) as e:
                row["error"] = str(e)
                row["total_ms"] = (time.perf_counter() - start) * 1000
            rows.append(row)
            (a.output / "results.json").write_text(
                json.dumps(
                    {
                        "provenance": provenance,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "config": cfg,
                        "results": rows,
                    },
                    indent=2,
                    default=str,
                )
            )
            print(
                row["name"],
                normalize,
                {
                    k: v
                    for k, v in row.items()
                    if k in ["point_error_mm", "angle_error_deg", "total_ms", "error", "surface_median_mm"]
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
