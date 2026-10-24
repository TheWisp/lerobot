"""Replay-only P2P worker that filters SDF pixels before fusion.

The sealed bridge and pinned Point2Pose checkout remain untouched. All other
pipeline stages see the original frame. Filter parameters and log destination
come from explicit environment variables; this is not a runtime default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-bridge", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("sealed_p2p_bridge_temporal_sdf", args.sealed_bridge)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load sealed Point2Pose bridge")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)

    filter_mode = os.environ["FLASH_P2P_SDF_FILTER"]
    if filter_mode not in {"temporal", "depth_component"}:
        raise ValueError(f"unknown SDF input filter: {filter_mode}")
    voxel_m = float(os.environ["FLASH_P2P_TEMPORAL_VOXEL_M"]) if filter_mode == "temporal" else None
    prior_count = int(os.environ["FLASH_P2P_TEMPORAL_PRIOR_KEYFRAMES"]) if filter_mode == "temporal" else None
    edge_m = float(os.environ["FLASH_P2P_DEPTH_EDGE_M"]) if filter_mode == "depth_component" else None
    if filter_mode == "temporal" and (voxel_m <= 0 or prior_count <= 0):
        raise ValueError("temporal SDF voxel size and prior keyframe count must be positive")
    if filter_mode == "depth_component" and edge_m <= 0:
        raise ValueError("depth-component edge tolerance must be positive")
    log_path = Path(os.environ["FLASH_P2P_TEMPORAL_LOG"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    ordinary_init = bridge.Session.init
    snapshot_name = os.environ.get("FLASH_P2P_SDF_SNAPSHOT")
    snapshot_frame = int(os.environ["FLASH_P2P_SDF_SNAPSHOT_FRAME"]) if snapshot_name else None
    ordinary_answer = bridge.Session._answer

    def capture_snapshot(session, frame, milliseconds):
        answer = ordinary_answer(session, frame, milliseconds)
        if snapshot_name and frame.id == snapshot_frame:
            obj = session.pipe.objects[0]
            volume = obj.sdf_volume
            if volume.gpu_mode:
                raise RuntimeError("diagnostic snapshot requires the CPU TSDF backend")
            np.savez_compressed(
                snapshot_name,
                tsdf=volume._tsdf_vol_cpu,
                weight=volume._weight_vol_cpu,
                origin=volume._vol_origin,
                voxel_size=volume._voxel_size,
                pose=answer["delta"],
            )
        return answer

    bridge.Session._answer = capture_snapshot

    def install_filter(session, rgb, depth_m, intrinsics, mask):
        from point2pose.modules.reconstruction.sdf_builder import SDFBuilder

        history: dict[int, deque[set[tuple[int, int, int]]]] = {}
        seen_keyframes: set[int] = set()
        ordinary_integrate = SDFBuilder.integrate_keyframe

        def integrate(builder, obj, keyframe):
            frame = keyframe.frame
            if frame is None or frame.depth is None or frame.mask is None:
                return ordinary_integrate(builder, obj, keyframe)
            obj_id = int(keyframe.obj_id)
            mask_array = frame.mask[obj_id, 0]
            if hasattr(mask_array, "detach"):
                mask_array = mask_array.detach().cpu().numpy()
            mask_array = np.asarray(mask_array) > 0
            raw_depth = np.asarray(frame.depth)
            y, x = np.nonzero(mask_array & (raw_depth > 0))
            z = raw_depth[y, x] / float(frame.depth_factor)
            if filter_mode == "temporal":
                k = np.asarray(frame.intrinsics)
                points = np.stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z), axis=1)
                pose = np.asarray(keyframe.pose)
                object_points = (points - pose[:3, 3]) @ pose[:3, :3]
                previous = history.setdefault(obj_id, deque(maxlen=prior_count))
                voxels = np.floor(object_points / voxel_m).astype(np.int32)
                candidate = {tuple(voxel) for voxel in voxels}
                prior = set().union(*previous)
                supported = (
                    np.array([tuple(voxel) in prior for voxel in voxels], dtype=bool)
                    if previous
                    else np.ones(len(voxels), dtype=bool)
                )
            elif obj_id in seen_keyframes:
                from compare_p2p_sdf_input_filters import largest_depth_component

                masked_depth_m = np.zeros(raw_depth.shape, dtype=np.float32)
                masked_depth_m[y, x] = z
                supported = largest_depth_component(masked_depth_m, edge_m)[y, x]
            else:
                supported = np.ones(len(z), dtype=bool)
            selected_depth = raw_depth.copy()
            selected_depth[y[~supported], x[~supported]] = 0
            frame.depth = selected_depth
            try:
                integrated = ordinary_integrate(builder, obj, keyframe)
            finally:
                frame.depth = raw_depth
            if integrated:
                seen_keyframes.add(obj_id)
                if filter_mode == "temporal":
                    previous.append(candidate)
            with log_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "frame_id": int(frame.id),
                            "object_id": obj_id,
                            "filter_mode": filter_mode,
                            "candidate_pixels": len(z),
                            "supported_pixels": int(supported.sum()),
                            "integrated": bool(integrated),
                        }
                    )
                    + "\n"
                )
            return integrated

        SDFBuilder.integrate_keyframe = integrate
        return ordinary_init(session, rgb, depth_m, intrinsics, mask)

    bridge.Session.init = install_filter
    sys.argv = [str(args.sealed_bridge), "--repo", str(args.repo), "--config", str(args.config)]
    bridge.main()


if __name__ == "__main__":
    main()
