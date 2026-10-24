"""Replay-only P2P worker for controlled SDF input and surface ablations.

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
import time
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
    if filter_mode not in {"temporal", "depth_component", "ablation"}:
        raise ValueError(f"unknown SDF input filter: {filter_mode}")
    voxel_m = float(os.environ["FLASH_P2P_TEMPORAL_VOXEL_M"]) if filter_mode == "temporal" else None
    prior_count = int(os.environ["FLASH_P2P_TEMPORAL_PRIOR_KEYFRAMES"]) if filter_mode == "temporal" else None
    edge_m = float(os.environ["FLASH_P2P_DEPTH_EDGE_M"]) if filter_mode == "depth_component" else None
    if filter_mode == "ablation":
        edge_m = float(os.environ.get("FLASH_P2P_DEPTH_EDGE_M", "0"))
        boundary_margin_px = float(os.environ.get("FLASH_P2P_SDF_BOUNDARY_MARGIN_PX", "0"))
        occlusion_radius_px = int(os.environ.get("FLASH_P2P_SDF_OCCLUSION_RADIUS_PX", "0"))
        occlusion_gap_m = float(os.environ.get("FLASH_P2P_SDF_OCCLUSION_GAP_M", "0"))
        replace_nearer_old = os.environ.get("FLASH_P2P_SDF_REPLACE_NEARER_OLD", "0") == "1"
        surface_tsdf_max = float(os.environ.get("FLASH_P2P_SDF_SURFACE_TSDF_MAX", "0.15"))
        if min(edge_m, boundary_margin_px, occlusion_radius_px, occlusion_gap_m) < 0:
            raise ValueError("ablation filter settings must be nonnegative")
        if not 0 < surface_tsdf_max <= 1:
            raise ValueError("surface TSDF tolerance must be in (0, 1]")
    if filter_mode == "temporal" and (voxel_m <= 0 or prior_count <= 0):
        raise ValueError("temporal SDF voxel size and prior keyframe count must be positive")
    if filter_mode == "depth_component" and edge_m <= 0:
        raise ValueError("depth-component edge tolerance must be positive")
    log_path = Path(os.environ["FLASH_P2P_TEMPORAL_LOG"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    if filter_mode == "ablation":
        log_path.with_suffix(".settings.json").write_text(
            json.dumps(
                {
                    "mode": filter_mode,
                    "depth_edge_m": edge_m,
                    "boundary_margin_px": boundary_margin_px,
                    "occlusion_radius_px": occlusion_radius_px,
                    "occlusion_gap_m": occlusion_gap_m,
                    "surface_tsdf_max": surface_tsdf_max,
                    "replace_nearer_old": replace_nearer_old,
                    "first_keyframe_unfiltered": True,
                },
                indent=2,
            )
            + "\n"
        )
    ordinary_init = bridge.Session.init
    snapshot_name = os.environ.get("FLASH_P2P_SDF_SNAPSHOT")
    snapshot_frame = int(os.environ["FLASH_P2P_SDF_SNAPSHOT_FRAME"]) if snapshot_name else None
    snapshot_dir_name = os.environ.get("FLASH_P2P_SDF_KEYFRAME_SNAPSHOT_DIR")
    snapshot_dir = Path(snapshot_dir_name) if snapshot_dir_name else None
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
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
            started = time.perf_counter()
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
            elif filter_mode == "depth_component" and obj_id in seen_keyframes:
                from compare_p2p_sdf_input_filters import largest_depth_component

                masked_depth_m = np.zeros(raw_depth.shape, dtype=np.float32)
                masked_depth_m[y, x] = z
                supported = largest_depth_component(masked_depth_m, edge_m)[y, x]
            elif filter_mode == "ablation":
                from sdf_boundary_occlusion_filters import (
                    inside_mask_margin,
                    old_surface_depth_image,
                    reject_nearer_than_old_surface,
                )

                supported = np.ones(len(z), dtype=bool)
                stage_counts = {}
                if obj_id in seen_keyframes and edge_m > 0:
                    from compare_p2p_sdf_input_filters import largest_depth_component

                    masked_depth_m = np.zeros(raw_depth.shape, dtype=np.float32)
                    masked_depth_m[y, x] = z
                    supported &= largest_depth_component(masked_depth_m, edge_m)[y, x]
                    stage_counts["after_depth_component"] = int(supported.sum())
                if obj_id in seen_keyframes and boundary_margin_px > 0:
                    supported &= inside_mask_margin(mask_array, boundary_margin_px)[y, x]
                    stage_counts["after_boundary"] = int(supported.sum())
                if obj_id in seen_keyframes and occlusion_radius_px > 0:
                    predicted = old_surface_depth_image(
                        obj.sdf_volume,
                        keyframe.pose,
                        frame.intrinsics,
                        raw_depth.shape,
                        surface_tsdf_max,
                    )
                    measured = np.zeros(raw_depth.shape, dtype=np.float32)
                    measured[y, x] = z
                    rejected, comparable = reject_nearer_than_old_surface(
                        measured, predicted, occlusion_radius_px, occlusion_gap_m
                    )
                    stage_counts["old_surface_comparable"] = int(
                        np.count_nonzero(comparable[y, x] & supported)
                    )
                    stage_counts["rejected_nearer"] = int(np.count_nonzero(rejected[y, x] & supported))
                    supported &= ~rejected[y, x]
                    stage_counts["after_occlusion"] = int(supported.sum())
            else:
                supported = np.ones(len(z), dtype=bool)
            if filter_mode != "ablation":
                stage_counts = {}
            selected_depth = raw_depth.copy()
            selected_depth[y[~supported], x[~supported]] = 0
            restore = None
            if filter_mode == "ablation" and obj_id in seen_keyframes and replace_nearer_old:
                from sdf_boundary_occlusion_filters import old_voxels_in_front_of_new_depth

                accepted_depth = np.zeros(raw_depth.shape, dtype=np.float32)
                accepted_depth[y[supported], x[supported]] = z[supported]
                volume = obj.sdf_volume
                indices, surface_count = old_voxels_in_front_of_new_depth(
                    volume,
                    accepted_depth,
                    keyframe.pose,
                    frame.intrinsics,
                    occlusion_radius_px,
                    occlusion_gap_m,
                    surface_tsdf_max,
                )
                stage_counts["cleared_old_voxels"] = int(len(indices))
                stage_counts["cleared_old_surface_voxels"] = surface_count
                if len(indices):
                    index = tuple(indices.T)
                    restore = (
                        volume,
                        index,
                        volume._tsdf_vol_cpu[index].copy(),
                        volume._weight_vol_cpu[index].copy(),
                        volume._color_vol_cpu[index].copy(),
                    )
                    volume._tsdf_vol_cpu[index] = 1
                    volume._weight_vol_cpu[index] = 0
                    volume._color_vol_cpu[index] = 0
            preprocessing_ms = (time.perf_counter() - started) * 1000
            frame.depth = selected_depth
            try:
                integrated = ordinary_integrate(builder, obj, keyframe)
            except Exception:
                if restore is not None:
                    volume, index, old_tsdf, old_weight, old_color = restore
                    volume._tsdf_vol_cpu[index] = old_tsdf
                    volume._weight_vol_cpu[index] = old_weight
                    volume._color_vol_cpu[index] = old_color
                raise
            finally:
                frame.depth = raw_depth
            if not integrated and restore is not None:
                volume, index, old_tsdf, old_weight, old_color = restore
                volume._tsdf_vol_cpu[index] = old_tsdf
                volume._weight_vol_cpu[index] = old_weight
                volume._color_vol_cpu[index] = old_color
            if integrated:
                seen_keyframes.add(obj_id)
                if filter_mode == "temporal":
                    previous.append(candidate)
                if snapshot_dir is not None:
                    volume = obj.sdf_volume
                    if volume.gpu_mode:
                        raise RuntimeError("diagnostic snapshot requires the CPU TSDF backend")
                    np.savez_compressed(
                        snapshot_dir / f"object_{obj_id}_frame_{int(frame.id):04d}.npz",
                        tsdf=volume._tsdf_vol_cpu,
                        weight=volume._weight_vol_cpu,
                        origin=volume._vol_origin,
                        voxel_size=volume._voxel_size,
                        selected_depth=selected_depth,
                        mask=mask_array,
                        pose=keyframe.pose,
                        intrinsics=frame.intrinsics,
                    )
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
                            "preprocessing_ms": round(preprocessing_ms, 3),
                            "hook_total_ms": round((time.perf_counter() - started) * 1000, 3),
                            **stage_counts,
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
