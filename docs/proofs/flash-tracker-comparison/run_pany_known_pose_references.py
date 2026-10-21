"""Test PANY point maps with several causal RGB-D references of known pose.

This is a controlled multi-reference ablation, not PANY's released unposed
multi-view pipeline: reference poses come from a separate causal pose trace.
The released aggregator and point head are strictly loaded from its checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from run_pany_single_reference import LoRALinear, read_frame
from torch import nn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--pany-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--anchor-trace", required=True, type=Path)
    parser.add_argument("--reference-frames", required=True, type=int, nargs="+")
    parser.add_argument("--frames", required=True, type=int, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sorted(set(args.reference_frames)) != args.reference_frames:
        parser.error("reference frames must be strictly increasing")
    if sorted(set(args.frames)) != args.frames or args.frames[0] != args.reference_frames[-1]:
        parser.error("query frames must start at the latest reference and be strictly increasing")
    if args.reference_frames[0] != 100:
        parser.error("the recorded frame-100 object model is the common reference")
    sys.path.insert(0, str(args.pany_repo))

    from inference_utils.pose_estimation import robust_umeyama  # noqa: PLC0415
    from inference_utils.utils import center_crop, crop_input, to_tensor  # noqa: PLC0415
    from vggt.heads.dpt_head import DPTHead  # noqa: PLC0415
    from vggt.models.aggregator import Aggregator  # noqa: PLC0415
    from vggt.utils.geometry import (  # noqa: PLC0415
        depth_to_cam_coords_points,
        unproject_depth_map_to_point_map,
    )

    class UsedPANY(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.aggregator = Aggregator(
                img_size=518, patch_size=14, patch_embed="dinov2_vitl14_reg", depth=24
            )
            self.point_head = DPTHead(
                dim_in=2048,
                output_dim=4,
                intermediate_layer_idx=[4, 11, 17, 23],
                activation="inv_log",
                conf_activation="expp1",
            )
            for block in self.aggregator.frame_blocks + self.aggregator.global_blocks:
                for name in ("qkv", "proj"):
                    setattr(block.attn, name, LoRALinear(getattr(block.attn, name)))

    model = UsedPANY()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    source = checkpoint["state_dict"]
    used = {key: value for key, value in source.items() if key.startswith(("aggregator.", "point_head."))}
    model.load_state_dict(used, strict=True)
    del checkpoint, source, used
    model = model.to("cuda").eval()

    intrinsics = np.loadtxt(args.dataset / "cam_K.txt").reshape(3, 3)
    with np.load(args.anchor_trace) as trace:
        anchors = {
            int(frame): pose.copy()
            for frame, pose, lost in zip(trace["frames"], trace["poses"], trace["lost"], strict=True)
            if not lost
        }
    if any(frame not in anchors for frame in args.reference_frames):
        parser.error("all references need a valid causal pose in the anchor trace")

    references = []
    for frame in args.reference_frames:
        rgb, depth, mask = read_frame(args.dataset, args.masks, args.role, frame)
        image, ref_mask, ref_depth, ref_k, ref_extrinsic = center_crop(
            rgb, mask, depth, intrinsics, np.eye(4), target_image_shape=(518, 518)
        )
        measured = unproject_depth_map_to_point_map(ref_depth[None], ref_extrinsic[None], ref_k[None])[0]
        in_common = np.linalg.inv(anchors[frame])
        measured = measured @ in_common[:3, :3].T + in_common[:3, 3]
        valid = ref_mask.astype(bool) & (ref_depth > 0)
        if valid.sum() < 3:
            raise ValueError(f"reference frame {frame} has too few depth-backed pixels")
        tensor = to_tensor(Image.fromarray(image.astype(np.uint8)))
        tensor *= torch.from_numpy(ref_mask.astype(np.float32))[None]
        references.append((tensor, valid, measured))

    rows = []
    for frame in args.frames:
        rgb, depth, mask = read_frame(args.dataset, args.masks, args.role, frame)
        image, query_mask, query_depth, query_k, valid_query = crop_input(
            rgb, mask, depth, intrinsics, target_size=(518, 518)
        )
        query_valid = query_mask.astype(bool) & (query_depth > 0)
        pose = np.eye(4, dtype=np.float64)
        lost = True
        elapsed_ms = float("nan")
        if valid_query and query_valid.sum() >= 3:
            images = torch.stack((image, *(reference[0] for reference in references)), dim=0)[None].to("cuda")
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        tokens, patch_start = model.aggregator(images)
                    point_maps, point_conf = model.point_head(tokens, images, patch_start)
                points = point_maps[0].float().cpu().numpy()
                confidence = point_conf[0].float().cpu().numpy()
                pred_refs = np.concatenate(
                    [points[i + 1][valid] for i, (_, valid, _) in enumerate(references)], axis=0
                )
                measured_refs = np.concatenate([measured[valid] for _, valid, measured in references], axis=0)
                conf_refs = np.concatenate(
                    [confidence[i + 1][valid] for i, (_, valid, _) in enumerate(references)], axis=0
                )
                scale, align_r, align_t = robust_umeyama(
                    pred_refs, measured_refs, conf_refs, conf_refs, with_scaling=True
                )
                pred_query = scale * (points[0] @ align_r.T) + align_t
                measured_query = depth_to_cam_coords_points(query_depth, query_k)
                query_conf = confidence[0][query_valid]
                _, rotation, translation = robust_umeyama(
                    pred_query[query_valid],
                    measured_query[query_valid],
                    query_conf,
                    query_conf,
                    with_scaling=False,
                )
                pose[:3, :3] = rotation
                pose[:3, 3] = translation
                lost = not (np.isfinite(pose).all() and np.linalg.det(rotation) > 0)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"frame {frame}: alignment failed: {exc}", flush=True)
            finally:
                del images
        rows.append((frame, pose, lost, int(query_valid.sum()), elapsed_ms))
        print(f"frame {frame}: lost={lost} ms={elapsed_ms:.1f}", flush=True)

    poses_raw = np.stack([row[1] for row in rows])
    if rows[0][2]:
        raise ValueError("cannot normalize a lost self-pose at the latest reference")
    self_pose_correction = np.linalg.inv(poses_raw[0]) @ anchors[args.reference_frames[-1]]
    poses_seed_normalized = poses_raw @ self_pose_correction
    if not np.allclose(poses_seed_normalized[0], anchors[args.reference_frames[-1]], atol=1e-8):
        raise AssertionError("normalized self-pose must equal its causal anchor")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frames=np.asarray([row[0] for row in rows], dtype=np.int32),
        poses=poses_raw,
        poses_seed_normalized=poses_seed_normalized,
        self_pose_correction=self_pose_correction,
        lost=np.asarray([row[2] for row in rows], dtype=bool),
        depth_pixels=np.asarray([row[3] for row in rows], dtype=np.int32),
        ms=np.asarray([row[4] for row in rows], dtype=np.float64),
        metadata=np.asarray(
            json.dumps(
                {
                    "method": "PANY_RGBD_known_pose_references",
                    "reference_frames": args.reference_frames,
                    "role": args.role,
                    "anchor_trace": str(args.anchor_trace),
                }
            )
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()
