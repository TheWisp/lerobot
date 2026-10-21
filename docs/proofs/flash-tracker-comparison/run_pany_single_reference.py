"""Replay official PANY RGB-D single-reference inference on causal demo pairs.

Only PANY's aggregator and point head are instantiated because the official
RGB-D single-reference inference path uses exactly those modules. Its unused
training/matching head requires an older custom PointNet2 CUDA extension.
The loaded checkpoint must match every parameter in the two used modules.
This is an offline comparison, not a live FLASH backend.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


class LoRALinear(nn.Module):
    """Match the LoRA wrappers in PANY's released inference model."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.r = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.original(value) + self.dropout(self.lora_B(self.lora_A(value))) * self.scaling


def read_frame(
    dataset: Path, masks: Path, role: str, frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"{frame:06d}"
    rgb = cv2.imread(str(dataset / "rgb" / f"{stem}.jpg"))
    depth = cv2.imread(str(dataset / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(masks / role / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or depth is None or mask is None or depth.shape != mask.shape:
        raise ValueError(f"missing or unaligned RGB-D/mask at frame {frame}")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth.astype(np.float32) * 0.001, mask > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--pany-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sorted(set(args.frames)) != args.frames or args.frames[0] < args.anchor_frame:
        parser.error("frames must be strictly increasing and no earlier than the causal anchor")
    if not args.checkpoint.is_file():
        parser.error(f"missing PANY checkpoint: {args.checkpoint}")
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
    anchor_rgb, anchor_depth, anchor_mask = read_frame(args.dataset, args.masks, args.role, args.anchor_frame)
    anchor_rgb, anchor_mask, anchor_depth, anchor_k, anchor_extrinsic = center_crop(
        anchor_rgb,
        anchor_mask,
        anchor_depth,
        intrinsics,
        np.eye(4, dtype=np.float64),
        target_image_shape=(518, 518),
    )
    anchor_points = unproject_depth_map_to_point_map(
        anchor_depth[None], anchor_extrinsic[None], anchor_k[None]
    )[0]
    anchor_tensor = to_tensor(Image.fromarray(anchor_rgb.astype(np.uint8)))
    anchor_valid = anchor_mask.astype(bool) & (anchor_depth > 0)
    if anchor_valid.sum() < 3:
        raise ValueError("anchor has too few depth-backed mask pixels")

    rows = []
    for frame in args.frames:
        rgb, depth, mask = read_frame(args.dataset, args.masks, args.role, frame)
        image, query_mask, query_depth, query_k, valid_query = crop_input(
            rgb, mask, depth, intrinsics, target_size=(518, 518)
        )
        pose = np.eye(4, dtype=np.float64)
        elapsed_ms = float("nan")
        lost = True
        masked_pixels = int(query_mask.astype(bool).sum())
        depth_pixels = int(np.count_nonzero(query_mask.astype(bool) & (query_depth > 0)))
        if valid_query and depth_pixels >= 3:
            images = torch.stack((anchor_tensor, image), dim=0)[None].to("cuda")
            start = time.perf_counter()
            try:
                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        tokens, patch_start = model.aggregator(images)
                    point_maps, point_conf = model.point_head(tokens, images, patch_start)
                points = point_maps[0].float().cpu().numpy()
                confidence = point_conf[0].float().cpu().numpy()
                anchor_pred = points[0][anchor_valid]
                anchor_measured = anchor_points[anchor_valid]
                anchor_confidence = confidence[0][anchor_valid]
                scale, align_r, align_t = robust_umeyama(
                    anchor_pred, anchor_measured, anchor_confidence, anchor_confidence, with_scaling=True
                )
                query_pred = scale * (points[1] @ align_r.T) + align_t
                query_measured = depth_to_cam_coords_points(query_depth, query_k)
                query_valid = query_mask.astype(bool) & (query_depth > 0)
                query_confidence = confidence[1][query_valid]
                _, rotation, translation = robust_umeyama(
                    query_pred[query_valid],
                    query_measured[query_valid],
                    query_confidence,
                    query_confidence,
                    with_scaling=False,
                )
                pose[:3, :3] = rotation
                pose[:3, 3] = translation
                lost = not (np.isfinite(pose).all() and np.linalg.det(rotation) > 0)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"frame {frame}: alignment failed: {exc}", flush=True)
            finally:
                del images
        rows.append((frame, pose, lost, masked_pixels, depth_pixels, elapsed_ms))
        print(
            f"frame {frame}: lost={lost} ms={elapsed_ms:.1f} mask={masked_pixels} depth={depth_pixels}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frames=np.asarray([row[0] for row in rows], dtype=np.int32),
        poses=np.stack([row[1] for row in rows]),
        lost=np.asarray([row[2] for row in rows], dtype=bool),
        mask_pixels=np.asarray([row[3] for row in rows], dtype=np.int32),
        depth_pixels=np.asarray([row[4] for row in rows], dtype=np.int32),
        ms=np.asarray([row[5] for row in rows], dtype=np.float64),
        metadata=np.asarray(
            json.dumps(
                {"method": "PANY_RGBD_single_reference", "anchor_frame": args.anchor_frame, "role": args.role}
            )
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()
