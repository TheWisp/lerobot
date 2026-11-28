"""Rank masked demo object views against a recorded live object view.

This retrieves candidates only. A caller must verify a transferred landmark
against the live mask, measured depth, and temporal/rigid consistency.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from flash_cross_episode_trackeverything import _live_masks
from transformers import Dinov2Model


def _foreground_crop(image: np.ndarray, mask: np.ndarray, side: int = 224) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("object mask is empty")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    color = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    foreground = mask[y0:y1, x0:x1]
    height, width = color.shape[:2]
    scale = 0.8 * side / max(height, width)
    out_width, out_height = max(1, round(width * scale)), max(1, round(height * scale))
    color = cv2.resize(color, (out_width, out_height), interpolation=cv2.INTER_AREA)
    foreground = (
        cv2.resize(foreground.astype(np.uint8), (out_width, out_height), interpolation=cv2.INTER_NEAREST) > 0
    )
    crop = np.full((side, side, 3), 127, np.uint8)
    shape = np.zeros((side, side), np.uint8)
    left, top = (side - out_width) // 2, (side - out_height) // 2
    crop[top : top + out_height, left : left + out_width][foreground] = color[foreground]
    shape[top : top + out_height, left : left + out_width][foreground] = 1
    return crop, shape


def _features(images: list[np.ndarray], model_path: Path, device: str, batch_size: int) -> np.ndarray:
    model = Dinov2Model.from_pretrained(model_path, local_files_only=True).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(np.stack(images[start : start + batch_size])).to(device)
            batch = (batch.permute(0, 3, 1, 2).float() / 255 - mean) / std
            output = model(pixel_values=batch).last_hidden_state[:, 0]
            features.append(torch.nn.functional.normalize(output, dim=1).cpu().numpy())
    return np.vstack(features)


def _rank(scores: np.ndarray, frames: list[int], spacing: int, limit: int) -> list[int]:
    selected = []
    for index in np.argsort(-scores):
        if all(abs(frames[index] - frames[other]) >= spacing for other in selected):
            selected.append(int(index))
        if len(selected) >= limit:
            break
    return selected


def run(args: argparse.Namespace) -> dict:
    if args.demo_end < args.demo_start or args.spacing < 1 or args.limit < 1 or args.batch_size < 1:
        raise ValueError("invalid frame range or ranking parameters")
    eligible = None
    if args.demo_landmark_evidence is not None:
        with args.demo_landmark_evidence.open(newline="") as handle:
            evidence = list(csv.DictReader(handle))
        eligible = {
            int(row["frame"])
            for row in evidence
            if row["mask_hit"].lower() == "true"
            and float(row["visibility_logit"]) > 0
            and float(row["surface_depth_error_mm"]) <= args.source_surface_tolerance_m * 1000
        }
    demo_frames, crops, silhouettes = [], [], []
    for frame in range(args.demo_start, args.demo_end + 1):
        if eligible is not None and frame not in eligible:
            continue
        image_path = args.demo_recording / "rgb" / f"{frame:06d}.jpg"
        mask_path = args.demo_mask_dir / f"{frame:06d}.png"
        if not image_path.exists() or not mask_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or not np.any(mask):
            continue
        crop, silhouette = _foreground_crop(image, mask > 0)
        demo_frames.append(frame)
        crops.append(crop)
        silhouettes.append(silhouette)
    if not demo_frames:
        raise ValueError("no eligible demo object views")
    table = pq.read_table(
        args.live_root / "data/chunk-000/file-000.parquet", columns=["observation.images.overhead"]
    )
    live_image = cv2.imdecode(
        np.frombuffer(table[0][args.live_frame].as_py()["bytes"], np.uint8), cv2.IMREAD_COLOR
    )
    if live_image is None:
        raise ValueError("could not decode live image")
    live_mask = _live_masks(args.live_root, [args.live_frame], live_image.shape[:2], args.role).get(
        args.live_frame
    )
    if live_mask is None:
        raise ValueError("live object mask is unavailable for this camera frame")
    live_crop, live_shape = _foreground_crop(live_image, live_mask)
    features = _features([*crops, live_crop], args.dino_model, args.device, args.batch_size)
    cosine = features[:-1] @ features[-1]
    live_shape = live_shape.astype(bool)
    shape_iou = np.array(
        [
            np.logical_and(live_shape, shape).sum() / np.logical_or(live_shape, shape).sum()
            for shape in silhouettes
        ]
    )
    result = {
        "live_frame": args.live_frame,
        "eligible_demo_frames": len(demo_frames),
        "object_only": True,
        "dino": [
            {
                "demo_frame": demo_frames[index],
                "cosine": float(cosine[index]),
                "shape_iou": float(shape_iou[index]),
            }
            for index in _rank(cosine, demo_frames, args.spacing, args.limit)
        ],
        "silhouette": [
            {
                "demo_frame": demo_frames[index],
                "cosine": float(cosine[index]),
                "shape_iou": float(shape_iou[index]),
            }
            for index in _rank(shape_iou, demo_frames, args.spacing, args.limit)
        ],
        "note": "Similarity proposes source views. It does not verify landmark identity or metric pose.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ranked-views.json").write_text(json.dumps(result, indent=2) + "\n")
    cv2.imwrite(str(args.output / "live-object-crop.png"), cv2.cvtColor(live_crop, cv2.COLOR_RGB2BGR))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-mask-dir", type=Path, required=True)
    parser.add_argument("--demo-start", type=int, required=True)
    parser.add_argument("--demo-end", type=int, required=True)
    parser.add_argument("--demo-landmark-evidence", type=Path)
    parser.add_argument("--source-surface-tolerance-m", type=float, default=0.01)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--live-frame", type=int, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--spacing", type=int, default=12)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
