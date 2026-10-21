"""Track fixed object pixels with CoTracker3's online window API.

Only the last frame of each completed 16-frame window is emitted. It depends
on no later frame, but observations between the eight-frame endpoints are not
available from this model and must not be scored as causal per-frame output.
Run with Meta's co-tracker checkout on PYTHONPATH in the LeRobot environment.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from cotracker.predictor import CoTrackerOnlinePredictor


def rgb(dataset: Path, frame_id: int) -> np.ndarray:
    image = cv2.imread(str(dataset / "rgb" / f"{frame_id:06d}.jpg"))
    if image is None:
        raise ValueError(f"missing RGB frame {frame_id}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def seed_points(
    dataset: Path, masks: Path, role: str, frame_id: int, rgb_image: np.ndarray, max_query_points: int
) -> np.ndarray:
    mask = cv2.imread(str(masks / role / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(dataset / "depth" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
    if mask is None or depth is None or mask.shape != rgb_image.shape[:2] or depth.shape != mask.shape:
        raise ValueError(f"{role} seed RGB, mask and depth must align")
    corners = cv2.goodFeaturesToTrack(
        cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY),
        maxCorners=max_query_points,
        qualityLevel=0.005,
        minDistance=5,
        mask=mask,
    )
    seed_uv = corners.reshape(-1, 2) if corners is not None else np.empty((0, 2), dtype=np.float32)
    if len(seed_uv):
        pixels = np.rint(seed_uv).astype(int)
        seed_uv = seed_uv[depth[pixels[:, 1], pixels[:, 0]] > 0]
    if len(seed_uv) < 3:
        raise ValueError(f"{role} seed mask has fewer than three depth-backed queries")
    return seed_uv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--context-role", action="append", default=[])
    parser.add_argument("--seed-frame", required=True, type=int)
    parser.add_argument("--last-frame", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--max-query-points", type=int, default=120)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.last_frame <= args.seed_frame or not args.checkpoint.is_file() or args.max_query_points < 3:
        parser.error("need valid frame interval, checkpoint, and point count")
    if args.role in args.context_role or len(set(args.context_role)) != len(args.context_role):
        parser.error("context roles must differ from the primary role and each other")
    model = CoTrackerOnlinePredictor(checkpoint=str(args.checkpoint)).cuda().eval()
    window = model.step * 2
    if args.last_frame - args.seed_frame + 1 < window:
        parser.error("recording must contain at least one complete model window")
    seed_rgb = rgb(args.dataset, args.seed_frame)
    seed_uv = seed_points(
        args.dataset, args.masks, args.role, args.seed_frame, seed_rgb, args.max_query_points
    )
    context_uv = [
        seed_points(args.dataset, args.masks, role, args.seed_frame, seed_rgb, args.max_query_points)
        for role in args.context_role
    ]
    all_seed_uv = np.concatenate((seed_uv, *context_uv))
    queries = torch.from_numpy(
        np.column_stack((np.zeros(len(all_seed_uv), dtype=np.float32), all_seed_uv))[None]
    ).cuda()

    def video(first: int, last: int) -> torch.Tensor:
        images = np.stack([rgb(args.dataset, frame) for frame in range(first, last + 1)])
        return torch.from_numpy(images).permute(0, 3, 1, 2).float()[None].cuda()

    model(video(args.seed_frame, args.seed_frame + model.step - 1), is_first_step=True, queries=queries)
    rows = []
    for endpoint in range(args.seed_frame + window - 1, args.last_frame + 1, model.step):
        chunk = video(endpoint - window + 1, endpoint)
        torch.cuda.synchronize()
        started = time.perf_counter()
        tracks, visibility = model(chunk)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if tracks is None or visibility is None or tracks.shape[-2] != len(all_seed_uv):
            raise ValueError("online model did not return all real point tracks")
        xy = tracks[0, -1].cpu().numpy()
        visible = visibility[0, -1].cpu().numpy()
        rows.append((endpoint, xy, visible, elapsed_ms))
        if endpoint % 32 == args.seed_frame % 32:
            print(endpoint, "visible", int(visible[: len(seed_uv)].sum()), "/", len(seed_uv), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        role=np.asarray(args.role),
        roles=np.asarray((args.role, *args.context_role)),
        role_query_counts=np.asarray((len(seed_uv), *(len(points) for points in context_uv))),
        context_roles=np.asarray(args.context_role),
        context_query_points=np.asarray([len(points) for points in context_uv]),
        seed_frame=np.asarray(args.seed_frame),
        seed_uv=seed_uv,
        all_seed_uv=all_seed_uv,
        window_frames=np.asarray(window),
        stride_frames=np.asarray(model.step),
        frames=np.asarray([row[0] for row in rows]),
        uv=np.stack([row[1][: len(seed_uv)] for row in rows]),
        visible=np.stack([row[2][: len(seed_uv)] for row in rows]),
        all_uv=np.stack([row[1] for row in rows]),
        all_visible=np.stack([row[2] for row in rows]),
        step_ms=np.asarray([row[3] for row in rows]),
    )
    print(args.output, "endpoints", len(rows), "point queries", len(seed_uv))


if __name__ == "__main__":
    main()
