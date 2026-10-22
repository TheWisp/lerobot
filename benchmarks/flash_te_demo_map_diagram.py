"""Draw the actual demo TrackEverything observations used by the map pilot."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from flash_cross_episode_trackeverything import _image, _live_masks, _project
from flash_te_demo_map_consistency import DEMO_VIEWS, robust_rigid, supported_ids
from scipy.spatial.distance import pdist


def farthest_ids(ids: np.ndarray, xy: np.ndarray, count: int) -> np.ndarray:
    """Pick a few spread-out persistent tracks to label without visual clutter."""
    chosen = [int(np.argmin(xy[:, 0] + xy[:, 1]))]
    while len(chosen) < count:
        distances = np.linalg.norm(xy[:, None] - xy[chosen][None], axis=2)
        chosen.append(int(np.argmax(distances.min(axis=1))))
    return ids[chosen]


def put_lines(canvas: np.ndarray, lines: list[str], x: int, y: int) -> None:
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
        y += 28


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    base = Path("/home/feit/.cache/huggingface/lerobot/demos")
    demo = base / "insertion/recording"
    growth = base / "insertion/flash/v1/tracker-runs/trackeverything-growth-ui-20261021-a"
    output = repo / "docs/proofs/flash-cross-episode/te-demo-map/demo-tracked-points.jpg"
    intrinsics = np.loadtxt(demo / "cam_K.txt").reshape(3, 3)
    with np.load(growth / "scene.npz", allow_pickle=False) as raw:
        scene = {name: raw[name] for name in raw.files}
    mask_root = growth / "segmentation/masks/0"
    support = {
        frame: supported_ids(frame, scene, demo, mask_root, intrinsics)[0]
        for frame in range(min(DEMO_VIEWS), max(DEMO_VIEWS) + 1)
    }
    persistent = np.flatnonzero(np.stack(list(support.values())).all(axis=0))
    first_index = int(np.flatnonzero(scene["frames"] == min(DEMO_VIEWS))[0])
    last_index = int(np.flatnonzero(scene["frames"] == max(DEMO_VIEWS))[0])
    first_points = scene["query_coords_m"][first_index, persistent]
    last_points = scene["query_coords_m"][last_index, persistent]
    _, _, rigid_inliers, _ = robust_rigid(first_points, last_points)
    pair_drift_mm = float(np.median(np.abs(pdist(first_points) - pdist(last_points))) * 1000)
    anchor_index = int(np.flatnonzero(scene["frames"] == 500)[0])
    anchor_xy = _project(scene["query_coords_m"][anchor_index, persistent], intrinsics)
    highlighted = farthest_ids(persistent, anchor_xy, 5)
    panels = []
    for frame in DEMO_VIEWS:
        frame_index = int(np.flatnonzero(scene["frames"] == frame)[0])
        rgb = cv2.imread(str(demo / "rgb" / f"{frame:06d}.jpg"))
        crop = cv2.resize(rgb[90:230, 165:285], (360, 420), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((480, 360, 3), dtype=np.uint8)
        canvas[42:462] = crop
        cv2.putText(
            canvas,
            f"DEMO {frame}  |  {int(support[frame].sum())} points",
            (9, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )
        ids = np.flatnonzero(support[frame])
        xy = _project(scene["query_coords_m"][frame_index, ids], intrinsics)
        for x, y in xy:
            px, py = int(round((x - 165) * 3)), int(round((y - 90) * 3 + 42))
            cv2.circle(canvas, (px, py), 3, (255, 230, 60), -1)
        for letter, point_id in zip("ABCDE", highlighted, strict=True):
            x, y = _project(scene["query_coords_m"][frame_index, point_id : point_id + 1], intrinsics)[0]
            px, py = int(round((x - 165) * 3)), int(round((y - 90) * 3 + 42))
            cv2.circle(canvas, (px, py), 7, (0, 0, 0), 2)
            cv2.circle(canvas, (px, py), 5, (30, 230, 255), -1)
            cv2.putText(canvas, letter, (px + 9, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 3)
            cv2.putText(canvas, letter, (px + 9, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)
        panels.append(canvas)
    key = np.zeros_like(panels[0])
    put_lines(
        key,
        [
            "CYAN: supported TE points",
            "A-E: same five point IDs",
            "in every frame 475-540",
            "",
            f"{len(persistent)} IDs are supported",
            "on every intervening frame.",
            "",
            "This is DEMO tracking only.",
            "No TE replay track was run.",
        ],
        18,
        42,
    )
    panels.append(key)
    grid = np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:])))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    trial = base / "flash_insertion_trial_20261020_020-20261020-203310-655401"
    results = json.loads((output.parent / "results.json").read_text())
    frames = [item["live"] for item in results["rows"]]
    table = pq.read_table(trial / "data/chunk-000/file-000.parquet", columns=["observation.images.overhead"])
    masks = _live_masks(trial, frames, (480, 848), "dowel")
    replay_panels = []
    for row in results["rows"]:
        frame = row["live"]
        selected = row["selected"]
        rgb = _image(table["observation.images.overhead"][frame].as_py()["bytes"], cv2.IMREAD_COLOR)
        contours, _ = cv2.findContours(
            masks[frame].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgb, contours, -1, (0, 255, 255), 1)
        crop = cv2.resize(rgb[145:315, 220:370], (300, 340), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((380, 300, 3), dtype=np.uint8)
        canvas[36:376] = crop
        title = f"REPLAY {frame}  ->  DEMO {selected['demo']}"
        cv2.putText(canvas, title, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for number, (x, y) in enumerate(selected["marker_xy"], start=1):
            px, py = int(round((x - 220) * 2)), int(round((y - 145) * 2 + 36))
            cv2.circle(canvas, (px, py), 7, (0, 0, 0), 2)
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(canvas, str(number), (px + 8, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 3)
            cv2.putText(
                canvas, str(number), (px + 8, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1
            )
        replay_panels.append(canvas)
    replay_grid = np.vstack((np.hstack(replay_panels[:5]), np.hstack(replay_panels[5:])))
    replay_output = output.parent / "replay-frame-matches.jpg"
    cv2.imwrite(str(replay_output), replay_grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(
        f"wrote {output}; continuously supported IDs: {len(persistent)}; "
        f"highlighted IDs: {highlighted.tolist()}; rigid inliers: {rigid_inliers}; "
        f"median pair-distance change: {pair_drift_mm:.1f} mm"
    )
    print(f"wrote {replay_output}; replay frames: {frames}")


if __name__ == "__main__":
    main()
