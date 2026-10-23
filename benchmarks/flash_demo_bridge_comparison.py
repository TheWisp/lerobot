"""Compare one-shot and consecutive masked-cloud transfers within a FLASH demo.

This diagnostic uses saved segmentation masks and RGB-D only. It never reads
TrackEverything point trajectories or object poses. The consecutive path
re-fits a small rigid transform for every neighboring pair of demo frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from flash_best_view_transfer import fit, paint, reference_points, refine, screen
from flash_cross_episode_trackeverything import _project
from flash_rgbd_registration import apply, cloud
from scipy.spatial import cKDTree

START_FRAME = 500
END_FRAME = 540
DISTANCE_M = 0.006  # Offline surface-comparison band, not a robot acceptance threshold.


def surface_score(source: np.ndarray, target: np.ndarray, r: np.ndarray, t: np.ndarray) -> dict:
    moved = apply(source, r, t)
    source_distance = cKDTree(target).query(moved)[0]
    target_distance = cKDTree(moved).query(target)[0]
    return {
        "source_6mm": float(np.mean(source_distance <= DISTANCE_M)),
        "target_6mm": float(np.mean(target_distance <= DISTANCE_M)),
        "median_mm": [float(np.median(source_distance) * 1000), float(np.median(target_distance) * 1000)],
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    base = Path("/home/feit/.cache/huggingface/lerobot/demos/insertion")
    recording = base / "recording"
    masks = base / "flash/v1/tracker-runs/trackeverything-growth-ui-20261021-a/segmentation/masks/0"
    output = repo / "docs/proofs/flash-cross-episode/best-view-transfer"
    intrinsics = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)

    frames = {}
    for frame in range(START_FRAME, END_FRAME + 1):
        rgb = cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg"))
        depth = screen(cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED))
        mask = cv2.imread(str(masks / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        frames[frame] = {"rgb": rgb, "mask": mask, "points": cloud(depth, mask, intrinsics, 5)}

    source = frames[START_FRAME]["points"]
    target = frames[END_FRAME]["points"]
    markers = reference_points(source)
    direct = fit(source, target)

    chain_r, chain_t = np.eye(3), np.zeros(3)
    links = []
    trace = [{"frame": START_FRAME, "marker_xy": _project(markers, intrinsics).tolist()}]
    for frame in range(START_FRAME, END_FRAME):
        a, b = frames[frame]["points"], frames[frame + 1]["points"]
        centroid_shift = np.median(b, axis=0) - np.median(a, axis=0)
        step_r, step_t = refine(a, b, np.eye(3), centroid_shift)
        evidence = surface_score(a, b, step_r, step_t)
        links.append({"from": frame, "to": frame + 1, **evidence})
        chain_r, chain_t = step_r @ chain_r, step_r @ chain_t + step_t
        trace.append(
            {"frame": frame + 1, "marker_xy": _project(apply(markers, chain_r, chain_t), intrinsics).tolist()}
        )

    mapped_direct = apply(markers, direct["rotation"], direct["translation"])
    mapped_chain = apply(markers, chain_r, chain_t)
    direct_xy = _project(mapped_direct, intrinsics)
    chain_xy = _project(mapped_chain, intrinsics)
    report = {
        "start_frame": START_FRAME,
        "end_frame": END_FRAME,
        "source_points": len(source),
        "target_points": len(target),
        "direct": surface_score(source, target, direct["rotation"], direct["translation"]),
        "sequential": surface_score(source, target, chain_r, chain_t),
        "direct_marker_xy": direct_xy.tolist(),
        "sequential_marker_xy": chain_xy.tolist(),
        "marker_separation_px": np.linalg.norm(direct_xy - chain_xy, axis=1).tolist(),
        "links": links,
        "sequential_trace": trace,
    }
    (output / "direct-vs-sequential-bridge.json").write_text(json.dumps(report, indent=2) + "\n")

    images = [
        paint(frames[START_FRAME]["rgb"], frames[START_FRAME]["mask"], markers, intrinsics, "demo 500"),
        paint(frames[END_FRAME]["rgb"], frames[END_FRAME]["mask"], mapped_direct, intrinsics, "direct"),
        paint(frames[END_FRAME]["rgb"], frames[END_FRAME]["mask"], mapped_chain, intrinsics, "consecutive"),
    ]
    x0, x1, y0, y1 = 150, 315, 55, 260  # Crop covering dowel and the direct fit's displaced point.
    figure = np.hstack([cv2.resize(image[y0:y1, x0:x1], (495, 615)) for image in images])
    cv2.imwrite(str(output / "direct-vs-sequential-bridge.jpg"), figure, [cv2.IMWRITE_JPEG_QUALITY, 90])
    key_frames = [500, 510, 520, 530, 540]
    trace_panels = []
    for frame in key_frames:
        marker_xyz = markers if frame == START_FRAME else None
        if marker_xyz is None:
            xy = np.asarray(trace[frame - START_FRAME]["marker_xy"])
            image = frames[frame]["rgb"].copy()
            contours, _ = cv2.findContours(
                frames[frame]["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, (0, 255, 255), 1)
            colors = [(30, 30, 255), (20, 140, 255), (20, 240, 240), (30, 220, 30), (255, 90, 20)]
            for index, (point, color) in enumerate(zip(xy, colors, strict=True)):
                x, y = np.rint(point).astype(int)
                cv2.circle(image, (x, y), 5, (0, 0, 0), 2)
                cv2.circle(image, (x, y), 4, color, -1)
                cv2.putText(image, str(index + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        else:
            image = paint(frames[frame]["rgb"], frames[frame]["mask"], marker_xyz, intrinsics, "demo 500")
        crop = cv2.resize(image[y0:y1, x0:x1], (330, 410))
        cv2.rectangle(crop, (0, 0), (330, 32), (0, 0, 0), -1)
        cv2.putText(crop, f"demo {frame}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        trace_panels.append(crop)
    cv2.imwrite(
        str(output / "sequential-bridge-trace.jpg"),
        np.hstack(trace_panels),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"links", "sequential_trace"}},
            indent=2,
        )
    )
    print(
        "minimum adjacent balanced coverage", min(min(row["source_6mm"], row["target_6mm"]) for row in links)
    )


if __name__ == "__main__":
    main()
