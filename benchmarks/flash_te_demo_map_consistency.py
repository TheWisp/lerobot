"""Offline TrackEverything demo map + masked RGB-D live registration pilot.

The model has already processed the demo. This reads its saved growing point
cohorts, keeps IDs supported by both demo masks and measured depth, fits their
rigid motion between demo views, then independently registers each demo view
to each saved live RGB-D frame. It does not run TrackEverything across the
episode cut or change the live FLASH controller.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from flash_best_view_transfer import fit, reference_points, screen
from flash_cross_episode_trackeverything import _image, _live_masks, _project, _rigid_fit
from flash_rgbd_registration import apply, cloud

DEMO_REFERENCE = 500
DEMO_VIEWS = [475, 490, 500, 510, 520, 530, 540]
LIVE_FRAMES = [472, 477, 482, 485, 490, 496, 502, 506, 507, 515]
DEPTH_SUPPORT_M = 0.01  # Same offline visible-surface diagnostic as the saved growth run.
RIGID_INLIER_M = 0.005  # Offline co-motion diagnostic, not a controller threshold.


def supported_ids(
    frame: int,
    scene: dict,
    demo: Path,
    mask_root: Path,
    intrinsics: np.ndarray,
    role: str = "dowel",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_index = int(np.flatnonzero(scene["frames"] == frame)[0])
    mask = cv2.imread(str(mask_root / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
    depth = cv2.imread(str(demo / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    points = scene["query_coords_m"][frame_index]
    valid_xyz = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    safe = np.where(valid_xyz[:, None], points, np.array([0, 0, 1]))
    projected = np.rint(_project(safe, intrinsics)).astype(int)
    x, y = projected[:, 0], projected[:, 1]
    supported = valid_xyz & (scene["query_visibility_logits"][frame_index] > 0)
    supported &= scene["query_points_2d"][:, 0] <= frame_index
    supported &= scene["query_roles"] == role
    supported &= (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    inside = np.flatnonzero(supported)
    supported[inside] &= mask[y[inside], x[inside]] & (depth[y[inside], x[inside]] > 0)
    inside = np.flatnonzero(supported)
    supported[inside] &= np.abs(points[inside, 2] - depth[y[inside], x[inside]] * 0.001) <= DEPTH_SUPPORT_M
    return supported, mask, depth


def robust_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    rng = np.random.default_rng(0)
    best = None
    for _ in range(500):
        indices = rng.choice(len(source), 3, replace=False)
        if np.linalg.matrix_rank(source[indices] - source[indices].mean(axis=0)) < 2:
            continue
        rotation, translation = _rigid_fit(source[indices], target[indices])
        residual = np.linalg.norm(apply(source, rotation, translation) - target, axis=1)
        inliers = residual <= RIGID_INLIER_M
        rank = (int(inliers.sum()), -float(np.median(residual[inliers])) if inliers.any() else -1)
        if best is None or rank > best[0]:
            best = (rank, inliers)
    if best is None or best[1].sum() < 3:
        raise ValueError("too few rigidly consistent shared TrackEverything points")
    rotation, translation = _rigid_fit(source[best[1]], target[best[1]])
    residual = np.linalg.norm(apply(source, rotation, translation) - target, axis=1)
    inliers = residual <= RIGID_INLIER_M
    return rotation, translation, int(inliers.sum()), float(np.median(residual) * 1000)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    base = Path("/home/feit/.cache/huggingface/lerobot/demos")
    demo = base / "insertion/recording"
    growth = base / "insertion/flash/v1/tracker-runs/trackeverything-growth-ui-20261021-a"
    trial = base / "flash_insertion_trial_20261020_020-20261020-203310-655401"
    output = repo / "docs/proofs/flash-cross-episode/te-demo-map"
    output.mkdir(parents=True, exist_ok=True)
    intrinsics = np.loadtxt(demo / "cam_K.txt").reshape(3, 3)
    with np.load(growth / "scene.npz", allow_pickle=False) as source:
        scene = {key: source[key] for key in source.files}
    mask_root = growth / "segmentation/masks/0"
    demo_data = {}
    for frame in sorted(set(DEMO_VIEWS + [DEMO_REFERENCE])):
        support, mask, depth = supported_ids(frame, scene, demo, mask_root, intrinsics)
        demo_data[frame] = {
            "support": support,
            "mask": mask,
            "cloud": cloud(screen(depth), mask, intrinsics, 5),
        }
    reference = demo_data[DEMO_REFERENCE]
    markers = reference_points(reference["cloud"])
    reference_index = int(np.flatnonzero(scene["frames"] == DEMO_REFERENCE)[0])
    bridges = {}
    for frame in DEMO_VIEWS:
        index = int(np.flatnonzero(scene["frames"] == frame)[0])
        shared = np.flatnonzero(reference["support"] & demo_data[frame]["support"])
        if frame == DEMO_REFERENCE:
            rotation, translation, inliers, median = np.eye(3), np.zeros(3), len(shared), 0.0
        else:
            rotation, translation, inliers, median = robust_rigid(
                scene["query_coords_m"][reference_index, shared], scene["query_coords_m"][index, shared]
            )
        bridges[frame] = {
            "rotation": rotation,
            "translation": translation,
            "shared_ids": len(shared),
            "rigid_inliers_5mm": inliers,
            "median_pair_residual_mm": median,
        }
    # The endpoint fit can conceal a lost track that reappeared on the surface.
    # Count IDs whose mask and measured-depth support persists at every frame.
    continuous = np.stack(
        [supported_ids(frame, scene, demo, mask_root, intrinsics)[0] for frame in range(500, 541)]
    ).all(axis=0)
    bridges[540]["continuous_supported_ids"] = int(continuous.sum())

    table = pq.read_table(
        trial / "data/chunk-000/file-000.parquet",
        columns=["observation.images.overhead", "observation.images.overhead_depth"],
    )
    first_depth = _image(
        table["observation.images.overhead_depth"][LIVE_FRAMES[0]].as_py()["bytes"], cv2.IMREAD_UNCHANGED
    )
    masks = _live_masks(trial, LIVE_FRAMES, first_depth.shape, "dowel")
    rows, panels = [], []
    for live_frame in LIVE_FRAMES:
        depth = _image(
            table["observation.images.overhead_depth"][live_frame].as_py()["bytes"], cv2.IMREAD_UNCHANGED
        )
        rgb = _image(table["observation.images.overhead"][live_frame].as_py()["bytes"], cv2.IMREAD_COLOR)
        mask = masks[live_frame]
        target = cloud(screen(depth), mask, intrinsics, 5)
        candidates = []
        for demo_frame in DEMO_VIEWS:
            direct = fit(demo_data[demo_frame]["cloud"], target)
            bridge = bridges[demo_frame]
            rotation = direct["rotation"] @ bridge["rotation"]
            translation = direct["rotation"] @ bridge["translation"] + direct["translation"]
            mapped = apply(markers, rotation, translation)
            candidate = {
                "demo": demo_frame,
                "surface_6mm": direct["rank"][0],
                "marker_xy": _project(mapped, intrinsics).tolist(),
                "marker_xyz_m": mapped.tolist(),
            }
            candidates.append(candidate)
        selected = max(candidates, key=lambda item: item["surface_6mm"])
        point = np.rint(selected["marker_xy"][0]).astype(int)
        if 0 <= point[0] < mask.shape[1] and 0 <= point[1] < mask.shape[0]:
            selected["point1_inside_visible_mask"] = bool(mask[point[1], point[0]])
            measured = depth[point[1], point[0]]
            selected["point1_depth_delta_mm"] = (
                float(selected["marker_xyz_m"][0][2] * 1000 - measured) if measured > 0 else None
            )
        else:
            selected["point1_inside_visible_mask"] = False
            selected["point1_depth_delta_mm"] = None
        rows.append({"live": live_frame, "selected": selected, "candidates": candidates})
        chosen = np.asarray(selected["marker_xy"])
        display = rgb.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, (0, 255, 255), 1)
        colors = [(30, 30, 255), (20, 140, 255), (20, 240, 240), (30, 220, 30), (255, 90, 20)]
        for index, (xy, color) in enumerate(zip(chosen, colors, strict=True)):
            x, y = np.rint(xy).astype(int)
            cv2.circle(display, (x, y), 5, (0, 0, 0), 2)
            cv2.circle(display, (x, y), 4, color, -1)
            cv2.putText(display, str(index + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        crop = cv2.resize(display[145:315, 220:370], (300, 340))
        cv2.rectangle(crop, (0, 0), (300, 28), (0, 0, 0), -1)
        title = f"live {live_frame} / demo {selected['demo']}"
        cv2.putText(crop, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)
        panels.append(crop)
    (output / "results.json").write_text(
        json.dumps(
            {
                "demo_reference": DEMO_REFERENCE,
                "demo_views": DEMO_VIEWS,
                "bridges": {
                    str(frame): {
                        key: value for key, value in bridge.items() if key not in {"rotation", "translation"}
                    }
                    for frame, bridge in bridges.items()
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    grid = np.vstack([np.hstack(panels[:5]), np.hstack(panels[5:])])
    cv2.imwrite(str(output / "selected-live-frames.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Contrast the previously failing distant-frame ICP bridge with two
    # independently supported TrackEverything demo-map paths at live frame 502.
    comparison = []
    live_frame = 502
    row = next(item for item in rows if item["live"] == live_frame)
    rgb = _image(table["observation.images.overhead"][live_frame].as_py()["bytes"], cv2.IMREAD_COLOR)
    for demo_frame in [530, 540]:
        candidate = next(item for item in row["candidates"] if item["demo"] == demo_frame)
        points_xy = np.asarray(candidate["marker_xy"])
        image = rgb.copy()
        for index, (xy, color) in enumerate(zip(points_xy, colors, strict=True)):
            x, y = np.rint(xy).astype(int)
            cv2.circle(image, (x, y), 5, (0, 0, 0), 2)
            cv2.circle(image, (x, y), 4, color, -1)
            cv2.putText(image, str(index + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        crop = cv2.resize(image[145:315, 220:370], (450, 510))
        cv2.rectangle(crop, (0, 0), (450, 34), (0, 0, 0), -1)
        title = f"demo {demo_frame}, TE demo map: {candidate['surface_6mm']:.1%}"
        cv2.putText(crop, title, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        comparison.append(crop)
    cv2.imwrite(
        str(output / "live-502-two-demo-views.jpg"), np.hstack(comparison), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    # A point can look correctly placed in RGB yet contradict aligned depth.
    live_frame = 472
    row = next(item for item in rows if item["live"] == live_frame)
    chosen = row["selected"]
    rgb = _image(table["observation.images.overhead"][live_frame].as_py()["bytes"], cv2.IMREAD_COLOR)
    depth = _image(
        table["observation.images.overhead_depth"][live_frame].as_py()["bytes"], cv2.IMREAD_UNCHANGED
    )
    depth_gray = np.clip((depth.astype(np.float32) - 340) * 255 / 120, 0, 255).astype(np.uint8)
    depth_display = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
    point = np.rint(chosen["marker_xy"][0]).astype(int)
    panels = []
    for title, source in [("RGB", rgb), ("aligned depth", depth_display)]:
        display = source.copy()
        if title == "RGB":
            contours, _ = cv2.findContours(
                masks[live_frame].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(display, contours, -1, (0, 255, 255), 1)
        cv2.circle(display, tuple(point), 5, (0, 0, 0), 2)
        cv2.circle(display, tuple(point), 4, (0, 0, 255), -1)
        crop = cv2.resize(display[145:315, 220:370], (450, 510))
        cv2.rectangle(crop, (0, 0), (450, 58), (0, 0, 0), -1)
        cv2.putText(crop, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(
            crop,
            f"map {chosen['marker_xyz_m'][0][2] * 1000:.0f} / pixel {depth[point[1], point[0]]} mm",
            (6, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )
        panels.append(crop)
    cv2.imwrite(
        str(output / "live-472-depth-disagreement.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    print(
        json.dumps(
            {
                "selected": [
                    (item["live"], item["selected"]["demo"], item["selected"]["marker_xy"][0])
                    for item in rows
                ],
                "bridges": {
                    frame: (item["shared_ids"], item["rigid_inliers_5mm"]) for frame, item in bridges.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
