"""Audit dowel-shaft pose during the actual demo turn (frames 280–400).

The image-space reference is the long direction of the current SAM3 mask.
The recorded gripper delta is a second, conditional reference: it assumes the
dowel did not slip inside the grasp. Neither is full 6D ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.visible_points import surface_support
from lerobot.policies.flash.visible_rigid_tracking import (
    VisibleRGBDFrame,
    fit_visible_trace,
    visible_surface_cloud,
)

LABELS = {
    "point2pose": "Point2Pose",
    "bundlesdf": "BundleSDF",
    "trackeverything_native": "TrackEverything 3D",
    "litetracker": "LiteTracker + fit",
}
PANEL_KEYS = tuple(LABELS)
PANEL_WIDTH = 480
PANEL_HEIGHT = 420
CROP = (40, 50, 340, 285)


def project_line(
    pose: np.ndarray, center: np.ndarray, axis: np.ndarray, intrinsics: np.ndarray, half_length_m: float
) -> np.ndarray | None:
    current_center = pose[:3, :3] @ center + pose[:3, 3]
    current_axis = pose[:3, :3] @ axis
    endpoints = np.stack(
        (current_center - half_length_m * current_axis, current_center + half_length_m * current_axis)
    )
    if not np.isfinite(endpoints).all() or np.any(endpoints[:, 2] <= 0):
        return None
    return np.column_stack(
        (
            intrinsics[0, 0] * endpoints[:, 0] / endpoints[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * endpoints[:, 1] / endpoints[:, 2] + intrinsics[1, 2],
        )
    )


def direction_error(line: np.ndarray | None, observed_angle_deg: float) -> float | None:
    if line is None:
        return None
    delta = line[1] - line[0]
    angle = np.rad2deg(np.arctan2(delta[1], delta[0])) % 180
    return float(abs((angle - observed_angle_deg + 90) % 180 - 90))


def _median(rows: list[float]) -> float | None:
    return float(np.median(rows)) if rows else None


def audit(
    experiment: Path,
    point_result: Path,
    pose_trace: Path,
    output: Path,
    raw_video: Path,
    still_dir: Path,
    min_mask_elongation: float,
) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    with np.load(experiment / "queries.npz", allow_pickle=False) as query:
        frames, intrinsics = query["frames"], query["intrinsics"]
    point_rows = json.loads(point_result.read_text())["results"]
    with np.load(pose_trace, allow_pickle=False) as saved:
        source_axis = saved["dowel_reference_shaft_camera"].copy()
        source_center = saved["dowel_reference_center_camera_m"].copy()
        trace_frames = saved["frames"].copy()
        p2p_delta = saved["p2p_dowel_delta"].copy()
        p2p_lost = saved["p2p_dowel_lost"].copy()
        bundle_delta = saved["bundlesdf_dowel_delta"].copy()
        bundle_failed = saved["bundlesdf_dowel_failed"].copy()
        camera_from_base = np.linalg.inv(saved["base_from_camera"])
        gripper_base = saved["demo_ee_pose_base"].copy()
    index_by_frame = {int(frame): index for index, frame in enumerate(trace_frames)}
    source_p2p = p2p_delta[index_by_frame[int(frames[0])]]
    point_axis = source_p2p[:3, :3] @ source_axis
    point_center = source_p2p[:3, :3] @ source_center + source_p2p[:3, 3]
    gripper = camera_from_base[None] @ gripper_base[frames]
    inverse_initial_gripper = np.linalg.inv(gripper[0])

    rgbs, masks, rgbd, mask_evidence = [], [], [], []
    for frame in frames:
        rgb = cv2.imread(str(Path(manifest["recording"]) / "rgb" / f"{frame:06d}.jpg"))
        depth_raw = cv2.imread(str(Path(manifest["recording"]) / "depth" / f"{frame:06d}.png"), -1)
        mask = cv2.imread(str(Path(manifest["mask_dir"]) / f"{frame:06d}.png"), 0) > 0
        if rgb is None or depth_raw is None or depth_raw.shape != mask.shape:
            raise ValueError(f"missing aligned source frame {frame}")
        depth = depth_raw.astype(float) * 0.001
        surface = visible_surface_cloud(depth, mask, intrinsics, edge_pixels=2, max_points=1000)
        y, x = np.nonzero(mask)
        center = np.array([x.mean(), y.mean()])
        values, directions = np.linalg.eigh(np.cov(np.column_stack((x - center[0], y - center[1])).T))
        shaft = directions[:, -1]
        angle = float(np.rad2deg(np.arctan2(shaft[1], shaft[0])) % 180)
        mask_evidence.append(
            {"center": center.tolist(), "angle_deg": angle, "elongation": float(values[-1] / values[-2])}
        )
        rgbs.append(rgb)
        masks.append(mask)
        rgbd.append(VisibleRGBDFrame(surface, mask, depth, intrinsics))

    with np.load(experiment / "trackeverything.npz", allow_pickle=False) as prediction:
        native_xyz, native_visible = prediction["xyz"], prediction["visible"]
    native_support = np.stack(
        [
            surface_support(
                native_xyz[i], native_visible[i].astype(float), masks[i], rgbd[i].depth, intrinsics, 0.01
            )[0]
            for i in range(len(frames))
        ]
    )
    native = fit_visible_trace(
        native_xyz,
        native_support,
        tolerance_m=0.005,
        min_inliers=12,
        trials=100,
        reference_surface=rgbd[0].surface_m,
        rgbd_frames=rgbd,
        depth_m_per_unit=1.0,
        surface_tolerance_m=0.01,
        icp_iterations=20,
        trim_fraction=0.75,
    )
    methods: dict[str, list[np.ndarray | None]] = {
        "point2pose": [
            None if p2p_lost[index_by_frame[int(frame)]] else p2p_delta[index_by_frame[int(frame)]]
            for frame in frames
        ],
        "bundlesdf": [
            None if bundle_failed[index_by_frame[int(frame)]] else bundle_delta[index_by_frame[int(frame)]]
            for frame in frames
        ],
        "trackeverything_native": [
            None if native["lost"][i] else native["poses"][i] for i in range(len(frames))
        ],
    }
    for name, result in point_rows.items():
        if name == "trackeverything":
            name = "trackeverything_depth_lift"
        methods[name] = [
            None if row is None else np.asarray(row["pose_camera_from_source"], dtype=float)
            for row in result["mask_depth_surface"]["rows"]
        ]

    report: dict = {
        "frames": [int(frames[0]), int(frames[-1])],
        "inputs": {"manifest": str(experiment / "manifest.json"), "pose_trace": str(pose_trace)},
        "min_mask_elongation": min_mask_elongation,
        "mask": mask_evidence,
        "methods": {},
    }
    for name, poses in methods.items():
        center, axis = (
            (source_center, source_axis)
            if name in {"point2pose", "bundlesdf"}
            else (point_center, point_axis)
        )
        rows = []
        for i, (frame, pose) in enumerate(zip(frames, poses, strict=True)):
            mask = mask_evidence[i]
            if pose is None:
                rows.append(None)
                continue
            line = project_line(pose, center, axis, intrinsics, 0.045)
            if line is None:
                rows.append(None)
                continue
            center_px = line.mean(axis=0)
            gripper_delta = gripper[i] @ inverse_initial_gripper
            predicted_axis = pose[:3, :3] @ axis
            gripper_axis = gripper_delta[:3, :3] @ point_axis
            axis_error_deg = float(np.rad2deg(np.arccos(np.clip(abs(predicted_axis @ gripper_axis), -1, 1))))
            rows.append(
                {
                    "frame": int(frame),
                    "line_px": line.tolist(),
                    "center_to_mask_px": float(np.linalg.norm(center_px - mask["center"])),
                    "shaft_to_mask_deg": direction_error(line, mask["angle_deg"]),
                    "shaft_to_gripper_3d_deg": axis_error_deg,
                    "mask_elongation": mask["elongation"],
                }
            )
        visible = [row for row in rows if row is not None]
        elongated = [row for row in visible if row["mask_elongation"] >= min_mask_elongation]
        report["methods"][name] = {
            "pose_frames": len(visible),
            "median_shaft_to_mask_deg_when_elongated": _median(
                [row["shaft_to_mask_deg"] for row in elongated]
            ),
            "median_center_to_mask_px": _median([row["center_to_mask_px"] for row in visible]),
            "median_shaft_to_gripper_3d_deg": _median([row["shaft_to_gripper_3d_deg"] for row in visible]),
            "rows": rows,
        }
    gripper_rows = []
    for i in range(len(frames)):
        line = project_line(gripper[i] @ inverse_initial_gripper, point_center, point_axis, intrinsics, 0.045)
        if mask_evidence[i]["elongation"] >= min_mask_elongation:
            gripper_rows.append(direction_error(line, mask_evidence[i]["angle_deg"]))
    report["gripper_heading_to_mask_deg_median_when_elongated"] = _median(gripper_rows)

    crop_x0, crop_y0, crop_x1, crop_y1 = CROP
    video_width = 2 * PANEL_WIDTH + 36
    video_height = 2 * PANEL_HEIGHT + 70
    raw_video.parent.mkdir(parents=True, exist_ok=True)
    still_dir.mkdir(parents=True, exist_ok=True)
    stream = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (video_width, video_height))
    if not stream.isOpened():
        raise RuntimeError("could not open video writer")
    try:
        for i, frame in enumerate(frames):
            canvas = np.full((video_height, video_width, 3), (16, 17, 19), dtype=np.uint8)
            for panel_index, name in enumerate(PANEL_KEYS):
                image = rgbs[i].copy()
                contours, _ = cv2.findContours(
                    masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
                observed = mask_evidence[i]
                direction = np.array(
                    [np.cos(np.deg2rad(observed["angle_deg"])), np.sin(np.deg2rad(observed["angle_deg"]))]
                )
                mask_center = np.asarray(observed["center"])
                yellow_line = np.rint(
                    np.stack((mask_center - 45 * direction, mask_center + 45 * direction))
                ).astype(int)
                cv2.line(image, tuple(yellow_line[0]), tuple(yellow_line[1]), (30, 225, 255), 3)
                row = report["methods"][name]["rows"][i]
                if row is not None:
                    line = np.rint(row["line_px"]).astype(int)
                    cv2.arrowedLine(image, tuple(line[0]), tuple(line[1]), (255, 255, 0), 3, tipLength=0.17)
                    cv2.circle(image, tuple(np.rint(np.mean(line, axis=0)).astype(int)), 4, (255, 255, 0), -1)
                view = cv2.resize(
                    image[crop_y0:crop_y1, crop_x0:crop_x1],
                    (PANEL_WIDTH, PANEL_HEIGHT - 45),
                    interpolation=cv2.INTER_CUBIC,
                )
                panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (25, 25, 25), dtype=np.uint8)
                panel[45:] = view
                cv2.putText(panel, LABELS[name], (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (245, 245, 245), 2)
                if row is None:
                    cv2.putText(panel, "NO POSE", (330, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 80, 255), 2)
                else:
                    cv2.putText(
                        panel,
                        f"{row['shaft_to_mask_deg']:.0f} deg",
                        (345, 31),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (220, 220, 220),
                        1,
                    )
                col, panel_row = panel_index % 2, panel_index // 2
                px, py = 12 + col * (PANEL_WIDTH + 12), 12 + panel_row * (PANEL_HEIGHT + 12)
                canvas[py : py + PANEL_HEIGHT, px : px + PANEL_WIDTH] = panel
            cv2.putText(
                canvas,
                "WHITE mask   YELLOW visible shaft   CYAN predicted shaft",
                (16, video_height - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.61,
                (238, 238, 238),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"frame {frame}",
                (video_width - 165, video_height - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.61,
                (238, 238, 238),
                1,
                cv2.LINE_AA,
            )
            stream.write(canvas)
            if int(frame) in {280, 320, 340, 400}:
                cv2.imwrite(str(still_dir / f"rotation_{frame}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 89])
    finally:
        stream.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--point-result", type=Path, required=True)
    parser.add_argument("--pose-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--still-dir", type=Path, required=True)
    parser.add_argument("--min-mask-elongation", type=float, default=3.0)
    args = parser.parse_args()
    report = audit(
        args.experiment,
        args.point_result,
        args.pose_trace,
        args.output,
        args.raw_video,
        args.still_dir,
        args.min_mask_elongation,
    )
    print(
        json.dumps(
            {
                name: {key: value[key] for key in ("pose_frames", "median_shaft_to_mask_deg_when_elongated")}
                for name, value in report["methods"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
