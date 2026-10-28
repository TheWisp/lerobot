"""Import a synchronized external RGB-D clip as a regular LeRobot episode.

This is an observation import, not a FLASH skill compiler. It preserves source
timestamps and camera intrinsics without inventing robot actions or calibration.
Example input has recording/{rgb,depth}/000000.*, times.txt and cam_K.txt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def import_rgbd(
    recording: Path,
    destination: Path,
    *,
    repo_id: str,
    task: str,
    source: str,
    source_frame_offset: int = 0,
) -> Path:
    recording = recording.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    times = np.atleast_1d(np.loadtxt(recording / "times.txt", dtype=np.float64))
    intrinsics = np.loadtxt(recording / "cam_K.txt", dtype=np.float64).reshape(3, 3)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("source timestamps must be finite and strictly increasing")
    if not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("source camera intrinsics are invalid")
    rgb_paths = sorted((recording / "rgb").glob("*"))
    depth_paths = sorted((recording / "depth").glob("*"))
    if len(rgb_paths) != len(times) or len(depth_paths) != len(times):
        raise ValueError("RGB, depth and timestamp counts differ")

    def frame(index: int) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.imread(str(rgb_paths[index]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_paths[index]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.ndim != 3 or depth.ndim != 2:
            raise ValueError(f"frame {index}: missing RGB or single-channel depth")
        if rgb.shape[:2] != depth.shape or depth.dtype != np.uint16:
            raise ValueError(f"frame {index}: RGB/depth alignment or millimetre depth is invalid")
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth

    first_rgb, first_depth = frame(0)
    height, width = first_depth.shape
    # LeRobot's video writer expects an integral nominal rate. Keep the
    # irregular source clock separately, one original timestamp per frame.
    nominal_fps = max(1, round(float(1.0 / np.median(np.diff(times)))))
    features = {
        "observation.images.rgb": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.depth": {
            "dtype": "video",
            "shape": (height, width, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        },
        "flash.source_time_s": {"dtype": "float64", "shape": (1,), "names": None},
        "flash.source_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=nominal_fps,
        features=features,
        root=destination,
        robot_type="external_rgbd_observation",
        # Source uses zero for no return. The default encoder's 10 mm minimum
        # would silently convert every invalid pixel into a positive depth.
        depth_encoder=DepthEncoderConfig(depth_min=0.0),
    )
    for index in range(len(times)):
        rgb, depth = (first_rgb, first_depth) if index == 0 else frame(index)
        dataset.add_frame(
            {
                "task": task,
                "observation.images.rgb": rgb,
                "observation.images.depth": depth[:, :, None],
                "flash.source_time_s": np.asarray([times[index]], dtype=np.float64),
                "flash.source_frame_index": np.asarray([source_frame_offset + index], dtype=np.int64),
            }
        )
    dataset.save_episode()
    dataset.finalize()
    extension = destination / "flash" / "v1"
    extension.mkdir(parents=True)
    (extension / "source.json").write_text(
        json.dumps(
            {
                "schema": "external-rgbd-observation-v1",
                "repo_id": repo_id,
                "source": source,
                "source_recording": str(recording),
                "source_frame_offset": source_frame_offset,
                "source_time_unit": "seconds",
                "depth_unit": "millimetres",
                "camera_intrinsics": intrinsics.tolist(),
                "status": "perception_only",
                "missing_for_spatial_skill": [
                    "camera_to_robot_transform",
                    "measured_end_effector_trajectory",
                    "robot_actions",
                    "task_timeline_annotations",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-frame-offset", type=int, default=0)
    args = parser.parse_args()
    print(import_rgbd(**vars(args)))


if __name__ == "__main__":
    main()
