"""An external RGB-D import must preserve depth validity and source timing."""

import json

import cv2
import numpy as np

from benchmarks.flash_import_external_rgbd import import_rgbd
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def test_import_preserves_invalid_depth_and_source_clock(tmp_path):
    recording = tmp_path / "recording"
    (recording / "rgb").mkdir(parents=True)
    (recording / "depth").mkdir()
    times = np.array([0.0, 0.04, 0.11], dtype=np.float64)
    np.savetxt(recording / "times.txt", times)
    np.savetxt(recording / "cam_K.txt", np.array([[90, 0, 32], [0, 90, 32], [0, 0, 1]]))
    for index in range(3):
        rgb = np.full((64, 64, 3), index * 30, dtype=np.uint8)
        depth = np.full((64, 64), 850 + index, dtype=np.uint16)
        depth[20:30, 30:40] = 0
        cv2.imwrite(str(recording / "rgb" / f"{index:06d}.png"), rgb)
        cv2.imwrite(str(recording / "depth" / f"{index:06d}.png"), depth)

    root = import_rgbd(
        recording,
        tmp_path / "dataset",
        repo_id="local/external_rgbd_test",
        task="test",
        source="synthetic RGB-D source",
        source_frame_offset=100,
    )
    dataset = LeRobotDataset("local/external_rgbd_test", root=root)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
    for index in range(3):
        row = dataset[index]
        saved_depth = row["observation.images.depth"].numpy().squeeze()
        assert np.array_equal(saved_depth == 0, depth == 0)
        assert np.isclose(row["flash.source_time_s"].item(), times[index], atol=1e-6)
        assert row["flash.source_frame_index"].item() == 100 + index
    source = json.loads((root / "flash" / "v1" / "source.json").read_text())
    assert source["status"] == "perception_only"
    assert "robot_actions" in source["missing_for_spatial_skill"]
