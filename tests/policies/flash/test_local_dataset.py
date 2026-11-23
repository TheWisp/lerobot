"""FLASH local browsing keeps the shared dataset implementation unchanged."""

import shutil

import numpy as np
import pytest
import torch

from benchmarks.flash_experiment_inputs import _local_dataset
from lerobot.datasets import dataset_metadata, lerobot_dataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.local_dataset import LocalDataset


@pytest.mark.parametrize("missing", [None, "meta/info.json", "meta/tasks.parquet", "meta/episodes", "data"])
def test_local_files_only_never_repairs_from_hub(tmp_path, lerobot_dataset_factory, monkeypatch, missing):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=3, use_videos=False, camera_features={}
    )

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Local inspection must not contact the Hub")

    monkeypatch.setattr(dataset_metadata, "get_safe_version", network_forbidden)
    monkeypatch.setattr(lerobot_dataset, "get_safe_version", network_forbidden)
    monkeypatch.setattr(LeRobotDatasetMetadata, "_pull_from_repo", network_forbidden)
    monkeypatch.setattr(LeRobotDataset, "_download", network_forbidden)
    if missing:
        target = dataset.root / missing
        shutil.rmtree(target) if target.is_dir() else target.unlink()
        with pytest.raises(FileNotFoundError):
            _local_dataset(dataset.root, dataset.repo_id, 0)
    else:
        local = _local_dataset(dataset.root, dataset.repo_id, 0)
        assert len(local) == len(dataset)
        for key, value in dataset[1].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(local[1][key], value)
            else:
                assert local[1][key] == value
        assert local.episode_rows(0) == dataset.episode_rows(0)
        np.testing.assert_array_equal(
            local.episode_column("timestamp", 0), dataset.episode_column("timestamp", 0)
        )


def test_missing_root_is_not_created(tmp_path):
    root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        LocalDataset("local/missing", root)
    assert not root.exists()


def test_local_recording_restores_rgb_depth_pixels_and_statistics(tmp_path):
    rgb_key, depth_key = "observation.images.rgb", "observation.images.depth"
    features = {
        rgb_key: {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channels"]},
        depth_key: {
            "dtype": "image",
            "shape": (8, 8, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True, "depth_unit": "mm"},
        },
    }
    dataset = LeRobotDataset.create("local/rgbd", 30, features, root=tmp_path / "rgbd", use_videos=False)
    for i in range(3):
        dataset.add_frame(
            {
                rgb_key: np.full((8, 8, 3), 30 + i * 30, np.uint8),
                depth_key: np.full((8, 8, 1), 400 + i * 100, np.uint16),
                "task": "record RGB-D",
            }
        )
    dataset.save_episode()
    dataset.finalize()

    local = LocalDataset(dataset.repo_id, dataset.root)
    ordinary = LeRobotDataset(dataset.repo_id, root=dataset.root)
    assert local.meta.stats[rgb_key]["mean"].shape == (3, 1, 1)
    assert local.meta.stats[depth_key]["mean"].shape == (1, 1, 1)
    np.testing.assert_allclose(local.meta.stats[rgb_key]["mean"], 60 / 255)
    np.testing.assert_allclose(local.meta.stats[depth_key]["mean"], 500)
    np.testing.assert_array_equal(local.meta.stats[depth_key]["count"], [3])
    for i in range(3):
        torch.testing.assert_close(local[i][rgb_key], ordinary[i][rgb_key])
        np.testing.assert_array_equal(local[i][depth_key], np.full((1, 8, 8), 400 + i * 100))


def test_local_reader_rejects_missing_video(tmp_path, lerobot_dataset_factory, monkeypatch):
    dataset = lerobot_dataset_factory(root=tmp_path / "video", total_episodes=1, total_frames=3)
    key = dataset.meta.video_keys[0]
    (dataset.root / dataset.meta.get_video_file_path(0, key)).unlink()

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Missing video must not trigger a Hub repair")

    monkeypatch.setattr(LeRobotDataset, "_download", network_forbidden)
    with pytest.raises(FileNotFoundError):
        LocalDataset(dataset.repo_id, dataset.root)
