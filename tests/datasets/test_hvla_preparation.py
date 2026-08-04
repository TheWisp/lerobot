# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for lerobot.datasets.hvla_preparation.

All tests use tiny synthetic datasets (2 episodes x 3 frames, 2 RGB cameras
at 32x48 / 48x32). No real dataset, no TRAIN, no performance claims.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("av", reason="av is required for video encoding (install lerobot[dataset])")

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pyav_utils import get_codec
from lerobot.datasets.video_utils import get_video_info

FPS = 5
FEATURES = {
    "observation.images.cam_a": {
        "dtype": "video",
        "shape": (32, 48, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.cam_b": {
        "dtype": "video",
        "shape": (48, 32, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}
VIDEO_KEYS = ["observation.images.cam_a", "observation.images.cam_b"]


@pytest.fixture
def prep():
    """Import hvla_preparation only when the local FFmpeg build has H.264.

    The module constructs its encoder config at import time, which validates
    against the local FFmpeg and raises when H.264 is unavailable.
    """
    if get_codec("h264") is None:
        pytest.skip("'h264' not in local FFmpeg build")
    from lerobot.datasets import hvla_preparation

    return hvla_preparation


def _make_frame() -> dict:
    frame = {"task": "dummy task"}
    for key, ft in FEATURES.items():
        if ft["dtype"] == "video":
            frame[key] = np.random.randint(0, 256, size=ft["shape"], dtype=np.uint8)
        else:
            frame[key] = torch.randn(ft["shape"])
    return frame


def _create_source(root: Path, episodes: int = 2, frames_per_episode: int = 3) -> None:
    dataset = LeRobotDataset.create(
        repo_id="test/hvla_prep_src",
        fps=FPS,
        features=FEATURES,
        root=root,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, g=2, preset="ultrafast"),
    )
    for _ in range(episodes):
        for _ in range(frames_per_episode):
            dataset.add_frame(_make_frame())
        dataset.save_episode()
    # finalize() flushes buffered episode metadata and info.json totals;
    # without it the source loads back with 0 episodes.
    dataset.finalize()


def _run_prepare(prep, src_root: Path, out_root: Path):
    return prep.prepare_hvla_dataset(
        source_repo_id="test/hvla_prep_src",
        source_root=src_root,
        output_repo_id="test/hvla_prep_out",
        output_root=out_root,
    )


class TestPrepareRoundtrip:
    def test_output_loads_and_matches_contract(self, prep, tmp_path):
        src_root = tmp_path / "src"
        out_root = tmp_path / "out"
        _create_source(src_root)
        src_info_before = (src_root / "meta" / "info.json").read_bytes()

        result = _run_prepare(prep, src_root, out_root)

        assert result == out_root
        # Output loads as a plain standard LeRobotDataset.
        dataset = LeRobotDataset("test/hvla_prep_out", root=out_root)
        assert dataset.meta.total_episodes == 2
        assert dataset.meta.total_frames == 6
        assert sorted(dataset.meta.video_keys) == sorted(VIDEO_KEYS)
        assert dataset.meta.fps == FPS
        for key in VIDEO_KEYS:
            assert list(dataset.meta.features[key]["shape"]) == [224, 224, 3]
            info = dataset.meta.features[key]["info"]
            assert info["video.codec"] == "h264"
            assert info["video.width"] == 224
            assert info["video.height"] == 224
            assert info["video.fps"] == FPS
        # Samples decode as CHW 224x224.
        for index in (0, len(dataset) - 1):
            sample = dataset[index]
            for key in VIDEO_KEYS:
                assert tuple(sample[key].shape) == (3, 224, 224)

        # Source dataset untouched.
        assert (src_root / "meta" / "info.json").read_bytes() == src_info_before

    def test_shared_video_file_processed_once_and_frames_preserved(self, prep, tmp_path):
        src_root = tmp_path / "src"
        out_root = tmp_path / "out"
        _create_source(src_root)  # small dataset: both episodes share one file per camera

        _run_prepare(prep, src_root, out_root)

        for key in VIDEO_KEYS:
            src_videos = sorted((src_root / "videos" / key).rglob("*.mp4"))
            out_videos = sorted((out_root / "videos" / key).rglob("*.mp4"))
            assert [p.relative_to(src_root) for p in src_videos] == [
                p.relative_to(out_root) for p in out_videos
            ]
            for src_video, out_video in zip(src_videos, out_videos, strict=True):
                assert get_video_info(out_video)["video.fps"] == get_video_info(src_video)["video.fps"]

    def test_progress_callback_reports_every_file(self, prep, tmp_path):
        src_root = tmp_path / "src"
        out_root = tmp_path / "out"
        _create_source(src_root)
        calls = []

        prep.prepare_hvla_dataset(
            source_repo_id="test/hvla_prep_src",
            source_root=src_root,
            output_repo_id="test/hvla_prep_out",
            output_root=out_root,
            progress=lambda done, total, current: calls.append((done, total, current)),
        )

        total_videos = len(list(src_root.rglob("videos/**/*.mp4")))
        assert calls
        assert all(total == total_videos for _, total, _ in calls)
        assert calls[-1][0] == total_videos


class TestRefusals:
    def test_refuses_existing_output(self, prep, tmp_path):
        src_root = tmp_path / "src"
        out_root = tmp_path / "out"
        _create_source(src_root)
        out_root.mkdir()
        with pytest.raises(FileExistsError):
            _run_prepare(prep, src_root, out_root)

    def test_refuses_source_as_output(self, prep, tmp_path):
        src_root = tmp_path / "src"
        _create_source(src_root)
        with pytest.raises(ValueError, match="differ"):
            _run_prepare(prep, src_root, src_root)

    def test_refuses_output_inside_source(self, prep, tmp_path):
        src_root = tmp_path / "src"
        _create_source(src_root)
        with pytest.raises(ValueError, match="inside the source"):
            _run_prepare(prep, src_root, src_root / "prepared")
        # No output or staging directory was created inside the source.
        assert not (src_root / "prepared").exists()
        assert not list(src_root.glob(".*staging*"))


class TestValidateSource:
    """Input-contract rejections that don't need a real encoded dataset."""

    def test_depth_refused(self, prep):
        meta = SimpleNamespace(
            video_keys=["observation.images.depth"],
            depth_keys=["observation.images.depth"],
            features={
                "observation.images.depth": {
                    "dtype": "video",
                    "shape": [32, 48, 1],
                    "info": {"is_depth_map": True},
                }
            },
            fps=FPS,
        )
        with pytest.raises(ValueError, match="[Dd]epth"):
            prep._validate_source(SimpleNamespace(meta=meta))

    def test_no_video_features_refused(self, prep):
        meta = SimpleNamespace(video_keys=[], depth_keys=[], features={}, fps=FPS)
        with pytest.raises(ValueError, match="no video features"):
            prep._validate_source(SimpleNamespace(meta=meta))

    def test_non_rgb_video_refused(self, prep):
        meta = SimpleNamespace(
            video_keys=["observation.images.cam"],
            depth_keys=[],
            features={
                "observation.images.cam": {
                    "dtype": "video",
                    "shape": [32, 48, 1],
                    "info": {"is_depth_map": False},
                }
            },
            fps=FPS,
        )
        with pytest.raises(ValueError, match="HWC RGB"):
            prep._validate_source(SimpleNamespace(meta=meta))


class TestCli:
    def test_cli_success_and_output_loadable(self, prep, tmp_path, monkeypatch, capsys):
        from lerobot.scripts.lerobot_prepare_hvla_dataset import main

        src_root = tmp_path / "src"
        out_root = tmp_path / "out"
        _create_source(src_root)
        monkeypatch.setattr(
            "sys.argv",
            [
                "lerobot-prepare-hvla-dataset",
                "--source-repo-id",
                "test/hvla_prep_src",
                "--source-root",
                str(src_root),
                "--output-repo-id",
                "test/hvla_prep_out",
                "--output-root",
                str(out_root),
            ],
        )
        assert main() == 0
        dataset = LeRobotDataset("test/hvla_prep_out", root=out_root)
        assert dataset.meta.total_episodes == 2
        assert json.loads((out_root / "meta" / "info.json").read_text())["features"][
            "observation.images.cam_a"
        ]["shape"] == [224, 224, 3]

    def test_cli_failure_returns_nonzero(self, prep, tmp_path, monkeypatch):
        """CLI maps a core failure to exit code 1 without creating output.

        The core function is stubbed out: this test only covers the
        argument->core->exit-code mapping, not real failure modes, and must
        not depend on Hub behaviour for missing repos.
        """
        from lerobot.scripts.lerobot_prepare_hvla_dataset import main

        def failing_prepare(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(prep, "prepare_hvla_dataset", failing_prepare)
        monkeypatch.setattr(
            "sys.argv",
            [
                "lerobot-prepare-hvla-dataset",
                "--source-repo-id",
                "test/src",
                "--output-repo-id",
                "test/out",
                "--output-root",
                str(tmp_path / "out"),
            ],
        )
        assert main() == 1
        assert not (tmp_path / "out").exists()
