# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the dataset-preparation GUI API router.

Job state machine tests monkeypatch the core function; one boundary test
runs the real prepare_hvla_dataset end to end on a tiny synthetic dataset.
No real dataset, no TRAIN.
"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("av", reason="av is required for video encoding (install lerobot[dataset])")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.pyav_utils import get_codec
from lerobot.gui.api import dataset_preparation as dp

FPS = 5
FEATURES = {
    "observation.images.cam": {
        "dtype": "video",
        "shape": (32, 48, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(dp.router)
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def _clean_jobs():
    dp._jobs.clear()
    yield
    dp._jobs.clear()


def _wait_for_terminal(client: TestClient, job_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.get(f"/api/dataset-preparation/jobs/{job_id}").json()
        if job["status"] in ("complete", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state: {job}")


def _patch_core(monkeypatch, fn):
    # Patching imports hvla_preparation, which builds its encoder config at
    # import time and validates H.264 against the local FFmpeg build.
    if get_codec("h264") is None:
        pytest.skip("'h264' not in local FFmpeg build")
    monkeypatch.setattr("lerobot.datasets.hvla_preparation.prepare_hvla_dataset", fn)


class TestJobStateMachine:
    def test_post_returns_job_id_and_completes(self, client, monkeypatch, tmp_path):
        def fake_prepare(*, source_repo_id, source_root, output_repo_id, output_root, progress=None):
            if progress:
                progress(0, 2, "videos/cam/chunk-000/file-000.mp4")
                progress(2, 2, "")
            return Path(output_root)

        _patch_core(monkeypatch, fake_prepare)
        res = client.post(
            "/api/dataset-preparation/hvla",
            json={"source_repo_id": "test/src", "output_root": str(tmp_path / "out")},
        )
        assert res.status_code == 201
        job_id = res.json()["job_id"]

        job = _wait_for_terminal(client, job_id)
        assert job["status"] == "complete"
        assert job["done"] == job["total"] == 2
        assert job["output_repo_id"] == "test/src_hvla224"

    def test_second_concurrent_job_rejected(self, client, monkeypatch, tmp_path):
        gate = threading.Event()

        def blocking_prepare(**kwargs):
            gate.wait(timeout=10)

        _patch_core(monkeypatch, blocking_prepare)
        try:
            res1 = client.post(
                "/api/dataset-preparation/hvla",
                json={"source_repo_id": "test/a", "output_root": str(tmp_path / "a")},
            )
            assert res1.status_code == 201
            res2 = client.post(
                "/api/dataset-preparation/hvla",
                json={"source_repo_id": "test/b", "output_root": str(tmp_path / "b")},
            )
            assert res2.status_code == 409
        finally:
            gate.set()

    def test_existing_output_rejected(self, client, tmp_path):
        existing = tmp_path / "out"
        existing.mkdir()
        res = client.post(
            "/api/dataset-preparation/hvla",
            json={"source_repo_id": "test/src", "output_root": str(existing)},
        )
        assert res.status_code == 409

    def test_failure_is_readable(self, client, monkeypatch, tmp_path):
        def failing_prepare(**kwargs):
            raise RuntimeError("boom: ffmpeg exploded")

        _patch_core(monkeypatch, failing_prepare)
        res = client.post(
            "/api/dataset-preparation/hvla",
            json={"source_repo_id": "test/src", "output_root": str(tmp_path / "out")},
        )
        job = _wait_for_terminal(client, res.json()["job_id"])
        assert job["status"] == "failed"
        assert "boom" in job["error"]

    def test_unknown_job_404(self, client):
        assert client.get("/api/dataset-preparation/jobs/nope").status_code == 404


class TestApiToCoreBoundary:
    """One real end-to-end pass through API -> core -> loadable dataset."""

    @pytest.mark.skipif(get_codec("h264") is None, reason="'h264' not in local FFmpeg build")
    def test_real_tiny_dataset(self, client, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        src_root = tmp_path / "src"
        dataset = LeRobotDataset.create(
            repo_id="test/api_prep_src",
            fps=FPS,
            features=FEATURES,
            root=src_root,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, g=2, preset="ultrafast"),
        )
        for _ in range(2):
            for _ in range(3):
                dataset.add_frame(
                    {
                        "task": "dummy task",
                        "observation.images.cam": np.random.randint(0, 256, size=(32, 48, 3), dtype=np.uint8),
                        "action": torch.randn(2),
                    }
                )
            dataset.save_episode()
        # finalize() flushes buffered episode metadata and info.json totals.
        dataset.finalize()

        out_root = tmp_path / "out"
        res = client.post(
            "/api/dataset-preparation/hvla",
            json={
                "source_repo_id": "test/api_prep_src",
                "source_root": str(src_root),
                "output_root": str(out_root),
            },
        )
        assert res.status_code == 201
        job = _wait_for_terminal(client, res.json()["job_id"])
        assert job["status"] == "complete"

        prepared = LeRobotDataset("test/api_prep_src_hvla224", root=out_root)
        assert prepared.meta.total_episodes == 2
        assert list(prepared.meta.features["observation.images.cam"]["shape"]) == [224, 224, 3]
        assert tuple(prepared[0]["observation.images.cam"].shape) == (3, 224, 224)
