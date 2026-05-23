# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the per-file Hub transfer loops in ``hub_jobs.py``.

The sync functions are exercised directly with monkeypatched HF imports —
we don't hit the network. Endpoint tests then verify the FastAPI plumbing
returns ``job_id`` synchronously and surfaces job state via the polling
endpoint.
"""

from __future__ import annotations

import asyncio
import threading
import time
import types

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.hub_jobs import (
    enumerate_upload_files,
    make_job,
    run_download_sync,
    run_upload_sync,
)
from lerobot.gui.state import AppState

# ── enumerate_upload_files ──────────────────────────────────────────────────


class TestEnumerateUploadFiles:
    """Pure path-walking logic — no HF involved."""

    def test_returns_every_regular_file(self, tmp_path):
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        (tmp_path / "data.parquet").write_bytes(b"abc")
        (tmp_path / "video.mp4").write_bytes(b"xyz")

        files = enumerate_upload_files(tmp_path)
        rels = sorted(p.relative_to(tmp_path).as_posix() for p in files)
        assert rels == ["data.parquet", "meta/info.json", "video.mp4"]

    def test_skips_cache_and_lerobot_metadata(self, tmp_path):
        """``.cache/`` and ``.lerobot_gui_edits.json`` are never pushed."""
        (tmp_path / "data.parquet").write_bytes(b"x")
        (tmp_path / ".lerobot_gui_edits.json").write_text("{}")
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "huggingface").mkdir()
        (tmp_path / ".cache" / "huggingface" / "tmp").write_bytes(b"junk")

        files = enumerate_upload_files(tmp_path)
        rels = [p.relative_to(tmp_path).as_posix() for p in files]
        assert rels == ["data.parquet"]

    def test_sorted_for_determinism(self, tmp_path):
        """File order is sorted — the progress display advances predictably."""
        for name in ["z.bin", "a.bin", "m.bin"]:
            (tmp_path / name).write_bytes(b"x")

        files = enumerate_upload_files(tmp_path)
        rels = [p.relative_to(tmp_path).as_posix() for p in files]
        assert rels == ["a.bin", "m.bin", "z.bin"]

    def test_missing_root_asserts(self, tmp_path):
        with pytest.raises(AssertionError):
            enumerate_upload_files(tmp_path / "nope")


# ── run_upload_sync ─────────────────────────────────────────────────────────


class _FakeHfApi:
    """Stands in for ``huggingface_hub.HfApi``; records every call."""

    instances: list[_FakeHfApi] = []

    def __init__(self):
        self.create_repo_calls: list[dict] = []
        self.dataset_info_calls: list[dict] = []
        _FakeHfApi.instances.append(self)

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)

    def whoami(self):
        return {"name": "test-user"}

    def dataset_info(self, repo_id, files_metadata=False):
        # Configurable per-test via class attribute. The default mirrors a
        # small two-file dataset, enough to exercise the sibling loop.
        return _FakeHfApi._dataset_info_payload

    _dataset_info_payload: types.SimpleNamespace = types.SimpleNamespace(siblings=[])


@pytest.fixture(autouse=True)
def reset_fake_api_instances():
    """Each test starts with an empty instance log so assertions stay scoped."""
    _FakeHfApi.instances.clear()
    yield
    _FakeHfApi.instances.clear()


@pytest.fixture
def patch_hf_upload(monkeypatch):
    """Patch the HF imports used by ``run_upload_sync``.

    Returns the list of recorded ``upload_file`` invocations so tests can
    assert which files were pushed and in what order.
    """
    import huggingface_hub

    uploads: list[dict] = []

    def fake_upload_file(**kwargs):
        uploads.append(kwargs)
        return f"https://huggingface.co/{kwargs['repo_id']}/blob/main/{kwargs['path_in_repo']}"

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "upload_file", fake_upload_file)
    return uploads


class TestRunUploadSync:
    def test_populates_totals_then_increments_per_file(self, tmp_path, patch_hf_upload):
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        (tmp_path / "data.parquet").write_bytes(b"x" * 100)

        job = make_job("ds1", "upload", "user/ds1")
        run_upload_sync(root=tmp_path, repo_id="user/ds1", job=job)

        assert job.status == "complete" or job.status == "running"
        # Status is left at "running" by the sync function on success — the
        # caller (in datasets.py) flips it to "complete". So we accept both
        # to keep the sync function's contract self-contained.
        assert job.files_total == 2
        assert job.files_done == 2
        assert job.bytes_done == job.bytes_total  # all bytes accounted
        assert job.bytes_total >= 100  # at minimum, the parquet's bytes
        # Files uploaded in sorted order — order matters for predictable progress.
        rels = [u["path_in_repo"] for u in patch_hf_upload]
        assert rels == ["data.parquet", "meta/info.json"]
        # Repo was created once before the loop.
        assert len(_FakeHfApi.instances) == 1
        assert _FakeHfApi.instances[0].create_repo_calls == [
            {"repo_id": "user/ds1", "repo_type": "dataset", "exist_ok": True, "private": True}
        ]

    def test_current_file_cleared_after_completion(self, tmp_path, patch_hf_upload):
        """The 'in flight' indicator goes blank when nothing is uploading."""
        (tmp_path / "a.bin").write_bytes(b"x")
        job = make_job("ds", "upload", "user/ds")
        run_upload_sync(root=tmp_path, repo_id="user/ds", job=job)
        assert job.current_file is None

    def test_cancel_between_files_stops_loop(self, tmp_path, monkeypatch):
        """Setting ``cancel_event`` mid-loop short-circuits subsequent uploads."""
        import huggingface_hub

        # Three files; cancel after the first completes.
        for name in ["a.bin", "b.bin", "c.bin"]:
            (tmp_path / name).write_bytes(b"x" * 10)

        uploads: list[str] = []
        job = make_job("ds", "upload", "user/ds")

        def fake_upload_file(**kwargs):
            uploads.append(kwargs["path_in_repo"])
            # Trigger cancellation after the first file. The cancel check
            # at the TOP of the next iteration should short-circuit.
            if len(uploads) == 1:
                job.cancel_event.set()

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        monkeypatch.setattr(huggingface_hub, "upload_file", fake_upload_file)

        run_upload_sync(root=tmp_path, repo_id="user/ds", job=job)

        assert job.status == "cancelled"
        assert uploads == ["a.bin"]  # second & third never reached
        assert job.files_done == 1
        assert job.files_total == 3

    def test_create_repo_can_be_skipped(self, tmp_path, patch_hf_upload):
        """Some flows (e.g. re-upload after a prior create) want to skip create_repo."""
        (tmp_path / "x.bin").write_bytes(b"x")
        job = make_job("ds", "upload", "user/ds")
        run_upload_sync(root=tmp_path, repo_id="user/ds", job=job, create_repo=False)
        assert _FakeHfApi.instances == []  # HfApi never instantiated


# ── run_download_sync ───────────────────────────────────────────────────────


@pytest.fixture
def patch_hf_download(monkeypatch):
    """Patch HF download imports + dataset_info to return a fixed sibling list."""
    import huggingface_hub

    downloads: list[dict] = []

    def fake_hf_hub_download(**kwargs):
        downloads.append(kwargs)
        # Write a stub file so a follow-up read can verify the download
        # happened — mirrors HF's real "writes file under local_dir" behaviour.
        from pathlib import Path as _Path

        target = _Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub")
        return str(target)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    return downloads


def _set_dataset_info(siblings: list[dict]) -> None:
    """Configure the next ``HfApi.dataset_info`` return value."""
    sib_objs = [types.SimpleNamespace(rfilename=s["rfilename"], size=s.get("size", 0)) for s in siblings]
    _FakeHfApi._dataset_info_payload = types.SimpleNamespace(siblings=sib_objs)


class TestRunDownloadSync:
    def test_iterates_siblings_and_calls_hf_hub_download(self, tmp_path, patch_hf_download):
        _set_dataset_info(
            [
                {"rfilename": "meta/info.json", "size": 50},
                {"rfilename": "data/chunk-000/file_000000.parquet", "size": 1000},
            ]
        )

        job = make_job("ds", "download", "user/ds")
        run_download_sync(root=tmp_path / "ds", repo_id="user/ds", job=job)

        assert job.files_total == 2
        assert job.files_done == 2
        assert job.bytes_total == 1050
        assert job.bytes_done == 1050
        assert [d["filename"] for d in patch_hf_download] == [
            "meta/info.json",
            "data/chunk-000/file_000000.parquet",
        ]
        # Files actually written under root — full posix subpath preserved.
        assert (tmp_path / "ds" / "meta" / "info.json").exists()
        assert (tmp_path / "ds" / "data" / "chunk-000" / "file_000000.parquet").exists()

    def test_empty_repo_completes_with_zero_totals(self, tmp_path, patch_hf_download):
        """Repo with no siblings is a valid no-op, not an error."""
        _set_dataset_info([])
        job = make_job("ds", "download", "user/ds")
        run_download_sync(root=tmp_path / "ds", repo_id="user/ds", job=job)
        assert job.files_total == 0
        assert job.files_done == 0
        assert job.bytes_total == 0

    def test_cancel_stops_after_current_file(self, tmp_path, monkeypatch):
        import huggingface_hub

        _set_dataset_info([{"rfilename": f"f{i}.bin", "size": 10} for i in range(4)])
        downloads: list[str] = []
        job = make_job("ds", "download", "user/ds")

        def fake_hf_hub_download(**kwargs):
            downloads.append(kwargs["filename"])
            if len(downloads) == 2:
                job.cancel_event.set()

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

        run_download_sync(root=tmp_path, repo_id="user/ds", job=job)

        assert job.status == "cancelled"
        assert downloads == ["f0.bin", "f1.bin"]  # third never reached
        assert job.files_done == 2


# ── HubJobState ─────────────────────────────────────────────────────────────


class TestHubJobState:
    def test_to_dict_omits_internal_event(self):
        """The JSON snapshot must not leak the cancel_event (not serialisable)."""
        job = make_job("ds", "upload", "user/ds")
        snap = job.to_dict()
        assert "cancel_event" not in snap
        assert snap["job_id"] == job.job_id
        assert snap["direction"] == "upload"
        assert snap["status"] == "pending"

    def test_make_job_generates_unique_ids(self):
        ids = {make_job("ds", "upload", "x/y").job_id for _ in range(50)}
        assert len(ids) == 50


# ── AppState.active_hub_job_for ─────────────────────────────────────────────


class TestActiveHubJobLookup:
    def test_returns_pending_or_running_job(self):
        state = AppState(frame_cache=FrameCache(max_bytes=1_000))
        job = make_job("ds1", "upload", "u/r")
        state.hub_jobs[job.job_id] = job

        assert state.active_hub_job_for("ds1") is job
        assert state.active_hub_job_for("ds2") is None

        job.status = "running"
        assert state.active_hub_job_for("ds1") is job

        job.status = "complete"
        assert state.active_hub_job_for("ds1") is None

        job.status = "failed"
        assert state.active_hub_job_for("ds1") is None


# ── Endpoint integration ────────────────────────────────────────────────────


@pytest.fixture
def app_with_state(monkeypatch):
    """FastAPI app + clean module state; patches HF auth + transfers to mocks."""
    import huggingface_hub

    app = FastAPI()
    app.include_router(datasets_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    datasets_module.set_app_state(state)

    # Always-OK auth so the endpoint's pre-flight whoami doesn't 401 us.
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)

    yield app, state, monkeypatch


def _make_open_dataset(state: AppState, dataset_id: str, root) -> None:
    """Register a minimal fake dataset on AppState so the endpoint can find it.

    We don't need a real LeRobotDataset for these endpoint tests — the
    upload/download endpoints only read ``.root`` and ``.repo_id`` off it.
    """
    state.datasets[dataset_id] = types.SimpleNamespace(
        root=str(root),
        repo_id=dataset_id,
        meta=types.SimpleNamespace(total_episodes=0, total_frames=0),
    )


class TestHubUploadEndpoint:
    def test_returns_job_id_synchronously(self, app_with_state, tmp_path, monkeypatch):
        """POST returns immediately with a job_id; the transfer runs in the background."""
        import huggingface_hub

        app, state, _ = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "x.bin").write_bytes(b"x" * 10)
        _make_open_dataset(state, "user/ds", ds_root)

        # Block the upload until we say so — proves the endpoint is non-blocking.
        upload_gate = threading.Event()
        upload_seen: list[str] = []

        def fake_upload_file(**kwargs):
            upload_seen.append(kwargs["path_in_repo"])
            assert upload_gate.wait(timeout=2.0), "upload_gate never released"

        monkeypatch.setattr(huggingface_hub, "upload_file", fake_upload_file)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                t0 = time.monotonic()
                resp = await client.post(
                    "/api/datasets/user%2Fds/hub/upload",
                    json={"repo_id": "user/ds"},
                )
                elapsed = time.monotonic() - t0
                # The endpoint must NOT wait for the upload — request returns
                # while the worker is still gated.
                assert elapsed < 1.0, f"endpoint blocked for {elapsed:.2f}s"
                assert resp.status_code == 200, resp.text
                job_id = resp.json()["job_id"]

                # Job is registered immediately.
                assert job_id in state.hub_jobs

                # Poll once: status is pending or running, files_done < total.
                progress = await client.get(f"/api/datasets/hub/progress/{job_id}")
                assert progress.status_code == 200
                snap = progress.json()
                assert snap["direction"] == "upload"
                assert snap["status"] in ("pending", "running")

                # Let the worker complete; poll until done.
                upload_gate.set()
                for _ in range(50):  # up to ~2.5s
                    await asyncio.sleep(0.05)
                    snap = (await client.get(f"/api/datasets/hub/progress/{job_id}")).json()
                    if snap["status"] in ("complete", "failed", "cancelled"):
                        break
                assert snap["status"] == "complete", snap
                assert snap["files_done"] == 1
                assert upload_seen == ["x.bin"]

        asyncio.run(run())

    def test_second_upload_returns_409_with_existing_job_id(self, app_with_state, tmp_path, monkeypatch):
        """Concurrent transfer requests surface the in-flight job rather than racing."""
        import huggingface_hub

        app, state, _ = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "x.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        gate = threading.Event()
        monkeypatch.setattr(
            huggingface_hub,
            "upload_file",
            lambda **kw: gate.wait(timeout=2.0),
        )

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = await client.post("/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"})
                assert first.status_code == 200
                first_job = first.json()["job_id"]

                # Give the worker a tick to flip the job to "running" before
                # the second call; otherwise the second call could observe
                # "pending" and the race-prevention assertion is still valid
                # but less interesting.
                await asyncio.sleep(0.05)

                second = await client.post("/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"})
                assert second.status_code == 409, second.text
                detail = second.json()["detail"]
                assert detail["job_id"] == first_job

                gate.set()

        asyncio.run(run())


class TestHubProgressEndpoint:
    def test_unknown_job_returns_404(self, app_with_state):
        app, _state, _ = app_with_state

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/progress/nope-not-a-job")
                assert resp.status_code == 404

        asyncio.run(run())

    def test_cancel_sets_cancel_event(self, app_with_state):
        app, state, _ = app_with_state
        job = make_job("ds", "upload", "user/ds")
        state.hub_jobs[job.job_id] = job
        assert not job.cancel_event.is_set()

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{job.job_id}/cancel")
                assert resp.status_code == 200
                assert resp.json()["status"] == "cancel_requested"
                assert job.cancel_event.is_set()

        asyncio.run(run())

    def test_cancel_unknown_job_returns_404(self, app_with_state):
        app, _state, _ = app_with_state

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/datasets/hub/progress/nope/cancel")
                assert resp.status_code == 404

        asyncio.run(run())
