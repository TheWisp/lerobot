import json
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
import torch

from lerobot.diagnostics.replay_capture import ReplayCapture


def test_roundtrip_owned_snapshots_and_rerecord(tmp_path):
    capture = ReplayCapture(tmp_path, {})
    x = torch.arange(24).reshape(1, 24).float()
    rng = torch.random.get_rng_state().clone()
    for disposition in ["discarded", "saved"]:
        capture.begin_episode(0)
        capture.set_frame(0)
        capture.record_model_call(state=x, images=[x], noise=x, actions=x)
        capture.finish_episode(disposition, 1)
    x.zero_()
    assert capture.close()
    assert torch.equal(rng, torch.random.get_rng_state())
    for i, disposition in enumerate(["discarded", "saved"]):
        folder = capture.root / f"attempt_{i:06d}"
        manifest = json.loads((folder / "attempt.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["disposition"] == disposition
        payload = torch.load(next(folder.glob("*.pt")), weights_only=True)
        assert payload["state"].sum() == 276
        assert payload["capture"]["frame_index"] == 0


def test_slow_disk_bounds_pending_memory_and_stop(tmp_path):
    capture = ReplayCapture(tmp_path, {}, max_pending_bytes=16)
    gate = threading.Event()
    entered = threading.Event()
    original = capture._save_prediction

    def slow(path, payload):
        entered.set()
        gate.wait(5)
        original(path, payload)

    capture._save_prediction = slow
    capture.begin_episode(0)
    capture.set_frame(0)
    capture.record_model_call(state=torch.zeros(4))
    assert entered.wait(2)
    for i in range(10):
        capture.set_frame(i + 1)
        capture.record_model_call(state=torch.zeros(4))
    assert capture.peak_pending_bytes == 16
    capture.finish_episode("saved", 11)
    start = time.perf_counter()
    assert not capture.close(timeout=0.02)
    assert time.perf_counter() - start < 0.5
    gate.set()
    assert capture.close()
    manifest = json.loads((capture.root / "attempt_000000/attempt.json").read_text())
    assert manifest["status"] == "incomplete" and manifest["missing"] == 10


def test_write_failure_never_raises_into_prediction(tmp_path):
    capture = ReplayCapture(tmp_path, {})

    def fail(*args):
        raise OSError("simulated full disk")

    capture._save_prediction = fail
    capture.begin_episode(1)
    capture.set_frame(0)
    capture.record_model_call(state=torch.zeros(4))
    capture.finish_episode("saved", 1)
    assert not capture.close()
    manifest = json.loads((capture.root / "attempt_000000/attempt.json").read_text())
    assert manifest["status"] == "incomplete"
    assert json.loads((capture.root / "session.json").read_text())["status"] == "incomplete"


def test_aborted_take_and_no_reset_capture(tmp_path):
    capture = ReplayCapture(tmp_path, {})
    capture.record_model_call(state=torch.ones(1))
    capture.begin_episode(2)
    capture.set_frame(0)
    capture.record_model_call(state=torch.ones(1))
    assert capture.close()
    manifests = list(capture.root.glob("attempt_*/attempt.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["disposition"] == "aborted"


def test_gui_defaults_and_explicit_flag(tmp_path):
    import asyncio

    from lerobot.gui.api import run

    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"type":"smolvla"}')
    request = run.RecordRequest(
        robot={}, repo_id="eval/eval_test", single_task="pick", policy_path=str(checkpoint)
    )
    assert request.replay_capture is False
    for enabled in [False, True]:
        request.replay_capture = enabled
        launch = AsyncMock()
        with (
            patch.object(run, "_ensure_no_active_process"),
            patch.object(run, "_release_preview_cameras", AsyncMock()),
            patch.object(run, "_profile_to_cli_args", return_value=[]),
            patch.object(run, "_launch_subprocess", launch),
            patch.object(run, "_active_process") as process,
        ):
            process.pid = 123
            asyncio.run(run.start_record(request))
        args = launch.call_args.args[0]
        assert ("--replay_capture=true" in args) == enabled
    (checkpoint / "config.json").write_text('{"type":"act"}')
    with (
        patch.object(run, "_ensure_no_active_process"),
        patch.object(run, "_release_preview_cameras", AsyncMock()),
    ):
        with pytest.raises(Exception, match="local SmolVLA"):
            asyncio.run(run.start_record(request))


def test_unsupported_modes_fail_before_capture_or_hardware(tmp_path):
    from types import SimpleNamespace

    from lerobot.diagnostics.replay_capture import create_smolvla_capture

    cfg = SimpleNamespace(type="smolvla", rtc_config=None, compile_model=False)
    policy = SimpleNamespace(config=cfg)
    record = SimpleNamespace(interpolation_multiplier=1)
    dataset = SimpleNamespace(root=tmp_path)
    cfg.compile_model = True
    with pytest.raises(ValueError, match="compile_model"):
        create_smolvla_capture(policy, dataset, record)
    cfg.compile_model = False
    cfg.rtc_config = object()
    with pytest.raises(ValueError, match="no RTC"):
        create_smolvla_capture(policy, dataset, record)
    cfg.rtc_config = None
    record.interpolation_multiplier = 2
    with pytest.raises(ValueError, match="interpolation_multiplier"):
        create_smolvla_capture(policy, dataset, record)
    assert not (tmp_path / "replay_capture").exists()
