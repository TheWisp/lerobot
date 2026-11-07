"""History must preserve the real RGB-D evidence without slowing or mutating tracking."""

import json
import shutil
import struct
from dataclasses import replace
from threading import Event

import numpy as np

from benchmarks.flash_experiment_inputs import decode_mask
from lerobot.gui.flash_live_scene import encode_rgbd_scene, frame_evidence
from lerobot.policies.flash.tracking_history import (
    RollingHistory,
    decode_record,
    read_index,
    read_packet,
    save_window,
)
from tests.policies.flash.test_shared_image_tracking import frame


def packet(index, dependencies=()):
    measured = replace(frame(index), source_time_s=float(index), available_time_s=float(index))
    mask = np.ones(measured.depth.shape, bool)
    mask[10:20, 30:40] = False  # Interior holes must survive, not just the outline.
    row = {"frame": index, "time_s": float(index), "objects": {"object": {"group_growing": None}}}
    return encode_rgbd_scene(
        measured,
        {
            "frame": index,
            "time_s": float(index),
            **frame_evidence(row, {"object": mask}),
            "objects": {
                "object": {
                    "name": "test",
                    "status": "lost",
                    "surface": [],
                    "axis_points": [],
                    "diagnostics": {"surface_frames_used": list(dependencies)},
                }
            },
        },
    )


def new_history(path, **kwargs):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"settings": "exact run configuration"}))
    return RollingHistory(path, reserve_bytes=0, **kwargs)


def submit(history, *indices):
    for i in indices:
        history.submit(packet(i))
        history.queue.join()


def test_rolling_frames_and_saved_window_preserve_exact_evidence(tmp_path):
    recent = tmp_path / "recent"
    history = new_history(recent, seconds=2)
    submit(history, 0, 1, 2)
    saved = tmp_path / save_window(recent, tmp_path)
    submit(history, 3, 4, 5)
    history.close()
    assert [r["frame"] for r in read_index(recent)["frames"]] == [3, 4, 5]
    assert [r["frame"] for r in read_index(saved)["frames"]] == [0, 1, 2]
    shutil.rmtree(recent)  # Starting another session cannot erase a saved window.
    restored = read_packet(saved, 1)
    assert restored == packet(1)
    restored_frame, restored_masks, result, _ = decode_record(restored)
    np.testing.assert_array_equal(restored_frame.rgb, frame(1).rgb)
    np.testing.assert_array_equal(restored_frame.metric_depth(), frame(1).metric_depth())
    assert result["frame"] == restored_frame.frame_id == 1
    assert not restored_masks["object"][12, 32]
    n = struct.unpack("<I", restored[:4])[0]
    header = json.loads(restored[4 : 4 + n])
    mask = decode_mask(header["masks_rle"]["object"], (96, 128))
    assert mask.sum() == 96 * 128 - 100
    assert header["result"]["objects"]["object"]["group_growing"] is None
    assert json.loads((saved / "config.json").read_text())["settings"] == "exact run configuration"


def test_clearing_recent_history_keeps_saved_clip_and_resumes_recording(tmp_path):
    recent = tmp_path / "recent"
    history = new_history(recent)
    submit(history, 0, 1)
    saved = tmp_path / save_window(recent, tmp_path)
    history.clear()
    assert not read_index(recent)["frames"]
    assert not list(recent.glob("*.bin.z"))
    assert read_packet(saved, 1) == packet(1)
    submit(history, 2)
    history.close()
    assert [row["frame"] for row in read_index(recent)["frames"]] == [2]


def test_recent_history_keeps_direct_old_measurement_dependencies(tmp_path):
    history = new_history(tmp_path / "recent", seconds=1)
    submit(history, 0, 1)
    history.submit(packet(4, dependencies=[0]))
    history.queue.join()
    history.close()
    rows = read_index(history.directory)["frames"]
    assert [r["frame"] for r in rows] == [0, 4]
    assert rows[0]["context_only"]
    assert not rows[1]["missing_dependencies"]
    assert not (history.directory / "0000000001.bin.z").exists()


def test_retention_expires_by_time_through_long_sessions(tmp_path):
    history = new_history(tmp_path / "recent", seconds=3)
    submit(history, *range(40))
    history.close()
    index = read_index(history.directory)
    assert [r["frame"] for r in index["frames"]] == [36, 37, 38, 39]
    assert len(list(history.directory.glob("*.bin.z"))) == 4
    assert index["bytes"] == sum(p.stat().st_size for p in history.directory.glob("*.bin.z"))


def test_slow_writer_drops_logs_instead_of_blocking_tracker(tmp_path, monkeypatch):
    started, release = Event(), Event()
    original = RollingHistory._write

    def slow(self, value):
        started.set()
        assert release.wait(5)
        return original(self, value)

    monkeypatch.setattr(RollingHistory, "_write", slow)
    history = new_history(tmp_path / "recent")
    history.submit(packet(0))
    assert started.wait(5)
    for i in range(1, 10):
        history.submit(packet(i))
    assert history.queue.qsize() == 2
    assert history.dropped == 7
    release.set()
    history.close()
    assert read_index(history.directory)["dropped_frames"] == 7


def test_disk_full_is_reported_without_hanging_or_changing_results(tmp_path, monkeypatch):
    history = new_history(tmp_path / "recent")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    history.submit(packet(0))
    history.close()
    index = read_index(history.directory)
    assert not index["frames"]
    assert index["dropped_frames"] == 1
    assert "disk reserve" in index["error"]
