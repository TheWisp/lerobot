"""History must preserve the real RGB-D evidence without slowing or mutating tracking."""

import asyncio
import gzip
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
    export_episode,
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


def test_filter_revision_is_preserved_in_raw_history_and_saved_episode(tmp_path):
    from lerobot.policies.flash.tracking_history import capture_packet

    measured = replace(
        frame(7),
        source_time_s=7.0,
        available_time_s=7.0,
        depth_processing={"spatial": True, "temporal": False, "hole_filling": True},
        depth_filter_revision=4,
    )
    history = new_history(tmp_path / "recent")
    history.submit_frame(measured)
    history.queue.join()
    history.close()
    row = read_index(history.directory)["frames"][0]
    assert row["depth_processing"] == measured.depth_processing
    assert row["depth_filter_revision"] == 4
    restored, _, _, _ = decode_record(read_packet(history.directory, 7))
    assert restored.depth_processing == measured.depth_processing
    assert restored.depth_filter_revision == 4
    assert decode_record(capture_packet(measured))[0].depth_processing == measured.depth_processing

    # Processing results are attached later. They must not rewrite provenance
    # captured with the camera frame, even if a stale result reports otherwise.
    overlay = history.directory / "0000000007.overlay.json"
    overlay.write_text(json.dumps({"depth_processing": {"spatial": False}, "depth_filter_revision": 99}))
    restored, _, _, _ = decode_record(read_packet(history.directory, 7))
    assert restored.depth_processing == measured.depth_processing
    assert restored.depth_filter_revision == 4

    root = export_episode(history.directory)
    source = json.loads((root / "flash" / "v1" / "source_frames.json").read_text())["frames"][0]
    assert source["depth_processing"] == measured.depth_processing
    assert source["depth_filter_revision"] == 4


def test_pre_filter_depth_survives_recent_history_and_episode(tmp_path):
    from lerobot.policies.flash.tracking_history import capture_packet

    measured = replace(frame(8), source_time_s=8.0, available_time_s=8.0)
    before = measured.depth.copy()
    before[4, 5] = 0
    measured = replace(
        measured,
        depth_before_filters=before,
        depth_processing={"spatial": False, "temporal": False, "hole_filling": True},
    )
    restored, _, _, _ = decode_record(capture_packet(measured))
    np.testing.assert_array_equal(restored.depth, measured.depth)
    np.testing.assert_array_equal(restored.depth_before_filters, before)

    history = new_history(tmp_path / "recent")
    history.submit_frame(measured)
    unfiltered = replace(
        measured,
        frame_id=9,
        source_time_s=9.0,
        available_time_s=9.0,
        depth_before_filters=None,
        depth_processing={"spatial": False, "temporal": False, "hole_filling": False},
    )
    history.submit_frame(unfiltered)
    history.queue.join()
    history.close()
    restored, _, _, _ = decode_record(read_packet(history.directory, 8))
    np.testing.assert_array_equal(restored.depth_before_filters, before)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = export_episode(history.directory)
    dataset = LeRobotDataset(f"local/flash-{history.directory.name}", root=root, local_files_only=True)
    np.testing.assert_array_equal(
        np.asarray(dataset[0]["observation.images.camera_depth_before_filters"])[0], before
    )
    np.testing.assert_array_equal(
        np.asarray(dataset[1]["observation.images.camera_depth_before_filters"])[0], unfiltered.depth
    )


def test_model_depth_and_stereo_pair_stay_separate_from_raw_camera_record(tmp_path):
    raw = replace(
        frame(10),
        source_time_s=10.0,
        available_time_s=10.0,
        infrared_left=np.full((96, 128), 31, np.uint8),
        infrared_right=np.full((96, 128), 41, np.uint8),
    )
    model_depth = raw.depth.copy()
    model_depth[5, 6] += 7
    model_frame = replace(raw, depth=model_depth, depth_before_filters=raw.depth)
    overlay = encode_rgbd_scene(
        model_frame,
        {
            "frame": 10,
            "time_s": 10.0,
            "depth_source": "foundation_stereo",
            "objects": {},
            "scene_group_tracks": [],
            "masks_rle": {},
            "result": {},
        },
    )
    history = new_history(tmp_path / "recent")
    history.submit_frame(raw)
    history.queue.join()
    history.attach_result(overlay)
    history.queue.join()
    history.close()

    packet = read_packet(history.directory, 10)
    n = struct.unpack("<I", packet[:4])[0]
    header = json.loads(packet[4 : 4 + n])
    assert header["depth_source"] == "sensor"
    assert header["tracking_depth_source"] == "foundation_stereo"
    assert header["model_depth_bytes"] == model_depth.size * 4
    restored, _, _, _ = decode_record(packet)
    np.testing.assert_array_equal(restored.depth, raw.depth)
    np.testing.assert_array_equal(restored.infrared_left, raw.infrared_left)
    np.testing.assert_array_equal(restored.infrared_right, raw.infrared_right)
    offset = (
        4 + n + header["depth_bytes"] + header["depth_before_filter_bytes"] + 2 * header["infrared_bytes"]
    )
    model = np.frombuffer(packet[offset : offset + header["model_depth_bytes"]], "<f4")
    np.testing.assert_allclose(model.reshape(raw.depth.shape), model_frame.metric_depth())

    saved = tmp_path / save_window(history.directory, tmp_path)
    assert (saved / "0000000010.model_depth.zst").is_file()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = export_episode(saved)
    dataset = LeRobotDataset(f"local/flash-{saved.name}", root=root, local_files_only=True)
    np.testing.assert_array_equal(
        np.asarray(dataset[0]["observation.images.camera_depth_model"])[0], model_depth
    )
    assert "observation.images.camera_infrared_left" in dataset.meta.video_keys
    assert "observation.images.camera_infrared_right" in dataset.meta.video_keys


def test_saved_window_exports_a_lerobot_episode_with_depth_and_tracking(tmp_path, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.gui.api import datasets as datasets_api, flash_tracking
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.state import AppState

    recent = tmp_path / "recent"
    history = new_history(recent)
    for i in range(3):
        history.submit_frame(replace(frame(i), source_time_s=float(i), available_time_s=float(i)))
        history.queue.join()
    history.attach_result(packet(1))
    history.queue.join()
    streams = {
        "color": {"width": 128, "height": 96, "fps": 30},
        "depth": {"width": 848, "height": 480, "fps": 30},
    }
    (recent / "camera_streams.json").write_text(json.dumps(streams))
    saved = tmp_path / save_window(recent, tmp_path)
    history.close()
    assert json.loads((saved / "camera_streams.json").read_text()) == streams

    root = export_episode(saved)
    assert export_episode(saved) == root
    dataset = LeRobotDataset(f"local/flash-{saved.name}", root=root, local_files_only=True)
    assert dataset.meta.total_episodes == 1
    assert dataset.episode_rows(0) == (0, 3)
    assert dataset.meta.video_keys == ["observation.images.camera"]
    assert "observation.images.camera_depth" in dataset.meta.depth_keys
    np.testing.assert_array_equal(
        np.asarray(dataset[1]["observation.images.camera_depth"]), frame(1).depth[None]
    )
    assert (root / "flash" / "v1" / "overlays" / "000001.json").exists()
    assert json.loads((root / "flash" / "v1" / "camera_streams.json").read_text()) == streams
    assert json.loads((root / "flash" / "v1" / "camera.json").read_text())["width"] == 128
    assert (
        json.loads((root / "flash" / "v1" / "source_frames.json").read_text())["frames"][1]["source_frame_id"]
        == 1
    )
    monkeypatch.setenv("LEROBOT_FLASH_HISTORY_DIR", str(tmp_path))
    listing = asyncio.run(flash_tracking.history(saved.name))
    assert listing["recordings"][0]["episode_root"] == str(root)
    assert listing["overlay_indices"] == [1]
    assert listing["episode_camera"]["height"] == 96
    assert listing["camera_streams"] == streams
    overlay = asyncio.run(flash_tracking.history_overlay(saved.name, 2))
    assert overlay["source_index"] == 1 and overlay["frame"] == 1
    old_state = datasets_api._app_state
    state = AppState(frame_cache=FrameCache(max_bytes=4_000_000))
    datasets_api.set_app_state(state)
    datasets_api.ensure_executors()
    state.datasets[str(root)] = dataset
    try:
        response = asyncio.run(
            datasets_api.get_frame_depth(str(root), 0, 1, "observation.images.camera_depth")
        )
        saved_depth = np.frombuffer(gzip.decompress(response.body), "<u2").reshape(96, 128)
        np.testing.assert_array_equal(saved_depth, frame(1).depth)
    finally:
        datasets_api._app_state = old_state


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
    for i in range(1, 80):
        history.submit(packet(i))
    assert history.queue.qsize() == 64
    assert history.dropped == 15
    release.set()
    history.close()
    assert read_index(history.directory)["dropped_frames"] == 15


def test_disk_full_is_reported_without_hanging_or_changing_results(tmp_path, monkeypatch):
    history = new_history(tmp_path / "recent")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    history.submit(packet(0))
    history.close()
    index = read_index(history.directory)
    assert not index["frames"]
    assert index["dropped_frames"] == 1
    assert "disk reserve" in index["error"]


def test_camera_frames_precede_selection_and_tracking_attaches_without_replacing_pixels(tmp_path):
    recent = tmp_path / "recent"
    history = new_history(recent, seconds=10)
    originals = [replace(frame(i), source_time_s=float(i), available_time_s=float(i)) for i in range(3)]
    for measured in originals:
        history.submit_frame(measured)
    history.queue.join()
    assert [r["frame"] for r in read_index(recent)["frames"]] == [0, 1, 2]
    before, masks, result, _ = decode_record(read_packet(recent, 1))
    assert masks == {} and result is None
    np.testing.assert_array_equal(before.rgb, originals[1].rgb)
    np.testing.assert_array_equal(before.depth, originals[1].depth)

    history.attach_result(packet(1))
    history.queue.join()
    after, masks, result, _ = decode_record(read_packet(recent, 1))
    np.testing.assert_array_equal(after.rgb, before.rgb)
    np.testing.assert_array_equal(after.depth, before.depth)
    assert result["frame"] == 1 and masks["object"].any()
    assert not read_index(recent)["frames"][0]["objects"]
    assert read_index(recent)["frames"][1]["objects"]["object"]["status"] == "lost"

    saved = tmp_path / save_window(recent, tmp_path)
    history.close()
    assert decode_record(read_packet(saved, 1))[2] == result
    assert decode_record(read_packet(saved, 0))[2] is None


def test_capture_copies_camera_buffers_before_they_are_reused(tmp_path):
    recent = tmp_path / "recent"
    history = new_history(recent)
    measured = frame(0)
    original_rgb, original_depth = measured.rgb.copy(), measured.depth.copy()
    history.submit_frame(measured)
    measured.rgb[:] = 0
    measured.depth[:] = 0
    history.close()
    restored, _, _, _ = decode_record(read_packet(recent, 0))
    np.testing.assert_array_equal(restored.rgb, original_rgb)
    np.testing.assert_array_equal(restored.depth, original_depth)


def test_camera_history_reports_disk_full_without_shortening_its_window(tmp_path, monkeypatch):
    recent = tmp_path / "recent"
    history = new_history(recent, seconds=60)
    history.submit_frame(replace(frame(0), source_time_s=0.0, available_time_s=0.0))
    history.queue.join()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    history.submit_frame(replace(frame(1), source_time_s=1.0, available_time_s=1.0))
    history.close()
    index = read_index(recent)
    assert [row["frame"] for row in index["frames"]] == [0]
    assert index["dropped_frames"] == 1
    assert "disk reserve" in index["error"]
