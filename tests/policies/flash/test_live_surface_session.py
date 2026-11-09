"""Same measured frames must take the same path through live and offline adapters."""

import gc
import json
import subprocess
import sys
from dataclasses import replace
from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest

from benchmarks.flash_tracker_experiment import run_case
from lerobot.gui.flash_group_overlay import SceneGroupOverlay
from lerobot.gui.flash_live_scene import live_scene_packet, loss_reason, object_evidence
from lerobot.policies.flash.depth_enhancement import (
    DepthOutput,
    LiveDepthStream,
    tracking_frame_with_depth_source,
)
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.motion_groups import GroupConfig
from lerobot.policies.flash.perception import LatestRGBDFrameStore
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_session_config import defaults
from lerobot.policies.flash.tracking_history import RollingHistory, decode_record, read_packet
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig
from tests.policies.flash.test_shared_image_tracking import TranslationModel, frame, masks


@pytest.mark.parametrize(
    "check,expected",
    [
        ({"scene_depth_coherent_protrusion": True}, "Fit protrudes into measured empty space"),
        ({"has_visible_overlap": False}, "Fit misses the visible surface"),
        ({"geometry_accepted": False}, "Fit disagrees with measured depth"),
        ({"geometry_accepted": True, "accepted": False}, "Too little overlap with the saved surface"),
    ],
)
def test_live_loss_reports_which_measured_geometry_check_rejected_the_fit(check, expected):
    details = {"reason": "no_observation_compatible_candidate", "fit": {"candidate_check": check}}
    assert loss_reason("lost", details) == expected
    assert loss_reason("predicted", details) is None


def test_live_loss_reports_rejected_point_group_prediction():
    details = {
        "reason": "no_observation_compatible_candidate",
        "hint_source": "object_point_group",
        "prediction_geometry": {"has_visible_overlap": False, "accepted": False},
    }
    assert loss_reason("lost", details) == "Prediction misses the visible surface"


def test_group_vetoed_candidate_remains_visible_only_as_rejected_evidence():
    pose = np.eye(4).tolist()
    result = {
        "group_growing": None,
        "group_growing_status": "lost",
        "group_growing_details": {
            "reason": "surface_group_disagreement",
            "candidate_camera_from_reference": pose,
        },
    }
    evidence = object_evidence(result, "group_growing")
    assert evidence["pose"] is None
    assert evidence["rejected_poses"] == {"fit": pose, "prediction": None}


def test_live_startup_heap_excludes_imports_without_disabling_new_cycle_collection():
    # The policy is process-wide, so verify it in an isolated worker process.
    code = """
import gc
import weakref
from lerobot.policies.flash.live_surface_worker import _freeze_startup_heap
_freeze_startup_heap()
assert gc.isenabled() and gc.get_freeze_count() > 0
class Node: pass
node = Node()
node.cycle = node
ref = weakref.ref(node)
del node
gc.collect()
assert ref() is None
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def sequence():
    # Empty observation between valid frames, actual point tracking, pose/map updates.
    for i in range(5):
        current = frame(i)
        measured = masks()
        if i == 2:
            measured["left"][:] = False
        if i >= 3:
            current = replace(current, depth=current.depth + 2)
        yield current, measured


class SuppliedMasks:
    prompts = {"left": "left object", "right": "right object"}

    def __init__(self, frames):
        self.values = {frame.frame_id: mask for frame, mask in frames}
        self.calls = []

    def masks(self, frame):
        self.calls.append(frame.frame_id)
        return self.values[frame.frame_id]


@pytest.mark.parametrize("backend", ["lk", "ufm"])
def test_live_timing_command_loads_the_selected_image_matcher(tmp_path, monkeypatch, backend):
    import sys

    import torch

    from benchmarks import flash_live_runtime as runtime

    # Run the timing command through the real session, stubbing only neural inference
    # and file input. Previously UFM was selectable but its model was never supplied.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(runtime, "samples", lambda _: iter(sequence()))
    model, loads = TranslationModel(), []

    def load():
        loads.append(backend)
        return model

    monkeypatch.setattr(runtime, "load_ufm", load)
    enabled_before = gc.isenabled()
    gc_during_frames = []
    original_step = LiveSurfaceSession.step

    def step_with_gc_check(self, *args, **kwargs):
        gc_during_frames.append(gc.isenabled())
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(LiveSurfaceSession, "step", step_with_gc_check)
    settings = defaults()
    settings["shared_matching"]["backend"] = backend
    settings["config"]["compiled_surface_fit"] = False
    settings["config"]["reuse_surface_search"] = False
    case = {"id": "live-command", "kind": "recorded_rgbd", "roles": ["left", "right"]}
    case_path, settings_path, output = tmp_path / "case.json", tmp_path / "settings.json", tmp_path / "run"
    case_path.write_text(json.dumps(case))
    settings_path.write_text(json.dumps(settings))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live-runtime",
            "--case",
            str(case_path),
            "--settings",
            str(settings_path),
            "--output",
            str(output),
            "--frames",
            "5",
        ],
    )
    runtime.main()
    report = json.loads((output / "summary.json").read_text())
    assert report["frames"] == 5
    archived = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    assert [row["frame"] for row in archived] == list(range(5))
    assert all(set(row["objects"]) == {"left", "right"} for row in archived)
    assert loads == (["ufm"] if backend == "ufm" else [])
    assert model.calls == (4 if backend == "ufm" else 0)
    assert gc_during_frames == [enabled_before] * 5
    assert gc.isenabled() == enabled_before


@pytest.mark.parametrize("backend", ["lk", "ufm"])
@pytest.mark.parametrize("matcher", ["visible_icp"])
@pytest.mark.parametrize("split_mask_measurement", [False, True])
def test_live_and_recorded_inputs_have_identical_poses_surfaces_groups_and_covariance(
    tmp_path, monkeypatch, backend, matcher, split_mask_measurement
):
    import torch

    # Stub just GPU neural inference; all tracking/fitting/validation is real.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    frames = list(sequence())
    enabled_before = gc.isenabled()

    def recorded_frames(_):
        for item in frames:
            assert gc.isenabled() == enabled_before
            yield item

    monkeypatch.setattr("benchmarks.flash_tracker_experiment.samples", recorded_frames)
    settings = defaults()
    settings["config"]["shape"]["registration"]["backend"] = matcher
    settings["shared_matching"]["backend"] = backend
    settings["group_experiment"]["support_confirmation_s"] = 0.03
    method = "group_growing"
    source = SuppliedMasks(frames)
    live_model, recorded_model = TranslationModel(), TranslationModel()
    live = LiveSurfaceSession(settings, source, live_model)
    live_rows, maps = [], []
    history = RollingHistory(tmp_path / "history", reserve_bytes=0)
    group_overlay = SceneGroupOverlay()
    for i, (f, _) in enumerate(frames):
        measured_masks = live.measure_masks(f) if split_mask_measurement else None
        row, matches, mask = live.step(f, measured_masks=measured_masks)
        live_rows.append(row)
        maps.append(
            {role: live.engine.candidates[method][role].store.points.copy() for role in source.prompts}
        )
        packet = live_scene_packet(
            f,
            live,
            row,
            mask,
            matches,
            group_overlay=group_overlay,
            depth_processing={"decimation": False, "spatial": True, "temporal": False, "hole_filling": False},
        )
        history.submit(packet)
        history.queue.join()
        import struct

        n = struct.unpack("<I", packet[:4])[0]
        header = json.loads(packet[4 : 4 + n])
        assert header["frame"] == f.frame_id
        assert header["time_s"] == f.source_time_s
        assert header["depth_processing"]["spatial"] is True
        assert [g["id"] for g in header["scene_group_tracks"]] == [
            g["id"] for g in row.get("scene_group_tracks", [])
        ]
        assert all("links" in g for g in header["scene_group_tracks"])
        assert all("links" not in g for g in row.get("scene_group_tracks", []))
        for role, obj in header["objects"].items():
            assert (
                obj["diagnostics"]["map_version_after"]
                == row["objects"][role][method + "_details"]["map_version_after"]
            )
            assert "group" not in obj["diagnostics"]  # Already in group_tracks.
            if obj["pose"] is not None:
                assert obj["diagnostics"]["unpublished_camera_from_reference"] is None
        if i == 2 and header["objects"]["left"]["status"] == "lost":
            assert header["objects"]["left"]["reason"] == "Object mask missing"
            assert header["objects"]["left"]["reason_code"] == "no_mask"
        np.testing.assert_array_equal(
            np.frombuffer(packet[4 + n : 4 + n + header["depth_bytes"]], "<f4").reshape(f.depth.shape),
            f.metric_depth().astype("f4"),
        )
    with pytest.raises(ValueError, match="increase"):
        live.engine.step(frames[-1][0], frames[-1][1])
    live.close()
    history.close()
    case = {
        "id": "source-parity",
        "kind": "recorded_rgbd",
        "roles": list(source.prompts),
        "expected_frames": len(frames),
    }
    run_case(
        case,
        tmp_path / "recorded",
        recorded_model,
        VisionPipelineConfig.from_dict(settings["config"]),
        None,
        None,
        single_method=method,
        record_correspondences=True,
        surface_experiment=GrowingSurfaceConfig(**settings["surface_experiment"]),
        group_experiment=GroupConfig.from_dict(settings["group_experiment"]),
        shared_matching_config=SharedMatchingConfig.from_dict(settings["shared_matching"]),
    )
    assert gc.isenabled() == enabled_before
    recorded = [json.loads(line) for line in (tmp_path / "recorded/frames.jsonl").read_text().splitlines()]
    for i, (actual, expected) in enumerate(zip(live_rows, recorded, strict=True)):
        _, retained_masks, evidence, _ = decode_record(
            read_packet(tmp_path / "history", frames[i][0].frame_id)
        )
        assert evidence == actual  # Includes exact timings from that run; a rerun has different timings.
        for role, mask in frames[i][1].items():
            np.testing.assert_array_equal(retained_masks[role], mask)
        for role in source.prompts:
            a, b = actual["objects"][role], expected["objects"][role]
            for key in (method, method + "_status", method + "_details"):
                assert a[key] == b[key]
                assert evidence["objects"][role][key] == b[key]
            version = b[method + "_details"]["map_version_after"]
            with np.load(
                tmp_path / "recorded/surface_models" / method / role / f"{version:06d}.npz"
            ) as saved:
                np.testing.assert_array_equal(maps[i][role], saved["points"])
    assert source.calls == list(range(5))
    assert live_model.calls == recorded_model.calls == (4 if backend == "ufm" else 0)


def test_live_mask_measurement_uses_raw_frame_and_rejects_another_depth_frame(monkeypatch):
    frames = list(sequence())
    raw_frame = frames[0][0]
    model_frame = replace(raw_frame, depth=raw_frame.depth + 2)
    source = SuppliedMasks(frames)
    live = LiveSurfaceSession(defaults(), source)
    measured = live.measure_masks(raw_frame)
    assert measured[0] == raw_frame.frame_id

    def engine_step(frame, masks, *, mask_ms):
        assert frame is model_frame
        assert mask_ms >= 0
        assert source.calls == [raw_frame.frame_id]
        np.testing.assert_array_equal(masks["left"], frames[0][1]["left"])
        return {"frame": frame.frame_id}, {}

    monkeypatch.setattr(live.engine, "step", engine_step)
    try:
        row, _, _ = live.step(model_frame, measured_masks=measured)
        assert row["frame"] == raw_frame.frame_id
        with pytest.raises(ValueError, match="same camera frame"):
            live.step(frames[1][0], measured_masks=measured)
        assert source.calls == [raw_frame.frame_id]
    finally:
        live.close()


def test_saved_camera_history_replays_through_the_live_frame_session(tmp_path, monkeypatch):
    import sys

    from benchmarks import flash_live_runtime as runtime

    settings = defaults()
    settings["config"]["compiled_surface_fit"] = False
    settings["config"]["reuse_surface_search"] = False
    frames = [
        (replace(current, frame_id=10 + 2 * i), measured)
        for i, (current, measured) in enumerate(list(sequence())[:3])
    ]
    source = SuppliedMasks(frames)
    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "config.json").write_text(json.dumps({"pipeline_settings": settings}))
    history = RollingHistory(recording, reserve_bytes=0)
    original = LiveSurfaceSession(settings, source)
    expected = []
    for current, _ in frames:
        row, matches, masks = original.step(current)
        expected.append(row)
        history.submit_frame(current)
        history.queue.join()
        history.attach_result(live_scene_packet(current, original, row, masks, matches, rgb_format="jpeg"))
        history.queue.join()
    history.submit_frame(replace(frame(3), frame_id=16))  # Capture without a tracking tick.
    history.queue.join()
    original.close()
    history.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flash-live-runtime",
            "--recording",
            str(recording),
            "--output",
            str(tmp_path / "replayed"),
            "--frames",
            "3",
        ],
    )
    runtime.main()
    report = json.loads((tmp_path / "replayed/summary.json").read_text())
    assert report["source_recording"] == str(recording)
    assert report["depth_source"] == "sensor"
    assert report["recorded_camera_frames"] == 4
    assert report["recorded_tracking_frames"] == 3
    replayed = [json.loads(line) for line in (tmp_path / "replayed/frames.jsonl").read_text().splitlines()]
    saved_packet = read_packet(recording, 10)
    metadata_length = int.from_bytes(saved_packet[:4], "little")
    assert json.loads(saved_packet[4 : 4 + metadata_length])["rgb_format"] == "raw_rgb8"
    np.testing.assert_array_equal(decode_record(saved_packet)[0].rgb, frames[0][0].rgb)
    assert [row["frame"] for row in replayed] == [10, 12, 14]
    for role in source.prompts:
        measured_availability = sum(row["objects"][role]["group_growing"] is not None for row in replayed[1:])
        assert report["quality"]["roles"][role]["reference_frame"] == 10
        assert (
            report["quality"]["roles"][role]["methods"]["group_growing"]["available_after_initialization"]
            == measured_availability
        )
    for before, after in zip(expected, replayed, strict=True):
        for role in source.prompts:
            assert (
                before["objects"][role]["group_growing_status"]
                == after["objects"][role]["group_growing_status"]
            )
            assert before["objects"][role]["group_growing"] == after["objects"][role]["group_growing"]


def test_learned_depth_runs_on_camera_frames_without_an_object_tracker():
    camera = LatestRGBDFrameStore()
    stop = Event()
    calls, completed = [], []

    class Depth:
        def predict(self, incoming):
            calls.append(float(incoming.sensor_depth_m[0, 0]))
            return DepthOutput(incoming.sensor_depth_m + 0.002, "fixture", incoming.sensor_depth_m > 0)

    stream = LiveDepthStream(camera, Depth(), None, stop, on_result=completed.append)
    stream.start()
    try:
        for index in (7, 8):
            raw = frame(index)
            camera.publish(raw)
            deadline = monotonic() + 2
            while monotonic() < deadline:
                selected, depth_ms = stream.snapshot()
                if selected is not None and selected.frame_id == index:
                    break
                sleep(0.005)
            else:
                pytest.fail(f"Depth stream did not publish camera frame {index}")
            np.testing.assert_array_equal(selected.depth, raw.depth + 2)
            np.testing.assert_array_equal(selected.depth_before_filters, raw.depth)
            assert depth_ms >= 0
        deadline = monotonic() + 2
        while len(completed) < 2 and monotonic() < deadline:
            sleep(0.005)
        assert len(calls) == 2
        assert [result.frame_id for result in completed] == [7, 8]
        np.testing.assert_array_equal(completed[-1].depth, frame(8).depth + 2)
        assert stream.error is None
    finally:
        stop.set()
        stream.join(2)


def test_prefit_and_finished_live_packets_can_share_one_encoded_image():
    from lerobot.gui.flash_live_scene import encode_rgbd_image, encode_rgbd_scene

    measured = frame(6)
    header = {"frame": 6, "objects": {}, "scene_group_tracks": []}
    encoded = encode_rgbd_image(measured, "jpeg")
    assert encode_rgbd_scene(measured, header, rgb_format="jpeg", encoded_rgb=encoded) == (
        encode_rgbd_scene(measured, header, rgb_format="jpeg")
    )


def test_live_and_recorded_depth_selection_preserve_camera_measurement():
    measured = frame(6)

    class Depth:
        def predict(self, incoming):
            np.testing.assert_array_equal(incoming.sensor_depth_m, measured.metric_depth())
            return DepthOutput(incoming.sensor_depth_m + 0.002, "fixture", measured.depth > 0)

    selected = tracking_frame_with_depth_source(measured, Depth())
    np.testing.assert_array_equal(selected.depth, measured.depth + 2)
    np.testing.assert_array_equal(selected.depth_before_filters, measured.depth)
    assert selected.frame_id == measured.frame_id

    metric = replace(measured, depth=measured.metric_depth(), depth_m_per_unit=1.0)
    selected_metric = tracking_frame_with_depth_source(metric, Depth())
    np.testing.assert_allclose(selected_metric.depth, metric.depth + 0.002, rtol=0, atol=1e-7)


def test_sensor_depth_is_the_camera_frame_without_an_extra_source():
    captured = frame(7)
    assert tracking_frame_with_depth_source(captured, None) is captured


def test_new_selection_has_no_old_surface_or_point_memory():
    settings = defaults()
    settings["shared_matching"]["backend"] = "lk"
    frames = list(sequence())
    old = LiveSurfaceSession(settings, SuppliedMasks(frames))
    old.step(frames[0][0])
    old.step(frames[1][0])
    old.close()
    new = LiveSurfaceSession(settings, SuppliedMasks(frames))
    assert not new.engine.references and new.engine.shared.previous is None
    assert new.engine.candidates["group_growing"]["left"].reference is None
    new.close()


def test_scene_groups_update_once_before_all_objects_and_survive_rejected_pose(monkeypatch):
    from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession
    from lerobot.policies.flash.motion_groups import SceneMotionGroups
    from lerobot.policies.flash.object_group_association import ObjectGroupAssociation

    frames = list(sequence())
    events = []
    update = SceneMotionGroups.update
    predict = ObjectGroupAssociation.query
    fit = GrowingSurfaceSession.update

    def group_update(self, scene):
        events.append((scene.frame_id, "group"))
        return update(self, scene)

    def query(self, snapshot, frame, *args, **kwargs):
        assert snapshot.frame_id == frame.frame_id
        events.append((frame.frame_id, "query"))
        return predict(self, snapshot, frame, *args, **kwargs)

    monkeypatch.setattr(SceneMotionGroups, "update", group_update)
    monkeypatch.setattr(ObjectGroupAssociation, "query", query)

    def run(reject):
        live = LiveSurfaceSession(defaults(), SuppliedMasks(frames))
        snapshots = []

        def result(self, prepared, *args, **kwargs):
            value = fit(self, prepared, *args, **kwargs)
            if reject and prepared["frame_id"] > 0:
                value = {**value, "status": "lost", "camera_from_reference": None}
            return value

        monkeypatch.setattr(GrowingSurfaceSession, "update", result)
        try:
            for frame, _ in frames:
                live.step(frame)
                snapshot = live.engine.scene_groups.frame
                snapshots.append(
                    {gid: (g.ids.copy(), g.base_xyz.copy()) for gid, g in snapshot.groups.items()}
                )
        finally:
            live.close()
        return snapshots

    baseline, rejected = run(False), run(True)
    assert any(baseline), "The invariant must exercise real groups"
    for a, b in zip(baseline, rejected, strict=True):
        assert a.keys() == b.keys()
        for gid in a:
            np.testing.assert_array_equal(a[gid][0], b[gid][0])
            np.testing.assert_allclose(a[gid][1], b[gid][1])
    for sample, _ in frames:
        order = [event for index, event in events if index == sample.frame_id]
        assert order == ["group", "query", "query"] * 2
