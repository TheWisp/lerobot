"""Same measured frames must take the same path through live and offline adapters."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from benchmarks.flash_tracker_experiment import run_case
from lerobot.gui.flash_group_overlay import SceneGroupOverlay
from lerobot.gui.flash_live_scene import live_scene_packet, loss_reason
from lerobot.policies.flash.depth_enhancement import DepthOutput
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.live_surface_worker import _prepare_measurements
from lerobot.policies.flash.motion_groups import GroupConfig
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
    assert loads == (["ufm"] if backend == "ufm" else [])
    assert model.calls == (4 if backend == "ufm" else 0)


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
    monkeypatch.setattr("benchmarks.flash_tracker_experiment.samples", lambda _: iter(frames))
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


def test_learned_depth_and_sam_measure_the_same_frame_concurrently():
    captured = frame(7)
    mask_started, depth_started = Event(), Event()

    class Masks:
        def measure_masks(self, incoming):
            assert incoming is captured
            mask_started.set()
            assert depth_started.wait(2), "depth did not start while SAM was active"
            return incoming.frame_id, masks(), 1.0

    class Depth:
        def predict(self, incoming):
            np.testing.assert_array_equal(incoming.rgb, captured.rgb)
            np.testing.assert_array_equal(incoming.sensor_depth_m, captured.metric_depth())
            depth_started.set()
            assert mask_started.wait(2), "SAM did not start while depth was active"
            return DepthOutput(incoming.sensor_depth_m + 0.002, "fixture", captured.depth > 0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        tracking_frame, measured, depth_ms = _prepare_measurements(captured, Masks(), Depth(), None, executor)
    assert measured[0] == tracking_frame.frame_id == captured.frame_id
    np.testing.assert_array_equal(tracking_frame.depth, captured.depth + 2)
    np.testing.assert_array_equal(tracking_frame.depth_before_filters, captured.depth)
    assert depth_ms >= 0


def test_sensor_depth_does_not_start_a_second_mask_path():
    captured = frame(7)
    tracking_frame, measured, depth_ms = _prepare_measurements(captured, None, None, None, None)
    assert tracking_frame is captured
    assert measured is None
    assert depth_ms == 0


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
