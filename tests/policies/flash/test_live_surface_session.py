"""Same measured frames must take the same path through live and offline adapters."""

import json
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.flash_tracker_experiment import run_case
from lerobot.gui.flash_group_overlay import GroupOverlay
from lerobot.gui.flash_live_scene import live_scene_packet, loss_reason
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_group_hints import GroupConfig
from lerobot.policies.flash.surface_session_config import defaults
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
def test_live_and_recorded_inputs_have_identical_poses_surfaces_groups_and_covariance(
    tmp_path, monkeypatch, backend, matcher
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
    for i, (f, _) in enumerate(frames):
        row, _, mask = live.step(f)
        live_rows.append(row)
        maps.append(
            {role: live.engine.candidates[method][role].store.points.copy() for role in source.prompts}
        )
        packet = live_scene_packet(f, live, row, mask, defaultdict(GroupOverlay))
        import struct

        n = struct.unpack("<I", packet[:4])[0]
        header = json.loads(packet[4 : 4 + n])
        assert header["frame"] == f.frame_id
        assert header["time_s"] == f.source_time_s
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
        for role in source.prompts:
            a, b = actual["objects"][role], expected["objects"][role]
            for key in (method, method + "_status", method + "_details"):
                assert a[key] == b[key]
            version = b[method + "_details"]["map_version_after"]
            with np.load(
                tmp_path / "recorded/surface_models" / method / role / f"{version:06d}.npz"
            ) as saved:
                np.testing.assert_array_equal(maps[i][role], saved["points"])
    assert source.calls == list(range(5))
    assert live_model.calls == recorded_model.calls == (4 if backend == "ufm" else 0)


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


def test_group_checks_and_matching_share_the_current_frame_geometry(monkeypatch):
    from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
    from lerobot.policies.flash.surface_group_hints import GroupHints

    prepared = {}
    consumed = []
    original_predict = GroupHints.predict
    original_correct = ImageConditionedFit.correct_surface_first

    def predict(self, *args, **kwargs):
        geometry = kwargs.get("prepared_geometry")
        if geometry is not None:
            prepared[id(geometry.observation)] = geometry
        return original_predict(self, *args, **kwargs)

    def correct(self, *args, **kwargs):
        assert self.geometry is prepared[id(self.observation)]
        consumed.append(self.geometry)
        return original_correct(self, *args, **kwargs)

    monkeypatch.setattr(GroupHints, "predict", predict)
    monkeypatch.setattr(ImageConditionedFit, "correct_surface_first", correct)
    frames = list(sequence())
    live = LiveSurfaceSession(defaults(), SuppliedMasks(frames))
    try:
        for frame, _ in frames:
            live.step(frame)
    finally:
        live.close()
    assert len(consumed) >= 2
    assert len({id(g.observation) for g in consumed}) == len(consumed)


def test_worker_module_has_a_real_cli_entrypoint():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "lerobot.policies.flash.live_surface_worker", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert result.returncode == 0
    assert "directory" in result.stdout and "usage:" in result.stdout


def test_shared_defaults_enable_the_validated_geometry_guards():
    settings = defaults()
    config = VisionPipelineConfig.from_dict(settings["config"])
    surface = GrowingSurfaceConfig(**settings["surface_experiment"])
    assert settings["shared_matching"]["backend"] == "lk"
    assert config.shape.scene_depth_constraints
    assert config.shape.prefer_observed_overlap
    assert surface.partial_surface_matching
    assert not surface.uncertainty.reset_on_geometry_conflict


def test_settings_reference_matches_the_actual_shared_defaults():
    from benchmarks.flash_pipeline_settings import catalog

    def flatten(value, prefix=""):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                yield from flatten(item, name)
            else:
                yield name, item

    assert {row["path"]: row["default"] for row in catalog()} == dict(flatten(defaults()))
