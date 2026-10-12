# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""The Point2Pose bridge's wire protocol: whole frames over a pipe, which hands out short reads."""

import importlib.util
import json
import os
import pathlib
import threading

import numpy as np
import pytest

BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "p2p_bridge.py"


def _bridge_module():
    spec = importlib.util.spec_from_file_location("p2p_bridge", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def upstream_pose_result(monkeypatch):
    """The defective upstream return contract, isolated from GPU models.

    The same six-point case was reproduced through the real installed register:
    returned x=20 mm, refined x=0, reported inliers=6, actual returned inliers=0.
    Tests independently recompute residuals from the pose the caller receives.
    """
    import sys
    import types

    source = np.array([[x, y, 0.4] for x in [-0.01, 0, 0.01] for y in [-0.01, 0.01]])
    seed = np.eye(4)
    seed[0, 3] = 0.02
    refined = np.eye(4)
    reported = np.zeros(len(source))
    result = {
        "clusters": [{"T": seed.copy()}, {"T": refined.copy()}],
        "best_cluster_idx": 1,
        "sdf_refine": {"applied": True},
        "final_inlier_gate": {"applied": False},
        "residuals": reported,
        "inliers": reported <= 0.005,
    }

    class Register:
        def register(self, *args, **kwargs):
            self._last_selected_T = seed.copy()
            return seed.copy(), result

    names = [
        "point2pose",
        "point2pose.modules",
        "point2pose.modules.register",
        "point2pose.modules.register.svd_cluster_ransac_register",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[names[-1]].SVDClusterRANSACRegister = Register
    bridge = _bridge_module()
    monkeypatch.setattr(bridge, "_unchanged", lambda fn, pin: pin == "97600960c87a")
    return bridge, Register, source, seed, refined, result


def test_sdf_refinement_returns_the_pose_that_its_fit_quality_describes(upstream_pose_result):
    bridge, register, source, seed, refined, stats = upstream_pose_result
    bridge._fix_sdf_pose_result()
    instance = register()
    pose, result = instance.register(source, source)
    actual_residuals = np.linalg.norm(source @ pose[:3, :3].T + pose[:3, 3] - source, axis=1)
    np.testing.assert_allclose(actual_residuals, result["residuals"], atol=1e-12)
    np.testing.assert_array_equal(actual_residuals <= 0.005, result["inliers"])
    np.testing.assert_array_equal(instance._last_selected_T, pose)
    np.testing.assert_array_equal(pose, refined)
    assert not np.shares_memory(pose, stats["clusters"][1]["T"])


def test_refinement_repair_preserves_a_real_object_rotation(upstream_pose_result):
    bridge, register, source, _, _, stats = upstream_pose_result
    angle = np.deg2rad(25)
    refined = np.eye(4)
    refined[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    centre = source.mean(axis=0)
    refined[:3, 3] = centre - refined[:3, :3] @ centre + [0.01, -0.02, 0.03]
    target = source @ refined[:3, :3].T + refined[:3, 3]
    stats["clusters"][1]["T"] = refined
    bridge._fix_sdf_pose_result()
    pose, result = register().register(source, target)
    np.testing.assert_allclose(source @ pose[:3, :3].T + pose[:3, 3], target, atol=1e-12)
    np.testing.assert_array_equal(result["inliers"], np.ones(len(source), bool))


@pytest.mark.parametrize("case", ["unsupported", "skipped", "disabled", "no_candidates"])
def test_refinement_repair_preserves_existing_fallbacks(upstream_pose_result, case):
    bridge, register, source, seed, _, stats = upstream_pose_result
    # Reports that score the seed must still return that seed, even when a
    # discarded refined candidate remains in the diagnostic cluster list.
    if case == "unsupported":
        stats["final_inlier_gate"]["applied"] = True
    elif case in ("skipped", "disabled"):
        stats["sdf_refine"].update(applied=False, enabled=case != "disabled")
    else:
        stats.clear()
    stats["residuals"] = np.full(len(source), 0.02)
    stats["inliers"] = np.zeros(len(source), bool)
    bridge._fix_sdf_pose_result()
    pose, result = register().register(source, source)
    np.testing.assert_array_equal(pose, seed)
    actual_residuals = np.linalg.norm(source @ pose[:3, :3].T + pose[:3, 3] - source, axis=1)
    np.testing.assert_allclose(actual_residuals, result["residuals"], atol=1e-12)


def test_refinement_repair_is_installed_by_bridge_startup_only_once(upstream_pose_result, monkeypatch):
    bridge, register, source, *_ = upstream_pose_result
    for name in (
        "_speed_up_sdf_refine",
        "_speed_up_sdf_costs",
        "_speed_up_ransac",
        "_speed_up_depth",
        "_speed_up_sam2_input",
        "_speed_up_tapir",
        "_tapir_query_chunk",
    ):
        monkeypatch.setattr(bridge, name, lambda *args: None)
    bridge._speedups(None)
    once = register.register
    bridge._speedups(None)
    assert register.register is once
    pose, stats = register().register(source, source)
    assert pose[0, 3] == 0 and stats["sdf_refine"]["applied"]


def test_refinement_repair_leaves_an_unreviewed_upstream_version_untouched(upstream_pose_result, monkeypatch):
    bridge, register, *_ = upstream_pose_result
    monkeypatch.setattr(bridge, "_unchanged", lambda *args: False)
    original = register.register
    bridge._fix_sdf_pose_result()
    assert register.register is original


def test_a_frame_larger_than_the_pipe_buffer_arrives_whole():
    bridge = _bridge_module()
    r_fd, w_fd = os.pipe()
    reader, writer = os.fdopen(r_fd, "rb", buffering=0), os.fdopen(w_fd, "wb", buffering=0)
    rgb = np.random.default_rng(0).integers(0, 255, (480, 848, 3), dtype=np.uint8)
    depth = np.random.default_rng(1).random((480, 848), dtype=np.float32)

    # The writer runs beside the reader: a pipe holds far less than one frame.
    t = threading.Thread(
        target=bridge._write, args=(writer,), kwargs={"kind": "step", "rgb": rgb, "depth": depth}
    )
    t.start()
    got = bridge._read(reader)
    t.join()
    writer.close()
    assert got is not None
    assert str(got["kind"]) == "step"
    np.testing.assert_array_equal(got["rgb"], rgb)
    np.testing.assert_array_equal(got["depth"], depth)
    assert bridge._read(reader) is None, "a closed pipe reads as the end of the conversation"
    reader.close()


def test_a_reply_round_trips_its_json_and_arrays():
    bridge = _bridge_module()
    r_fd, w_fd = os.pipe()
    reader, writer = os.fdopen(r_fd, "rb"), os.fdopen(w_fd, "wb")
    meta = {"ok": True, "lost": False, "n_visible": 60}
    bridge._write(writer, meta=json.dumps(meta), delta=np.eye(4), live_uv=np.zeros((60, 2), np.float32))
    writer.close()
    got = bridge._read(reader)
    reader.close()
    assert json.loads(str(got["meta"])) == meta
    assert got["delta"].shape == (4, 4) and got["live_uv"].shape == (60, 2)


@pytest.mark.parametrize("segmented", [False, None, True])
def test_visible_tracks_with_no_segmentation_remain_explainable_in_the_act_record(
    tmp_path, monkeypatch, segmented
):
    """A held object had many visible tracks but no fit: an empty mask vetoed them all. Preserve that distinction
    through bridge, worker and recording, without treating missing diagnostics as an empty observation."""
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(BRIDGE.parent))
    import pregrasp_worker as pw

    from lerobot.gui.api import pregrasp

    bridge = _bridge_module()
    session = bridge.Session(None)
    uv = np.arange(16, dtype=np.float32).reshape(8, 2)
    table = SimpleNamespace(
        visible=np.ones(8, bool), track_2d=uv, valid=np.ones(8, bool), obj2track_map={0: np.arange(8)}
    )
    obj = SimpleNamespace(
        lost=True, lost_streak=9, mean_residual=-1, key_points=np.ones((8, 3)), pose=np.eye(4)
    )
    session.pipe = SimpleNamespace(track_table=table, objects=[obj])
    support = {
        "extract_val_obj_mask": np.array([True] * 6 + [False] * 2),
        "extract_uncer_obj_mask": np.array([True] * 5 + [False] * 3),
        "extract_valid_kp_mask": np.ones(8, bool),
        "extract_inside_mask": np.full(8, segmented or False),
    }
    session._fe = SimpleNamespace(valid_stats={0: support}, reg_stats={}, valid_indices={})
    mask = None if segmented is None else np.full((1, 1, 8, 8), segmented, bool)
    wire = session._answer(SimpleNamespace(intrinsics=np.eye(3), mask=mask), 1.0)
    result = {**json.loads(wire.pop("meta")), **wire}
    tracker = pw.Tracker(None, None, None, None, p2p=SimpleNamespace(step=lambda *args: result))
    rgb, depth = np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 0.4, np.float32)
    answer = tracker.step(pw._Frame(rgb, depth, "partial"), "p2p")
    monkeypatch.setattr(pregrasp._state, "demo", None)
    monkeypatch.setattr(pregrasp._state, "groups_with_acts", False)
    run = pregrasp._Run(tmp_path, {})
    run.frame(SimpleNamespace(rgb=rgb, depth_m=depth, created=1.0), answer, {"ok": False}, "hold", None, None)
    pregrasp._RUN_EXECUTOR.submit(lambda: None).result()
    recorded = run.frames[0]
    assert recorded["ok"] is False and recorded["n_matches"] == 8 and recorded["fit_points"] == 0
    assert recorded["mask_pixels"] == (None if segmented is None else 64 * segmented)
    assert recorded["fit_support"] == {
        "depth_valid": 6,
        "certain": 5,
        "mapped": 8,
        "inside_mask": 8 * bool(segmented),
    }
    session._fe = None
    assert session._fit_support(0) == {}, "a skipped front end must not reuse the previous frame's counts"


def test_a_new_pipeline_does_not_wait_for_the_collector_to_free_the_last_one(monkeypatch):
    """Every find starts a pipeline with its own models on the GPU. The wrapper that reads the front end's result
    refers back to the front end, so each old pipeline stayed alive until a garbage collection that did not come:
    the bridge grew with every find until CUDA ran out of memory."""
    import gc
    import sys
    import types
    import weakref

    bridge = _bridge_module()

    class FrontEnd:
        def step(self, frame):
            return {"tracks": 0}

    class Pipeline:
        def __init__(self, cfg):
            self.frontend = FrontEnd()

        def step(self, frame):
            self.frontend.step(frame)

    fakes = {
        "point2pose": types.ModuleType("point2pose"),
        "point2pose.data_types": types.ModuleType("point2pose.data_types"),
        "point2pose.data_types.frame": types.ModuleType("point2pose.data_types.frame"),
        "point2pose.pipeline": types.ModuleType("point2pose.pipeline"),
        "point2pose.pipeline.modular_pipeline": types.ModuleType("point2pose.pipeline.modular_pipeline"),
    }
    fakes["point2pose.data_types.frame"].Frame = lambda **kw: types.SimpleNamespace(**kw)
    fakes["point2pose.pipeline.modular_pipeline"].ModularPipeline = Pipeline
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(bridge.Session, "_answer", lambda self, frame, ms: {})

    session = bridge.Session(cfg=None)
    rgb, depth = np.zeros((48, 84, 3), np.uint8), np.full((48, 84), 0.45, np.float32)
    mask = np.ones((48, 84), dtype=bool)
    gc.disable()  # no collection between the two finds, as on the rig
    try:
        session.init(rgb, depth, np.eye(3), mask)
        first = weakref.ref(session.pipe.frontend)
        session.init(rgb, depth, np.eye(3), mask)
        assert first() is None, "the previous pipeline outlived the next find"
        assert session.pipe.frontend is not None
    finally:
        gc.enable()


FAKE_BRIDGE = """
import io, json, os, struct, sys
import numpy as np

inp, out = sys.stdin.buffer, sys.stdout.buffer


def send(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    out.write(struct.pack(">I", len(data)) + data)
    out.flush()


send(meta=json.dumps({"ready": True, "shared_models": os.environ.get("FAKE_SHARED") == "1"}))
while True:
    head = inp.read(4)
    if len(head) < 4:
        sys.exit(0)
    n, body = struct.unpack(">I", head)[0], b""
    while len(body) < n:
        chunk = inp.read(n - len(body))
        if not chunk:
            sys.exit(0)
        body += chunk
    req = np.load(io.BytesIO(body))
    send(meta=json.dumps({"ok": True, "kind": str(req["kind"]), "pid": os.getpid()}))
"""


@pytest.mark.parametrize("shared", [True, False])
def test_a_find_restarts_tracking_in_the_running_bridge_unless_it_cannot_share_its_models(
    tmp_path, monkeypatch, shared
):
    """Every find after the first started a new bridge process, seconds in which nothing was tracked: a pipeline
    replaced inside one process left part of its models on the GPU. A bridge that loads its models once and shares them
    with every pipeline keeps running; one that cannot (Point2Pose not the version it was written against) still
    starts a fresh process for each later init, and its old process exits with its GPU memory."""
    import sys

    sys.path.insert(0, str(BRIDGE.parent))
    import pregrasp_worker as pw

    from lerobot.showservo.pose import CameraIntrinsics

    fake = tmp_path / "fake_bridge.py"
    fake.write_text(FAKE_BRIDGE)
    monkeypatch.setattr(pw, "P2P_PYTHON", sys.executable)
    monkeypatch.setattr(pw, "P2P_BRIDGE", fake)
    monkeypatch.setenv("FAKE_SHARED", "1" if shared else "0")
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=424.0, cy=240.0)
    rgb, depth = np.zeros((48, 84, 3), np.uint8), np.full((48, 84), 0.45, np.float32)
    mask = np.ones((48, 84), dtype=bool)
    bridge = pw.P2PBridge(tmp_path / "unused.yaml")
    try:
        first = bridge.init(rgb, depth, mask, intr)["pid"]
        assert bridge.step(rgb, depth)["pid"] == first, "steps stay in the process of their init"
        old = bridge.proc
        second = bridge.init(rgb, depth, mask, intr)["pid"]
        if shared:
            assert second == first and old.poll() is None, (
                "a later find restarts tracking in the running bridge"
            )
        else:
            assert second != first, "a later find runs in a fresh process"
            assert old.poll() is not None, "the old process has exited, and its GPU memory with it"
        assert bridge.step(rgb, depth)["pid"] == second
    finally:
        bridge.close()
