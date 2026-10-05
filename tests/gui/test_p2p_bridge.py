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

BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "p2p_bridge.py"


def _bridge_module():
    spec = importlib.util.spec_from_file_location("p2p_bridge", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


send(meta=json.dumps({"ready": True}))
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


def test_every_find_after_the_first_gets_a_fresh_bridge_process(tmp_path, monkeypatch):
    """A pipeline replaced inside one bridge process left part of its models on the GPU, and every act's find added
    more until CUDA ran out. The memory comes back only when the process exits, so each later init starts one."""
    import sys

    sys.path.insert(0, str(BRIDGE.parent))
    import pregrasp_worker as pw

    from lerobot.showservo.pose import CameraIntrinsics

    fake = tmp_path / "fake_bridge.py"
    fake.write_text(FAKE_BRIDGE)
    monkeypatch.setattr(pw, "P2P_PYTHON", sys.executable)
    monkeypatch.setattr(pw, "P2P_BRIDGE", fake)
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=424.0, cy=240.0)
    rgb, depth = np.zeros((48, 84, 3), np.uint8), np.full((48, 84), 0.45, np.float32)
    mask = np.ones((48, 84), dtype=bool)
    bridge = pw.P2PBridge(tmp_path / "unused.yaml")
    try:
        first = bridge.init(rgb, depth, mask, intr)["pid"]
        assert bridge.step(rgb, depth)["pid"] == first, "steps stay in the process of their init"
        old = bridge.proc
        second = bridge.init(rgb, depth, mask, intr)["pid"]
        assert second != first, "a later find runs in a fresh process"
        assert old.poll() is not None, "the old process has exited, and its GPU memory with it"
        assert bridge.step(rgb, depth)["pid"] == second
    finally:
        bridge.close()
