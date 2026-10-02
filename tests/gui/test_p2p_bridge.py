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
