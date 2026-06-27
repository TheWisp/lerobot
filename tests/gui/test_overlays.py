# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the Overlays backend logic — regression coverage for bugs hit
while building it: the model-key rename, camera-filter resolution, and the
frame/PNG conversions. (The frontend race / display filter and the standalone's
new-frame gating are covered end-to-end, not here.)"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.gui.api import overlays
from lerobot.policies.debug_vision import adapters, standalone


def test_every_step_key_resolves_to_an_adapter():
    # The picker maps each step key -> build_adapter(key); a rename that desyncs
    # them (sam3_video -> sam3_track) would 400 at run time. This guards it.
    for step in overlays._STEPS:
        assert step["key"] in adapters.ADAPTERS, f"overlay step {step['key']!r} has no adapter"


def test_sam3_step_external_label_is_sam3():
    step = next(s for s in overlays._STEPS if s["key"] == "sam3_track")
    assert step["label"] == "SAM3"


def test_no_step_advertises_the_misleading_video_key():
    # The old "sam3_video" name implied the OOM-prone Sam3VideoModel; it must not
    # be what the panel exposes.
    assert all(s["key"] != "sam3_video" for s in overlays._STEPS)


@pytest.mark.parametrize(
    "filt,cams,expected",
    [
        (None, ["o.top", "o.front"], {"o.top", "o.front"}),  # None -> all
        ([], ["o.top", "o.front"], {"o.top", "o.front"}),  # empty -> all
        (["top"], ["o.top", "o.front"], {"o.top"}),  # substring
        (["o.top"], ["o.top", "o.front"], {"o.top"}),  # exact
        (["nope"], ["o.top", "o.front"], {"o.top", "o.front"}),  # no match -> all
        (["top", "front"], ["o.top", "o.front", "o.wrist"], {"o.top", "o.front"}),  # multi
        (["TOP"], ["o.top"], {"o.top"}),  # case-insensitive
    ],
)
def test_resolve_active(filt, cams, expected):
    assert standalone._resolve_active(filt, cams) == expected


def test_frame_rgb_chw_float_to_hwc_uint8():
    t = torch.zeros(3, 4, 5)  # CHW float in [0,1]
    t[0] = 1.0  # full red
    out = overlays._frame_rgb({"cam": t}, "cam")
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8
    assert (out[..., 0] == 255).all() and (out[..., 1] == 0).all()


def test_frame_rgb_hwc_uint8_passthrough():
    t = torch.randint(0, 256, (8, 6, 3), dtype=torch.uint8)  # H=8 not in {1,3,4}
    out = overlays._frame_rgb({"cam": t}, "cam")
    assert out.shape == (8, 6, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, t.numpy())


def test_frame_rgb_rgba_chw_drops_alpha():
    t = torch.zeros(4, 4, 5)  # CHW with 4 channels
    assert overlays._frame_rgb({"cam": t}, "cam").shape == (4, 5, 3)


def test_frame_rgb_grayscale_expands_to_rgb():
    t = torch.zeros(4, 5)  # HW
    assert overlays._frame_rgb({"cam": t}, "cam").shape == (4, 5, 3)


def test_png_roundtrip_preserves_rgba():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[2:6, 2:6] = (255, 0, 0, 255)
    png = overlays._png(rgba)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    back = np.array(Image.open(io.BytesIO(png)))
    assert back.shape == (8, 8, 4)
    np.testing.assert_array_equal(back, rgba)


def test_proc_sm_none_is_zero():
    assert overlays._proc_sm(None) == 0


def test_proc_sm_returns_int_in_range():
    import os

    u = overlays._proc_sm(os.getpid())
    assert isinstance(u, int) and 0 <= u <= 100


# (_require_cuda was removed with the in-process data path; the CUDA gate now lives in the
# out-of-process worker — standalone.py shows a red badge when CUDA is unavailable.)


def test_seed_drops_degenerate_object_instead_of_killing_the_rest():
    """Regression: a degenerate seed mask makes the SAM3 tracker reject the whole
    conditioning frame ("maskmem_features ... empty"), which used to take every
    co-seeded object down with it (real bug: chess piece from the top view killed
    'robot arm'). _seed must drop the smallest-area object and retry."""
    import contextlib

    a = object.__new__(adapters.Sam3TrackByDetectionAdapter)  # skip the heavy __init__

    class _Torch:
        float16 = "float16"  # only used as the session dtype arg

        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    a._torch = _Torch()
    a.device = "cpu"

    seeded: list[int] = []

    class _Sess:
        def add_new_frame(self, pv):
            return 0

    class _Proc:
        def init_video_session(self, **kw):
            seeded.clear()
            return _Sess()

        def process_new_mask_for_video_frame(self, inference_session, frame_idx, obj_ids, input_masks):
            seeded.append(obj_ids[0])

    a.trk_proc = _Proc()

    def _trk(inference_session, frame_idx):
        if len(seeded) > 1:  # the multi-object conditioning is what the bug rejects
            raise ValueError(
                "maskmem_features in conditioning outputs cannot be empty when not is_initial..."
            )
        return object()

    a.trk = _trk
    a._read_output = lambda track, out, h, w: None

    track = {"session": None, "objs": {}, "masks": {}, "scores": {}, "since_flush": 0}
    seeds = {"robot arm": np.ones((20, 20), bool), "chess piece": np.ones((3, 3), bool)}  # piece is smaller
    a._seed(track, seeds, pv=None, h=20, w=20)

    assert track["session"] is not None, "should recover, not leave the track sessionless"
    assert "robot arm" in track["objs"], "the good object must survive"
    assert "chess piece" not in track["objs"], "the smallest (degenerate) object should be dropped"


def test_seed_flags_every_object_for_conditioning():
    """Regression for the real multi-object bug: process_new_mask_for_video_frame
    REPLACES the session's "new input" set each call, so after seeding N objects only
    the last is flagged — the tracker then conditions only that one and crashes on the
    rest with "maskmem_features ... empty". _seed must re-flag ALL seeded objects before
    the tracker runs, so every object is conditioned (and e.g. +/- carving works)."""
    import contextlib

    a = object.__new__(adapters.Sam3TrackByDetectionAdapter)

    class _Torch:
        float16 = "float16"

        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    a._torch = _Torch()
    a.device = "cpu"

    class _Sess:
        def __init__(self):
            self.obj_with_new_inputs = []

        def add_new_frame(self, pv):
            return 0

    sess = _Sess()

    class _Proc:
        def init_video_session(self, **kw):
            return sess

        def process_new_mask_for_video_frame(self, inference_session, frame_idx, obj_ids, input_masks):
            inference_session.obj_with_new_inputs = list(obj_ids)  # the real bug: replace, not append

    a.trk_proc = _Proc()
    saw = {}

    def _trk(inference_session, frame_idx):
        saw["flags"] = list(inference_session.obj_with_new_inputs)  # what the tracker actually sees
        return object()

    a.trk = _trk
    a._read_output = lambda track, out, h, w: None

    track = {"session": None, "objs": {}, "masks": {}, "scores": {}, "since_flush": 0}
    seeds = {"metal plate": np.ones((20, 20), bool), "meat": np.ones((10, 10), bool)}  # plate larger
    a._seed(track, seeds, pv=None, h=20, w=20)

    assert saw.get("flags") == [1, 2], (
        f"all seeded objects must be flagged for the tracker, got {saw.get('flags')!r}"
    )
    assert set(track["objs"].values()) == {1, 2}, "both objects must survive the seed"


# ---- composite / parse semantics (the +/- carving + control parsing) ----


class _FakeCV2:
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 0

    @staticmethod
    def findContours(*a, **k):  # noqa: N802 — mimics cv2's camelCase API
        return [], None

    @staticmethod
    def drawContours(*a, **k):  # noqa: N802
        return None


def _box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_composite_positive_fills_and_negative_carves():
    """The heart of +/-: a positive is filled in its colour; a negative is carved OUT of
    overlapping positives and never drawn itself (what 'cube - reflections' relies on)."""
    h, w = 12, 12
    pos, neg = _box(h, w, 2, 10, 2, 10), _box(h, w, 4, 8, 4, 8)  # neg sits inside pos
    rgba = adapters._composite_concepts(
        h, w, {"a": [pos]}, ["a"], {"a": (10, 20, 30)}, {"a": "+"}, None, _FakeCV2()
    )
    assert tuple(rgba[3, 3, :3]) == (10, 20, 30) and rgba[3, 3, 3] > 0  # positive filled in its colour
    assert rgba[0, 0, 3] == 0  # outside is transparent

    carved = adapters._composite_concepts(
        h,
        w,
        {"a": [pos], "b": [neg]},
        ["a", "b"],
        {"a": (10, 20, 30), "b": (90, 90, 90)},
        {"a": "+", "b": "-"},
        None,
        _FakeCV2(),
    )
    assert carved[6, 6, 3] == 0, "the negative region must be carved out of the positive"
    assert carved[3, 3, 3] > 0, "the rest of the positive must remain"
    assert not (carved[..., :3] == (90, 90, 90)).any(), "the negative concept itself is never drawn"


def test_composite_background_fills_the_inverse():
    h, w = 8, 8
    pos = _box(h, w, 2, 6, 2, 6)
    rgba = adapters._composite_concepts(
        h, w, {"a": [pos]}, ["a"], {"a": (1, 2, 3)}, {"a": "+"}, (200, 100, 50), _FakeCV2()
    )
    assert tuple(rgba[0, 0, :3]) == (200, 100, 50) and rgba[0, 0, 3] > 0, (
        "background fills the inverse region"
    )
    assert rgba[3, 3, 3] > 0, "the detection itself is still drawn on top"


def test_parse_objects_names_colours_signs():
    names, colors, signs = adapters._parse_objects(
        {"objects": [{"name": "ring", "color": [1, 2, 3], "sign": "-"}, {"name": "arm"}]}, 6
    )
    assert names == ["ring", "arm"]
    assert colors == {"ring": (1, 2, 3)}  # arm omitted -> palette fallback downstream
    assert signs == {"ring": "-", "arm": "+"}
    assert adapters._parse_objects({"objects": []}, 6) == (None, None, None)  # nothing usable -> keep state
    assert adapters._parse_objects({}, 6) == (None, None, None)
    capped, _, _ = adapters._parse_objects({"objects": [{"name": f"o{i}"} for i in range(9)]}, 3)
    assert len(capped) == 3  # capped at max_objects


def test_parse_background_color_transparent_unset():
    assert adapters._parse_background({"background": {"color": [4, 5, 6]}}) == (4, 5, 6)
    assert adapters._parse_background({"background": {"color": None}}) is None  # transparent
    assert adapters._parse_background({}) is adapters._BG_UNSET  # absent -> keep current


def test_concept_color_user_then_palette():
    assert adapters._concept_color("x", ["x"], {"x": (7, 8, 9)}) == (7, 8, 9)  # user choice wins
    assert adapters._concept_color("x", ["x"], {}) == adapters._CONCEPT_PALETTE[0]  # else palette by position
    assert adapters._concept_color("y", ["x", "y"], {}) == adapters._CONCEPT_PALETTE[1]


# --- arbitrary observation-stream camera keys --------------------------------
# The overlay path must be camera-key-agnostic: the real robot uses short keys
# ("front", "top"), dataset feeders use dotted keys ("observation.images.front"),
# a custom stream could use anything. The producer key must equal the consumer
# key end-to-end, and distinct keys must never alias to the same shm block.
@pytest.mark.parametrize(
    "keys",
    [
        ["front", "left_wrist", "right_wrist", "top"],  # the real bi_so107 robot
        ["observation.images.front", "observation.images.top"],  # dotted dataset keys
        ["cam.0", "cam.1", "weird/name", "UPPER"],  # arbitrary / punctuated
    ],
)
def test_overlay_buffer_roundtrips_arbitrary_camera_keys(keys):
    from lerobot.policies.debug_vision.overlay_ipc import SharedOverlayBuffer

    cams = dict.fromkeys(keys, (4, 6))
    writer = SharedOverlayBuffer(cameras=cams, model="t", create=True)
    reader = None
    try:
        for i, k in enumerate(keys):
            writer.write_overlay(k, np.full((4, 6, 4), i + 1, dtype=np.uint8))  # distinct fill per camera
        reader = SharedOverlayBuffer(create=False)
        assert set(reader.cameras) == set(keys), "camera keys did not round-trip through the meta block"
        for i, k in enumerate(keys):
            assert reader.overlay_seq(k) == 1, f"{k!r} overlay invisible to the reader"
            rgba, _ts = reader.read_overlay(k)
            assert int(rgba[0, 0, 0]) == i + 1, f"{k!r} read another camera's overlay (shm key collision)"
    finally:
        if reader is not None:
            reader.cleanup()
        writer.cleanup()


# --- /live/frame: loud failure on a camera-key mismatch ----------------------
def _fake_overlay_reader(cameras, seqs):
    class _R:
        def __init__(self):
            self.cameras = cameras

        def overlay_seq(self, cam):
            return seqs.get(cam, 0)

        def read_overlay(self, cam):
            if seqs.get(cam, 0) == 0:
                return None
            h, w = cameras[cam]
            return np.zeros((h, w, 4), dtype=np.uint8), 0.0

    return _R()


@pytest.fixture
def overlay_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(overlays.router)
    yield TestClient(app)
    overlays._live_reader = None
    overlays._live_png_cache = {}
    overlays._live_frame_warned.clear()
    overlays._live_frame_served.clear()
    overlays._live_proc = None
    overlays._live_model = None
    overlays._live_stopping = False
    overlays._machines.clear()


def test_live_frame_serves_a_produced_camera(overlay_client):
    overlays._live_reader = _fake_overlay_reader({"front": (4, 6)}, {"front": 1})
    assert overlay_client.get("/api/overlays/live/frame/front").status_code == 200


def test_live_frame_unknown_key_is_loud_404_not_silent_204(overlay_client):
    # The bug class behind "no overlay shows up": the frontend requests a camera the
    # producer never made. A silent 204 (warming) hides it; a loud 404 + log surfaces it.
    overlays._live_reader = _fake_overlay_reader({"front": (4, 6)}, {"front": 1})
    assert overlay_client.get("/api/overlays/live/frame/observation.images.front").status_code == 404
    assert "observation.images.front" in overlays._live_frame_warned  # the mismatch was logged


def test_live_frame_known_but_warming_is_204(overlay_client):
    overlays._live_reader = _fake_overlay_reader({"front": (4, 6)}, {"front": 0})  # known, no overlay yet
    assert overlay_client.get("/api/overlays/live/frame/front").status_code == 204


def test_live_status_renders_the_per_model_machine(overlay_client, monkeypatch):
    """live_status is driven by the per-model state machine — inactive when nothing runs,
    loading while the standalone warms, active once it reports phase 'active' (LOADED fired by
    _observe). The endpoint assembles no string state itself."""
    from lerobot.policies.debug_vision.overlay_state import Event

    overlays._live_proc = None
    overlays._live_model = None
    overlays._machines.clear()
    assert overlay_client.get("/api/overlays/live/status").json()["state"] == "inactive"

    class _Proc:
        returncode = None
        pid = 1

    overlays._live_proc = _Proc()
    overlays._live_model = "sam3_track"
    overlays._machine("sam3_track").fire(Event.START)  # inactive -> loading
    monkeypatch.setattr(overlays, "_get_live_reader", lambda: None)
    monkeypatch.setattr(overlays, "_read_status", lambda: {"phase": "loading"})
    assert overlay_client.get("/api/overlays/live/status").json()["state"] == "loading"
    # the standalone now reports 'active' -> _observe fires LOADED -> active, with live fps
    monkeypatch.setattr(overlays, "_read_status", lambda: {"phase": "active", "fps": 5.0, "vram": 3.7})
    r = overlay_client.get("/api/overlays/live/status").json()
    assert r["state"] == "active" and r["available"] is True and r["fps"] == 5.0


# --- C-case: standalone re-attaches when the publisher (teleop) restarts -----
def test_try_reattach_swaps_only_on_publisher_restart():
    """teleop stop+start creates a *fresh* obs-stream segment; the standalone must
    re-attach to it (not stay stuck on the dead one) — but a merely PAUSED stream (same
    segment, same high seq) must NOT trigger a swap. Regression for the lifecycle 'C'
    bug where a restarted teleop left the overlay frozen idle."""
    from lerobot.robots.obs_stream import ObservationStream, ObservationStreamReader

    keys = ["front", "top"]
    feats = dict.fromkeys(keys, (8, 8, 3))
    frame = {k: np.zeros((8, 8, 3), dtype=np.uint8) for k in keys}

    s1 = ObservationStream(feats, {})
    s2 = old = new = None
    try:
        for _ in range(5):
            s1.write_obs(frame)  # advance the segment's seq to 5
        old = ObservationStreamReader()
        assert standalone._try_reattach(old, keys) is None  # paused/live: same segment -> no swap

        s1.cleanup()
        s1 = None  # teleop stops; the old reader still maps the now-dead segment
        s2 = ObservationStream(feats, {})  # teleop starts again -> fresh segment, seq resets
        s2.write_obs(frame)
        new = standalone._try_reattach(old, keys)
        assert new is not None, "did not re-attach to the restarted publisher (C bug)"
        assert max(new.image_seq(c) for c in keys) == 1  # reads the fresh segment, not the dead one
    finally:
        for r in (old, new):
            if r is not None:
                r.close()
        for st in (s1, s2):
            if st is not None:
                st.cleanup()


# --- data publisher: the generation/no-op contract the worker depends on -----
def test_publish_data_frame_generation_and_noop(monkeypatch):
    """The data path's whole correctness-vs-thrash contract lives in publish_data_frame. The worker
    resets its tracker (re-runs the ~200ms/cam detector) on every `generation` bump, so this must
    bump ONLY on a real discontinuity. The 3fps regression was a same-frame re-publish (pause / the
    500ms status poll) being read as a new stream — guarded by the no-op cases here."""
    writes: list[dict] = []
    controls: list[int] = []

    class _Stream:
        def write_obs(self, obs):
            writes.append(obs)

    monkeypatch.setattr(overlays, "_data_pub", _Stream())
    monkeypatch.setattr(overlays, "_data_pub_dataset", "ds")
    monkeypatch.setattr(
        overlays, "_data_pub_cameras", []
    )  # empty -> skip _frame_rgb; assay the decision logic
    monkeypatch.setattr(overlays, "_data_pub_last_pos", None)
    monkeypatch.setattr(overlays, "_data_pub_generation", 0)
    monkeypatch.setattr(
        overlays, "_write_data_control", lambda: controls.append(overlays._data_pub_generation)
    )

    def pub(ep, fr):
        overlays.publish_data_frame("ds", ep, fr, {})

    pub(0, 0)  # first frame -> new stream: bump + write
    assert overlays._data_pub_generation == 1 and len(writes) == 1
    pub(0, 0)  # SAME frame (pause / poll re-publish) -> no bump, NO write
    assert overlays._data_pub_generation == 1 and len(writes) == 1
    pub(0, 1)  # +1 advance (playback) -> continuous: no bump, write
    assert overlays._data_pub_generation == 1 and len(writes) == 2
    pub(0, 1)  # same frame again mid-playback -> no-op
    assert overlays._data_pub_generation == 1 and len(writes) == 2
    pub(0, 2)  # +1 -> continuous
    assert overlays._data_pub_generation == 1 and len(writes) == 3
    pub(0, 50)  # forward scrub -> new stream: bump + write
    assert overlays._data_pub_generation == 2 and len(writes) == 4
    pub(0, 0)  # backward (wrap to loop start) -> new stream
    assert overlays._data_pub_generation == 3 and len(writes) == 5
    pub(1, 1)  # episode change (frame looks like +1 but other episode) -> new stream
    assert overlays._data_pub_generation == 4 and len(writes) == 6
    # the control (the reset signal) is pushed on EVERY discontinuity, never on a continuation/no-op
    assert controls == [1, 2, 3, 4]


def test_publish_data_frame_inactive_is_noop(monkeypatch):
    writes: list[dict] = []

    class _Stream:
        def write_obs(self, obs):
            writes.append(obs)

    monkeypatch.setattr(overlays, "_data_pub", _Stream())
    monkeypatch.setattr(overlays, "_data_pub_dataset", "ds")
    monkeypatch.setattr(overlays, "_data_pub_cameras", [])
    monkeypatch.setattr(overlays, "_data_pub_last_pos", None)
    monkeypatch.setattr(overlays, "_data_pub_generation", 0)
    overlays.publish_data_frame("OTHER", 0, 0, {})  # different dataset -> no-op
    assert writes == [] and overlays._data_pub_generation == 0
    monkeypatch.setattr(overlays, "_data_pub", None)  # no publisher at all -> no-op
    overlays.publish_data_frame("ds", 0, 0, {})
    assert writes == []
