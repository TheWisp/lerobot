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
from lerobot.overlays import adapters, standalone


def test_every_step_key_resolves_to_an_adapter():
    # The picker maps each step key -> build_adapter(key); a rename that desyncs
    # them (sam3_video -> sam3_track) would 400 at run time. This guards it.
    for step in overlays._STEPS:
        assert step["key"] in adapters.ADAPTERS, f"overlay step {step['key']!r} has no adapter"


def test_sam3_step_external_label_is_sam3():
    step = next(s for s in overlays._STEPS if s["key"] == "sam3_track")
    assert step["label"] == "SAM3"


def test_video_step_is_the_bounded_streaming_adapter():
    # "sam3_video" was once banned: it implied naive Sam3VideoModel use, whose session
    # retains every streamed frame + per-frame outputs forever (OOM on long streams).
    # It is now a real step — but only because the adapter bounds memory the same way
    # the two-tier does: per-frame eviction + a rolling session rebuild.
    assert any(s["key"] == "sam3_video" for s in overlays._STEPS)
    from lerobot.overlays.adapters import Sam3VideoUnifiedAdapter

    assert Sam3VideoUnifiedAdapter.FLUSH_EVERY > 0


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


def test_png_zeroes_rgb_under_fully_transparent_pixels():
    """A transparent-diff overlay keeps the whole camera frame in its RGB planes, so PNG
    used to compress a full photo the viewer never sees. Measured on a live run-tab
    overlay: 782 KB vs 88 KB. Oversized overlays could not finish downloading before the
    next pull replaced them, so the tile drew nothing at all."""
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., :3] = 200  # a "photo" everywhere...
    rgba[8:12, 8:12, 3] = 255  # ...visible only in this small patch
    png = overlays._png(rgba)
    back = np.array(Image.open(io.BytesIO(png)))
    invisible = back[..., 3] == 0
    assert invisible.any(), "test needs transparent pixels to be meaningful"
    assert not back[invisible][..., :3].any(), "RGB must be zeroed where alpha is 0"
    np.testing.assert_array_equal(back[8:12, 8:12], rgba[8:12, 8:12])  # visible pixels untouched


def test_png_leaves_a_fully_opaque_overlay_alone():
    """The zeroing must be scoped to invisible pixels: a data-tab WYSIWYG composite is
    opaque everywhere and must survive byte-for-byte."""
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., :3] = 123
    rgba[..., 3] = 255
    back = np.array(Image.open(io.BytesIO(overlays._png(rgba))))
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


# ---- parse semantics (control parsing) ----


def test_parse_objects_names_signs():
    # Colors are no longer part of the control contract: object identity in the
    # chrome is auto-assigned, never user-chosen (unified rows retired the palette).
    names, signs = adapters._parse_objects({"objects": [{"name": "ring", "sign": "-"}, {"name": "arm"}]}, 6)
    assert names == ["ring", "arm"]
    assert signs == {"ring": "-", "arm": "+"}
    assert adapters._parse_objects({"objects": []}, 6) == (None, None)  # nothing usable -> keep state
    assert adapters._parse_objects({}, 6) == (None, None)
    capped, _ = adapters._parse_objects({"objects": [{"name": f"o{i}"} for i in range(9)]}, 3)
    assert len(capped) == 3  # capped at max_objects


def test_concept_color_is_stable_by_position():
    assert adapters._concept_color("x", ["x"]) == adapters._CONCEPT_PALETTE[0]
    assert adapters._concept_color("y", ["x", "y"]) == adapters._CONCEPT_PALETTE[1]
    assert adapters._concept_color("stranger", ["x"]) == adapters._color_for("stranger")


# ---- control updates are PARTIAL: an absent key must not clear state ----


def test_absent_treatment_keys_keep_the_current_treatments():
    """The panel pushes partial control updates, so "absent" must mean "unchanged". This
    invariant used to be pinned for the retired background COLOUR (_BG_UNSET); it applies
    just as much to the treatments that replaced it. If absence read as "cleared", moving
    an unrelated control would silently wipe the background treatment mid-session."""
    bg = {"key": "blur", "params": {}}
    objs = {"ring": {"key": "tint", "params": {"color": [1, 2, 3]}}}
    new_bg, new_objs, changed = standalone._apply_treatments({"style": "x"}, bg, objs)
    assert new_bg == bg and new_objs == objs and not changed


def test_present_treatment_keys_replace_and_report_the_change():
    bg = {"key": "blur", "params": {}}
    new_bg, _objs, changed = standalone._apply_treatments(
        {"background_treatment": {"key": "none", "params": {}}}, bg, {}
    )
    assert new_bg == {"key": "none", "params": {}}, "an explicit None DOES clear it"
    assert changed, "the caller must re-render the parked frame"

    same_bg, _objs, changed = standalone._apply_treatments({"background_treatment": bg}, bg, {})
    assert same_bg == bg and not changed, "an identical value is not a change"


# ---- the transparent-diff overlay's alpha ----


def _region(h, w, y0, y1, x0, x1, value=1.0):
    a = np.zeros((h, w), dtype=np.float32)
    a[y0:y1, x0:x1] = value
    return a


def test_diff_alpha_is_transparent_where_nothing_was_treated_or_drawn():
    """The point of the diff: untreated pixels stay fully transparent so the run tab's
    live feed shows through instead of being frozen under an opaque copy of itself."""
    regions = [(_region(8, 8, 0, 8, 0, 8), {"key": "none"})]
    out = standalone._diff_alpha(regions, None, 8, 8)
    assert out.shape == (8, 8) and out.dtype == np.uint8
    assert not out.any(), "a 'none' treatment must not make anything opaque"


def test_diff_alpha_unions_treated_regions_and_ignores_untreated_ones():
    treated = _region(8, 8, 0, 4, 0, 8)
    untreated = _region(8, 8, 4, 8, 0, 8)
    out = standalone._diff_alpha([(treated, {"key": "blur"}), (untreated, {"key": "none"})], None, 8, 8)
    assert (out[0:4] == 255).all(), "the treated region is opaque"
    assert (out[4:8] == 0).all(), "the untreated region stays transparent"


def test_diff_alpha_rounds_full_feather_to_opaque():
    """Regression: feathered alpha reaches 0.9999998 at a large region's centre, and
    truncating gave 254 — every committed-looking pixel leaked a sliver of the untreated
    frame. Same truncation class as the composite_regions bug."""
    almost = _region(4, 4, 0, 4, 0, 4, value=np.float32(0.9999998))
    out = standalone._diff_alpha([(almost, {"key": "tint"})], None, 4, 4)
    assert (out == 255).all(), f"expected fully opaque, got {out.max()}"


def test_diff_alpha_marks_chrome_opaque_even_over_untreated_pixels():
    """Chrome is display-only but must be visible: the run tab defaults every treatment
    to None, so chrome is the ONLY opaque thing in the overlay."""
    chrome = np.zeros((8, 8), dtype=bool)
    chrome[2:4, 2:4] = True
    out = standalone._diff_alpha([(_region(8, 8, 0, 8, 0, 8), {"key": "none"})], chrome, 8, 8)
    assert (out[2:4, 2:4] == 255).all()
    assert out.sum() == 4 * 255, "only the chrome pixels are opaque"


def test_draw_detection_chrome_reports_its_own_footprint():
    """The mask must come from what chrome DREW, not from diffing output against input:
    a diff silently misses chrome drawn in the colour already underneath it."""
    pytest.importorskip("cv2")
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=bool)
    mask[40:80, 60:100] = True
    out, drawn = standalone._draw_detection_chrome(rgb, {"ring": mask})
    assert drawn.shape == (120, 160) and drawn.dtype == bool
    assert drawn.any(), "chrome drew something, so the footprint cannot be empty"
    changed = (out != rgb).any(axis=2)
    assert drawn[changed].all(), "every visibly changed pixel must be inside the footprint"
    assert not drawn[:20].any(), "chrome must not claim regions far from any object"


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
    from lerobot.overlays.overlay_ipc import SharedOverlayBuffer

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
    from lerobot.overlays.overlay_state import Event

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


def test_stream_identity_tracks_segment_presence(tmp_path, monkeypatch):
    """_stream_identity returns the meta segment's inode when present, None when absent —
    the signal the worker uses to tell 'replaced/orphaned' from 'same segment, paused'."""
    monkeypatch.setattr(standalone, "_SHM_DIR", str(tmp_path))
    meta = tmp_path / f"{standalone.SHM_PREFIX}meta"
    assert standalone._stream_identity() is None  # absent -> publisher gone
    meta.write_bytes(b"x")
    assert isinstance(standalone._stream_identity(), int)  # present -> an inode
    meta.unlink()
    assert standalone._stream_identity() is None  # removed again


# --- C-case: standalone re-attaches when the publisher (teleop) restarts -----
def test_try_reattach_swaps_only_on_publisher_restart():
    """teleop stop+start creates a *fresh* obs-stream segment (new inode); the standalone
    must re-attach to it (not stay stuck on the dead one) — but a merely PAUSED stream
    (same segment, same inode) must NOT trigger a swap, even at a high seq. Regression for
    the lifecycle 'C' bug. Identity comes from the reader's OWN fd (`_reader_inode`), so it
    names the segment we're bound to even when a separate path stat would race onto the new
    one (the startup-race / TOCTOU windows the adversarial review flagged)."""
    from lerobot.robots.obs_stream import ObservationStream, ObservationStreamReader

    keys = ["front", "top"]
    feats = dict.fromkeys(keys, (8, 8, 3))
    frame = {k: np.zeros((8, 8, 3), dtype=np.uint8) for k in keys}

    s1 = ObservationStream(feats, {})
    s2 = old = new_reader = None
    try:
        for _ in range(5):
            s1.write_obs(frame)  # advance the segment's seq to 5
        old = ObservationStreamReader()
        held_ino = standalone._reader_inode(old)  # the segment OLD is actually bound to (its fd)
        assert held_ino is not None and held_ino == standalone._stream_identity()
        # paused/live: same segment (same inode) -> no swap, despite the advanced seq
        assert standalone._try_reattach(keys, held_ino) is None

        s1.cleanup()
        s1 = None  # teleop stops; the old reader still maps the now-dead segment (pins its inode)
        s2 = ObservationStream(feats, {})  # teleop restarts -> fresh segment, NEW inode
        s2.write_obs(frame)
        # old reader's fd still names the ORPHAN while the path now names the live segment —
        # the startup-race / TOCTOU shape; held-from-fd != current-path must trigger a swap:
        assert standalone._reader_inode(old) == held_ino
        assert standalone._stream_identity() != held_ino
        swapped = standalone._try_reattach(keys, held_ino)
        assert swapped is not None, "did not re-attach to the restarted publisher (C bug)"
        new_reader, new_ino = swapped
        assert new_ino != held_ino  # bound to the fresh segment's identity, not the orphan's
        assert new_ino == standalone._reader_inode(new_reader)  # new_ino is the new reader's own fd inode
        assert max(new_reader.image_seq(c) for c in keys) == 1  # reads the fresh segment, not the dead one
    finally:
        for r in (old, new_reader):
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


def test_same_model_respawn_ends_active_not_permanently_off(overlay_client, monkeypatch):
    """A same-model respawn (what a resolution change does) shares ONE state machine.
    The regression: _spawn_worker fired START before tearing down the old worker, so
    the old worker's STOP/STOPPED knocked the machine from loading back to inactive,
    the new worker's LOADED was invalid from there and dropped, and the badge read
    'off' forever while the worker served fine. Teardown must complete BEFORE START."""
    import asyncio as _asyncio

    from lerobot.overlays.overlay_state import Event, State

    class _Proc:
        returncode = None
        pid = 1

        def terminate(self):
            self.returncode = 0

        async def wait(self):
            return 0

    async def _fake_exec(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(overlays, "_get_live_reader", lambda: None)
    overlays._machines.clear()
    overlays._live_proc = None
    overlays._live_model = None

    async def _respawn_flow():
        await overlays._spawn_worker("sam3_track", resolution=672)
        overlays._machine("sam3_track").fire(Event.LOADED)  # worker reports active
        # Resolution change: same model, different resolution -> full respawn.
        await overlays._spawn_worker("sam3_track", resolution=504)
        overlays._machine("sam3_track").fire(Event.LOADED)  # new worker reports active

    _asyncio.run(_respawn_flow())
    assert overlays._machine("sam3_track").state is State.ACTIVE
    assert overlays._live_resolution == 504


def test_unlink_stale_segments_removes_only_overlay_segments(tmp_path):
    """An uncleanly-killed worker leaves its shm segments behind; the fixed-name
    status segment frozen at phase "active" makes the NEXT spawn report loaded
    instantly (badge "active", zero overlays — the "SAM3 failed to load" report).
    The sweep removes every lerobot_overlay_* segment and nothing else."""
    from lerobot.overlays.overlay_ipc import unlink_stale_segments

    for name in ("lerobot_overlay_status", "lerobot_overlay_meta", "lerobot_overlay_img_cam"):
        (tmp_path / name).write_bytes(b"stale")
    (tmp_path / "lerobot_obs_meta").write_bytes(b"other-subsystem")
    assert unlink_stale_segments(root=str(tmp_path)) == 3
    assert [p.name for p in tmp_path.iterdir()] == ["lerobot_obs_meta"]
    assert unlink_stale_segments(root=str(tmp_path)) == 0  # idempotent


def test_spawn_sweeps_stale_segments_before_starting(overlay_client, monkeypatch, tmp_path):
    """_spawn_worker must sweep stale overlay segments AFTER teardown and BEFORE the
    subprocess starts, so every segment that exists post-spawn belongs to the new
    worker — _observe can then trust phase='active' unconditionally."""
    import asyncio as _asyncio

    from lerobot.overlays import overlay_ipc

    order = []
    monkeypatch.setattr(
        overlay_ipc, "unlink_stale_segments", lambda root="/dev/shm": order.append("sweep") or 0
    )

    class _Proc:
        returncode = None
        pid = 1

    async def _fake_exec(*args, **kwargs):
        order.append("spawn")
        return _Proc()

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", _fake_exec)
    overlays._machines.clear()
    overlays._live_proc = None
    overlays._live_model = None
    _asyncio.run(overlays._spawn_worker("sam3_track", resolution=672))
    assert order == ["sweep", "spawn"]


def _spawn_argv(monkeypatch, **kwargs) -> list[str]:
    """Run _spawn_worker with the subprocess stubbed out and return the argv it built."""
    import asyncio as _asyncio

    from lerobot.overlays import overlay_ipc

    monkeypatch.setattr(overlay_ipc, "unlink_stale_segments", lambda root="/dev/shm": 0)
    captured: list[str] = []

    class _Proc:
        returncode = None
        pid = 1

    async def _fake_exec(*args, **kwargs):
        captured.extend(args)
        return _Proc()

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", _fake_exec)
    overlays._machines.clear()
    overlays._live_proc = None
    overlays._live_model = None
    _asyncio.run(overlays._spawn_worker("sam3_track", **kwargs))
    return captured


@pytest.mark.parametrize("multi,expected", [(True, "--multi-instance=1"), (False, "--multi-instance=0")])
def test_spawn_seeds_the_instance_policy_on_the_command_line(overlay_client, monkeypatch, multi, expected):
    """The instance policy must be SEEDED at spawn, not left to the control channel: a
    control push is a no-op until the worker's buffer exists. The data tab got away with
    it because its config is re-pushed on every status poll; the run tab has no re-push,
    so an unseeded value would never arrive and the worker would silently disagree with
    the panel about whether both arms are segmented."""
    assert expected in _spawn_argv(monkeypatch, multi_instance=multi)


def test_spawn_omits_the_instance_flag_when_unset(overlay_client, monkeypatch):
    """None means "say nothing" so the adapter default stands — callers that don't care
    must not silently force a policy."""
    assert not [a for a in _spawn_argv(monkeypatch) if str(a).startswith("--multi-instance")]


def test_live_start_request_carries_the_instance_policy():
    """Parity gap this closes: the data tab had multi_instance and the run tab did not, so
    the same objects at the same resolution segmented both arms on one tab and one arm on
    the other, with nothing in the UI to explain it."""
    assert "multi_instance" in overlays.LiveStartRequest.model_fields
    assert overlays.LiveStartRequest(model="sam3_track").multi_instance is False
    assert overlays.ConfigureRequest(dataset_id="d", model="sam3_track").multi_instance is True


def _mk_proc(rc):
    class _P:
        returncode = rc
        pid = 1

    return _P()


def test_observe_maps_every_exit_to_an_event(overlay_client, monkeypatch):
    """State-machine audit: a worker exit must NEVER be silently ignored. rc!=0 fires
    CRASH (badge 'error', model kept for restart-from-error); an UNCOMMANDED clean/
    SIGTERM exit resets to inactive and clears ownership — previously both a crashed
    worker exiting 0 and any self-exit left the badge frozen on a dead process."""
    from lerobot.overlays.overlay_state import Event, State

    monkeypatch.setattr(overlays, "_read_status", lambda: {})
    # Abnormal death -> ERROR, model kept.
    overlays._machines.clear()
    overlays._live_model = "sam3_track"
    overlays._live_proc = _mk_proc(1)
    overlays._machine("sam3_track").fire(Event.START)
    overlays._machine("sam3_track").fire(Event.LOADED)
    overlays._observe()
    assert overlays._machine("sam3_track").state is State.ERROR
    assert overlays._live_model == "sam3_track" and overlays._live_proc is None

    # Uncommanded clean exit -> INACTIVE, ownership cleared.
    overlays._machines.clear()
    overlays._live_model = "sam3_track"
    overlays._live_proc = _mk_proc(0)
    overlays._machine("sam3_track").fire(Event.START)
    overlays._machine("sam3_track").fire(Event.LOADED)
    overlays._observe()
    assert overlays._machine("sam3_track").state is State.INACTIVE
    assert overlays._live_model is None and overlays._live_proc is None


def test_observe_logs_desync_instead_of_dropping_it(overlay_client, monkeypatch, caplog):
    """A live worker reporting 'active' while the machine says inactive is a broken
    invariant (this exact silence hid the stale-segment and respawn-order bugs).
    It must be logged, not swallowed; the machine must not move."""
    import logging as _logging

    from lerobot.overlays.overlay_state import State

    overlays._machines.clear()
    overlays._live_model = "sam3_track"
    overlays._live_proc = _mk_proc(None)
    monkeypatch.setattr(overlays, "_read_status", lambda: {"phase": "active"})
    with caplog.at_level(_logging.WARNING, logger="lerobot.gui.api.overlays"):
        overlays._observe()
    assert overlays._machine("sam3_track").state is State.INACTIVE
    assert any("desync" in r.message for r in caplog.records)


def test_spawn_missing_sidecar_is_400_and_error_state(overlay_client, monkeypatch):
    """sam3_1 without its sidecar venv must fail the spawn with the setup recipe (400)
    and land the machine in `error` — the previous code fired a nonexistent
    Event.ERROR, which raised AttributeError (a 500 with no recipe) instead."""
    import asyncio as _asyncio

    from fastapi import HTTPException

    from lerobot.overlays.overlay_state import State

    monkeypatch.setenv("LEROBOT_SAM31_PYTHON", "/nonexistent/python")
    overlays._machines.clear()
    overlays._live_proc = None
    overlays._live_model = None
    with pytest.raises(HTTPException) as ei:
        _asyncio.run(overlays._spawn_worker("sam3_1"))
    assert ei.value.status_code == 400 and "sidecar" in ei.value.detail
    assert overlays._machine("sam3_1").state is State.ERROR


def test_spawn_exec_failure_is_500_and_error_state(overlay_client, monkeypatch):
    """If the worker subprocess cannot exec at all, the machine must not sit in
    `loading` forever with no process behind it."""
    import asyncio as _asyncio

    from fastapi import HTTPException

    from lerobot.overlays.overlay_state import State

    async def _boom(*a, **k):
        raise FileNotFoundError("no such interpreter")

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", _boom)
    overlays._machines.clear()
    overlays._live_proc = None
    overlays._live_model = None
    with pytest.raises(HTTPException) as ei:
        _asyncio.run(overlays._spawn_worker("sam3_track"))
    assert ei.value.status_code == 500
    assert overlays._machine("sam3_track").state is State.ERROR
