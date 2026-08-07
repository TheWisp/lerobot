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

import asyncio
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


def test_concept_color_survives_removing_an_earlier_object():
    """The colour was the concept's index in the list handed to the drawing pass — which is
    the set currently VISIBLE. So deleting one object's row recoloured every object after it,
    and an object merely dropping out of view for a frame recoloured its neighbours. An
    object's colour must depend on nothing but the object."""
    adapters._COLOR_BY_CONCEPT.clear()
    first, second, third = (adapters._concept_color(n) for n in ("a", "b", "c"))
    assert len({first, second, third}) == 3, "distinct objects get distinct colours"

    # 'a' is deleted and 'b' is briefly lost; neither may disturb anyone else.
    assert adapters._concept_color("c") == third
    assert adapters._concept_color("b") == second
    assert adapters._concept_color("d") not in {first, second, third}, "new objects take free slots"


def test_concept_color_falls_back_to_a_hash_past_the_palette():
    adapters._COLOR_BY_CONCEPT.clear()
    for i in range(len(adapters._CONCEPT_PALETTE)):
        adapters._concept_color(f"c{i}")
    assert adapters._concept_color("overflow") == adapters._color_for("overflow")


def test_deleting_a_row_returns_its_colour_to_the_palette():
    """Assign-and-never-free is stable but exhausts an 8-entry palette within one session of
    adding and removing objects, after which every new object falls back to a hash and they
    stop being distinguishable. Deleting a row frees; losing an object for a few frames
    must not, which is the whole reason the map exists."""
    adapters._COLOR_BY_CONCEPT.clear()
    a = object.__new__(adapters.Sam3TrackByDetectionAdapter)
    a._init_click_state()  # this branch reads click state in set_control
    a.prompt = "ring . dowel"
    a._signs = {}
    a._seed_multi = False
    a._tracks = {}
    a._restart_tracking = lambda: None

    ring, dowel = adapters._concept_color("ring"), adapters._concept_color("dowel")
    assert ring != dowel

    a.set_control({"objects": [{"name": "dowel", "sign": "+"}]})  # the 'ring' row is deleted
    assert "ring" not in adapters._COLOR_BY_CONCEPT, "a deleted row frees its palette entry"
    assert adapters._concept_color("dowel") == dowel, "the survivor is untouched"
    assert adapters._concept_color("fresh") == ring, "the freed entry is reusable"


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


def test_data_configure_respawns_on_shape_change_not_dataset_change(overlay_client, monkeypatch):
    """A dataset switch used to tear the worker down unconditionally — throwing away the
    ~6 s SAM3 load even for a dataset and its own __preview, whose camera shapes are equal
    by construction (observed live as "switching dataset reloads the model"). The teardown
    is keyed on the stream SHAPE now: same {cam: (h, w)} map reuses the live worker (it
    re-attaches to the replaced segments); a different shape still respawns, because the
    worker's buffers are sized by it."""
    from types import SimpleNamespace

    calls = {"teardown": 0, "spawn": 0}

    async def fake_teardown():
        calls["teardown"] += 1

    async def fake_spawn(model, **kw):
        calls["spawn"] += 1

    monkeypatch.setattr(overlays, "SLOT", type(overlays.SLOT)())  # isolate the aux-GPU slot singleton
    monkeypatch.setattr(overlays, "_teardown_current", fake_teardown)
    monkeypatch.setattr(overlays, "_spawn_worker", fake_spawn)
    monkeypatch.setattr(overlays, "start_data_publisher", lambda *a, **k: True)
    monkeypatch.setattr(overlays, "_write_data_control", lambda: None)
    monkeypatch.setattr(overlays, "_live_model", "sam3_track")
    monkeypatch.setattr(overlays, "_live_resolution", None)
    monkeypatch.setattr(overlays, "_live_proc", SimpleNamespace(returncode=None, pid=1))
    # _app_state is installed by server startup, which this client fixture skips.
    monkeypatch.setattr(
        overlays, "_app_state", SimpleNamespace(datasets={"A": object(), "B": object(), "C": object()})
    )

    dims = {"cur": {"top": (480, 640), "wrist": (480, 640)}}
    monkeypatch.setattr(overlays, "_dataset_camera_dims", lambda ds: dict(dims["cur"]))
    monkeypatch.setattr(overlays, "_data_worker_dims", dict(dims["cur"]))

    def configure(ds_id):
        r = overlay_client.post(
            "/api/overlays/data/configure",
            json={"dataset_id": ds_id, "model": "sam3_track"},
            headers={"X-Overlay-Session": "shape-test"},
        )
        assert r.status_code == 200, r.text
        return r

    configure("A")
    configure("B")  # different dataset, SAME shape — the __preview case
    assert calls["teardown"] == 0, "same-shape switch must keep the worker (and its model)"
    assert calls["spawn"] == 2  # spawn is a control push for a live same-model worker

    dims["cur"] = {"cam": (240, 320)}  # different names AND dims
    configure("C")
    assert calls["teardown"] == 1, "a shape change must still respawn"
    assert overlays._data_worker_dims == dims["cur"], "the tracked shape follows the worker"


def test_data_cancel_parks_the_worker_for_the_next_configure(overlay_client, monkeypatch):
    """Cancel fires on every switch to a dataset with no overlay config, so killing the worker
    there made a plain dataset bounce cost a full SAM3 reload (observed live as "switching
    dataset unloads and reloads the model" — the shape-keyed configure fix never engaged
    because the teardown came from the STOP path). Cancel now releases the slot and stops the
    publisher but PARKS the worker; the next same-shape configure reuses it. /data/free stays
    the explicit kill."""
    from types import SimpleNamespace

    calls = {"teardown": 0, "spawn": 0, "stop_pub": 0}

    async def fake_teardown():
        calls["teardown"] += 1

    async def fake_spawn(model, **kw):
        calls["spawn"] += 1

    monkeypatch.setattr(overlays, "SLOT", type(overlays.SLOT)())  # isolate the aux-GPU slot singleton
    monkeypatch.setattr(overlays, "_teardown_current", fake_teardown)
    monkeypatch.setattr(overlays, "_spawn_worker", fake_spawn)
    monkeypatch.setattr(overlays, "start_data_publisher", lambda *a, **k: True)
    monkeypatch.setattr(
        overlays, "stop_data_publisher", lambda: calls.__setitem__("stop_pub", calls["stop_pub"] + 1)
    )
    monkeypatch.setattr(overlays, "_write_data_control", lambda: None)
    monkeypatch.setattr(overlays, "_live_model", "sam3_track")
    monkeypatch.setattr(overlays, "_live_resolution", None)
    monkeypatch.setattr(overlays, "_live_proc", SimpleNamespace(returncode=None, pid=1))
    monkeypatch.setattr(overlays, "_app_state", SimpleNamespace(datasets={"A": object(), "B": object()}))
    shape = {"top": (480, 640), "wrist": (480, 640)}
    monkeypatch.setattr(overlays, "_dataset_camera_dims", lambda ds: dict(shape))
    monkeypatch.setattr(overlays, "_data_worker_dims", dict(shape))

    r = overlay_client.post("/api/overlays/data/cancel", headers={"X-Overlay-Session": "park-test"})
    assert r.status_code == 200, r.text
    assert r.json()["parked"] is True, "an alive worker must be reported parked, not torn down"
    assert calls["stop_pub"] == 1, "cancel must still stop the obs-stream publisher"
    assert calls["teardown"] == 0, "cancel must NOT tear the worker down"

    r = overlay_client.post(
        "/api/overlays/data/configure",
        json={"dataset_id": "B", "model": "sam3_track"},
        headers={"X-Overlay-Session": "park-test"},
    )
    assert r.status_code == 200, r.text
    assert calls["teardown"] == 0, "same-shape configure after a park must reuse the warm worker"
    assert calls["spawn"] == 1  # control push on the live worker, not a process start


def test_live_start_evicts_a_data_parked_worker(overlay_client, monkeypatch):
    """A parked worker is bound to a DATASET's stream shape, and the worker refuses to re-attach
    across a shape change by design — same-model reuse inside _spawn_worker would hand the run
    overlay a worker that waits forever on teleop's differently-shaped stream. live_start must
    evict it (a run-tab worker, _data_worker_dims is None, is still reused as before)."""
    from types import SimpleNamespace

    calls = {"teardown": 0, "spawn": 0}

    async def fake_teardown():
        calls["teardown"] += 1

    async def fake_spawn(model, **kw):
        calls["spawn"] += 1

    monkeypatch.setattr(overlays, "SLOT", type(overlays.SLOT)())  # isolate the aux-GPU slot singleton
    monkeypatch.setattr(overlays, "_teardown_current", fake_teardown)
    monkeypatch.setattr(overlays, "_spawn_worker", fake_spawn)
    monkeypatch.setattr(overlays, "_live_model", "sam3_track")
    monkeypatch.setattr(overlays, "_live_resolution", None)
    monkeypatch.setattr(overlays, "_live_proc", SimpleNamespace(returncode=None, pid=1))
    monkeypatch.setattr(overlays, "_data_worker_dims", {"top": (480, 640)})
    r = overlay_client.post("/api/overlays/live/start", json={"model": "sam3_track"})
    assert r.status_code == 200, r.text
    assert calls["teardown"] == 1, "a data-shaped worker must be evicted before the run overlay"

    monkeypatch.setattr(overlays, "_data_worker_dims", None)
    r = overlay_client.post("/api/overlays/live/start", json={"model": "sam3_track"})
    assert r.status_code == 200, r.text
    assert calls["teardown"] == 1, "a run-tab worker (no data shape) is reused, not evicted"


def test_a_dropped_frame_during_playback_is_not_a_new_stream(monkeypatch):
    """Continuity was `pos == last + 1` exactly, so ONE dropped frame counted as a new
    stream: it bumped generation, which resets the worker's tracker and re-runs the detector
    for every concept (~30 ms each). Playback advances on a timer while inference runs
    slower, so skips are routine and the overlay was re-seeding constantly. Backwards, a
    different episode, and a long jump must still reset."""
    writes: list[dict] = []
def test_data_control_write_carries_the_latest_click_op(monkeypatch):
    """The control block is a single latched slot and the data tab re-writes it wholesale on
    every status poll, so a click/box op POSTed to /live/control was erased within ~1 s —
    data-tab gestures would have worked only by luck. The latest op must ride along on those
    writes (safe to repeat: the worker applies each op once, gated on click_seq)."""
    written = []

    class _Reader:
        def write_control(self, block):
            written.append(block)

    monkeypatch.setattr(overlays, "_get_live_reader", lambda: _Reader())
    monkeypatch.setattr(overlays, "_data_pub_config", {"objects": []})
    monkeypatch.setattr(overlays, "_data_pub_generation", 7)
    monkeypatch.setattr(overlays, "_last_click_op", {})

    overlays._write_data_control()
    assert "clicks" not in written[-1], "no op yet — nothing to carry"

    op = {"clicks": {"top": [[10, 20, 1]]}, "click_name": {"top": "object_1"}, "click_seq": 5}
    monkeypatch.setattr(overlays, "_last_click_op", dict(op))
    overlays._write_data_control()
    carried = written[-1]
    assert carried["clicks"] == op["clicks"], "the click must survive the config re-push"
    assert carried["click_seq"] == 5, "the seq must ride along or the worker re-applies it"
    assert carried["generation"] == 7 and carried["config"] == {"objects": []}, "config still written"


def test_live_control_remembers_only_click_ops(overlay_client, monkeypatch):
    """/live/control is the one click transport for BOTH tabs. It must remember click/box ops
    (so _write_data_control can carry them) and must not mistake an ordinary control push —
    a prompt or camera change — for one, or a stale op would be replayed forever."""

    class _Reader:
        def write_control(self, block):
            pass

    monkeypatch.setattr(overlays, "_get_live_reader", lambda: _Reader())
    monkeypatch.setattr(overlays, "_last_click_op", {})

    r = overlay_client.post("/api/overlays/live/control", json={"prompt": "green ring"})
    assert r.status_code == 200, r.text
    assert overlays._last_click_op == {}, "a plain control push is not a click op"

    op = {"boxes": {"top": [[1, 2, 30, 40]]}, "click_name": {"top": "object_2"}}
    r = overlay_client.post("/api/overlays/live/control", json={**op, "cameras": ["top"]})
    assert r.status_code == 200, r.text
    remembered = dict(overlays._last_click_op)
    assert remembered.pop("click_seq", None) is not None, "the server stamps the sequence"
    assert remembered == op, "the box op must be remembered, without the camera field"


def test_live_control_carries_the_remembered_click_op(overlay_client, monkeypatch):
    """The run tab pushes an op once, then keeps sending ordinary control updates. Because the
    control block is a single latched slot that each write replaces wholesale, an op that the
    worker had not yet sampled was erased by the next prompt/treatment push. The server rides
    the remembered op along on every write — the client must not, so that exactly one place
    owns how long an op lives (see test_click_op_does_not_outlive_its_worker)."""
    written = []

    class _Reader:
        def write_control(self, block):
            written.append(block)

    monkeypatch.setattr(overlays, "_get_live_reader", lambda: _Reader())
    monkeypatch.setattr(overlays, "_last_click_op", {})

    op = {"clicks": {"top": [[10, 20, 1]]}, "click_name": {"top": "object_1"}}
    assert overlay_client.post("/api/overlays/live/control", json=op).status_code == 200
    assert written[-1]["clicks"] == op["clicks"]
    first_seq = written[-1]["click_seq"]

    # An ordinary push afterwards must not erase it.
    assert overlay_client.post("/api/overlays/live/control", json={"prompt": "ring"}).status_code == 200
    assert written[-1]["prompt"] == "ring", "the actual update must still be written"
    assert written[-1]["click_seq"] == first_seq, "the unsampled op must survive the next write"

    # A newer op in the body wins over the remembered one rather than being merged under it.
    newer = {"boxes": {"top": [[1, 2, 3, 4]]}}
    assert overlay_client.post("/api/overlays/live/control", json=newer).status_code == 200
    assert written[-1]["boxes"] == newer["boxes"]
    assert written[-1]["click_seq"] > first_seq, "each op must advance the sequence"
    assert "clicks" not in written[-1], "a newer op replaces the old one, not merges under it"


def test_click_op_does_not_outlive_its_worker(monkeypatch):
    """A remembered op is addressed to ONE worker process. The replacement starts at click_seq
    0, so anything still remembered would pass its idempotency gate and be applied again — a
    click made on the previous dataset, at that frame's pixel coordinates, seeding a tracker on
    whatever now lies under them. Teardown must forget it, exactly as it forgets the stream shape."""
    monkeypatch.setattr(overlays, "_last_click_op", {"clicks": {"top": [[10, 20, 1]]}, "click_seq": 9})
    monkeypatch.setattr(overlays, "_data_worker_dims", {"top": (480, 640)})
    monkeypatch.setattr(overlays, "_live_model", None)  # nothing running: the early-return path

    asyncio.run(overlays._teardown_current())

    assert overlays._last_click_op == {}, "a click op must not survive the worker it was aimed at"
    assert overlays._data_worker_dims is None


def test_live_start_forwards_the_box_method(overlay_client, monkeypatch):
    """box_method is a LOAD-TIME argv seed for the worker (like multi_instance): the control
    re-push only reaches a process that already exists, so a value dropped here means the run
    tab silently runs the default box API until some later edit happens to re-push."""
    seen = {}

    async def fake_spawn(model, **kw):
        seen.update(kw)

    monkeypatch.setattr(overlays, "SLOT", type(overlays.SLOT)())  # isolate the aux-GPU slot singleton
    monkeypatch.setattr(overlays, "_spawn_worker", fake_spawn)
    monkeypatch.setattr(overlays, "_data_worker_dims", None)

    r = overlay_client.post(
        "/api/overlays/live/start", json={"model": "sam3_track", "box_method": "exemplar"}
    )
    assert r.status_code == 200, r.text
    assert seen.get("box_method") == "exemplar", "the run tab's box API choice must reach the worker"
    assert seen.get("text_detection") is True, "the sibling knob must keep travelling too"


def test_worker_reuse_keeps_an_unread_click_op(monkeypatch):
    """_spawn_worker's reuse path writes the control block directly rather than through
    live_control, so it used to erase a click/box op the worker had not sampled yet — the
    data tab hits this on every config push, which is how a row could vanish from the panel
    while its detection stayed in the scene."""
    import asyncio
    from types import SimpleNamespace

    written = []

    class _Reader:
        def write_control(self, block):
            written.append(block)

    monkeypatch.setattr(overlays, "_get_live_reader", lambda: _Reader())
    monkeypatch.setattr(overlays, "_live_model", "sam3_track")
    monkeypatch.setattr(overlays, "_live_resolution", None)
    monkeypatch.setattr(overlays, "_live_proc", SimpleNamespace(returncode=None, pid=1))
    op = {"clicks_remove": {"front": ["object_2"]}, "click_seq": 11}
    monkeypatch.setattr(overlays, "_last_click_op", dict(op))

    asyncio.run(overlays._spawn_worker("sam3_track", objects=[], resolution=None))

    assert written, "the reuse path must push control rather than respawn"
    assert written[-1]["clicks_remove"] == op["clicks_remove"], "an unread op must survive"
    assert written[-1]["click_seq"] == 11
    assert "config" in written[-1], "the config it was called for must still be written"


def test_two_clients_gestures_both_survive(overlay_client, monkeypatch):
    """The bug that cost a whole session. click_seq used to be Date.now()-derived per client,
    and the worker ignores any id it has already passed — so with two tabs (same millisecond)
    or two laptops (clock skew), one client's gestures were dropped in silence. Removals from
    the losing client never landed, so rows vanished from the panel while their detections
    stayed in the scene. The server assigns the id now, so ordering cannot depend on a clock
    it does not own."""
    written = []

    class _Reader:
        def write_control(self, block):
            written.append(block)

    monkeypatch.setattr(overlays, "_get_live_reader", lambda: _Reader())
    monkeypatch.setattr(overlays, "_last_click_op", {})
    monkeypatch.setattr(overlays, "_click_seq", 0)

    # Client A clicks; client B (a second tab, an hour-skewed laptop) removes a row. Both
    # send a bare op — neither proposes an id, and a proposed one must not be honoured.
    overlay_client.post("/api/overlays/live/control", json={"clicks": {"front": [[10, 20, 1]]}})
    seq_a = written[-1]["click_seq"]
    overlay_client.post(
        "/api/overlays/live/control",
        json={"clicks_remove": {"front": ["object_1"]}, "click_seq": 1},  # stale id, ignored
    )
    seq_b = written[-1]["click_seq"]

    assert written[-1]["clicks_remove"] == {"front": ["object_1"]}
    assert seq_b > seq_a, "the second client's op must advance the sequence, not lose to it"


def test_a_dropped_gesture_is_logged_but_a_replay_is_not(caplog):
    """The gate that drops a stale op returns in silence, which is exactly why the lost
    removals took a session to notice: a dropped gesture and a gesture that did nothing look
    identical. A replay of the applied op is normal — it rides along on every control write —
    so only a DIFFERENT op that cannot advance the counter is worth a warning."""
    a = object.__new__(adapters.Sam3TrackByDetectionAdapter)
    a._init_click_state()
    a.prompt = "green ring"

    a.set_control({"clicks": {"front": [[1, 2, 1]]}, "click_seq": 5})
    assert a._last_click_seq == 5

    caplog.clear()
    with caplog.at_level("WARNING"):
        a.set_control({"clicks": {"front": [[1, 2, 1]]}, "click_seq": 5})  # the ride-along
    assert not caplog.records, "replaying the applied op is expected and must stay quiet"

    with caplog.at_level("WARNING"):
        a.set_control({"clicks_remove": {"front": ["object_1"]}, "click_seq": 5})
    assert any("DROPPED" in r.message for r in caplog.records), (
        "a different op that cannot advance the counter is a lost gesture — say so"
    )


def test_clicked_and_typed_objects_share_one_cap():
    """The tracker carries text concepts and clicked ones in a single session, so capping the
    two lists separately allowed twice MAX_OBJECTS masklets — and the panel then offered rows
    the worker silently refused."""
    a = object.__new__(adapters.Sam3TrackByDetectionAdapter)
    a._init_click_state()
    a.prompt = " . ".join(f"thing{i}" for i in range(adapters.ConceptMaskAdapter.MAX_OBJECTS - 1))
    a._cam = "front"
    track = {"masks": {}, "scores": {}, "objs": {}, "session": None, "since_flush": 0}
    seeded = []
    a._seed = lambda tr, seeds, pv, h, w: seeded.append(sorted(seeds))

    mask = np.ones((4, 4), bool)
    assert a._admit_clicked_mask(track, mask, pv=None, h=4, w=4) is True, "one slot is left"
    assert a._admit_clicked_mask(track, mask, pv=None, h=4, w=4) is False, (
        "the cap counts typed concepts too, not just clicked ones"
    )
    assert len(a._click_names["front"]) == 1


def test_force_republishes_the_same_frame_without_resetting_tracking(monkeypatch):
    """A gesture on a paused episode has to hand the worker a frame, because the worker only
    reads the control block when it processes one. But the re-publish must NOT bump
    generation: that resets the tracker, which would destroy the very clicked objects the
    gesture is modifying. Force means 'publish again', not 'new stream'."""
    writes: list[dict] = []
    controls: list[int] = []

    class _Stream:
        def write_obs(self, obs):
            writes.append(obs)

    monkeypatch.setattr(overlays, "_data_pub", _Stream())
    monkeypatch.setattr(overlays, "_data_pub_dataset", "ds")
    monkeypatch.setattr(overlays, "_data_pub_cameras", [])
    monkeypatch.setattr(overlays, "_data_pub_last_pos", None)
    monkeypatch.setattr(overlays, "_data_pub_generation", 0)
    monkeypatch.setattr(overlays, "_write_data_control", lambda: None)

    overlays.publish_data_frame("ds", 0, 400, {})  # first frame is a new stream
    start = overlays._data_pub_generation

    for fr in (401, 402, 404, 405, 412):  # smooth, then dropped frames of increasing size
        overlays.publish_data_frame("ds", 0, fr, {})
    assert overlays._data_pub_generation == start, "a dropped frame is still the same video"

    overlays.publish_data_frame("ds", 0, 460, {})
    assert overlays._data_pub_generation == start + 1, "a long jump is a scrub"

    gen = overlays._data_pub_generation
    overlays.publish_data_frame("ds", 0, 459, {})
    assert overlays._data_pub_generation == gen + 1, "backwards is never continuous"

    gen = overlays._data_pub_generation
    overlays.publish_data_frame("ds", 1, 460, {})
    assert overlays._data_pub_generation == gen + 1, "a different episode is a new stream"


def test_losing_an_object_is_logged(caplog):
    """An object leaves the drawn set the moment its score falls under LOST_THRESH, and that
    happened with no trace in any log — so a run showing no seed[] and no flush[] lines
    proved nothing, since silent loss leaves no line either. It cost several wrong
    diagnoses."""
    a = object.__new__(adapters.Sam3TrackByDetectionAdapter)
    a._cam = "top"
    a._concepts = ["ring", "dowel"]
    mask = np.ones((4, 4), bool)
    track = {"masks": {"ring": mask, "dowel": mask}, "scores": {"ring": 0.9, "dowel": 0.9}}

    a._live_masks(track)  # first pass establishes the held set; nothing to report yet
    caplog.clear()
    with caplog.at_level("INFO"):
        a._live_masks(track)
    assert not caplog.records, "an unchanged held set must stay quiet"

    track["scores"]["dowel"] = 0.0  # the tracker loses confidence
    with caplog.at_level("INFO"):
        a._live_masks(track)
    assert any("lost" in r.message and "dowel" in str(r.args) for r in caplog.records), caplog.text

    track["scores"]["dowel"] = 0.9  # and finds it again
    caplog.clear()
    with caplog.at_level("INFO"):
        a._live_masks(track)
    assert any("recovered" in r.message for r in caplog.records), caplog.text


def test_each_gui_process_gets_its_own_worker_log(monkeypatch):
    """A fixed /tmp name meant a second GUI server — another port, a test instance —
    truncated the first's worker log the moment it spawned a worker, destroying the evidence
    of a session still in progress. A live bug report was diagnosed against a log that had
    already been overwritten, and the root cause that came out of it was wrong."""
    import inspect

    # A static check: the path is computed inline inside the spawn function, which starts a
    # real subprocess, so there is nothing to call. Pairing it with the log endpoint keeps
    # the two in step — the endpoint reads the same variable, so it follows automatically.
    src = inspect.getsource(overlays._spawn_worker)
    assert "getpid" in src, "the worker log path must be scoped to this GUI process"
    assert '"lerobot_overlays.log"' not in src, "a fixed name is shared between servers"
def test_a_box_never_borrows_text_from_other_object_rows():
    """The two box modes differ in which model reads the box and in nothing else. An earlier
    version passed the typed concepts to the detector alongside the box because it scored
    better in clutter — but those words belong to other rows, so the same gesture meant
    different things depending on state it had nothing to do with, and boxing a dowel while
    'green ring' sat in another row biased the result toward the ring."""
    from unittest.mock import MagicMock

    a = object.__new__(adapters.Sam3TrackByDetectionAdapter)
    a._init_click_state()
    a.prompt = "green ring . robot arm"  # other rows the user typed
    a._cam = "front"
    a._torch = torch
    a._Image = Image
    a.device = "cpu"
    a._det_threshold = 0.5
    a._proc_size = {"height": 672, "width": 672}

    seen_text = []

    def _proc(*args, **kw):
        if "text" in kw or (args and isinstance(args[0], str)):
            seen_text.append(kw.get("text", args[0] if args else None))
        out = MagicMock()
        out.to.return_value = {"pixel_values": MagicMock(), "input_ids": MagicMock(), "attention_mask": None}
        return out

    a.det_proc = MagicMock(side_effect=_proc)
    a.det_proc.post_process_instance_segmentation.return_value = [{"masks": []}]
    a.det = MagicMock(return_value=MagicMock())
    a.det.parameters.return_value = iter([torch.zeros(1)])

    a._mask_from_exemplar(np.zeros((8, 8, 3), np.uint8), (1, 1, 5, 5), 8, 8)

    assert seen_text, "the detector is still prompted, just not with another row's words"
    assert all(not t for t in seen_text), f"a box must carry no text prompt, got {seen_text!r}"


def test_deleting_an_object_returns_its_colour_to_the_palette():
    """Found by driving the GUI, not by reading it: colours are assigned and never
    reassigned so they stay stable, but the palette is 8 long, and one session of clicking
    and deleting exhausted it — after which every new object fell back to a hash and they
    stopped being reliably distinguishable. Deleting frees; merely losing an object for a
    few frames must not, which is the whole reason the registry exists."""
    a = object.__new__(adapters.Sam3TrackByDetectionAdapter)
    a._init_click_state()
    a.prompt = ""
    a._text_detection = False
    a._tracks = {}
    adapters._COLOR_BY_CONCEPT.clear()

    a._click_names = {"top": ["a", "b"], "front": ["b"]}
    colour_a, colour_b = adapters._concept_color("a"), adapters._concept_color("b")
    assert colour_a != colour_b

    # 'b' is still designated on 'front', so removing it from 'top' must NOT free its colour.
    a.set_control({"clicks_remove": {"top": ["a", "b"]}, "click_seq": 1})
    assert "b" in adapters._COLOR_BY_CONCEPT, "a name another camera still uses keeps its colour"
    assert "a" not in adapters._COLOR_BY_CONCEPT, "the deleted object frees its palette entry"
    assert adapters._concept_color("fresh") == colour_a, "the freed entry is reusable"

    # Clearing a camera deletes everything on it.
    a.set_control({"clicks": {"front": []}, "click_seq": 2})
    assert "b" not in adapters._COLOR_BY_CONCEPT, "clearing a camera frees its objects' colours"
