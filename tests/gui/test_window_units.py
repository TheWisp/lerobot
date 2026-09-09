"""The window builder's helpers, driven directly.

These are the pieces the end-to-end tests exercise only through a real dataset,
a real encoder and a browser: the frame boundaries handed to the page's
decoder, the codec string its decoder is configured with, the ffmpeg arguments
a rung turns into, the envelope the bundle carries, the cache key, and how a
camera's mask column is resolved. A defect in any of them shows up as a
decoder error or a wrong picture, three layers away from its cause.

Every stream here is built byte by byte, so a test says what the format is
rather than what one encoder happened to emit.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from lerobot.gui.api import window_playback as wp

# ── H.264 Annex B, split at access-unit delimiters ──────────────────────────

AUD = b"\x00\x00\x00\x01\x09\x10"  # 4-byte start code, NAL type 9
SPS = b"\x00\x00\x00\x01\x67\x64\x00\x1f"  # NAL type 7, profile 0x64 level 0x1f
SLICE = b"\x00\x00\x01\x65"  # 3-byte start code, NAL type 5 (IDR)


def test_access_units_are_split_at_every_delimiter():
    stream = AUD + SPS + SLICE + b"\xaa" * 10 + AUD + SLICE + b"\xbb" * 4
    sizes = wp._split_access_units(stream)
    assert sizes == [len(AUD) + len(SPS) + len(SLICE) + 10, len(AUD) + len(SLICE) + 4]
    assert sum(sizes) == len(stream)


def test_a_three_byte_start_code_is_a_delimiter_too():
    """The encoder emits either form. A splitter that assumes the four-byte one
    starts each frame a byte early: the count still looks right, so the sizes
    are what this checks."""
    aud3 = b"\x00\x00\x01\x09\x10"
    first = aud3 + SLICE + b"\xaa" * 6
    second = aud3 + SLICE + b"\xbb" * 6
    sizes = wp._split_access_units(first + second)
    assert sizes == [len(first), len(second)]
    assert sum(sizes) == len(first + second)


def test_a_delimiter_shaped_byte_inside_a_frame_does_not_split_it():
    """Only a NAL whose type is 9 is a delimiter: a payload byte that happens
    to look like one must not end a frame."""
    stream = AUD + SLICE + b"\x00\x00\x01\x41" + b"\xcc" * 8
    assert wp._split_access_units(stream) == [len(stream)]


def test_a_stream_without_delimiters_is_refused():
    with pytest.raises(RuntimeError, match="no access-unit delimiters"):
        wp._split_access_units(SPS + SLICE + b"\xaa" * 10)


# ── AV1 low-overhead OBU stream, split at temporal delimiters ───────────────


def obu(kind: int, payload: bytes = b"") -> bytes:
    """One OBU with a size field: header byte, then a leb128 size."""
    size = len(payload)
    leb = b""
    while True:
        byte = size & 0x7F
        size >>= 7
        leb += bytes([byte | (0x80 if size else 0)])
        if not size:
            break
    return bytes([(kind << 3) | 0x02]) + leb + payload


TD = obu(2)  # temporal delimiter, empty payload
FRAME = obu(6, b"\x11" * 20)  # a frame OBU


def test_temporal_units_are_split_at_every_delimiter():
    stream = TD + FRAME + TD + obu(6, b"\x22" * 5)
    sizes = wp._split_temporal_units(stream)
    assert sizes == [len(TD) + len(FRAME), len(TD) + len(obu(6, b"\x22" * 5))]
    assert sum(sizes) == len(stream)


def test_a_payload_byte_that_looks_like_a_delimiter_does_not_split():
    """OBUs are walked by their size fields, so the parser never guesses."""
    stream = TD + obu(6, bytes([(2 << 3) | 0x02, 0x00]) * 4)
    assert wp._split_temporal_units(stream) == [len(stream)]


def test_a_multi_byte_size_is_read_whole():
    """A frame over 127 bytes carries a two-byte leb128 size; reading one byte
    of it walks into the middle of the frame and every later split is wrong."""
    big = obu(6, b"\x33" * 300)
    stream = TD + big + TD + FRAME
    assert wp._split_temporal_units(stream) == [len(TD) + len(big), len(TD) + len(FRAME)]


def test_an_obu_without_a_size_field_is_refused():
    with pytest.raises(RuntimeError, match="OBU without a size field"):
        wp._split_temporal_units(bytes([2 << 3]) + b"\x00")


def test_a_stream_without_a_delimiter_is_refused():
    with pytest.raises(RuntimeError, match="no temporal delimiters"):
        wp._split_temporal_units(FRAME)


def test_a_stored_sample_gains_the_delimiter_the_container_dropped():
    """An ISOBMFF sample is a temporal unit with its delimiter removed; the raw
    stream the page decodes needs it back, and only when it is missing."""
    sample = obu(6, b"\x44" * 8)
    with_td = wp._av1_temporal_unit(sample)
    assert with_td == TD + sample
    assert wp._av1_temporal_unit(with_td) == with_td, "a delimiter must not be added twice"


# ── the codec string the page's decoder is configured with ──────────────────


def test_the_codec_string_comes_from_the_sequence_parameter_set():
    assert wp._h264_codec_string(AUD + SPS + SLICE) == "avc1.64001f"


def test_a_stream_with_no_parameter_set_has_no_codec_string():
    assert wp._h264_codec_string(AUD + SLICE + b"\xaa" * 8) is None


# ── the ffmpeg arguments a rung turns into ──────────────────────────────────


def encode_args(rung: str, **enc) -> tuple[list[str], list[str]]:
    options = {"codec": "h264", "rc": "cbr", "q": 26, "preset": "veryfast"}
    options.update(enc)
    return wp._encode_args(rung, options, 15)


def test_a_rung_scales_down_but_never_up():
    """A 960-wide camera at the 1280 rung stays 960 wide: upscaling costs bytes
    and adds nothing, and it made the tab's tiles jump between sizes."""
    vf, _ = encode_args("640")
    assert vf == ["-vf", "scale=w='min(iw,640)':h=-2"]


def test_the_full_rung_encodes_at_the_source_size_and_a_fixed_quality():
    """`full` has no width and no cap: reached here only for composited frames,
    where the archive's own samples cannot be served."""
    vf, video = encode_args("full")
    assert vf == []
    assert "-crf" in video and video[video.index("-crf") + 1] == "18"
    assert "-b:v" not in video and "-maxrate" not in video


def test_constant_bitrate_caps_the_stream_and_constant_quality_caps_the_peak():
    _, cbr = encode_args("320", rc="cbr")
    _, crf = encode_args("320", rc="crf", q=26)
    assert cbr[cbr.index("-b:v") + 1] == "300k" and cbr[cbr.index("-maxrate") + 1] == "300k"
    assert "-b:v" not in crf
    assert crf[crf.index("-crf") + 1] == "26" and crf[crf.index("-maxrate") + 1] == "300k"


def test_h264_windows_start_with_a_keyframe_and_carry_delimiters():
    """Every window is decoded on its own, so the group of pictures is the
    window and there are no B-frames to reorder."""
    _, video = encode_args("320")
    assert video[video.index("-g") + 1] == "15" and video[video.index("-keyint_min") + 1] == "15"
    assert video[video.index("-bf") + 1] == "0"
    assert "h264_metadata=aud=insert" in video


def test_av1_takes_its_cap_through_its_own_parameter_string():
    _, cbr = encode_args("320", codec="av1", rc="cbr", preset="8")
    _, crf = encode_args("320", codec="av1", rc="crf", q=34, preset="8")
    assert "-svtav1-params" in cbr and cbr[cbr.index("-svtav1-params") + 1] == "rc=1"
    assert crf[crf.index("-svtav1-params") + 1] == "mbr=300k"
    assert "-maxrate" not in cbr and "-maxrate" not in crf


def test_an_unknown_rung_is_refused():
    with pytest.raises(AssertionError):
        encode_args("2160")


# ── the bundle's envelope ───────────────────────────────────────────────────


def test_a_short_episode_is_sent_frame_for_frame():
    arr = np.arange(12, dtype=float).reshape(6, 2)
    e = wp._envelope(arr, columns=8)
    assert e == {"columns": 6, "lo": arr.tolist(), "hi": None}


def test_a_long_episode_is_summarised_by_column():
    arr = np.stack([np.arange(1000, dtype=float), -np.arange(1000, dtype=float)], axis=1)
    e = wp._envelope(arr, columns=10)
    assert e["columns"] == 10 and len(e["lo"]) == 10 and len(e["hi"]) == 10
    # Each column spans exactly a hundred frames, and the pair brackets them.
    assert e["lo"][0] == [0.0, -99.0] and e["hi"][0] == [99.0, -0.0]
    assert e["lo"][-1] == [900.0, -999.0] and e["hi"][-1] == [999.0, -900.0]


def test_the_envelope_brackets_every_sample():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(5000, 3))
    e = wp._envelope(arr, columns=64)
    lo, hi = np.array(e["lo"]), np.array(e["hi"])
    assert lo.min(axis=0).tolist() == pytest.approx(arr.min(axis=0).round(4).tolist(), abs=1e-4)
    assert hi.max(axis=0).tolist() == pytest.approx(arr.max(axis=0).round(4).tolist(), abs=1e-4)
    assert (lo <= hi).all()


def test_a_scalar_feature_keeps_its_shape():
    e = wp._envelope(np.arange(10, dtype=float).reshape(10, 1), columns=5)
    assert all(len(v) == 1 for v in e["lo"]) and all(len(v) == 1 for v in e["hi"])


# ── the cache key ───────────────────────────────────────────────────────────

ENC = {"codec": "h264", "rc": "cbr", "q": 26, "preset": "veryfast"}


def key(**over):
    args = {
        "dataset_id": "user/ds",
        "episode_idx": 1,
        "start": 30,
        "seconds": 1.0,
        "rung": "320",
        "enc": ENC,
        "masks": "runs",
        "fingerprint": "",
        "generation": 7,
    }
    args.update(over)
    return wp._cache_key(**args)


def test_every_part_of_the_request_is_in_the_key():
    """Two windows that differ in any request term must be two entries, or one
    answers for the other and the operator sees the wrong pixels."""
    base = key()
    for over in (
        {"dataset_id": "user/other"},
        {"episode_idx": 2},
        {"start": 60},
        {"seconds": 2.0},
        {"rung": "640"},
        {"enc": {**ENC, "codec": "av1", "preset": "8"}},
        {"enc": {**ENC, "rc": "crf"}},
        {"enc": {**ENC, "q": 34}},
        {"masks": "none"},
        {"masks": "composited", "fingerprint": "cam:abc"},
    ):
        assert key(**over) != base, over
    assert key() == base, "the same request must key the same entry"


def test_a_composited_window_is_keyed_by_the_recipe():
    a = key(masks="composited", fingerprint="cam:abc")
    b = key(masks="composited", fingerprint="cam:def")
    assert a != b
    assert key(masks="composited", fingerprint="cam:abc") == a


def test_the_full_rung_ignores_encoder_options_it_does_not_use():
    """`full` is a remux: the encoder settings do not touch its bytes, so they
    must not multiply its cache entries."""
    a = key(rung="full", enc=ENC)
    b = key(rung="full", enc={**ENC, "rc": "crf", "q": 40})
    assert a == b
    assert key(rung="full", masks="composited", fingerprint="x") != a


def test_the_key_carries_the_format_version():
    assert f"v{wp.FORMAT_VERSION}" in key().name


def test_a_written_dataset_is_a_new_key():
    """A trim renumbers frames and a delete renumbers episodes, so a window
    built before an edit must not answer for one asked afterwards."""
    assert key(generation=111) != key(generation=222)
    assert key(generation=111) == key(generation=111)


def test_the_generation_is_the_metadata_mtime(tmp_path):
    """It has to survive a server restart, which an in-process counter would
    not, and change on every path that rewrites the dataset."""
    import types

    meta = tmp_path / "meta"
    meta.mkdir()
    info = meta / "info.json"
    info.write_text("{}")
    ds = types.SimpleNamespace(root=tmp_path)
    before = wp.dataset_generation(ds)
    assert before > 0
    info.write_text('{"total_frames": 1}')
    assert wp.dataset_generation(ds) > before
    assert wp.dataset_generation(types.SimpleNamespace(root=tmp_path / "gone")) == 0


# ── resolving a camera's mask column ────────────────────────────────────────


def fake_dataset(features: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(meta=types.SimpleNamespace(features=features))


MASK = {"dtype": "string", "mask_encoding": "coco_rle", "mask_labels": ["ball"]}
CAMERA = {"dtype": "video", "shape": (240, 480, 3)}


def test_a_mask_column_is_found_by_what_it_is():
    from lerobot.gui.api.datasets import mask_column_of

    ds = fake_dataset({"observation.images.top": CAMERA, "masks.top": MASK})
    assert mask_column_of(ds, "observation.images.top") == "masks.top"


def test_the_old_namespace_resolves_too():
    """A dataset written before the mask namespace moved carries
    `observation.masks.<camera>`; deriving `masks.<camera>` finds nothing on
    it, which is how playback served stored pixels under a composited URL."""
    from lerobot.gui.api.datasets import mask_column_of

    ds = fake_dataset({"observation.images.top": CAMERA, "observation.masks.top": MASK})
    assert mask_column_of(ds, "observation.images.top") == "observation.masks.top"


def test_a_camera_without_masks_resolves_to_nothing():
    from lerobot.gui.api.datasets import mask_column_of

    ds = fake_dataset(
        {"observation.images.top": CAMERA, "observation.images.wrist": CAMERA, "masks.top": MASK}
    )
    assert mask_column_of(ds, "observation.images.wrist") is None


def test_a_column_that_is_not_a_mask_is_not_one():
    """The encoding marks a mask column; a same-named feature added by hand is
    not one."""
    from lerobot.gui.api.datasets import mask_column_of

    ds = fake_dataset({"observation.images.top": CAMERA, "masks.top": {"dtype": "string"}})
    assert mask_column_of(ds, "observation.images.top") is None


def test_numeric_features_exclude_pixels_strings_and_masks():
    ds = fake_dataset(
        {
            "observation.state": {"dtype": "float32"},
            "reward": {"dtype": "float32"},
            "observation.images.top": CAMERA,
            "thumbnail": {"dtype": "image"},
            "task": {"dtype": "string"},
            "masks.top": MASK,
        }
    )
    assert wp._numeric_features(ds) == ["observation.state", "reward"]
