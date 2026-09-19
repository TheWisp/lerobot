# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What an episode's video actually is, read from the file rather than info.json.

A merged dataset does not have one video format. The dataset that motivated
this carries 241 episodes encoded h264 beside 33 encoded AV1, and info.json
reports a single answer for the whole set -- so the split was invisible in the
interface while the two groups trained differently.

These probe real files written by ffmpeg, because the thing under test is
whether the numbers the panel shows are the file's own.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)


def _encode(path, codec="libx264", width=160, height=120, fps=10, frames=10):
    """A real video file, so the probe reads a real container.

    160x120 rather than something smaller because SVT-AV1 silently writes a
    zero-byte file below its minimum dimensions, which would make the codec
    comparison below pass for the wrong reason.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate={fps}:duration={frames / fps}",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    assert path.stat().st_size > 0, f"{codec} wrote an empty file at {width}x{height}"
    return path


def test_the_probe_reports_the_file_s_own_properties(tmp_path):
    from lerobot.gui.api.datasets import _probe_video

    info = _probe_video(_encode(tmp_path / "a.mp4", width=160, height=120, fps=10))

    assert info is not None
    assert info.codec == "h264"
    assert (info.width, info.height) == (160, 120)
    assert info.pix_fmt == "yuv420p"
    assert info.fps == pytest.approx(10, abs=0.01)


def test_two_encodings_of_the_same_frames_are_told_apart(tmp_path):
    """The case this exists for: one dataset, two codecs, invisible until now."""
    from lerobot.gui.api.datasets import _probe_video

    h264 = _probe_video(_encode(tmp_path / "h264.mp4", codec="libx264"))
    av1 = _probe_video(_encode(tmp_path / "av1.mp4", codec="libsvtav1"))

    assert h264 is not None and av1 is not None
    assert h264.codec != av1.codec, "two codecs reported as the same thing"
    assert {h264.codec, av1.codec} == {"h264", "av1"}


def test_a_missing_or_unreadable_file_returns_none(tmp_path):
    """A probe failure must never break the panel that displays it -- a listing
    of 274 episodes should not go blank because one file is mid-write."""
    from lerobot.gui.api.datasets import _probe_video

    assert _probe_video(tmp_path / "absent.mp4") is None

    not_a_video = tmp_path / "junk.mp4"
    not_a_video.write_bytes(b"not a container")
    assert _probe_video(not_a_video) is None


def test_a_re_encode_is_not_served_from_the_cache(tmp_path):
    """The cache is keyed by path AND mtime. Keyed by path alone it would keep
    reporting the old codec after a re-encode, which is exactly the situation
    -- converting a dataset's videos -- where someone consults this panel."""
    from lerobot.gui.api.datasets import _probe_video

    path = tmp_path / "same-name.mp4"
    first = _probe_video(_encode(path, codec="libx264"))
    assert first is not None and first.codec == "h264"

    path.unlink()
    second = _probe_video(_encode(path, codec="libsvtav1"))
    assert second is not None
    assert second.codec == "av1", "a re-encode was served from the cache"


def test_the_listing_endpoint_probes_without_a_missing_name(tmp_path, lerobot_dataset_factory, monkeypatch):
    """Drives ``list_episodes``, not ``_probe_video``.

    The probe runs on a bounded executor, and the first version of this feature
    referred to one that arrives in a different change -- so the endpoint raised
    ``NameError`` at runtime while every direct test of the probe passed. Only
    calling the endpoint reaches that line.
    """
    import asyncio

    from lerobot.gui.api import datasets as datasets_api

    ds = lerobot_dataset_factory(
        root=tmp_path / "ds", repo_id="probe/demo", total_episodes=2, total_frames=20
    )

    class _State:
        """Only what list_episodes reaches for."""

        def __init__(self, mapping):
            self.datasets = mapping

    monkeypatch.setattr(datasets_api, "_app_state", _State({"probe/demo": ds}))

    rows = asyncio.run(datasets_api.list_episodes("probe/demo"))

    assert len(rows) == 2
    # The field exists on every row even when nothing could be probed; the panel
    # renders it unconditionally.
    for row in rows:
        assert hasattr(row, "video_streams")


def test_the_probe_cache_is_bounded():
    """Keyed by path AND mtime, so an unbounded cache grows by one entry per
    re-encode and never sheds the superseded one. A GUI process is long-lived
    and a dataset conversion re-encodes every file it owns."""
    from lerobot.gui.api.datasets import _probe_video_cached

    info = _probe_video_cached.cache_info()
    assert info.maxsize is not None, "the probe cache is unbounded"
    assert info.maxsize >= 512, f"too small to be useful across a dataset: {info.maxsize}"


def test_the_cache_key_includes_mtime_not_just_path(tmp_path):
    """The complement to the bound: capping a cache keyed on path alone would
    still serve a stale codec after a re-encode."""
    from lerobot.gui.api.datasets import _probe_video_cached

    path = _encode(tmp_path / "k.mp4", codec="libx264")
    first = _probe_video_cached(str(path), 1)
    second = _probe_video_cached(str(path), 2)
    assert first is not None and second is not None
    # Different mtimes are different entries even for one path.
    assert _probe_video_cached.cache_info().misses >= 2
