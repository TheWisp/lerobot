"""Tests for lerobot.gui.cache_invalidation.invalidate_caches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lerobot.gui.cache_invalidation import invalidate_caches


def _make_app_state():
    return SimpleNamespace()


def test_drops_this_datasets_cached_windows_and_no_others(tmp_path, monkeypatch):
    """A window is a function of pixels an edit may have rewritten, so an edit
    must drop this dataset's windows -- and only this dataset's."""
    monkeypatch.setenv("LEROBOT_WINDOW_CACHE_DIR", str(tmp_path))
    from lerobot.gui.api import window_playback

    enc = {"codec": "h264", "rc": "cbr", "q": 26, "preset": "veryfast"}
    mine = [
        window_playback._cache_key("user/ds", 0, 0, 1.0, "320", enc),
        window_playback._cache_key("user/ds", 3, 30, 2.0, "640", enc, "composited", "cam:abc"),
    ]
    other = window_playback._cache_key("user/other", 0, 0, 1.0, "320", enc)
    for f in [*mine, other]:
        f.write_bytes(b"x" * 100)

    with patch("lerobot.datasets.video_utils._default_decoder_cache"):
        invalidate_caches(_make_app_state(), "user/ds")

    assert not any(f.exists() for f in mine), [f.name for f in mine if f.exists()]
    assert other.exists(), "another dataset's windows were dropped too"


def test_clears_video_decoder_cache_when_nonempty():
    app_state = _make_app_state()
    with patch("lerobot.datasets.video_utils._default_decoder_cache") as cache:
        cache.size.return_value = 3
        invalidate_caches(app_state, "user/ds")
    cache.clear.assert_called_once()


def test_skips_video_decoder_clear_when_empty():
    """When the cache has no entries, skip .clear() to avoid noise."""
    app_state = _make_app_state()
    with patch("lerobot.datasets.video_utils._default_decoder_cache") as cache:
        cache.size.return_value = 0
        invalidate_caches(app_state, "user/ds")
    cache.clear.assert_not_called()


def test_calls_episode_index_invalidator_when_provided():
    app_state = _make_app_state()
    invalidator = MagicMock()
    with patch("lerobot.datasets.video_utils._default_decoder_cache"):
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=invalidator)
    invalidator.assert_called_once_with("user/ds")


def test_skips_episode_index_invalidator_when_none():
    app_state = _make_app_state()
    with patch("lerobot.datasets.video_utils._default_decoder_cache"):
        # Should not raise
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=None)


def test_window_cache_error_does_not_prevent_other_invalidations():
    """If the window cache cannot be dropped, the video decoder cache and the
    index cache must still be cleared."""
    app_state = _make_app_state()
    invalidator = MagicMock()
    with patch("lerobot.datasets.video_utils._default_decoder_cache") as cache:
        cache.size.return_value = 2
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=invalidator)
    cache.clear.assert_called_once()
    invalidator.assert_called_once_with("user/ds")


def test_video_decoder_error_does_not_prevent_index_invalidation():
    app_state = _make_app_state()
    invalidator = MagicMock()
    with patch("lerobot.datasets.video_utils._default_decoder_cache") as cache:
        cache.size.side_effect = RuntimeError("size failed")
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=invalidator)
    invalidator.assert_called_once_with("user/ds")


def test_episode_index_invalidator_error_swallowed():
    """A broken index invalidator must not leak out of invalidate_caches."""
    app_state = _make_app_state()
    invalidator = MagicMock(side_effect=RuntimeError("bad"))
    with patch("lerobot.datasets.video_utils._default_decoder_cache"):
        # Should not raise
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=invalidator)
