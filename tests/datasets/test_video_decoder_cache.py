#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Unit tests for ``lerobot.datasets.video_utils.VideoDecoderCache``.

These cover the LRU bounding + file-handle release behaviour added to prevent
unbounded growth when iterating over datasets with many distinct video files
(observed: ~35 GB anon-rss per DataLoader worker on an 8 k-file dataset).
"""

import shutil
import threading
from pathlib import Path

import pytest

pytest.importorskip("torchcodec", reason="torchcodec is required (install lerobot[dataset])")

from lerobot.datasets.video_utils import VideoDecoderCache  # noqa: E402

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "encoded_videos"
SRC_CLIP = TEST_ARTIFACTS_DIR / "clip_4frames.mp4"


def _make_distinct_clips(tmp_path: Path, n: int) -> list[Path]:
    """Copy the small reference mp4 to ``n`` distinct paths.

    The cache keys on absolute path, so distinct paths force distinct cache entries
    even though the file contents are identical.
    """
    assert SRC_CLIP.exists(), f"missing test artifact {SRC_CLIP}"
    paths = []
    for i in range(n):
        dst = tmp_path / f"clip_{i:04d}.mp4"
        shutil.copyfile(SRC_CLIP, dst)
        paths.append(dst)
    return paths


class TestVideoDecoderCacheBounded:
    def test_default_cache_is_bounded(self):
        """The default cache must have a finite ``max_size`` to bound RSS growth."""
        cache = VideoDecoderCache()
        assert cache.max_size is not None, "default cache must be bounded"
        assert cache.max_size > 0

    def test_size_capped_at_max_size(self, tmp_path):
        """``get_decoder`` for >``max_size`` distinct paths must NOT grow without bound."""
        paths = _make_distinct_clips(tmp_path, n=5)
        cache = VideoDecoderCache(max_size=2)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 2

    def test_evicts_least_recently_used(self, tmp_path):
        """Re-accessing an entry must promote it; the LRU entry is the one evicted."""
        paths = _make_distinct_clips(tmp_path, n=3)
        cache = VideoDecoderCache(max_size=2)

        cache.get_decoder(paths[0])
        cache.get_decoder(paths[1])
        cache.get_decoder(paths[0])  # promote paths[0] to MRU; paths[1] is now LRU
        cache.get_decoder(paths[2])  # should evict paths[1]

        assert str(paths[0]) in cache  # MRU stays
        assert str(paths[1]) not in cache  # LRU evicted
        assert str(paths[2]) in cache  # newest stays

    def test_eviction_closes_file_handle(self, tmp_path):
        """Evicting an entry must close its fsspec file handle (otherwise we leak FDs)."""
        paths = _make_distinct_clips(tmp_path, n=2)
        cache = VideoDecoderCache(max_size=1)

        cache.get_decoder(paths[0])
        # Reach into the cache to capture the handle before it is evicted. This is
        # the only assertion in the suite that touches a private attribute, and it
        # is the most direct way to prove the file descriptor is actually released.
        evicted_handle = cache._cache[str(paths[0])][1]
        assert evicted_handle.closed is False

        cache.get_decoder(paths[1])  # forces eviction of paths[0]

        assert evicted_handle.closed is True

    def test_clear_closes_all_file_handles(self, tmp_path):
        """``clear()`` must close every cached file handle."""
        paths = _make_distinct_clips(tmp_path, n=3)
        cache = VideoDecoderCache(max_size=10)

        for p in paths:
            cache.get_decoder(p)
        handles = [entry[1] for entry in cache._cache.values()]
        assert all(not h.closed for h in handles)

        cache.clear()

        assert cache.size() == 0
        assert all(h.closed for h in handles)

    def test_hit_does_not_reopen_or_evict(self, tmp_path):
        """A cache hit must return the same decoder instance without touching the cap."""
        paths = _make_distinct_clips(tmp_path, n=1)
        cache = VideoDecoderCache(max_size=2)

        first = cache.get_decoder(paths[0])
        second = cache.get_decoder(paths[0])

        assert first is second
        assert cache.size() == 1

    def test_unbounded_when_max_size_none(self, tmp_path):
        """``max_size=None`` preserves the legacy unbounded behaviour."""
        paths = _make_distinct_clips(tmp_path, n=4)
        cache = VideoDecoderCache(max_size=None)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 4

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        """``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` env var sets the default ``max_size``."""
        monkeypatch.setenv("LEROBOT_VIDEO_DECODER_CACHE_SIZE", "3")
        cache = VideoDecoderCache()
        assert cache.max_size == 3

        paths = _make_distinct_clips(tmp_path, n=5)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 3


class TestVideoDecoderCacheEvictionSafety:
    """Eviction must never close a decoder another thread is still using.

    The per-path lock that makes concurrent decoding safe is a *different* lock
    from the one guarding the cache, so LRU eviction could close a file handle
    out from under an in-flight ``get_frames_at`` — a segfault rather than an
    exception. Neither parent of the 2026-07 upstream merge had this: the fork's
    cache was unbounded so nothing was ever evicted, and upstream's had no
    per-path locking because nothing decoded concurrently.
    """

    def test_in_use_entry_is_not_evicted(self, tmp_path):
        """A decoder held by another thread survives pressure from new entries."""
        paths = _make_distinct_clips(tmp_path, n=3)
        cache = VideoDecoderCache(max_size=2)

        entered = threading.Event()
        may_finish = threading.Event()
        still_open = []

        def holder():
            with cache.decoder_for(str(paths[0])):
                entered.set()
                may_finish.wait(timeout=10)
                # Read the entry's handle directly: if eviction closed it, this
                # is what an in-flight decode would have been reading through.
                entry = cache._cache.get(str(paths[0]))
                still_open.append(entry is not None and not entry[1].closed)

        t = threading.Thread(target=holder)
        t.start()
        assert entered.wait(timeout=10), "holder thread never started decoding"

        # Age the held entry to LRU and push the cache over capacity.
        cache.get_decoder(paths[1])
        cache.get_decoder(paths[2])

        may_finish.set()
        t.join(timeout=10)

        assert still_open == [True], "an in-use decoder's file handle was closed by eviction"

    def test_capacity_is_still_enforced_when_nothing_is_held(self, tmp_path):
        """Skipping busy entries must not disable eviction for idle ones."""
        paths = _make_distinct_clips(tmp_path, n=5)
        cache = VideoDecoderCache(max_size=2)
        for p in paths:
            with cache.decoder_for(str(p)):
                pass
        assert cache.size() == 2, "eviction stopped working for unheld entries"

    def test_freshly_created_entry_is_not_evicted_before_use(self, tmp_path):
        """The entry a caller is about to receive must survive its own insertion.

        It is not locked yet at insertion time, so a naive "evict anything
        unlocked" rule could discard and close it before the caller decodes.
        """
        paths = _make_distinct_clips(tmp_path, n=3)
        cache = VideoDecoderCache(max_size=1)
        for p in paths:
            with cache.decoder_for(str(p)) as decoder:
                assert decoder is not None
                entry = cache._cache.get(str(p))
                assert entry is not None, "the just-created entry was evicted before use"
                assert not entry[1].closed, "the just-created entry's handle was closed before use"
