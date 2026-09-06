# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Which clip the video endpoint builds, and what it calls the cache entry.

The endpoint decides three things nothing else can see: whether a request gets
the composited build or the plain one, what the cached file is called, and
whether a recipe change reaches the viewer. All three cross a boundary -- an
HTTP query string on one side, a filename on the other -- so nothing type-checks
them, and a mistake shows up as the wrong video rather than an error.

The name carries the recipe's fingerprint on purpose: that is what makes an
effects edit land on a different entry instead of being served the stale one.
A raw clip and a composited clip of the same episode must not collide either,
or whichever was built first answers both.
"""

import asyncio
from types import SimpleNamespace

import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.datasets.test_saved_masks_training import masked_dataset_root  # noqa: F401

CAMERA = "observation.images.top"
SPEC = {
    "mask_size": [64, 96],
    "mask_labels": ["ring"],
    "mask_treatments": {"ring": {"key": "blur", "params": {}}},
    "mask_background": {"key": "none"},
}


@pytest.fixture
def endpoint(masked_dataset_root, tmp_path, monkeypatch):  # noqa: F811
    """The video endpoint with the transcodes stubbed and the cache redirected.

    Stubbing the two builders is the point: this pins which one is chosen and
    what path it is handed, not what ffmpeg produces (that is
    `test_composited_clip_build.py`). The cache goes to tmp_path so the test
    cannot write into the user's real ~/.cache.
    """
    from lerobot.gui.api import datasets as api

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    monkeypatch.setattr(api, "_app_state", SimpleNamespace(datasets={repo_id: ds}), raising=False)

    cache = tmp_path / "clips"
    cache.mkdir()
    monkeypatch.setattr(api, "_playback_cache_dir", lambda: cache)

    calls: list[tuple[str, object]] = []

    def plain(src, out, start_s, duration_s, profile):
        calls.append(("plain", out))
        out.write_bytes(b"plain")

    def composited(
        dataset, dataset_id, episode_idx, camera_key, src, out, start_s, duration_s, profile, spec
    ):
        calls.append(("composited", out))
        out.write_bytes(b"composited")

    monkeypatch.setattr(api, "_transcode_episode", plain)
    monkeypatch.setattr(api, "_transcode_episode_composited", composited)

    pruned: list = []
    monkeypatch.setattr(api, "_prune_playback_cache", lambda keep=None: pruned.append(keep) or 0)

    def call(masks="", recipe=SPEC):
        monkeypatch.setattr(api, "_effective_recipe", lambda *a, **kw: recipe)
        return asyncio.run(api.get_episode_video(repo_id, 0, camera=CAMERA, profile="low", masks=masks))

    return SimpleNamespace(call=call, calls=calls, pruned=pruned)


def test_a_plain_request_takes_the_plain_build(endpoint):
    endpoint.call(masks="")
    assert [kind for kind, _ in endpoint.calls] == ["plain"]
    assert "__m" not in endpoint.calls[0][1].name, endpoint.calls[0][1].name


def test_asking_for_the_composite_takes_the_composited_build(endpoint):
    endpoint.call(masks="composited")
    assert [kind for kind, _ in endpoint.calls] == ["composited"]
    assert "__m" in endpoint.calls[0][1].name, endpoint.calls[0][1].name


def test_the_raw_and_composited_clips_do_not_share_a_cache_entry(endpoint):
    """Same episode, same profile, different pixels — so different files."""
    endpoint.call(masks="")
    endpoint.call(masks="composited")
    names = [out.name for _, out in endpoint.calls]
    assert len(set(names)) == 2, names


def test_an_effects_edit_lands_on_a_different_entry(endpoint):
    """The invalidation claim: no cache clearing, the name simply changes."""
    endpoint.call(masks="composited", recipe=SPEC)
    other = {**SPEC, "mask_treatments": {"ring": {"key": "tint", "params": {"color": [1, 2, 3]}}}}
    endpoint.call(masks="composited", recipe=other)
    names = [out.name for _, out in endpoint.calls]
    assert len(set(names)) == 2, f"a treatment change reused the stale clip: {names}"


def test_a_dataset_with_no_recipe_still_plays(endpoint):
    """Asking for the composite on an unmasked camera must not fail or hang."""
    endpoint.call(masks="composited", recipe=None)
    assert [kind for kind, _ in endpoint.calls] == ["plain"]


def test_the_cache_is_pruned_after_a_build(endpoint):
    endpoint.call(masks="composited")
    assert endpoint.pruned == [endpoint.calls[0][1]], endpoint.pruned


# ── the cache ceiling ───────────────────────────────────────────────────────
#
# Every recipe edit orphans an episode's composited clips by construction (the
# fingerprint is in the name), so without a bound the directory only grows.


@pytest.fixture
def clip_cache(tmp_path, monkeypatch):
    """A cache directory with a small ceiling, holding clips of known age."""
    import os

    from lerobot.gui.api import datasets as api

    cache = tmp_path / "clips"
    cache.mkdir()
    monkeypatch.setattr(api, "_playback_cache_dir", lambda: cache)
    monkeypatch.setattr(api, "_PLAYBACK_CACHE_MAX_BYTES", 300)

    def add(name, size, atime):
        f = cache / name
        f.write_bytes(b"x" * size)
        os.utime(f, (atime, atime))
        return f

    return SimpleNamespace(dir=cache, add=add, prune=api._prune_playback_cache)


def test_a_cache_under_the_ceiling_is_left_alone(clip_cache):
    """The complement of the eviction test: "nothing was deleted" would also
    pass on a pruner that never deletes, so both directions are pinned."""
    a = clip_cache.add("a.mp4", 100, 1_000)
    b = clip_cache.add("b.mp4", 100, 2_000)
    assert clip_cache.prune() == 0
    assert a.is_file() and b.is_file()


def test_the_least_recently_used_clips_go_first(clip_cache):
    old = clip_cache.add("old.mp4", 200, 1_000)
    clip_cache.add("mid.mp4", 200, 2_000)
    new = clip_cache.add("new.mp4", 200, 3_000)

    freed = clip_cache.prune()

    assert freed > 0
    assert not old.is_file(), "the oldest clip survived"
    assert new.is_file(), "the newest clip was evicted before the oldest"
    remaining = sum(f.stat().st_size for f in clip_cache.dir.glob("*.mp4"))
    assert remaining <= 300, remaining


def test_the_clip_about_to_be_served_is_never_evicted(clip_cache):
    """`keep` is the entry the request is holding: evicting it would delete the
    file the response is about to stream."""
    keep = clip_cache.add("keep.mp4", 200, 1)  # oldest, so the first candidate
    clip_cache.add("other.mp4", 200, 9_999)

    clip_cache.prune(keep=keep)

    assert keep.is_file(), "the clip the response was about to serve was deleted"
