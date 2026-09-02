# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Building a composited clip across threads must not change the clip.

Compositing is the whole cost of preparing a masked episode for playback and
has no dependency between frames, so the build spreads it over a pool. That
introduces three ways to be wrong that a serial loop could not be, none of
which raises — each one just produces a different video:

  * **Order.** Frames finish out of order and must reach the encoder in the
    order they were read.
  * **The tail.** Frames still in flight when the reader runs out have to be
    drained, or the clip is short by up to one window.
  * **The episode's randomness.** A ``random`` treatment is drawn once and
    cached for the episode. Each frame builds its own generator from the same
    seed, but consumes it only for the regions that frame actually has, so the
    frame that populates the cache decides the draw. Serially that is frame 0;
    under a pool it would be whoever got there first.

The first two are pinned against a serial reference build. The third cannot be
observed through this fixture — its segmenter returns the same masks for every
frame, so every frame would draw identically — so it is pinned where it lives
instead: frame 0 composites on the calling thread, before anything is
submitted.
"""

import json
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.datasets.test_saved_masks_training import masked_dataset_root  # noqa: F401

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="clip building needs ffmpeg")

CAMERA = "observation.images.top"
RANDOM_BG = {"key": "random", "params": {}}


class _Inline:
    """The serial reference: every frame composites before the next is read."""

    def submit(self, fn, *a, **kw):
        fut: Future = Future()
        fut.set_result(fn(*a, **kw))
        return fut


def _reversing(real):
    """Wrap the compositor so the frames submitted first finish last.

    Waiting for a lucky interleaving makes a concurrency test that passes when
    the timing is kind, so the order is inverted outright: the nth frame to
    start sleeps for (window - n) ticks. Whatever the pool does with the
    results, they become available in the opposite order to submission, and an
    out-of-order write can no longer hide.
    """
    import itertools
    import time

    seq = itertools.count()

    def wrapper(*a, **kw):
        time.sleep(max(0.0, 0.05 - 0.004 * next(seq)))
        return real(*a, **kw)

    return wrapper


def _build(root, repo_id, out, executor, monkeypatch, *, jitter=False, workers=6):
    """Build episode 0's composited clip through `executor`."""
    from lerobot.datasets import mask_compositing
    from lerobot.gui.api import datasets as api

    ds = LeRobotDataset(repo_id, root=root)
    spec = dict(ds.meta.features[mask_compositing.mask_feature_of(CAMERA)])
    spec["mask_background"] = RANDOM_BG
    ep = ds.meta.episodes[0]

    # The build reads episode offsets out of app state; one episode starting at
    # zero is all this needs, and registering it keeps the real code path.
    monkeypatch.setattr(api, "_app_state", SimpleNamespace(datasets={str(root): ds}), raising=False)
    monkeypatch.setattr(api, "_episode_start_indices", {}, raising=False)
    monkeypatch.setattr(api, "_composite_executor", executor)
    monkeypatch.setattr(api, "_COMPOSITE_WORKERS", workers)
    if jitter:
        monkeypatch.setattr(
            mask_compositing, "composite_from_store", _reversing(mask_compositing.composite_from_store)
        )
    api._transcode_episode_composited(
        dataset=ds,
        dataset_id=str(root),
        episode_idx=0,
        camera_key=CAMERA,
        src=ds.root / ds.meta.get_video_file_path(0, CAMERA),
        out=out,
        start_s=float(ep.get(f"videos/{CAMERA}/from_timestamp", 0.0) or 0.0),
        duration_s=float(ep["length"]) / float(ds.fps),
        profile="low",
        spec=spec,
    )
    return out


def _frame_count(path) -> int:
    probe = shutil.which("ffprobe")
    if probe is None:
        pytest.skip("ffprobe not available")
    out = subprocess.run(  # noqa: S603
        [
            probe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(json.loads(out.stdout)["streams"][0]["nb_read_frames"])


def test_the_pool_produces_the_serial_clip(masked_dataset_root, tmp_path, monkeypatch):  # noqa: F811
    """A pooled build, with completions deliberately out of order, must match
    the frame-by-frame build byte for byte. This is the load-bearing test: a
    drain that wrote frames as they finished would differ here and nowhere
    else."""
    root, repo_id = masked_dataset_root
    serial = _build(root, repo_id, tmp_path / "serial.mp4", _Inline(), monkeypatch)
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        parallel = _build(root, repo_id, tmp_path / "pool.mp4", pool, monkeypatch, jitter=True, workers=3)
    finally:
        pool.shutdown()
    assert serial.read_bytes() == parallel.read_bytes(), (
        "compositing across threads changed the clip: the frames reached the "
        "encoder in an order the serial build would not have produced"
    )


def test_every_frame_of_the_episode_reaches_the_clip(masked_dataset_root, tmp_path, monkeypatch):  # noqa: F811
    """The bounded window drains, including whatever is still in flight when
    the decoder runs out. A missing final drain truncates the clip, which two
    equally-truncated builds would agree on."""
    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    expected = int(ds.meta.episodes["length"][0])
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        clip = _build(root, repo_id, tmp_path / "tail.mp4", pool, monkeypatch, workers=3)
    finally:
        pool.shutdown()
    assert _frame_count(clip) == expected


def test_the_first_frame_composites_before_anything_is_submitted(masked_dataset_root, tmp_path, monkeypatch):  # noqa: F811
    """Frame 0 fills the episode's randomized-draw cache, and every later frame
    reads it back, so it must not be one of several frames racing to fill it.
    Serial order is restored by composing it on the calling thread first; this
    checks that, since the fixture cannot show the race itself."""
    import threading

    from lerobot.datasets import mask_compositing
    from lerobot.gui.api import datasets as api

    root, repo_id = masked_dataset_root
    threads: list[str] = []
    real = mask_compositing.composite_from_store

    def recording(*a, **kw):
        threads.append(threading.current_thread().name)
        return real(*a, **kw)

    caller = threading.current_thread().name
    pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="worker")
    try:
        monkeypatch.setattr(mask_compositing, "composite_from_store", recording)
        _build(root, repo_id, tmp_path / "first.mp4", pool, monkeypatch, workers=3)
    finally:
        pool.shutdown()
    assert threads, "nothing was composited"
    assert threads[0] == caller, f"frame 0 composited on {threads[0]}, not the calling thread"
    assert any(t.startswith("worker") for t in threads[1:]), "no frame ran on the pool at all"
    assert api._COMPOSITE_WORKERS  # the module still exposes the width the build uses
