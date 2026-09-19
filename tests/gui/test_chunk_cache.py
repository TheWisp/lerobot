"""Chunks are cached on disk, and dropped when the data moves (design: C6, C7, R7).

"Nothing changed" is satisfied by "nothing ever changes", so every stability
assertion here is paired with the change that must be seen beside it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("av")

import requests  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    GuiServer,
    build_dataset,
    chunk_url,
)


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    base = tmp_path_factory.mktemp("cache")
    return build_dataset(base / "one"), build_dataset(base / "two")


@pytest.fixture(scope="module")
def server(roots, tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    ids = [srv.open_dataset(r) for r in roots]
    yield srv, ids
    srv.stop()


def test_the_second_request_is_served_from_disk(server):
    srv, (one, _) = server
    url = chunk_url(srv.base, one, 0, 0)
    first = requests.get(url, timeout=120)
    second = requests.get(url, timeout=120)
    assert first.headers["X-Chunk-Cache"] == "miss"
    assert second.headers["X-Chunk-Cache"] == "hit"
    assert second.content == first.content
    # The complement: a different chunk is a different entry, or the key is not a key.
    other = requests.get(chunk_url(srv.base, one, 0, 20), timeout=120)
    assert other.headers["X-Chunk-Cache"] == "miss"
    assert other.content != first.content


def test_an_edit_drops_that_datasets_chunks_and_no_others(server):
    """The GUI's shared invalidation hook -- what every edit path calls -- drops
    the edited dataset's chunks; the untouched dataset keeps its entries."""
    from lerobot.gui.api import chunk_playback
    from lerobot.gui.cache_invalidation import invalidate_caches
    from lerobot.gui.server import _app_state

    srv, (one, two) = server
    for ds_id in (one, two):
        requests.get(chunk_url(srv.base, ds_id, 0, 0), timeout=120)
        assert requests.get(chunk_url(srv.base, ds_id, 0, 0), timeout=120).headers["X-Chunk-Cache"] == "hit"

    def files(ds_id):
        return sorted(
            p.name for p in chunk_playback.cache_dir().glob(f"{chunk_playback._prefix(ds_id)}__*.bin")
        )

    assert files(one) and files(two)

    invalidate_caches(_app_state, one)

    assert files(one) == [], "the edited dataset's chunks are gone"
    assert files(two), "the untouched dataset keeps its entries"
    assert requests.get(chunk_url(srv.base, one, 0, 0), timeout=120).headers["X-Chunk-Cache"] == "miss"
    assert requests.get(chunk_url(srv.base, two, 0, 0), timeout=120).headers["X-Chunk-Cache"] == "hit"


def test_an_entry_that_vanishes_mid_request_is_rebuilt_not_a_500(server, monkeypatch):
    """The cache is looked up, then read; an edit or a prune can land between.

    The edit-storm suite caught this as `chunk 180: HTTP 500` mid-playback: the
    entry was there for `exists()` and gone by the read, and the page was
    served an error for a chunk the server could simply have rebuilt.
    """
    from lerobot.gui.api import chunk_playback

    srv, (one, _) = server
    assert requests.get(chunk_url(srv.base, one, 0, 0), timeout=120).status_code == 200
    assert requests.get(chunk_url(srv.base, one, 0, 0), timeout=120).headers["X-Chunk-Cache"] == "hit"

    # The window itself: the look succeeded, the read finds nothing.
    taken = []

    def _vanished(key):
        taken.append(key)
        raise FileNotFoundError(2, "No such file or directory", str(key))

    monkeypatch.setattr(chunk_playback, "_read_and_touch", _vanished)
    r = requests.get(chunk_url(srv.base, one, 0, 0), timeout=120)

    assert taken, "the cached path was never taken, so this proves nothing"
    assert r.status_code == 200, (r.status_code, r.text[:300])
    assert r.headers["X-Chunk-Cache"] == "miss", "a rebuilt chunk is a miss, whatever the lookup said"
    assert len(r.content) > 0


def test_the_ceiling_evicts_least_recently_used(tmp_path, monkeypatch):
    import os
    import time

    from lerobot.gui.api import chunk_playback

    monkeypatch.setenv("LEROBOT_CHUNK_CACHE_DIR", str(tmp_path))
    names = ["a", "b", "c"]
    for i, _n in enumerate(names):
        p = tmp_path / f"x__v1__ep0__f{i}__low.bin"
        p.write_bytes(b"0" * 100)
        t = 1_000_000 + i * 10
        os.utime(p, (t, t))
    time.sleep(0.01)
    removed = chunk_playback.prune_cache(ceiling=250)
    assert removed == 100
    left = sorted(p.name for p in tmp_path.glob("*.bin"))
    assert left == ["x__v1__ep0__f1__low.bin", "x__v1__ep0__f2__low.bin"], left  # the oldest went


def test_a_file_that_goes_while_the_pruner_walks_does_not_take_the_prune_with_it(tmp_path, monkeypatch):
    """An edit drops its dataset's chunks while the pruner walks the same
    directory, so a path the glob returned can be gone by the stat.

    The prune runs on the miss path, after a chunk has been built and stored,
    so raising here fails the request that did the work -- the same HTTP 500
    mid-playback the read side served. A dangling symlink is that window
    standing still: the glob lists it, the stat does not find it.
    """
    from lerobot.gui.api import chunk_playback

    monkeypatch.setenv("LEROBOT_CHUNK_CACHE_DIR", str(tmp_path))
    cold = tmp_path / "x__v1__ep0__f0__low.bin"
    cold.write_bytes(b"0" * 300)
    gone = tmp_path / "x__v1__ep0__f1__low.bin"
    gone.symlink_to(tmp_path / "a_chunk_an_edit_already_dropped.bin")
    assert gone in set(tmp_path.glob("*.bin")), "the walk must still list it, or nothing is proven"
    assert not gone.exists(), "and the stat must not find it"

    assert chunk_playback.prune_cache(ceiling=100) == 300
    assert not cold.exists(), "the file that was really there still went"


def test_prune_reads_the_ceiling_at_call_time(tmp_path, monkeypatch):
    """A default bound at import would ignore a ceiling set later by the server."""
    from lerobot.gui.api import chunk_playback

    monkeypatch.setenv("LEROBOT_CHUNK_CACHE_DIR", str(tmp_path))
    (tmp_path / "x__v1__ep0__f0__low.bin").write_bytes(b"0" * 100)
    monkeypatch.setattr(chunk_playback, "CACHE_CEILING_BYTES", 50)
    assert chunk_playback.prune_cache() == 100
    assert not list(tmp_path.glob("*.bin"))
