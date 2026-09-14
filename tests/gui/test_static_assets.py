# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Static-asset compression: what it encodes, what it leaves alone, and why.

The point of the mechanism is that it holds no list of formats -- it compresses
a file once and keeps the result only if it helped. Several tests below exist
specifically to pin that: an unknown extension is compressed on merit, and an
extension that *looks* like an image is compressed or not according to its
bytes rather than its name.
"""

import asyncio
import gzip
import struct
import threading

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui import static_assets
from lerobot.gui.static_assets import CompressedStaticFiles, _accepts_gzip

# Mesh-sized, and shaped like the geometry this actually serves: a run of
# float triples, which is what makes a binary STL compressible in the first
# place. Sized to the real vendored meshes rather than to a toy.
_MESH_BYTES = 3 * 1024 * 1024


def _meshlike(nbytes: int) -> bytes:
    """Binary that compresses the way real mesh geometry does."""
    triples = nbytes // 12
    return b"".join(struct.pack("<fff", i % 97 * 0.5, i % 31 * 0.25, i % 13 * 0.125) for i in range(triples))


def _incompressible(nbytes: int) -> bytes:
    """Bytes with no redundancy left -- what an already-compressed file is."""
    import random

    return random.Random(0).randbytes(nbytes)


@pytest.fixture(autouse=True)
def _clear_compression_cache():
    """Keep the module-global cache from leaking between tests."""
    static_assets._compressed.clear()
    static_assets._inflight.clear()
    static_assets._cached_bytes = 0
    static_assets._budget_reached = False
    yield
    static_assets._compressed.clear()
    static_assets._inflight.clear()
    static_assets._cached_bytes = 0
    static_assets._budget_reached = False


@pytest.fixture
def assets(tmp_path):
    """A directory of synthesized assets, mounted the way the server mounts one."""
    (tmp_path / "mesh.stl").write_bytes(_meshlike(_MESH_BYTES))
    (tmp_path / "model.dae").write_text("<collada>" + "<node name='x'/>" * 20000 + "</collada>")
    (tmp_path / "robot.urdf").write_text("<robot>" + "<link name='l'/>" * 5000 + "</robot>")
    (tmp_path / "already.bin").write_bytes(_incompressible(64 * 1024))
    (tmp_path / "tiny.txt").write_text("small")
    # No extension the mechanism could recognise, holding compressible bytes.
    (tmp_path / "geometry.xyzmesh").write_bytes(_meshlike(256 * 1024))
    # An image extension over bytes that *do* compress: the decision must come
    # from the bytes, not the name.
    (tmp_path / "notreally.png").write_bytes(_meshlike(256 * 1024))
    return tmp_path


@pytest.fixture
def client(assets):
    app = FastAPI()
    app.mount("/assets", CompressedStaticFiles(directory=assets), name="assets")
    with TestClient(app) as c:
        yield c


def _wire_size(response) -> int:
    """Bytes that crossed the wire, as opposed to what httpx decoded for us."""
    return int(response.headers["content-length"])


@pytest.mark.parametrize("name", ["mesh.stl", "model.dae", "robot.urdf"])
def test_compressible_asset_is_encoded(client, assets, name):
    r = client.get(f"/assets/{name}", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
    assert _wire_size(r) < (assets / name).stat().st_size


@pytest.mark.parametrize("name", ["mesh.stl", "model.dae", "robot.urdf"])
def test_bytes_survive_encoding(client, assets, name):
    """The client must end up with the file, exactly."""
    r = client.get(f"/assets/{name}", headers={"accept-encoding": "gzip"})
    assert r.content == (assets / name).read_bytes()


def test_compression_actually_saves_bytes(client, assets):
    """The header is not the point; the wire size is."""
    raw = (assets / "mesh.stl").stat().st_size
    r = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    assert _wire_size(r) < raw / 2


def test_incompressible_asset_is_served_raw(client, assets):
    """Nothing names this file as compressed -- it fails the saving test."""
    r = client.get("/assets/already.bin", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert "content-encoding" not in r.headers
    assert r.content == (assets / "already.bin").read_bytes()


def test_unknown_extension_is_compressed_on_merit(client, assets):
    """No format list: an extension the mechanism has never seen still wins."""
    r = client.get("/assets/geometry.xyzmesh", headers={"accept-encoding": "gzip"})
    assert r.headers["content-encoding"] == "gzip"
    assert r.content == (assets / "geometry.xyzmesh").read_bytes()


def test_decision_follows_bytes_not_media_type(client, assets):
    """An image extension over compressible bytes is still compressed.

    A media-type denylist would skip this. The measurement does not.
    """
    r = client.get("/assets/notreally.png", headers={"accept-encoding": "gzip"})
    assert r.headers["content-encoding"] == "gzip"
    assert r.content == (assets / "notreally.png").read_bytes()


def test_small_asset_is_not_encoded(client):
    r = client.get("/assets/tiny.txt", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert "content-encoding" not in r.headers


def test_client_without_gzip_gets_plain_bytes(client, assets):
    r = client.get("/assets/mesh.stl", headers={"accept-encoding": "identity"})
    assert "content-encoding" not in r.headers
    assert _wire_size(r) == (assets / "mesh.stl").stat().st_size


def test_range_request_is_not_encoded(client, assets):
    """An encoded body would make the requested byte offsets mean something else."""
    r = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip", "range": "bytes=0-99"})
    assert r.status_code == 206
    assert "content-encoding" not in r.headers
    assert r.content == (assets / "mesh.stl").read_bytes()[:100]


def test_encoded_variant_has_its_own_etag(client):
    """Two representations of one file must not share an entity tag."""
    encoded = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    raw = client.get("/assets/mesh.stl", headers={"accept-encoding": "identity"})
    assert encoded.headers["etag"] != raw.headers["etag"]
    assert encoded.headers["vary"] == "Accept-Encoding"


def test_conditional_request_still_revalidates(client):
    """The encoded variant's ETag round-trips to a 304.

    Regression: marking the tag without unmarking it on the way back in means
    ``StaticFiles`` can never match it, so every conditional request re-sends
    the whole mesh -- worse than shipping no marker at all.
    """
    first = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    again = client.get(
        "/assets/mesh.stl",
        headers={"accept-encoding": "gzip", "if-none-match": first.headers["etag"]},
    )
    assert again.status_code == 304
    # The 304 must name the representation the client is holding, not the raw one.
    assert again.headers["etag"] == first.headers["etag"]


def test_raw_variant_also_varies_on_encoding(client):
    """A shared cache must not serve one representation for the other."""
    raw = client.get("/assets/mesh.stl", headers={"accept-encoding": "identity"})
    assert raw.headers["vary"] == "Accept-Encoding"


def test_edited_file_is_not_served_from_the_cache(client, assets):
    """The cache key carries mtime and size, so an edit is picked up."""
    first = client.get("/assets/model.dae", headers={"accept-encoding": "gzip"})
    assert first.content.startswith(b"<collada>")

    (assets / "model.dae").write_text("<collada>" + "<node name='y'/>" * 20001 + "</collada>")
    second = client.get("/assets/model.dae", headers={"accept-encoding": "gzip"})
    assert second.content == (assets / "model.dae").read_bytes()
    assert second.content != first.content


def test_compression_does_not_run_on_the_event_loop(client, monkeypatch):
    """Mesh-sized compression on the loop would stall every other request."""
    seen: list[str] = []
    original = static_assets._compress

    def _record(path, size):
        seen.append(threading.current_thread().name)
        return original(path, size)

    monkeypatch.setattr(static_assets, "_compress", _record)
    client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    assert seen and all(name.startswith("gui-compress") for name in seen)


def test_second_request_reuses_the_cached_result(client, monkeypatch):
    """Compression is paid once per file, not once per request."""
    calls = []
    original = static_assets._compress

    def _count(path, size):
        calls.append(path)
        return original(path, size)

    monkeypatch.setattr(static_assets, "_compress", _count)
    client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("gzip", True),
        ("gzip, deflate", True),
        ("deflate, gzip;q=0.8", True),
        ("*", True),
        ("identity, *;q=0.5", True),
        ("", False),
        ("identity", False),
        ("deflate", False),
        ("*;q=0", False),
        # Naming gzip in order to refuse it. A substring test reads these as yes.
        ("gzip;q=0", False),
        ("gzip;q=0, identity", False),
        ("gzip;q=0.000", False),
        # An explicit refusal of gzip outranks a permissive wildcard.
        ("gzip;q=0, *", False),
    ],
)
def test_accept_encoding_is_negotiated_not_substring_matched(header, expected):
    assert _accepts_gzip(header) is expected


def test_client_refusing_gzip_by_qvalue_gets_plain_bytes(client, assets):
    """Regression: ``gzip;q=0`` is a refusal, and a substring test misreads it."""
    r = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip;q=0, identity"})
    assert "content-encoding" not in r.headers
    assert r.content == (assets / "mesh.stl").read_bytes()


def test_concurrent_requests_compress_once(assets, monkeypatch):
    """Two arms of one robot load the same meshes at the same time.

    Regression: without coalescing, each request compressed the file again and
    each charged it to the cache budget, so the budget tripped on a multiple of
    the bytes actually held.
    """
    calls: list[str] = []
    original = static_assets._compress

    def _count(path, size):
        calls.append(str(path))
        return original(path, size)

    monkeypatch.setattr(static_assets, "_compress", _count)

    app = FastAPI()
    app.mount("/assets", CompressedStaticFiles(directory=assets), name="assets")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            return await asyncio.gather(
                *(c.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"}) for _ in range(4))
            )

    responses = asyncio.run(run())

    assert all(r.status_code == 200 for r in responses)
    assert all(r.content == (assets / "mesh.stl").read_bytes() for r in responses)
    assert len(calls) == 1, f"compressed {len(calls)} times for one file"
    held = sum(len(b) for b in static_assets._compressed.values() if b is not None)
    assert static_assets._cached_bytes == held


def test_budget_exhaustion_serves_raw_without_recompressing(client, assets, monkeypatch):
    """A full cache must degrade to raw, not re-gzip on every request.

    Regression: refusing to record the result left the key absent, so the file
    was compressed again for every subsequent request, forever.
    """
    monkeypatch.setattr(static_assets, "CACHE_BUDGET_BYTES", 0)
    calls: list[str] = []
    original = static_assets._compress

    def _count(path, size):
        calls.append(str(path))
        return original(path, size)

    monkeypatch.setattr(static_assets, "_compress", _count)

    first = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    second = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})

    assert "content-encoding" not in first.headers
    assert first.content == (assets / "mesh.stl").read_bytes()
    assert second.content == (assets / "mesh.stl").read_bytes()
    assert len(calls) == 1, "a file that did not fit the budget was compressed again"


def test_editing_a_file_releases_what_the_old_version_held(client, assets):
    """Editing a mesh repeatedly must not fill the budget with dead copies."""
    for i in range(4):
        (assets / "model.dae").write_text("<collada>" + f"<node name='{i}'/>" * 20000 + "</collada>")
        r = client.get("/assets/model.dae", headers={"accept-encoding": "gzip"})
        assert r.content == (assets / "model.dae").read_bytes()

    versions = [k for k in static_assets._compressed if k[0].endswith("model.dae")]
    assert len(versions) == 1, "superseded versions are still cached"
    held = sum(len(b) for b in static_assets._compressed.values() if b is not None)
    assert static_assets._cached_bytes == held


def test_incompressible_result_is_cached_even_though_it_costs_nothing(client, monkeypatch):
    """A negative result is free to keep, so it must not be re-derived."""
    calls: list[str] = []
    original = static_assets._compress

    def _count(path, size):
        calls.append(str(path))
        return original(path, size)

    monkeypatch.setattr(static_assets, "_compress", _count)
    client.get("/assets/already.bin", headers={"accept-encoding": "gzip"})
    client.get("/assets/already.bin", headers={"accept-encoding": "gzip"})
    assert len(calls) == 1


def test_assets_still_serve_after_a_server_shutdown(client, assets):
    """A second server lifespan in one process must still compress.

    Regression: the compression pool was a module global that the server's
    shutdown hook closed, so every later request raised "cannot schedule new
    futures after shutdown". In a full test session one earlier test brings the
    GUI up and down, and the URDF viewer then never loads its meshes.
    """
    static_assets.shutdown_compress_executor()
    static_assets._compressed.clear()

    r = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
    assert r.content == (assets / "mesh.stl").read_bytes()


def test_every_description_mount_is_compressed():
    """The wiring, checked without naming a robot.

    Whatever ``*_description`` packages are vendored, each must be served by
    the compressing class -- a new one must not quietly arrive uncompressed.
    """
    from starlette.routing import Mount

    from lerobot.gui.server import app

    mounts = [r for r in app.routes if isinstance(r, Mount) and r.path.startswith("/urdf-assets/")]
    assert mounts, "no vendored robot descriptions are mounted"
    for mount in mounts:
        assert isinstance(mount.app, CompressedStaticFiles), mount.path


def test_missing_asset_is_unaffected(client):
    assert client.get("/assets/nope.stl", headers={"accept-encoding": "gzip"}).status_code == 404


def test_encoded_body_is_valid_gzip(client, assets):
    """Decodable by something other than the client that fetched it."""
    r = client.get("/assets/mesh.stl", headers={"accept-encoding": "gzip"})
    # httpx already decoded; re-encode the same file and check our own framing.
    blob = static_assets._compress(assets / "mesh.stl", (assets / "mesh.stl").stat().st_size)
    assert blob is not None
    assert gzip.decompress(blob) == (assets / "mesh.stl").read_bytes()
    assert r.content == gzip.decompress(blob)
