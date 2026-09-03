"""Asset delivery over a high-latency link.

These pin the two behaviours that make a remote GUI usable — immutable caching
for vendored assets, and compression that skips payloads where it would only
cost CPU — plus the two ways a naive gzip layer breaks this app: JPEG frames
and SSE streams.
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.testclient import TestClient

from lerobot.gui.static_assets import (
    IMMUTABLE_CACHE_CONTROL,
    ImmutableStaticFiles,
    SelectiveGZipMiddleware,
    is_compressible,
)

BIG_TEXT = ("lerobot " * 2000).encode()
BIG_JPEG_ISH = bytes(range(256)) * 40  # incompressible-ish, > minimum size


@pytest.fixture
def app(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "mesh.stl").write_bytes(BIG_TEXT)

    application = FastAPI()
    application.add_middleware(SelectiveGZipMiddleware)
    application.mount("/immutable", ImmutableStaticFiles(directory=assets), name="immutable")

    @application.get("/text")
    def text() -> PlainTextResponse:
        return PlainTextResponse(BIG_TEXT.decode())

    @application.get("/frame")
    def frame() -> Response:
        return Response(content=BIG_JPEG_ISH, media_type="image/jpeg")

    @application.get("/tiny")
    def tiny() -> PlainTextResponse:
        return PlainTextResponse("hi")

    @application.get("/events")
    def events() -> StreamingResponse:
        def stream():
            for i in range(3):
                yield f"data: {i}\n\n".encode()

        return StreamingResponse(stream(), media_type="text/event-stream")

    return application


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ============================================================================
# is_compressible
# ============================================================================


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("text/html", True),
        ("application/json", True),
        ("text/plain; charset=utf-8", True),
        ("model/vnd.collada+xml", True),
        ("", True),
        ("image/jpeg", False),
        ("IMAGE/JPEG", False),
        ("image/png", False),
        ("video/mp4", False),
        ("application/zip", False),
        ("application/octet-stream", False),
        ("text/event-stream", False),
    ],
)
def test_is_compressible(content_type, expected):
    assert is_compressible(content_type) is expected


# ============================================================================
# Compression
# ============================================================================


def test_text_is_compressed(client):
    res = client.get("/text", headers={"accept-encoding": "gzip"})
    assert res.headers["content-encoding"] == "gzip"
    assert "accept-encoding" in res.headers.get("vary", "").lower()
    assert res.content == BIG_TEXT  # httpx decodes transparently
    assert int(res.headers["content-length"]) < len(BIG_TEXT)


def test_jpeg_is_not_compressed(client):
    """The hottest route in the app must not pay for gzip."""
    res = client.get("/frame", headers={"accept-encoding": "gzip"})
    assert "content-encoding" not in res.headers
    assert res.content == BIG_JPEG_ISH


def test_small_responses_are_not_compressed(client):
    res = client.get("/tiny", headers={"accept-encoding": "gzip"})
    assert "content-encoding" not in res.headers


def test_client_without_gzip_gets_plain_bytes(client):
    res = client.get("/text", headers={"accept-encoding": "identity"})
    assert "content-encoding" not in res.headers
    assert res.content == BIG_TEXT


def test_event_stream_is_not_buffered(client):
    """Buffering an SSE response would stall live run status."""
    with client.stream("GET", "/events", headers={"accept-encoding": "gzip"}) as res:
        assert "content-encoding" not in res.headers
        chunks = [c for c in res.iter_raw() if c]
    assert b"data: 0" in b"".join(chunks)


def test_compression_actually_saves_bytes(client):
    """The header is not the point; the wire size is."""
    res = client.get("/text", headers={"accept-encoding": "gzip"})
    on_the_wire = int(res.headers["content-length"])
    assert on_the_wire < len(BIG_TEXT) / 4, f"{on_the_wire} vs {len(BIG_TEXT)} uncompressed"


# ============================================================================
# Immutable caching
# ============================================================================


def test_vendored_asset_is_immutable(client):
    res = client.get("/immutable/mesh.stl", headers={"accept-encoding": "gzip"})
    assert res.status_code == 200
    assert res.headers["cache-control"] == IMMUTABLE_CACHE_CONTROL


def test_immutable_header_is_present_on_304(client):
    """Without it, a revalidating browser falls back to heuristic freshness."""
    first = client.get("/immutable/mesh.stl")
    etag = first.headers["etag"]

    second = client.get("/immutable/mesh.stl", headers={"if-none-match": etag})
    assert second.status_code == 304
    assert second.headers["cache-control"] == IMMUTABLE_CACHE_CONTROL


def test_mesh_bytes_are_unchanged_by_the_caching_layer(client):
    res = client.get("/immutable/mesh.stl")
    assert res.content == BIG_TEXT
