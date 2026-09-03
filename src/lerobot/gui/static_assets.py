"""Static-asset delivery tuned for a GUI reached over a WAN.

The GUI is routinely opened against a robot host at the other end of a slow,
high-latency link — measured on one such host: 248 ms RTT and ~500 KB/s, with
parallel connections adding nothing because the pipe is already saturated. Two
defaults hurt badly there and cost nothing to fix.

**Immutable assets were revalidated on every load.** ``StaticFiles`` sends an
``ETag`` but no ``Cache-Control``, so browsers fall back to heuristic freshness
and in practice re-ask for every mesh. A 304 for one 1.9 MB mesh still costs a
full round trip, and a single robot description is ~10 MB across ten meshes.
Vendored robot descriptions and third-party libraries change only when the
package does, so they are served ``immutable`` and never revalidated.

**Nothing was compressed.** Measured on this repository's own files: 4.6x on a
Collada mesh (3.3 MB -> 0.7 MB), 2.6x on a binary STL, 4.1x on ``app.js``.

Compression is deliberately selective. The GUI's hottest response is a JPEG
frame; gzipping one burns CPU per request to make it marginally larger. Server
-sent event streams must not be buffered at all, or live run status stops
updating.
"""

from __future__ import annotations

import gzip
import io

from starlette.datastructures import Headers, MutableHeaders
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: A year, the conventional ceiling. Safe only for content whose URL changes
#: when the bytes do — here, because a new package release replaces the file.
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"

#: Below this, the gzip header costs more than the saving.
MIN_COMPRESS_BYTES = 1024

#: Refuse to buffer more than this for compression. Guards against a future
#: route serving something huge through here; it falls through uncompressed.
MAX_COMPRESS_BYTES = 32 * 1024 * 1024

_ALREADY_COMPRESSED_PREFIXES = ("image/", "video/", "audio/")
_ALREADY_COMPRESSED_TYPES = frozenset(
    {
        "application/zip",
        "application/gzip",
        "application/x-gzip",
        "application/x-bzip2",
        "application/zstd",
        "application/octet-stream",  # safetensors, .pt — large and incompressible
    }
)


def is_compressible(content_type: str) -> bool:
    """Whether gzipping a response of this content type is worth the CPU.

    Precondition: ``content_type`` is a raw header value; parameters such as
    ``; charset=utf-8`` are tolerated. An empty type is treated as compressible,
    matching the conservative-but-useful default for unlabelled text.
    """
    main = content_type.split(";", 1)[0].strip().lower()
    if main == "text/event-stream":
        return False  # buffering an SSE stream would stall live updates
    if main.startswith(_ALREADY_COMPRESSED_PREFIXES):
        return False
    return main not in _ALREADY_COMPRESSED_TYPES


class ImmutableStaticFiles(StaticFiles):
    """``StaticFiles`` that marks what it serves as immutable.

    Precondition: mount only on directories whose contents ship with the
    package — vendored robot descriptions, third-party libraries. Never on
    files edited during development: a browser that cached one will not ask
    again until the max-age expires.
    """

    async def get_response(self, path: str, scope: Scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        # Set on every path, including 304s: a revalidating client that got no
        # Cache-Control would drop back to heuristic freshness next load.
        response.headers["Cache-Control"] = IMMUTABLE_CACHE_CONTROL
        return response


class SelectiveGZipMiddleware:
    """Gzip responses that benefit, leave the rest untouched.

    Starlette's ``GZipMiddleware`` decides on size alone. This one also looks
    at the content type, so JPEG frames — the hottest route in the app — skip
    compression entirely, and SSE streams are never buffered.
    """

    def __init__(self, app: ASGIApp, minimum_size: int = MIN_COMPRESS_BYTES) -> None:
        self.app = app
        self.minimum_size = minimum_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or "gzip" not in Headers(scope=scope).get("accept-encoding", ""):
            await self.app(scope, receive, send)
            return
        await _GZipResponder(self.app, self.minimum_size)(scope, receive, send)


class _GZipResponder:
    """Buffers one response, compresses it if that helps, then sends it."""

    def __init__(self, app: ASGIApp, minimum_size: int) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.start: Message | None = None
        self.body = bytearray()
        self.passthrough = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.send = send
        await self.app(scope, receive, self._send)

    async def _send(self, message: Message) -> None:
        if self.passthrough:
            await self.send(message)
            return

        if message["type"] == "http.response.start":
            headers = Headers(raw=message["headers"])
            declared = headers.get("content-length")
            too_big = declared is not None and int(declared) > MAX_COMPRESS_BYTES
            if (
                "content-encoding" in headers  # already encoded; leave it alone
                or not is_compressible(headers.get("content-type", ""))
                or too_big
            ):
                self.passthrough = True
                await self.send(message)
                return
            self.start = message
            return

        if message["type"] != "http.response.body":
            await self.send(message)
            return

        self.body.extend(message.get("body", b""))
        if len(self.body) > MAX_COMPRESS_BYTES:
            # Bigger than expected (streamed, no content-length): give up on
            # compression rather than hold it all in memory.
            self.passthrough = True
            assert self.start is not None
            await self.send(self.start)
            await self.send({"type": "http.response.body", "body": bytes(self.body), "more_body": True})
            self.body.clear()
            return
        if message.get("more_body", False):
            return

        assert self.start is not None, "body before response start"
        if len(self.body) < self.minimum_size:
            await self.send(self.start)
            await self.send({"type": "http.response.body", "body": bytes(self.body)})
            return

        buf = io.BytesIO()
        with gzip.GzipFile(mode="wb", fileobj=buf, compresslevel=6, mtime=0) as f:
            f.write(self.body)
        compressed = buf.getvalue()

        start = dict(self.start)
        headers = MutableHeaders(raw=list(start["headers"]))
        headers["Content-Encoding"] = "gzip"
        headers["Content-Length"] = str(len(compressed))
        headers.add_vary_header("Accept-Encoding")
        start["headers"] = headers.raw

        await self.send(start)
        await self.send({"type": "http.response.body", "body": compressed})
