"""Compressed delivery of static files, for a GUI reached over a slow link.

The GUI is routinely opened against a robot host at the far end of a WAN. There
the server is not the bottleneck -- it answers in a fraction of a millisecond
and the bytes then take seconds -- so the only lever that helps is sending
fewer of them. Starlette's ``StaticFiles`` sends everything uncompressed, and
the largest thing the GUI serves is mesh geometry, which is either verbose text
or loosely packed binary and compresses several-fold.

:class:`CompressedStaticFiles` is a drop-in ``StaticFiles`` that gzips a
response when doing so actually saves bytes. It is deliberately ignorant of
what it is mounted on: it holds no list of formats, extensions or paths, and
decides per file by compressing it once and keeping the result only if it came
out enough smaller. Content that is already compressed fails that test on its
first request and is served raw from then on, whatever it happens to be called.
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

logger = logging.getLogger(__name__)

#: Below this, the gzip envelope costs more than it saves.
MIN_COMPRESS_BYTES = 1024

#: Refuse to hold more than this in memory for one file. A directory mounted
#: with something enormous in it falls through uncompressed rather than
#: buffering it.
MAX_COMPRESS_BYTES = 32 * 1024 * 1024

#: Keep the compressed form only when it is at least this much smaller. This,
#: not a list of media types, is what excludes already-compressed content.
MIN_SAVING = 0.05

#: Balanced setting: most of the achievable ratio, a fraction of the CPU of
#: the maximum. Paid once per file per process thanks to the cache.
COMPRESS_LEVEL = 6

#: Ceiling on what the compressed-bytes cache may retain. Sized to hold what a
#: package realistically vendors; a mount larger than this simply stops being
#: cached rather than evicting in a loop.
CACHE_BUDGET_BYTES = 64 * 1024 * 1024

# Compression is CPU-bound and, at mesh sizes, long enough to stall every other
# request if it ran on the event loop. It also must not go on the shared
# default executor, which is contended with frame decode and camera teardown.
# Single worker on purpose: a cold load asks for a whole directory at once, and
# serialising keeps that from saturating a robot host's CPU.
_compress_executor: ThreadPoolExecutor | None = None

# (path, mtime_ns, size) -> gzipped bytes, or None for "raw is better".
# The mtime and size in the key mean an edited file is recompressed rather than
# served from a stale entry.
_compressed: dict[tuple[str, int, int], bytes | None] = {}
_cached_bytes = 0
_budget_reached = False

# Compressions already running, so that concurrent requests for one file wait
# on the first rather than each redoing it. Two arms of a bimanual robot load
# from the same directory at the same time, so this is a normal load, not a
# corner case.
_inflight: dict[tuple[str, int, int], asyncio.Future[bytes | None]] = {}


def _executor() -> ThreadPoolExecutor:
    """The compression pool, created on demand.

    Created lazily, and created again after a shutdown. The GUI app can be
    brought up more than once in one process -- every test that starts it does
    -- and a pool that stayed shut down would fail every later request with
    "cannot schedule new futures after shutdown".
    """
    global _compress_executor
    if _compress_executor is None:
        _compress_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-compress")
    return _compress_executor


def shutdown_compress_executor() -> None:
    """Release the compression thread and drop queued work on server shutdown."""
    global _compress_executor
    if _compress_executor is not None:
        _compress_executor.shutdown(wait=False, cancel_futures=True)
        _compress_executor = None


def _compress(path: Path, size: int) -> bytes | None:
    """Gzip ``path``, returning ``None`` when the result is not worth serving.

    Precondition: ``path`` is a readable regular file of ``size`` bytes.
    Postcondition: the return value, when not ``None``, decompresses to exactly
    the file's bytes and is smaller than them by at least :data:`MIN_SAVING`.
    """
    blob = gzip.compress(path.read_bytes(), COMPRESS_LEVEL, mtime=0)
    if len(blob) > size * (1.0 - MIN_SAVING):
        return None
    return blob


def _remember(key: tuple[str, int, int], blob: bytes | None) -> None:
    """Record a compression result, within the cache's byte budget.

    Idempotent per key: two requests that raced on the same file must not both
    charge it, or the budget trips on a multiple of the bytes actually held.
    At most one version of a path is retained, so the accounting tracks what is
    really held rather than everything ever compressed.

    Postcondition: every key reaching here is recorded. A result that does not
    fit the budget is recorded as "serve raw" rather than left absent -- an
    absent key means gzipping that file again on every single request, which is
    a worse failure than not compressing it at all.
    """
    global _cached_bytes, _budget_reached
    if key in _compressed:
        return
    # An edit gives the file a new key. Release what the superseded version was
    # holding, or editing a mesh repeatedly fills the budget with dead copies.
    for stale in [k for k in _compressed if k[0] == key[0]]:
        superseded = _compressed.pop(stale)
        _cached_bytes -= len(superseded) if superseded is not None else 0
    cost = len(blob) if blob is not None else 0
    if cost and _cached_bytes + cost > CACHE_BUDGET_BYTES:
        if not _budget_reached:
            logger.warning("static compression cache is full; further assets will be served uncompressed")
            _budget_reached = True
        _compressed[key] = None
        return
    _compressed[key] = blob
    _cached_bytes += cost


async def _gzipped(path: Path, stat: os.stat_result) -> bytes | None:
    """Gzipped bytes for ``path``, or ``None`` if it should be served raw.

    Postcondition: never runs compression on the event loop, and the answer
    agrees with what was cached -- one file does not arrive encoded for one
    request and raw for the next.
    """
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _compressed:
        return _compressed[key]

    running = _inflight.get(key)
    if running is not None:
        blob = await asyncio.shield(running)
        # Prefer what was recorded: if the budget refused this file, every
        # request for it must agree and serve it raw.
        return _compressed.get(key, blob)

    loop = asyncio.get_running_loop()
    pending = loop.run_in_executor(_executor(), _compress, path, stat.st_size)
    _inflight[key] = pending
    try:
        blob = await asyncio.shield(pending)
    finally:
        _inflight.pop(key, None)
    _remember(key, blob)
    return _compressed.get(key, blob)


# Headers that describe the raw file and would be wrong on the encoded one.
_REPLACED_HEADERS = frozenset({"content-length", "content-encoding", "accept-ranges", "etag"})

#: Marks the entity tag of the encoded representation, so it cannot be confused
#: with the raw one by a cache that disregards ``Vary``.
_ETAG_MARKER = "-gzip"


def _accepts_gzip(header: str) -> bool:
    """Whether the client will actually take a gzip-encoded body.

    A substring test is not enough: ``gzip;q=0`` names gzip in order to refuse
    it, and a bare ``*`` accepts it without naming it at all.
    """
    gzip_q: float | None = None
    wildcard_q: float | None = None
    for element in header.split(","):
        coding, _, params = element.strip().partition(";")
        coding = coding.strip().lower()
        if coding not in ("gzip", "*"):
            continue
        q = 1.0
        for param in params.split(";"):
            name, _, value = param.partition("=")
            if name.strip().lower() == "q":
                try:
                    q = float(value.strip())
                except ValueError:
                    q = 0.0
        if coding == "gzip":
            gzip_q = q
        else:
            wildcard_q = q
    # An explicit mention of gzip decides it; the wildcard only fills the gap.
    if gzip_q is not None:
        return gzip_q > 0
    return wildcard_q is not None and wildcard_q > 0


def _encoded_etag(etag: str) -> str:
    """The entity tag of the encoded representation of ``etag``."""
    return f'{etag[:-1]}{_ETAG_MARKER}"' if etag.endswith('"') else f"{etag}{_ETAG_MARKER}"


def _unmark_conditional(scope: Scope) -> tuple[Scope, bool]:
    """Strip the encoded-variant marker from ``If-None-Match``.

    ``StaticFiles`` validates the header against the raw file's entity tag, so
    a marked tag can never match it. Left in place, revalidation of an encoded
    asset always misses and re-sends the whole file -- strictly worse than not
    marking at all.

    Returns the scope to delegate with, and whether the client was in fact
    revalidating an encoded variant.
    """
    marked = f'{_ETAG_MARKER}"'.encode()
    headers = scope.get("headers") or []
    if not any(name.lower() == b"if-none-match" and marked in value for name, value in headers):
        return scope, False
    rewritten = [
        (name, value.replace(marked, b'"') if name.lower() == b"if-none-match" else value)
        for name, value in headers
    ]
    return {**scope, "headers": rewritten}, True


class CompressedStaticFiles(StaticFiles):
    """``StaticFiles`` that gzips what benefits from it.

    Whether a file benefits is measured, not assumed: it is compressed once and
    the result kept only if it came out enough smaller, so the class needs no
    knowledge of formats and can be mounted on any directory.

    Precondition: the mount's contents are identified by ``(path, mtime, size)``
    -- true of files on disk, which is all ``StaticFiles`` serves.
    Postcondition: what the client ends up with is byte-identical to the file
    on disk, whether it arrives encoded or raw.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        headers = Headers(scope=scope)
        accepts_gzip = _accepts_gzip(headers.get("accept-encoding", ""))

        delegate_scope, revalidating_encoded = _unmark_conditional(scope) if accepts_gzip else (scope, False)
        response = await super().get_response(path, delegate_scope)

        if revalidating_encoded and response.status_code == 304:
            # Hand the marked tag back, so the client's cache entry keeps
            # naming the representation it actually holds.
            etag = response.headers.get("etag")
            if etag:
                response.headers["etag"] = _encoded_etag(etag)
            response.headers["vary"] = "Accept-Encoding"
            return response

        # 304 and the error responses carry no body to encode.
        if not isinstance(response, FileResponse) or response.status_code != 200:
            return response

        # Whether this asset comes back encoded depends on Accept-Encoding, so
        # every representation of it must say so -- including the raw one, or a
        # shared cache could hand one client's copy to another.
        response.headers["vary"] = "Accept-Encoding"

        if not accepts_gzip:
            return response
        # An encoded body invalidates the byte offsets a range request names,
        # so a client asking for one gets the file as it is on disk.
        if "range" in headers:
            return response

        # Without a stat there is no key that can detect the file changing, so
        # the safe answer is to serve it as it is. StaticFiles always supplies
        # one; this is a guard, not a path that is expected to run.
        stat = response.stat_result
        if stat is None or not MIN_COMPRESS_BYTES <= stat.st_size <= MAX_COMPRESS_BYTES:
            return response

        blob = await _gzipped(Path(response.path), stat)
        if blob is None:
            return response

        encoded = Response(content=blob, status_code=response.status_code)
        for name, value in response.headers.items():
            if name.lower() not in _REPLACED_HEADERS:
                encoded.headers[name] = value
        etag = response.headers.get("etag")
        if etag:
            # The encoded form is a different representation of the file and
            # must not share an entity tag with the raw one.
            encoded.headers["etag"] = _encoded_etag(etag)
        encoded.headers["content-encoding"] = "gzip"
        encoded.headers["accept-ranges"] = "none"
        encoded.headers["vary"] = "Accept-Encoding"
        return encoded
