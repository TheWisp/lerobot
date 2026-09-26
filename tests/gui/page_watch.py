"""What a browser page was still waiting on when a test gave up on it.

A wait that times out says only that the page never got there. Watching the
page from before it loads records what it asked the server for and never got
back, and what it reported as an error. The GUI server runs in the test
process, so the report also says, for each request the page never got back,
whether the server received it and what it did with it; whether it answers a
request now; and where its threads are.
"""

from __future__ import annotations

import collections
import sys
import threading
import time
import traceback
from urllib.parse import unquote, urlsplit

#: Every HTTP request the GUI app received in this process, newest last:
#: ``{"t", "method", "path", "query", "status", "answered"}``, the last two
#: filled in when the response starts. Bounded; a report only needs the recent.
_JOURNAL: collections.deque = collections.deque(maxlen=5000)
_journal_lock = threading.Lock()
_journaled = False


class _Journal:
    """ASGI layer around the GUI app that writes each request to the journal."""

    def __init__(self, inner) -> None:
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        entry = {
            "t": time.monotonic(),
            "method": scope["method"],
            "path": scope["path"],
            "query": scope.get("query_string", b"").decode("latin-1"),
            "status": None,
            "answered": None,
        }
        _JOURNAL.append(entry)

        async def recording_send(message):
            if message["type"] == "http.response.start":
                entry["status"] = message["status"]
                entry["answered"] = time.monotonic()
            await send(message)

        return await self.inner(scope, receive, recording_send)


def _journal_the_gui_app() -> None:
    """Route the GUI app's requests through the journal, once per process.
    Starlette calls whatever ``middleware_stack`` holds, and nothing adds
    middleware to this app after it starts, so wrapping the stack is safe."""
    global _journaled
    with _journal_lock:
        if _journaled:
            return
        from lerobot.gui import server

        app = server.app
        app.middleware_stack = _Journal(app.middleware_stack or app.build_middleware_stack())
        _journaled = True


class PageWatch:
    """Attach before ``goto``; put ``report()`` in the failure of a wait on the page."""

    def __init__(self, page) -> None:
        _journal_the_gui_app()
        self._page = page
        self._since = time.monotonic()
        self._unanswered: dict[int, tuple[float, str, str]] = {}
        self._failed: list[str] = []
        self._errors: list[str] = []
        page.on("request", lambda r: self._unanswered.__setitem__(id(r), (time.monotonic(), r.method, r.url)))
        page.on("requestfinished", lambda r: self._unanswered.pop(id(r), None))
        page.on("requestfailed", self._on_failed)
        page.on("console", lambda m: self._errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: self._errors.append(str(e)))

    def _on_failed(self, request) -> None:
        self._unanswered.pop(id(request), None)
        self._failed.append(f"{request.method} {request.url}: {request.failure}")

    def report(self) -> str:
        now = time.monotonic()
        unanswered = [
            f"{now - t:.1f}s {m} {u} -- {self._server_side(m, u, now)}"
            for t, m, u in self._unanswered.values()
        ]
        running = [
            f"{now - e['t']:.1f}s {e['method']} {e['path']}"
            for e in list(_JOURNAL)
            if e["t"] >= self._since and e["answered"] is None
        ]
        return (
            f"unanswered requests: {unanswered}\n"
            f"failed requests: {self._failed}\n"
            f"page errors: {self._errors}\n"
            f"server still working on: {running}\n"
            f"server answers now: {self._server_answers()}\n"
            f"server threads:\n{_server_threads()}"
        )

    def _server_side(self, method: str, url: str, now: float) -> str:
        """What the server did with a request the page never got back."""
        parts = urlsplit(url)
        path, query = unquote(parts.path), parts.query
        seen = [
            e
            for e in list(_JOURNAL)
            if e["t"] >= self._since and e["method"] == method and e["path"] == path and e["query"] == query
        ]
        if not seen:
            return "the server never received it"
        e = seen[-1]
        if e["answered"] is None:
            return f"the server received it {now - e['t']:.1f}s ago and has not answered"
        return f"the server answered {e['status']} after {e['answered'] - e['t']:.2f}s"

    def _server_answers(self) -> str:
        import requests

        try:
            r = requests.get(self._page.url, timeout=5)
        except requests.RequestException as e:
            return f"no: {e!r}"
        return f"{r.status_code} in {r.elapsed.total_seconds():.2f}s"


def _server_threads() -> str:
    """The tail of every stack in this process that is in lerobot's own code
    or uvicorn's: the GUI server's thread and its workers."""
    names = {t.ident: t.name for t in threading.enumerate()}
    out = []
    for ident, frame in sys._current_frames().items():
        if ident == threading.get_ident():
            continue
        lines = traceback.format_stack(frame)
        if any("/src/lerobot/" in ln or "/uvicorn/" in ln for ln in lines):
            out.append(f"--- {names.get(ident, ident)}\n" + "".join(lines[-8:]))
    return "\n".join(out) or "(none)"
