"""What a browser page was still waiting on when a test gave up on it.

A wait that times out says only that the page never got there. Watching the
page from before it loads records what it asked the server for and never got
back, and what it reported as an error. The GUI server runs in the test
process, so the report also says whether it answers a request now and where
its threads are.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback


class PageWatch:
    """Attach before ``goto``; put ``report()`` in the failure of a wait on the page."""

    def __init__(self, page) -> None:
        self._page = page
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
        unanswered = [f"{now - t:.1f}s {m} {u}" for t, m, u in self._unanswered.values()]
        return (
            f"unanswered requests: {unanswered}\n"
            f"failed requests: {self._failed}\n"
            f"page errors: {self._errors}\n"
            f"server answers now: {self._server_answers()}\n"
            f"server threads:\n{_server_threads()}"
        )

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
