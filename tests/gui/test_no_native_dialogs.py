# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GUI asks its questions in its own dialogs, not the browser's.

``window.confirm`` draws the operating system's dialog: another typeface, another
palette, a title bar naming the origin, and a position the page cannot control.
Every one in the GUI's own bundle was replaced by ``static/dialogs.js``. This is
the ratchet that keeps the next from being typed by reflex: at a call site the
two spellings look alike, and nothing but this marks one of them as "opens a
browser dialog".

It covers what it can see -- the files this app serves as its UI. One native
call deliberately survives outside that scope, in the server-rendered
``/ai_setup`` page (``lerobot/gui/api/ai_setup.py``), which does not load
``dialogs.js`` and whose ``onclick="return confirm(...)"`` gates a form submit
synchronously: a promise cannot stand in for that without also rewriting how the
form is submitted.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"

# Every call by these names is a candidate, whatever the receiver: bare
# `confirm(`, `window.confirm(`, and equally `globalThis.` / `self.` / `top.` /
# `parent.` or any alias someone binds later. Naming the receivers that ARE the
# global object would fail open on the first one nobody listed, so the match is
# inverted -- catch them all, and exempt the one object that legitimately owns
# methods by these names.
#
# `_confirm(` and `myPrompt(` are excluded by the leading class: a name that
# merely ends in one of these words is a different identifier.
NATIVE_CALL = re.compile(r"(?<![\w$])(?:([\w$.]+)\s*\.\s*)?(confirm|alert|prompt)\s*\(")

# The module that replaced them. Anything else calling `.confirm(` would be a
# new method by that name, which is worth a look rather than a silent pass.
ALLOWED_RECEIVERS = {"Dialogs", "window.Dialogs"}

# The module that exists to replace them defines methods by these names.
EXEMPT = {"dialogs.js"}

# HTML as well as JS: an inline `onclick="confirm(...)"` opens the same browser
# dialog, and a JS-only sweep would never see it. Recursive, so a future
# `static/<subdir>/` is covered the day it appears rather than the day someone
# remembers this file.
SCANNED = ("*.js", "*.html")


def _sources() -> list[Path]:
    found = {p for pattern in SCANNED for p in STATIC.rglob(pattern)}
    return sorted(p for p in found if p.name not in EXEMPT)


def test_the_static_bundle_has_no_browser_dialogs():
    offenders = []
    for path in _sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for m in NATIVE_CALL.finditer(line):
                receiver = m.group(1)
                if receiver in ALLOWED_RECEIVERS:
                    continue
                shown = f"{receiver}." if receiver else ""
                offenders.append(
                    f"{path.relative_to(STATIC)}:{lineno}: {shown}{m.group(2)}( -- {line.strip()}"
                )

    assert not offenders, (
        "browser dialogs found; use `await Dialogs.confirm/alert/prompt` from "
        "static/dialogs.js instead (note the caller must be async):\n  " + "\n  ".join(offenders)
    )


def test_the_replacement_is_actually_loaded():
    """A guard that only forbids is worth little if the alternative is unreachable."""
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.search(r"/static/dialogs\.js\?v=\d+", index), (
        "index.html does not load dialogs.js, so every converted call site would "
        "throw ReferenceError at the moment it asks the user a question"
    )


def _flagged(source: str) -> list[str]:
    """What the guard would report for one line of source."""
    return [m.group(2) for m in NATIVE_CALL.finditer(source) if m.group(1) not in ALLOWED_RECEIVERS]


def test_the_guard_would_catch_a_real_call():
    """The regex is the whole test; a typo in it would pass everything silently."""
    assert _flagged("if (!confirm('x')) return;") == ["confirm"]
    assert _flagged("const v = window.prompt('x');") == ["prompt"]
    assert _flagged('el.setAttribute("onclick", "confirm(\'x\')")') == ["confirm"]


def test_the_guard_catches_receivers_nobody_listed():
    """The whole point of matching every receiver and exempting one: an alias
    for the global object that this file never mentions still opens the OS
    dialog, and a list of known aliases would wave it through."""
    for spelling in (
        "globalThis.confirm('x')",
        "self.alert('x')",
        "top.prompt('x')",
        "parent.confirm('x')",
        "window.top.confirm('x')",
    ):
        assert _flagged(spelling), f"{spelling} slipped past the guard"


def test_the_replacement_and_ordinary_identifiers_are_not_flagged():
    assert _flagged("await Dialogs.confirm('x')") == []
    assert _flagged("await window.Dialogs.prompt('x')") == []
    assert _flagged("if (!_confirm(x)) return;") == []
    assert _flagged("const v = myPrompt('x');") == []
