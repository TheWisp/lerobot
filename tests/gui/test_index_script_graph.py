# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""index.html must load every module the loaded modules depend on.

The GUI has no bundler. Modules publish themselves onto `window` from a UMD
wrapper and read each other back off it, so the only thing that makes a
dependency resolve is a `<script>` tag in index.html, in the right place. Drop
the tag and the file is still in the repository, still imported by the node
tests, still passing review -- and dead in the browser.

That happened: a rebase resolved the script block by keeping one side, which
removed the tags for `bitset.js`, `timeline_lanes.js` and `track_render.js`
while adding `window_player.js`. `feature_editing.js` destructures
`window.Bitset` as it loads, so it threw on every page load and every feature
row, mask lane and flag lane stopped rendering.

The cache-busting ratchet next door could not see it. It compares index.html
against a recorded fingerprint file, and the same rebase dropped those three
from both -- two bookkeeping files agreeing with each other, about a module
neither of them still mentions. The checks here face the other way, at the
source: what a module reads is read out of the module, not out of a baseline,
so there is nothing to re-record and no way to make the check agree with the
break.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"
INDEX = STATIC / "index.html"

# `root.Foo = factory()` in the UMD wrapper, or a plain `window.Foo = ...`.
PUBLISHES = re.compile(r"(?:^|[;{}\s])(?:root|window|self)\.([A-Za-z_]\w*)\s*=")
# Any mention of another module's global, wherever it appears.
READS = re.compile(r"\bwindow\.([A-Z]\w*)")
# A binding whose entire value is the global: no `?.`, no call, no fallback.
# The UMD wrapper invokes the factory immediately, so one of these runs while
# the script runs and the provider has to have run already.
READS_AT_LOAD = re.compile(
    r"^ {0,4}(?:const|let|var)\s+(?:\{[^}]*\}|\w+)\s*=\s*window\.([A-Z]\w*)\s*;\s*$", re.M
)


def _sources() -> dict[str, str]:
    return {f.name: f.read_text() for f in sorted(STATIC.glob("*.js"))}


def _load_order() -> dict[str, int]:
    """Script name → its position in index.html."""
    names = re.findall(r"/static/([\w.]+\.js)\?v=\d+", INDEX.read_text())
    return {name: i for i, name in enumerate(names)}


def _providers(sources: dict[str, str]) -> dict[str, set[str]]:
    """Global name → the module files that publish it."""
    out: dict[str, set[str]] = {}
    for name, text in sources.items():
        for symbol in set(PUBLISHES.findall(text)):
            out.setdefault(symbol, set()).add(name)
    return out


def test_every_global_a_loaded_module_reads_has_a_loaded_provider():
    """The regression itself: the file exists, and nothing loads it."""
    sources = _sources()
    order, providers = _load_order(), _providers(sources)

    unreachable = []
    for name in sorted(order):
        for symbol in sorted(set(READS.findall(sources.get(name, "")))):
            owners = providers.get(symbol, set())
            # Globals nothing in static/ publishes are the browser's or the
            # page's own; this rule is only about module-to-module links.
            if owners and not owners & set(order):
                unreachable.append(f"{name} reads window.{symbol}, published only by {sorted(owners)}")

    assert not unreachable, (
        "index.html does not load a module that another loaded module needs, so the global "
        "is undefined in the browser while the file sits in the repository:\n  " + "\n  ".join(unreachable)
    )


def test_a_module_read_while_the_page_loads_is_loaded_first():
    """Reachable is not enough when the read happens during script execution."""
    sources = _sources()
    order, providers = _load_order(), _providers(sources)

    late = []
    for name in sorted(order):
        for symbol in sorted(set(READS_AT_LOAD.findall(sources.get(name, "")))):
            for owner in sorted(providers.get(symbol, set()) & set(order)):
                if order[owner] > order[name]:
                    late.append(
                        f"{name} (tag {order[name]}) binds window.{symbol} as it loads, "
                        f"but {owner} is tag {order[owner]}"
                    )

    assert not late, (
        "a module reads another module's global while the page loads, and its provider's "
        "<script> comes later, so the binding is undefined and the reader throws:\n  " + "\n  ".join(late)
    )


def test_the_rule_has_something_to_check():
    """Both rules above pass trivially against an empty graph, which is what a
    broken regex or a moved directory produces. Pin that the parse still finds
    the structure it is asserting over."""
    sources = _sources()
    order, providers = _load_order(), _providers(sources)

    assert len(order) > 20, f"index.html parsed to {len(order)} scripts; the parse is wrong"
    assert len(providers) > 20, f"only {len(providers)} globals found; the publish pattern is wrong"

    at_load = {
        (name, symbol)
        for name in order
        for symbol in READS_AT_LOAD.findall(sources.get(name, ""))
        if symbol in providers
    }
    assert at_load, "no module reads another's global at load time; the load-time rule checks nothing"
