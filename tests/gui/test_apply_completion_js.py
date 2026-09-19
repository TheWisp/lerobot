"""A stopped mask-apply run owes a schema refresh; the check is JS, so it runs
under node with this pytest wrapper (skipped when node is absent).

The defect it locks: an apply run may APPEND a label to the dataset's
vocabulary, and the client's copy of the schema is read when the dataset is
opened. Nothing refreshed it when a run finished, so a label the run created had
no timeline lane, no Inspector row and no entry in the fill-gaps dialog for the
rest of the session — the masks were on disk while the UI showed the run as
having done nothing. See apply_completion.test.js for the assertions.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_apply_completion_effects_js():
    test_js = Path(__file__).parent / "apply_completion.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_page_actually_loads_the_module():
    """The node test proves the module is right; nothing proves the page has it.

    A module that ships untested is one thing; one that is tested and never
    loaded is worse, because the green test reads as coverage. `overlays.js`
    calls `window.ApplyCompletion` unguarded, so a missing tag is a TypeError
    inside the poll — which is swallowed by its own catch and looks like the
    run simply never finishing.
    """
    index = (STATIC / "index.html").read_text()
    assert "apply_completion.js" in index, "index.html does not load apply_completion.js"
    order = [
        index.index('src="/static/apply_completion.js'),
        index.index('src="/static/overlays.js'),
    ]
    assert order == sorted(order), "apply_completion.js must load before overlays.js, which calls it"


def test_a_changed_bundle_busts_its_cache():
    """Every static tag carries `?v=N`; a stale browser copy of `overlays.js`
    would keep the old completion path while the file on disk is fixed."""
    index = (STATIC / "index.html").read_text()
    for name in ("overlays.js", "apply_completion.js"):
        assert re.search(rf'src="/static/{re.escape(name)}\?v=\d+"', index), f"{name} has no ?v= buster"


def test_the_apply_run_actually_calls_the_completion_effects():
    """The module exists because one effect was missing for the life of the
    feature -- and for a while nothing called the module either, so the defect
    it documents was still live: the filler's job path refreshed the schema and
    invalidated the mask cache, and apply-while-playing did neither.

    Checked at the source. The alternative is a browser test that arms Apply,
    plays a whole episode with a segmenter loaded and asserts a refresh landed;
    the wiring is one call, and this is the assertion that would have caught its
    absence.
    """
    import pathlib

    panel = (pathlib.Path(__file__).resolve().parents[2] / "src/lerobot/gui/static/overlays.js").read_text()
    assert "ApplyCompletion?.applyTerminalEffects?.(" in panel, (
        "the apply-while-playing run does not run its completion effects; a label the "
        "run appended will have no lane, no Inspector row and no dialog entry until reload"
    )
    # The one that was missing, named explicitly: the others could all be present
    # while this stayed absent, which is exactly how it shipped.
    tail = panel.split("ApplyCompletion?.applyTerminalEffects?.(", 1)[1][:600]
    assert "refreshSchema" in tail, "the schema refresh is not wired into the run's completion"
    assert "invalidateMasks" in tail, "the mask cache is not invalidated when the run stops"
