# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Every question about "which cameras" has exactly one place that answers it.

For the live preview that place is the overlay panel's own selection, read
through ``Overlays.dataQuery()``. For a dataset-wide fill it is the dataset's
own camera list, offered by the dialog -- the panel's default is the cameras
that already carry masks, which for a gap fill is the set with nothing to add.
Neither answer is ever read out of the rendered buttons.

Those buttons are a projection of the panel's state, and reading the projection
back is what broke a dataset-wide fill twice over: they are rendered only while
a segmenter is picked, so with SAM3 off a scrape found none and read none as
"every camera"; and the Run tab's panel renders the same class, so its live
teleop camera was counted into a Data-tab job.

A browser test can only catch that in a flow someone thought to drive. This
one is a source rule, so a NEW module that scrapes the buttons fails here even
with no browser coverage of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"

#: The class the panel renders its camera buttons with. Its presence in the
#: renderer is asserted below, so renaming it cannot quietly retire this test.
BUTTON_CLASS = "overlays-cam-btn"

#: How else the same buttons can be selected. The first version of this rule
#: keyed on the class alone, and `querySelectorAll('[data-cam].on')` -- the same
#: elements, on the same two panels -- reintroduced the whole bug while every
#: assertion stayed green.
BUTTON_SELECTORS = (BUTTON_CLASS, "data-cam")

#: The one module that owns the selection and renders it.
OWNER = "overlays.js"

#: The accessor the owner publishes, and the only supported read.
ACCESSOR = "dataQuery"


def _js_sources() -> dict[str, str]:
    """Every script the page can load, including subdirectories and `.mjs`.

    `glob("*.js")` was the first version and missed both: a module dropped in
    `static/lib/` or named `.mjs` escaped every rule below while the file's own
    docstring claimed otherwise.
    """
    return {
        str(p.relative_to(STATIC)): p.read_text()
        for p in sorted(STATIC.rglob("*.js")) + sorted(STATIC.rglob("*.mjs"))
    }


def _reads_the_lit_buttons(src: str) -> list[str]:
    """DOM queries that select the panel's camera buttons by their lit class.

    Keyed on both identifiers the buttons carry -- the class and `data-cam` --
    and on either order, because `.on.overlays-cam-btn` and
    `querySelectorAll('[data-cam].on')` select exactly the same elements as the
    original scrape did. A bare `data-cam` is deliberately not enough: the run
    view and the training view use that attribute for their own camera tiles.
    """
    src = _without_comments(src)
    found = [
        q
        for q in re.findall(r"(?:querySelectorAll|querySelector|getElementsByClassName)\([^)]*\)", src)
        if any(t in q for t in BUTTON_SELECTORS) and re.search(r"[.'\"]on\b", q)
    ]
    # The same read, spelled as a filter over the rendered buttons.
    if any(t in src for t in BUTTON_SELECTORS):
        found += re.findall(r"classList\.contains\(\s*['\"]on['\"]\s*\)", src)
    return found


def _without_comments(src: str) -> str:
    """Source with `//` and `/* */` removed.

    A rule that greps raw source is satisfiable by prose: a comment mentioning
    `Overlays.dataQuery()` made the "resolves through the accessor" assertion
    pass over a function that did nothing of the kind.
    """
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?m)//.*$", "", src)


def test_the_renderer_still_uses_the_class_this_rule_is_written_about():
    """Vacuity guard: with the class renamed, every assertion below would pass
    for the wrong reason — nothing would mention it at all."""
    sources = _js_sources()
    assert BUTTON_CLASS in sources[OWNER], (
        f"{OWNER} no longer renders .{BUTTON_CLASS}; this rule is checking a class that "
        "no longer exists. Point it at the new one."
    )
    assert len(sources) > 20, "the static module scan found almost nothing — check STATIC"


def test_only_the_owner_mentions_the_camera_buttons():
    """Every other module must ask the panel, not the page."""
    sources = _js_sources()
    # Naming the class at all is already wrong outside the renderer; reading the
    # lit ones is wrong anywhere, including by `[data-cam].on`.
    offenders = {
        name
        for name, src in sources.items()
        if name != OWNER and (BUTTON_CLASS in _without_comments(src) or _reads_the_lit_buttons(src))
    }
    assert not offenders, (
        f"{', '.join(sorted(offenders))} reaches for the camera buttons in the DOM. The rendered "
        f"buttons are a projection of the panel's selection, not the selection: they exist only "
        f"while a segmenter is picked, and BOTH the Data and Run panels render this class. "
        f"Ask window.Overlays.{ACCESSOR}() instead."
    )


def test_the_owner_only_writes_the_projection_it_renders():
    """The owner renders the buttons and wires their clicks; it must not read the
    selection back out of them either — its own Set is the state."""
    read_back = _reads_the_lit_buttons(_js_sources()[OWNER])
    assert not read_back, (
        f"{OWNER} reads its own rendered buttons back ({read_back}); the selection Set is the state."
    )


def test_the_job_prefers_the_list_its_caller_showed(_stream=None):
    """``selectedCams(explicit)`` must take an explicit list first.

    The dialog offers every camera of the dataset and the operator narrows it;
    that choice is the promise the confirmation made, so nothing may re-resolve
    it. The panel's own selection remains the answer for callers that show no
    list of their own (the live preview), and every-camera the answer for a
    panel that holds none -- one function, one order, one place to read it."""
    stream = _without_comments(_js_sources()["overlay_stream.js"])
    body = re.search(r"function selectedCams\(explicit\)\s*\{(.+?)\n    \}", stream, re.S)
    assert body, "selectedCams(explicit) is gone; the job resolves cameras somewhere else now"
    statements = [ln.strip() for ln in body.group(1).splitlines() if ln.strip()]
    first = statements[0] if statements else ""
    assert "explicit" in first, (
        f"selectedCams no longer prefers its caller's list first (`{first.strip()}`): the dialog "
        "showed a set of cameras and the job would run a different one."
    )
    assert f"Overlays.{ACCESSOR}" in body.group(1), (
        f"selectedCams no longer falls back to the panel through Overlays.{ACCESSOR}()"
    )


def test_the_dialog_offers_the_datasets_cameras_and_hands_that_list_to_the_job():
    """The whole point of the picker: the dialog must NOT inherit the panel's
    selection (the panel defaults to the cameras that already have masks -- the
    ones with nothing to fill), and the list it showed must travel to the job by
    value rather than being worked out again when OK is pressed."""
    dialog = _js_sources()["feature_editing.js"]
    picker = re.search(r"const allCams = ([^;]+);\s*\n\s*const chosen = new Set\(allCams\);", dialog)
    assert picker, "the fill-gaps dialog no longer builds its camera choice from the dataset"
    expr = picker.group(1).strip()
    assert "camera_keys" in expr, (
        f"the dialog's camera list comes from `{expr}` rather than the dataset's cameras; "
        "a fill must offer the cameras with gaps."
    )
    # Mentioning the dataset is not enough: `q.cameras.length ? q.cameras : ds.camera_keys`
    # mentions it too and is exactly the inheritance this dialog exists to stop.
    # `camerasForJob` was an accessor an earlier draft of this branch added for
    # the dialog and then deleted. It is named here so the reintroduction is
    # caught, not because it exists.
    borrowed = [t for t in ("q.cameras", ACCESSOR, "camerasForJob") if t in expr]
    assert not borrowed, (
        f"the dialog's camera list consults the panel ({', '.join(borrowed)}) in `{expr}`. "
        "The panel defaults to the cameras that ALREADY have masks -- the ones with nothing "
        "to fill -- which is the run this picker was added to prevent."
    )
    assert re.search(r"runFillGaps\([^)]*\[\.\.\.chosen\]\s*\)", dialog), (
        "the operator's ticks are not what reaches runFillGaps; the dialog's promise and the "
        "job's request can differ again."
    )
    assert re.search(r"^\s*cameras,\s*$", dialog, re.M), (
        "runFillGaps no longer forwards its cameras into the job request"
    )


def test_the_dialog_never_reads_the_panels_camera_list_at_all():
    """Not just in the assignment: anywhere in the function.

    Checking only the `allCams = ...` expression follows no indirection, and one
    local helper defeats it --

        const panelCams = () => (q.cameras && q.cameras.length) ? q.cameras : null;
        const allCams = panelCams() || ds.camera_keys || [];

    -- which is the inheritance this dialog exists to stop, spelled in two lines
    instead of one. The dialog legitimately reads the panel's `computeMs` for its
    time estimate, so the panel query itself stays; its camera list is what must
    never be consulted.
    """
    src = _without_comments(_js_sources()["feature_editing.js"])
    body = re.search(r"async function openFillGaps\(\)\s*\{(.+?)\n    \}\n", src, re.S)
    assert body, "openFillGaps is gone or was renamed; this rule now checks nothing"
    reads = re.findall(r"\bq\.cameras\b|dataQuery\(\)[^\n]*\.cameras\b", body.group(1))
    assert not reads, (
        f"the fill-gaps dialog reads the panel's camera list ({reads}). The panel defaults to "
        "the cameras that ALREADY have masks -- the ones with nothing to fill -- which is the "
        "run this dialog was rebuilt to prevent."
    )
    # Complement: the panel query is still reached for the estimate, so this rule
    # cannot pass merely because the dialog stopped talking to the panel entirely.
    assert "computeMs" in body.group(1), "the dialog no longer reads the panel at all; re-check this rule"
