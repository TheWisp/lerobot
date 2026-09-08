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
"""Which cameras the Data tab works on has ONE source: the overlay panel's own
selection, read through ``Overlays.dataQuery()``.

The rendered camera buttons are a projection of that state. Reading the
projection back is what broke a dataset-wide fill twice over: the buttons are
rendered only while a segmenter is picked, so with SAM3 off a scrape found
none and read none as "every camera"; and the Run tab's panel renders the same
class, so its live teleop camera was counted into a Data-tab job.

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

#: The one module that owns the selection and renders it.
OWNER = "overlays.js"

#: The accessor the owner publishes, and the only supported read.
ACCESSOR = "dataQuery"


def _js_sources() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(STATIC.glob("*.js"))}


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
    offenders = {name for name, src in _js_sources().items() if BUTTON_CLASS in src and name != OWNER}
    assert not offenders, (
        f"{', '.join(sorted(offenders))} reaches for .{BUTTON_CLASS} in the DOM. The rendered "
        f"buttons are a projection of the panel's selection, not the selection: they exist only "
        f"while a segmenter is picked, and BOTH the Data and Run panels render this class. "
        f"Ask window.Overlays.{ACCESSOR}() instead."
    )


def test_the_owner_only_writes_the_projection_it_renders():
    """The owner renders the buttons and wires their clicks; it must not read the
    selection back out of them either — its own Set is the state."""
    src = _js_sources()[OWNER]
    # A read-back looks like a query for the "on" ones, in any quote style.
    read_back = re.findall(rf"""querySelectorAll\(\s*['"`][^'"`]*{BUTTON_CLASS}\.on""", src)
    assert not read_back, (
        f"{OWNER} reads its own rendered buttons back ({read_back}); the selection Set is the state."
    )


def test_the_job_resolves_cameras_through_the_accessor():
    """The pass and the dialog that confirms it must reach the same resolver;
    two computations over one state is how they came to disagree."""
    stream = _js_sources()["overlay_stream.js"]
    assert f"Overlays.{ACCESSOR}" in stream, (
        f"overlay_stream.js no longer resolves cameras through Overlays.{ACCESSOR}() — "
        "the job and the dialog can disagree again."
    )
    assert "camerasForJob" in stream, "the dialog's read of what the job would run on is gone"

    dialog = _js_sources()["feature_editing.js"]
    assert "camerasForJob" in dialog, (
        "the fill-gaps dialog no longer asks what the job would run on; it is resolving "
        "the camera list a second time."
    )


def test_the_dialog_adds_no_fallback_of_its_own():
    """The resolver already answers "every camera" for a panel with no selection.
    A fallback chained onto its result is a second answer: an empty list from the
    resolver would show as every camera in the dialog while the job ran none --
    the exact disagreement this whole rule is about.

    Checked on the assignment rather than on the call, because the call is
    usually held in a local first: a rule that grepped for the accessor name
    passed happily while `resolve() || ds.camera_keys` sat one line below it.
    """
    src = _js_sources()["feature_editing.js"]
    assignments = re.findall(r"const dsCams\s*=\s*([^;]+);", src)
    assert assignments, "the fill-gaps dialog no longer resolves a camera list into dsCams"
    for expr in assignments:
        assert "||" not in expr, (
            f"the fill-gaps dialog falls back on its own: `dsCams = {expr.strip()}`. "
            "One resolver, one answer: let it answer, or refuse to open."
        )
