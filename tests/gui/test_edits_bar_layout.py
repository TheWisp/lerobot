"""The pending-edits bar must not reflow the timeline when it appears.

As an ordinary flex child of `.main` it took its 43px the moment the first edit
was staged, shifting every timeline row up by that much. Measured on the data
view at 1700x1050, before and after staging one edit:

    masks.front       938 -> 895
    masks.left_wrist  976 -> 933

so screen y=948 was inside `masks.front` before and inside `masks.left_wrist`
after. Editing a mask segment is click-to-toggle and toggling back is the
documented undo, so "click the bar, then click it again" is a normal gesture --
and the second click landed on the next row down, where it is not a toggle at
all but a seek, which also discarded the range selection. Nothing reported an
error.

This is asserted against the stylesheet because the defect is a layout property:
no unit test of the JS could see it, and the browser check that found it does
not run in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STYLE = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static" / "style.css"


def _rule(selector: str) -> str:
    """The declarations of the first rule whose selector matches exactly."""
    css = STYLE.read_text()
    m = re.search(rf"(?:^|\}}|\*/)\s*{re.escape(selector)}\s*\{{(.*?)\}}", css, re.S)
    assert m, f"no rule for {selector} in style.css"
    return m.group(1)


def test_the_edits_bar_is_out_of_flow():
    """Out of flow is what stops it displacing anything when it appears."""
    decls = _rule(".edits-bar")
    assert re.search(r"position:\s*(absolute|fixed)", decls), (
        "the edits bar is back in normal flow, so showing it shifts every timeline row "
        f"up by its own height: {decls.strip()[:120]}"
    )
    # Pinned to the bottom edge, or "out of flow" is satisfied by a bar floating
    # somewhere unrelated.
    assert re.search(r"bottom:\s*0", decls), decls


def test_the_bars_container_can_position_it():
    """`position: absolute` resolves against the nearest positioned ancestor.

    Without this the bar escapes to the viewport, which looks the same on a
    maximised window and wrong everywhere else.
    """
    assert re.search(r"position:\s*relative", _rule(".main")), (
        ".main is not a positioning context, so the absolutely positioned edits bar "
        "would resolve against the viewport instead"
    )


def test_the_scroll_area_reserves_room_for_it():
    """The complement: out of flow alone would let the bar cover the last row.

    Padding on a scroll container extends what can be scrolled to rather than
    moving content, so this reserves the space without reintroducing the shift.
    """
    decls = _rule(".feature-rows")
    m = re.search(r"padding-bottom:\s*(\d+)px", decls)
    assert m, f".feature-rows reserves no room for the overlaid bar: {decls.strip()[:120]}"
    # The bar measured 43px; the reservation has to cover it, not merely exist.
    assert int(m.group(1)) >= 43, f"reserved {m.group(1)}px for a 43px bar"


@pytest.mark.parametrize("selector", [".edits-bar", ".main", ".feature-rows"])
def test_the_rules_this_depends_on_exist(selector: str):
    """A renamed selector would make every assertion above vacuous."""
    assert _rule(selector).strip(), selector
