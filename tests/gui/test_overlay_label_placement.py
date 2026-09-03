# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Where the live overlay puts a detection's name, and how big it is.

Reported as "a weird large font (indicator in the camera cards) that stacks on
top of existing ones ... it should use the same style and prune the overlap".

Two faults with one cause: a name can be drawn by the worker (here) or by the
stored-mask layer in the browser, and the two sized it differently and neither
avoided the other's pixels. Objects near the top edge are the common case for
the stacking -- every label clamps to the same y, so only the last one drawn
stays readable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from lerobot.overlays.standalone import _draw_labels, _free_slot, label_font_px

PIL = pytest.importorskip("PIL", reason="label drawing needs Pillow")


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ── size ────────────────────────────────────────────────────────────────────


def test_the_label_scales_with_the_frame():
    assert label_font_px(720) == int(720 * 0.032)
    assert label_font_px(240) == 14, "small frames get the legibility floor"


def test_a_taller_frame_never_gets_a_smaller_label():
    sizes = [label_font_px(h) for h in (120, 240, 480, 720, 1080, 2160)]
    assert sizes == sorted(sizes), sizes


def test_the_browser_sizes_a_label_the_same_way():
    """The two layers that can draw a name must agree, or it changes size when
    the segmenter is toggled over the same frame. Compared by running the JS
    rule rather than by restating it here, which would just be the same
    constant typed twice.
    """
    js = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static" / "masks.js"
    heights = [120, 240, 361, 480, 720, 1080, 2160]
    script = (
        "global.window={addEventListener(){}};global.document={addEventListener(){}};"
        f"require({str(js)!r});"
        f"console.log(JSON.stringify({heights}.map(window.MaskOverlay.labelFontPx)));"
    )
    out = subprocess.run(  # noqa: S603
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,  # noqa: S607
    )
    assert out.returncode == 0, out.stdout + out.stderr
    import json

    assert json.loads(out.stdout) == [label_font_px(h) for h in heights]


# ── placement ───────────────────────────────────────────────────────────────


def test_a_label_that_collides_slides_clear():
    first = (10, 0, 100, 20)
    moved = _free_slot((10, 0, 100, 20), [first], 720, 20)
    assert not _overlap(moved, first), moved


def test_a_label_with_room_is_left_alone():
    """The complement: "always slides down" would pass the test above while
    moving every label away from the object it names."""
    box = (10, 300, 100, 320)
    assert _free_slot(box, [(200, 300, 300, 320)], 720, 20) == box
    assert _free_slot(box, [], 720, 20) == box


def test_a_label_is_not_slid_out_of_the_frame():
    box = (10, 700, 100, 719)
    assert _free_slot(box, [box], 720, 20) == box


def test_names_drawn_at_the_top_edge_do_not_land_on_each_other():
    """The reported case, through the drawing function rather than the helper:
    three objects whose tops are all at y=0, which is where the clamp used to
    put every pill on the same pixels."""
    rgb = np.zeros((360, 640, 3), np.uint8)
    labels = [("green ring", (12, 0)), ("yellow block", (14, 0)), ("light green cube", (16, 0))]

    placed: list[tuple] = []
    for text, at in labels:
        rgb, rects = _draw_labels(rgb, [(text, at)], (200, 200, 200), taken=placed)
        placed.extend(rects)

    assert len(placed) == 3
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            assert not _overlap(a, b), f"pills overlap: {a} vs {b}"


def test_every_name_still_reaches_the_frame():
    """Sliding must not be a way of quietly dropping a detection's name."""
    rgb = np.zeros((360, 640, 3), np.uint8)
    before = rgb.copy()
    placed: list[tuple] = []
    for text in ("one", "two", "three"):
        rgb, rects = _draw_labels(rgb, [(text, (10, 0))], (255, 255, 255), taken=placed)
        placed.extend(rects)
    assert (rgb != before).any(), "nothing was drawn at all"
    for x0, y0, x1, y1 in placed:
        assert 0 <= y0 < rgb.shape[0] and 0 <= x0 < rgb.shape[1], (x0, y0, x1, y1)
