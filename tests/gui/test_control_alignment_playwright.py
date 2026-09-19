# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Controls that share a row are the same size, and a tile's chrome is one set.

Each of these controls used to carry its own padding and its own font, so
neighbours came out different heights: Play stood taller than the speed picker
beside it, and the dataset search box -- which inherited the body's 16px --
stood a third taller than every row it filters.

The camera tile is the same complaint in another form. Its name was a header
row above the picture while its enlarge button floated inside the picture, so
the two halves of one control set read as two mechanisms, and a four-camera
grid spent a row of height per tile on labels.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

# Renders a tile grid without needing a dataset on disk: the grid builder reads
# camera_keys off the opened-dataset record and nothing else.
RENDER_TILES = """
() => {
  const id = '/tmp/fake';
  datasets[id] = {camera_keys: ['observation.images.front', 'observation.images.top'], fps: 30};
  currentDataset = id; currentEpisode = 0; currentFrame = 0; totalFrames = 10;
  renderCameraGrid();
}
"""


def _box(page, selector: str) -> dict:
    box = page.locator(selector).first.bounding_box()
    assert box is not None, f"{selector} is not laid out"
    return box


def test_the_transport_row_is_one_height(gui_page):
    play = _box(gui_page, "#play-btn")
    speed = _box(gui_page, "#speed-select")
    assert abs(play["height"] - speed["height"]) <= 1, (
        f"Play is {play['height']}px and the speed picker is {speed['height']}px"
    )


def test_the_transport_row_shares_a_baseline(gui_page):
    """Equal heights are not enough on their own -- two equal boxes can still
    sit at different offsets in the row."""
    play = _box(gui_page, "#play-btn")
    speed = _box(gui_page, "#speed-select")
    assert abs(play["y"] - speed["y"]) <= 1


def test_the_transport_labels_do_not_wrap(gui_page):
    """The buttons have a fixed height now, so a squeezed row clips a wrapped
    label rather than growing to fit it."""
    gui_page.set_viewport_size({"width": 900, "height": 900})
    gui_page.wait_for_timeout(200)
    play = gui_page.locator("#play-btn")
    line_height, height = gui_page.evaluate(
        "() => { const el = document.getElementById('play-btn');"
        "  const cs = getComputedStyle(el);"
        "  return [parseFloat(cs.lineHeight) || parseFloat(cs.fontSize), el.scrollHeight]; }"
    )
    assert play.bounding_box()["height"] >= height, (
        f"the Play label wrapped: content is {height}px inside a {play.bounding_box()['height']}px button"
    )
    assert line_height > 0


def test_the_dataset_filter_controls_are_one_height(gui_page):
    search = _box(gui_page, "#dataset-search")
    sort = _box(gui_page, "#dataset-sort")
    favourites = _box(gui_page, "#dataset-favorites-only")
    heights = [search["height"], sort["height"], favourites["height"]]
    assert max(heights) - min(heights) <= 1, (
        f"search {search['height']}px, sort {sort['height']}px, favourites {favourites['height']}px"
    )


def test_the_dataset_filter_controls_are_not_larger_than_what_they_filter(gui_page):
    """They inherited the body's 16px through `font: inherit`, which made the
    search box larger than every dataset row beneath it."""
    control_px, row_px = gui_page.evaluate(
        "() => [parseFloat(getComputedStyle(document.getElementById('dataset-search')).fontSize),"
        "       parseFloat(getComputedStyle(document.body).fontSize)]"
    )
    assert control_px <= 13, f"the search box is still {control_px}px"
    assert control_px < row_px, "the control is still taking the body font size"


def test_a_tile_carries_its_name_and_its_enlarge_button_as_one_chip_family(gui_page):
    gui_page.evaluate(RENDER_TILES)
    gui_page.wait_for_selector(".camera-panel .camera-title", timeout=5_000)

    inside = gui_page.evaluate(
        "() => [...document.querySelectorAll('#camera-grid .camera-title')]"
        "        .every((el) => el.closest('.camera-frame') !== null)"
    )
    assert inside, "a tile's name is still a row of its own above the picture"

    shared = gui_page.evaluate(
        "() => [...document.querySelectorAll('#camera-grid .camera-title, #camera-grid .obs-cam-zoom')]"
        "        .every((el) => el.classList.contains('camera-chip'))"
    )
    assert shared, "the name and the enlarge button are not drawn from the same class"


def test_a_tiles_name_and_enlarge_button_sit_on_the_same_line(gui_page):
    gui_page.evaluate(RENDER_TILES)
    gui_page.wait_for_selector(".camera-panel .camera-title", timeout=5_000)
    name = _box(gui_page, "#camera-grid .camera-panel .camera-title")
    zoom = _box(gui_page, "#camera-grid .camera-panel .obs-cam-zoom")
    assert abs(name["y"] - zoom["y"]) <= 1, "the two chips are at different heights in the tile"
    assert name["x"] < zoom["x"], "the name should be on the leading edge, the enlarge button trailing"


def test_only_the_enlarge_button_is_drawn_as_a_control(gui_page):
    """A border is what this UI uses to say "this is a control" -- the
    visualizer's ghost toggle carries one and its robot-name label does not. The
    tile's name is a label; drawn with a border it read as a button that does
    nothing when pressed."""
    gui_page.evaluate(RENDER_TILES)
    gui_page.wait_for_selector(".camera-panel .camera-title", timeout=5_000)
    name_border, zoom_border = gui_page.evaluate(
        "() => ['#camera-grid .camera-panel .camera-title',"
        "       '#camera-grid .camera-panel .obs-cam-zoom']"
        "  .map((sel) => getComputedStyle(document.querySelector(sel)).borderTopWidth)"
    )
    assert name_border == "0px", f"the tile name still draws a border ({name_border})"
    assert zoom_border != "0px", "the enlarge button lost the border that marks it a control"


def test_a_click_on_the_tiles_name_still_reaches_the_tile(gui_page):
    """The name overlays the frame, and the frame is a click target in its own
    right while segmentation is armed. It does not need `pointer-events: none`
    to stay out of the way -- the gesture resolves with `closest()`, which finds
    the tile from the label just as well -- and having them keeps the `title`
    reachable, which is the only way to read an ellipsised camera key."""
    gui_page.evaluate(RENDER_TILES)
    gui_page.wait_for_selector(".camera-panel .camera-title", timeout=5_000)
    resolved = gui_page.evaluate(
        "() => document.querySelector('#camera-grid .camera-title')"
        "        .closest('[data-cam-cell]')?.dataset.camCell"
    )
    assert resolved == "observation.images.front"
    assert (
        gui_page.evaluate(
            "() => getComputedStyle(document.querySelector('#camera-grid .camera-title')).pointerEvents"
        )
        != "none"
    ), "an element that cannot be hovered can never show its title"


def test_the_tiles_name_is_drawn_under_the_overlay_it_would_otherwise_hide(gui_page):
    """Detection labels are drawn clamped to the top edge of the frame, which is
    where the name plate sits. The plate is opaque, so painting it above them hid
    the very thing the Data tab exists to review."""
    gui_page.evaluate(RENDER_TILES)
    gui_page.wait_for_selector(".camera-panel .camera-title", timeout=5_000)
    name_z, overlay_z = gui_page.evaluate(
        "() => ['#camera-grid .camera-panel .camera-title',"
        "       '#camera-grid .camera-panel .overlay-layer']"
        "  .map((sel) => parseInt(getComputedStyle(document.querySelector(sel)).zIndex, 10))"
    )
    assert name_z < overlay_z, f"the name plate ({name_z}) paints over the overlay ({overlay_z})"
