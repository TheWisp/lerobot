# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Switching episodes must not cold-boot the URDF visualizer.

``renderCameraGrid`` used to rebuild the whole observation grid with
``grid.innerHTML = …`` on every ``selectEpisode``. That destroys the URDF
``<iframe>``, and the replacement re-downloads and re-parses the URDF and all
its meshes, resets the orbit camera to the hardcoded default pose, and throws
away the iframe's per-episode trajectory cache — once per episode click, for a
scene that is a property of the dataset's motor set and cannot change between
episodes of one dataset.

These tests count the *document loads* of ``urdf_viz.html``, which is what the
operator actually sees as "it reloaded", and pin the identity of the iframe
element across the switch. The robot half of the grid signature is covered too:
a dataset on a different arm must still rebuild, or the viewer would keep
showing the previous robot.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Stubs the parent page's probe of ``…/urdf-viz/meta`` so the test does not need
# a real on-disk dataset whose motor set resolves to a vendored description —
# the behaviour under test is the parent's grid lifecycle, not robot
# resolution (tests/gui/test_urdf_viz.py owns that). Every other request,
# including the iframe's own document load, goes to the real server.
_STUB_META = """
window.__urdfMetaRobot = 'SO-101';
window.fetch = new Proxy(window.fetch, { apply: (t, self, args) => {
  const url = String(args[0]);
  if (url.includes('/urdf-viz/meta')) {
    return Promise.resolve(new Response(
      JSON.stringify({available: true, name: window.__urdfMetaRobot,
                      urdf: '/urdf-assets/nope.urdf', urdf_right: null,
                      base_offsets: null, bimanual: false,
                      sources: ['state'], ee_link: null}),
      {status: 200, headers: {'Content-Type': 'application/json'}}));
  }
  return Reflect.apply(t, self, args);
}});
"""

# Renders the grid for one episode without dragging in renderTree / trim
# loading / the overlays panel, none of which bear on the iframe's lifetime.
_RENDER = """
([dsId, epIdx]) => {
  datasets[dsId] = datasets[dsId] || {camera_keys: ['observation.images.top'], fps: 30};
  currentDataset = dsId;
  currentEpisode = epIdx;
  currentFrame = 0;
  totalFrames = 10;
  renderCameraGrid();
}
"""


@pytest.fixture
def page():
    from lerobot.gui import server as gui_server_mod

    port = _free_port()
    config = uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import requests

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(base_url, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        pytest.fail("GUI server did not come up")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto(base_url)
        pg.wait_for_function("typeof renderCameraGrid === 'function'", timeout=10_000)
        pg.evaluate(_STUB_META)
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def _viz_loads(page) -> list[str]:
    """Document loads of the visualizer, newest last. Reset per test by the fixture."""
    return page.evaluate("window.__vizLoads || []")


def _track_loads(page) -> None:
    page.evaluate("window.__vizLoads = []")
    page.on(
        "request",
        lambda r: page.evaluate("u => window.__vizLoads.push(u)", r.url)
        if "urdf_viz.html" in r.url
        else None,
    )


def _render(page, dataset_id: str, episode_idx: int) -> None:
    page.evaluate(_RENDER, [dataset_id, episode_idx])
    # The probe is async; the tile is only settled once it has stamped the
    # resolved signature onto the grid.
    page.wait_for_function(
        "document.getElementById('camera-grid').dataset.tileSig?.endsWith('SO-101')",
        timeout=10_000,
    )


def test_episode_switch_keeps_the_same_iframe(page):
    """The regression: the iframe element must survive an episode change."""
    _track_loads(page)
    _render(page, "acme/pick", 0)
    page.evaluate("document.getElementById('urdf-viz-iframe').__marker = 'original'")

    _render(page, "acme/pick", 1)

    marker = page.evaluate("document.getElementById('urdf-viz-iframe')?.__marker")
    assert marker == "original", (
        "the URDF iframe was replaced on an episode switch — the grid was rebuilt, "
        "so the viewer re-downloads its meshes and resets the orbit camera"
    )


def test_episode_switch_does_not_reload_the_visualizer_document(page):
    """What the operator sees: the visualizer must load exactly once."""
    _track_loads(page)
    _render(page, "acme/pick", 0)
    page.wait_for_timeout(300)
    after_first = len(_viz_loads(page))
    assert after_first == 1, f"expected one initial visualizer load, saw {after_first}"

    for episode in (1, 2, 3):
        _render(page, "acme/pick", episode)
    page.wait_for_timeout(300)

    loads = _viz_loads(page)
    assert len(loads) == 1, (
        f"the visualizer document reloaded {len(loads) - 1} extra time(s) across three "
        f"episode switches; it is resolved from the dataset's motor set and cannot "
        f"change between episodes"
    )


def test_camera_tiles_are_not_recreated_either(page):
    """The camera <img> elements share the grid's fate; loadAllFrames rewrites their srcs."""
    _track_loads(page)
    _render(page, "acme/pick", 0)
    page.evaluate("document.getElementById('frame-observation-images-top').__marker = 'original'")

    _render(page, "acme/pick", 1)

    assert page.evaluate("document.getElementById('frame-observation-images-top')?.__marker") == (
        "original"
    ), "camera tiles were rebuilt on an episode switch"


def test_clearing_the_grid_forces_the_next_render_to_rebuild(page):
    """Closing the open dataset must not leave a signature that suppresses the rebuild.

    ``closeDataset`` puts the grid back to its empty state. If it does that with a
    direct ``innerHTML`` write, the tile signature stays stamped, and re-opening
    the same dataset matches it — the operator gets "Select an episode to view"
    where the tiles belong.
    """
    _track_loads(page)
    _render(page, "acme/pick", 0)

    # The real close path, with only its DELETE stubbed.
    page.evaluate(
        "window.fetch = new Proxy(window.fetch, { apply: (t, self, args) => {"
        "  if (String(args[0]).includes('/api/datasets/') && args[1] && args[1].method === 'DELETE')"
        "    return Promise.resolve(new Response('{}', {status: 200}));"
        "  return Reflect.apply(t, self, args);"
        "}})"
    )
    page.evaluate("episodes['acme/pick'] = []")
    page.evaluate("closeDataset('acme/pick', {stopPropagation() {}})")
    page.wait_for_selector("#camera-grid .empty-state", timeout=10_000)

    _render(page, "acme/pick", 0)
    assert page.evaluate("document.getElementById('urdf-viz-iframe') !== null"), (
        "re-opening the dataset left the empty state in place — the stale tile "
        "signature suppressed the rebuild"
    )


def test_a_different_robot_still_rebuilds(page):
    """The grid signature keys on the robot, not just the camera set.

    Two datasets can share a camera layout and be recorded on different arms.
    Reusing the tile there would leave the previous robot's URDF on screen.
    """
    _track_loads(page)
    _render(page, "acme/pick", 0)
    page.evaluate("document.getElementById('urdf-viz-iframe').__marker = 'original'")

    page.evaluate("window.__urdfMetaRobot = 'SO-107'")
    page.evaluate(_RENDER, ["acme/other-arm", 0])
    page.wait_for_function(
        "document.getElementById('camera-grid').dataset.tileSig?.endsWith('SO-107')",
        timeout=10_000,
    )
    page.wait_for_timeout(300)

    assert page.evaluate("document.getElementById('urdf-viz-iframe')?.__marker") is None, (
        "the tile was reused for a dataset on a different arm — it would still be "
        "showing the previous robot's URDF"
    )
    assert len(_viz_loads(page)) == 2, "the visualizer did not reload for the new robot"
