"""The write rule runs client-side, so its test runs under node.

A run may fill a (frame, label) only where the label is ABSENT. Filtering on the
client keeps the request proportional to what the run produced rather than to
the episode's existing coverage, and keeps the decision on the side that already
holds that coverage. See apply_run_filter.test.js for the assertions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_apply_run_filter_js():
    test_js = Path(__file__).parent / "apply_run_filter.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_page_loads_the_filter():
    """A module that is tested and never loaded is worse than one that ships
    untested: the green test reads as coverage."""
    index = (STATIC / "index.html").read_text()
    assert "apply_run_filter.js" in index, "index.html does not load apply_run_filter.js"


def test_the_drain_files_against_the_run_s_episode_not_the_one_in_view():
    """A scrub during an Apply run must not throw the run's work away.

    The run captures its episode when it starts and publishes frames for that
    episode. The drain used to re-read `window.currentEpisode` on every call, so
    scrubbing elsewhere made the filter compare the run's frames against the
    episode now in VIEW: every frame was dropped, nothing was stored, and since
    the lock-step wait only ends when a drained frame comes back, each frame then
    burned the full 8 s deadline. The run looked alive and wrote nothing.

    Checked at the source: the fix is which value is passed in, and the browser
    test that would catch it needs a loaded segmenter and a played episode.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[2] / "src/lerobot/gui/static/overlays.js").read_text()
    assert re.search(r"async function stageDrained\(\s*runDs\s*,\s*runEp\s*\)", src), (
        "stageDrained no longer takes the run's own dataset/episode"
    )
    calls = re.findall(r"stageDrained\(([^)]*)\)", src)
    invocations = [c.strip() for c in calls if "runDs" not in c]
    assert invocations, "no stageDrained call sites found — the pattern moved"
    for c in invocations:
        assert c == "ds, ep", (
            f"stageDrained called as stageDrained({c}) — it must be given the run's "
            "captured episode, or a scrub mid-run discards everything it computes"
        )


def test_leaving_the_episode_ends_the_apply_run():
    """An Apply run is one episode's work -- the design calls it "the frames you
    watch". Filing its output against the right episode (the test above) stops
    masks landing in the wrong place, but on its own it leaves the run
    segmenting an episode nobody is looking at any more.

    So selecting another episode ends the run, and the panel has to be told:
    `app.js` already notified the Inspector on episode change and had no
    equivalent for the overlay panel, which is why nothing could react.
    """
    import pathlib

    static = pathlib.Path(__file__).resolve().parents[2] / "src/lerobot/gui/static"
    app = (static / "app.js").read_text()
    panel = (static / "overlays.js").read_text()

    assert "Overlays?.onEpisodeSelected?.(" in app, (
        "app.js does not tell the overlay panel the episode changed; an Apply run "
        "will keep writing the episode the operator has navigated away from"
    )
    assert "onEpisodeSelected: () => _applyEpisodeHook?.()" in panel, (
        "the panel exports no episode-change hook for app.js to call"
    )
    assert "stopApplyRun();" in panel.split("function onEpisodeChanged()", 1)[1][:400], (
        "the episode-change hook does not stop the run"
    )
    # The guard that turns a silent drop into a visible one.
    assert "dropped ${foreign} frame(s) not belonging to episode" in panel, (
        "frames from a foreign episode are dropped without a word; that silence is "
        "what made the original bug invisible"
    )
