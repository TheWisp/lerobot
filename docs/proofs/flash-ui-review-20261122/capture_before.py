import json
from pathlib import Path

import pytest

SOURCE = (Path(__file__).resolve().parents[3] / "src/lerobot/gui/static/flash_editor.js").read_text()
MUTANT = SOURCE.replace(
    "full.disabled=keys.length<2;\n    if(full.disabled)full.checked=true;", "full.disabled=!keys.length;"
).replace("checked=data.document.keyframes.length<2;", "checked=!data.document.keyframes.length;")


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    outcome = yield
    if fixturedef.argname == "gui_page":
        page = outcome.get_result()
        page.route(
            "**/static/flash_editor.js?*",
            lambda route: route.fulfill(body=MUTANT, content_type="text/javascript"),
        )
        page.reload()
        page.wait_for_function("typeof showFlashEditor === 'function'")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        page = item.funcargs["gui_page"]
        page.locator("#flash-editor-play").click()
        page.wait_for_function("flashEditor.player.metrics.wraps.length >= 2")
        metrics = page.evaluate(
            "({frames:flashEditor.player.metrics.painted.map(p=>p.frame),wraps:flashEditor.player.metrics.wraps.length})"
        )
        assert set(metrics["frames"]) == {2}, metrics
        output = item.funcargs["tmp_path"]
        (output / "before-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        page.locator("#flash-editor-workspace").screenshot(path=str(output / "before-playback.png"))
