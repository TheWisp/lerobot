"""Static contracts for the browser-wide live-camera transport selector."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STATIC = _ROOT / "src" / "lerobot" / "gui" / "static"


def test_selector_is_anchored_left_of_hf_without_entering_action_layout() -> None:
    index = (_STATIC / "index.html").read_text()
    style = (_STATIC / "style.css").read_text()

    account = index.split('<span class="tab-bar-account">', 1)[1].split("<!-- Transfers indicator", 1)[0]
    assert account.index('id="camera-video-control"') < account.index('id="hf-auth-indicator"')
    assert 'value="auto"' in account
    assert 'value="full-quality"' in account
    assert 'value="low-bandwidth"' in account

    account_rule = style.split(".tab-bar-account {", 1)[1].split("}", 1)[0]
    control_rule = style.split(".camera-video-control {", 1)[1].split("}", 1)[0]
    assert "margin-left: auto" in account_rule
    assert "position: relative" in account_rule
    assert "position: absolute" in control_rule
    assert "right: 100%" in control_rule
    # Deliberately NOT asserting a narrow-width hide rule: it existed, and was
    # removed because hiding the control below 1320px made it unreachable in
    # half-screen layouts. Anchoring, not visibility, is what this test guards.


def test_mode_is_browser_local_and_emits_one_shared_change_event() -> None:
    app = (_STATIC / "app.js").read_text()

    assert "lerobot.cameraVideoMode" in app
    assert "localStorage.getItem(STORAGE_KEY)" in app
    assert "localStorage.setItem(STORAGE_KEY, value)" in app
    assert "fetch('/api/run/camera-video-mode'" in app
    assert "new CustomEvent('camera-video-mode-change'" in app
    assert "window.CameraVideoMode = CameraVideoMode" in app


def test_live_camera_consumers_follow_the_shared_effective_mode() -> None:
    run = (_STATIC / "run.js").read_text()
    robot = (_STATIC / "robot.js").read_text()

    assert "function _effectiveCameraVideoMode" in run
    assert "cameraVideoMode === 'low-bandwidth'" in run
    assert "window.addEventListener('camera-video-mode-change'" in run
    assert "video_mode=${encodeURIComponent(videoMode)}" in robot
    assert "window.addEventListener('camera-video-mode-change'" in robot
