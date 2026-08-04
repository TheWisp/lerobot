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

from pathlib import Path

_STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"


def test_camera_errors_render_in_per_device_preview_placeholders() -> None:
    robot_js = (_STATIC_DIR / "robot.js").read_text()
    style_css = (_STATIC_DIR / "style.css").read_text()

    assert "camera-preview-placeholder camera-preview-error" in robot_js
    assert "Camera access denied" in robot_js
    assert 'role="status"' in robot_js
    assert "cam.error ? String(cam.error) : ''" in robot_js
    assert 'title="${esc(error)}"' in robot_js
    assert "camera-preview-status" in robot_js
    assert "cam.error_action" in robot_js
    assert "cam.error_command" in robot_js
    assert "cam.error_code" in robot_js
    assert "seenRemediations" in robot_js
    assert "function copyCameraCommand(button)" in robot_js
    assert "data-command=" in robot_js
    assert ".camera-preview-error" in style_css
    assert ".camera-preview-status" in style_css
    assert ".camera-remediation" in style_css
    assert ".camera-command-copy" in style_css
    assert "background: rgba(0,0,0,0.6)" in style_css
    status_rule = style_css.split(".camera-preview-status {", 1)[1].split("}", 1)[0]
    assert "cursor:" not in status_rule


def test_camera_errors_are_not_polled_or_assignable() -> None:
    robot_js = (_STATIC_DIR / "robot.js").read_text()

    assert "!cam.error && !cam.preview_error_pending && Number.isInteger(cam.preview_index)" in robot_js
    assert (
        "if (cam.error || cam.preview_error_pending || !Number.isInteger(cam.preview_index)) return;"
        in robot_js
    )
    assert "camera-frame/${cam.preview_index}" in robot_js
    assert "if (detectedCameras[cameraIndex].error) return;" in robot_js


def test_late_frame_failures_become_stable_error_placeholders() -> None:
    robot_js = (_STATIC_DIR / "robot.js").read_text()

    assert 'src=""' not in robot_js
    assert 'onload="handleCameraPreviewLoaded(${i})"' in robot_js
    assert 'onerror="handleCameraPreviewError(${i})"' in robot_js
    assert "img.dataset.loading !== 'true'" in robot_js
    assert "img.dataset.loading = 'true'" in robot_js
    assert "function handleCameraPreviewLoaded(cameraIndex)" in robot_js
    assert "if (img) img.dataset.loading = 'false'" in robot_js
    assert "async function handleCameraPreviewError(cameraIndex)" in robot_js
    assert "!img.getAttribute('src')" in robot_js
    assert "cam.preview_error_pending = true" in robot_js
    assert "payload?.detail" in robot_js
    assert "cam.error = message" in robot_js
    assert "cam.error_code = 'preview_failed'" in robot_js
    assert "cam.error_summary = 'Preview unavailable'" in robot_js
    assert "cam.error_action = 'Select Detect Cameras to reopen the preview." in robot_js
    assert "renderCameraPreview();" in robot_js
