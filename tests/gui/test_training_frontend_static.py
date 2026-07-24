# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

_STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"


def test_training_log_has_copy_control() -> None:
    training_js = (_STATIC_DIR / "training.js").read_text()

    assert 'class="btn-small secondary training-log-copy"' in training_js
    assert 'logBtn.closest(".training-card")?.querySelector(".training-log")' in training_js
    assert '"Training log copied to clipboard"' in training_js


def test_training_log_copy_has_insecure_context_fallback() -> None:
    training_js = (_STATIC_DIR / "training.js").read_text()

    assert "navigator.clipboard?.writeText" in training_js
    assert 'document.execCommand("copy")' in training_js


def test_hvla_progress_has_live_log_compatibility_fallback() -> None:
    training_js = (_STATIC_DIR / "training.js").read_text()

    assert "function trainingProgressFromLog(text)" in training_js
    assert "function trainingLegacySamplesFromLog(text)" in training_js
    assert "function trainingEtaFromSamples(samples)" in training_js
    assert "function trainingMetricSeries(snap)" in training_js
    assert r"\bstep\s+(\d+)\s*\/\s*(\d+)\b" in training_js
    assert "logProgress?.step" in training_js
    assert "logProgress?.eta_seconds" in training_js
    assert "const metricsSeries = trainingMetricSeries(snap)" in training_js
    assert "const series = trainingMetricSeries(snap)" in training_js


def test_training_dashboard_exposes_checkpoint_resume() -> None:
    training_js = (_STATIC_DIR / "training.js").read_text()
    model_js = (_STATIC_DIR / "model.js").read_text()

    assert "async function trainingResumeRun(runId, checkpointStep)" in training_js
    assert "/api/training/runs/${runId}/resume" in training_js
    assert "This creates a new run and keeps the source checkpoint unchanged." in training_js
    assert "Resumed from" in training_js
    assert "snap.resumable_checkpoint_steps || []" in training_js
    assert "ckpt.has_training_state && run.run_id" in model_js
    assert "trainingResumeRun('${run.run_id}', ${ckpt.step})" in model_js


def test_training_terminal_states_have_accessible_explanations() -> None:
    training_js = (_STATIC_DIR / "training.js").read_text()
    style_css = (_STATIC_DIR / "style.css").read_text()

    assert 'const TERMINAL_STATES = new Set(["completed", "stopped", "failed"])' in training_js
    assert 'completed: "Training ended cleanly' in training_js
    assert 'stopped: "Training ended before completion' in training_js
    assert 'failed: "Training could not complete' in training_js
    assert 'tabindex="0"' in training_js
    assert 'tooltip.setAttribute("role", "tooltip")' in training_js
    assert ".training-state-tooltip.visible" in style_css
