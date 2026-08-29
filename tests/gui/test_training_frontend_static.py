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


def test_training_dashboard_curates_health_metrics() -> None:
    training_js = (_STATIC_DIR / "training.js").read_text()
    charts_js = (_STATIC_DIR / "charts.js").read_text()

    assert 'label: "Gradient norm"' in training_js
    assert 'label: "Peak GPU allocation (GB)"' in training_js
    assert 'label: "Step time (ms)"' in training_js
    assert 'trainingLatestMetricValue(metricsSeries, "samples_per_s", "smp/s")' in training_js
    assert "Peak GPU alloc." in training_js
    assert "Not logged by this run" in training_js
    assert "legacyByStep.get(sample.step)" in training_js
    assert "xValues: series.map((sample) => sample.step)" in training_js
    assert "function _chartStepAtIndex(group, index)" in charts_js


def test_training_dashboard_keeps_generalization_separate_from_dense_metrics() -> None:
    training_js = (_STATIC_DIR / "training.js").read_text()
    style_css = (_STATIC_DIR / "style.css").read_text()

    metrics_card = training_js.index("${trainingMetricsCardHtml(metricsSeries, isActive)}")
    generalization_card = training_js.index("${trainingGeneralizationCardHtml(metricsSeries)}")
    checkpoints_card = training_js.index('<h3 class="training-card-heading">Checkpoints</h3>')

    assert metrics_card < generalization_card < checkpoints_card
    assert 'syncGroup: "training-generalization"' in training_js
    assert "xValues: evaluations.map((sample) => sample.step)" in training_js
    assert '<details class="training-generalization-history">' in training_js
    assert ".training-generalization-summary" in style_css


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


def test_training_policy_advanced_fields_use_progressive_disclosure() -> None:
    training_js = (_STATIC_DIR / "training.js").read_text()

    assert "fields.filter((f) => !f.advanced)" in training_js
    assert "fields.filter((f) => f.advanced)" in training_js
    assert "Advanced policy and performance settings" in training_js


def _function_body(source: str, signature: str) -> str:
    """Body of the named function, matched by braces.

    Assertions about *where* a statement sits need the enclosing function, not
    the whole file — a substring search would happily match an identical line
    somewhere else.
    """
    start = source.index(signature)
    open_brace = source.index("{", start)
    depth = 0
    for i in range(open_brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace : i + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def test_force_full_rebuild_is_disarmed_on_every_terminal_outcome() -> None:
    """A forced rebuild must not survive the build that used it.

    Regression: the reset lived inside the success branch, so a *failed* forced
    build left the flag set while the <details> re-rendered collapsed. The next
    "Build now" then bypassed the layer cache with nothing on screen saying so —
    and failure is the outcome most likely to follow ticking the box.
    """
    body = _function_body(
        (_STATIC_DIR / "training.js").read_text(), "async function trainingCheckBuildStatus()"
    )

    disarm = body.index("_trainingForceFullRebuild = false")
    failed_branch = body.index("if (!failed)")
    assert disarm < failed_branch, (
        "the force-full-rebuild reset sits inside the success branch; a failed "
        "forced build would leave it armed for the next build"
    )


def test_force_full_rebuild_control_is_styled() -> None:
    """The disclosure sits beside .training-policy-advanced in the same form, so
    an unstyled one renders as a visibly different kind of control."""
    training_js = (_STATIC_DIR / "training.js").read_text()
    style_css = (_STATIC_DIR / "style.css").read_text()

    assert 'class="training-image-advanced"' in training_js
    assert 'name="image_force_full_rebuild"' in training_js
    assert ".training-image-advanced {" in style_css
    assert ".training-image-advanced > summary {" in style_css
