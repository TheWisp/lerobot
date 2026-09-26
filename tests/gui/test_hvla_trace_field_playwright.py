# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The Inference Trace Directory the operator types must reach the request.

tests/gui/test_cli_contract.py owns the rest of the chain — request field to
launcher flag. The link it cannot see is the one from the form to the request
body, and that is the link that has broken before: the HVLA launch path read
the model dropdown instead of the step dropdown for its checkpoint, and every
test passed because none of them looked at what was sent.

So these drive the real form and read the body of the POST. The launch is
refused once the body has been read: a launch the page believes succeeded
opens the output SSE and starts an obs-stream poll that outlive the assertion
and keep hitting a server the fixture is about to shut down.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

CHECKPOINT = "/runs/demo/checkpoints/050000/pretrained_model"

_ARM = """([checkpoint, traceDir]) => {
    const robot = document.getElementById('run-policy-robot');
    robot.innerHTML = '<option value="arm">arm</option>';
    robot.value = 'arm';
    const sel = document.getElementById('run-policy-checkpoint');
    sel.innerHTML = `<option value="${checkpoint}" data-run-path="/runs/demo"
                             data-policy-type="hvla_flow_s1">demo</option>`;
    sel.value = checkpoint;
    document.getElementById('run-hvla-task').value = 'pick up the ball';
    document.getElementById('run-hvla-inference-trace').value = traceDir;
    // The product's own handler for "a model was chosen". It is what reveals
    // the HVLA section, so without it the field under test is display:none.
    _onPolicyCheckpointChange();
}"""


def _arm_hvla_form(page, trace_dir=""):
    """Fill the policy form the way a chosen model and a typed path leave it.

    Arming and the field value go in one evaluate, and the page's own start-up
    model load has to have settled first: `_ensureModelDataLoaded` rebuilds the
    model select when its scan returns, which wipes an option armed while that
    was still in flight, and the launch then bails on "select a checkpoint".
    """
    page.evaluate("switchTab('run')")
    page.evaluate("() => renderRunForm()")
    page.evaluate("() => selectWorkflow('policy')")
    page.wait_for_load_state("networkidle")
    page.evaluate(_ARM, [CHECKPOINT, trace_dir])
    page.wait_for_selector("#run-hvla-inference-trace", state="visible", timeout=10_000)


def _launch_and_capture(page):
    sent: dict = {}

    page.route(
        "**/api/robot/profiles/*",
        lambda route: route.fulfill(json={"type": "so101_follower", "cameras": {}}),
    )

    def _hvla(route):
        sent.update(route.request.post_data_json)
        route.fulfill(status=409, json={"detail": "not launched by this test"})

    page.route("**/api/run/hvla", _hvla)
    page.evaluate("() => launchRun()")
    assert sent, "launchRun sent no request -- it bailed before the POST"
    return sent


def test_the_typed_trace_directory_is_what_gets_sent(gui_page):
    page = gui_page
    _arm_hvla_form(page, trace_dir="/tmp/hvla_trace_run1")

    body = _launch_and_capture(page)

    assert body.get("inference_trace_dir") == "/tmp/hvla_trace_run1"


def test_an_empty_field_sends_no_directory(gui_page):
    """Off is the default and has to stay off. An empty string would reach the
    launcher as a directory named '' and the run would write a trace nobody
    asked for."""
    page = gui_page
    _arm_hvla_form(page)

    body = _launch_and_capture(page)

    assert body.get("inference_trace_dir") is None
