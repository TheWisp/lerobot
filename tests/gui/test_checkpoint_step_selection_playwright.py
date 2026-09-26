# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A chosen checkpoint step survives, and reaches the process that is launched.

The failure this pins ran a rollout against checkpoint 50000 that was meant for
10000. Nothing errored: the option list is rebuilt after an async fetch, the
rebuild reselects the newest, and a value chosen before the fetch returned was
replaced by a different one that is equally valid-looking. The run then reports
the checkpoint it actually used, which is not the one that was picked.

Holding the value in the control is only half of it: the HVLA launch path read
the model dropdown instead, so the same wrong checkpoint ran from a control
that displayed the right one. The launch tests below assert on the body that
goes on the wire, not on a helper's return value.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

RUN = "/runs/demo"
CHECKPOINTS = [
    {"step": 10000, "policy_path": f"{RUN}/checkpoints/010000/pretrained_model", "is_last": False},
    {"step": 50000, "policy_path": f"{RUN}/checkpoints/050000/pretrained_model", "is_last": True},
]
AT_10K = CHECKPOINTS[0]["policy_path"]
AT_50K = CHECKPOINTS[1]["policy_path"]


def _open_policy_form(page):
    """The step select is rendered by renderRunForm, not present in index.html.

    selectWorkflow is what the Policy button calls, and it is what sets the
    module's `selectedWorkflow` — the launch path reads that to decide which
    robot select and which endpoint to use, so a test that skips it launches
    nothing.
    """
    page.evaluate("switchTab('run')")
    page.evaluate("() => renderRunForm()")
    page.evaluate("() => selectWorkflow('policy')")
    page.wait_for_selector("#run-policy-step", state="attached", timeout=10_000)


def _arm_and_refresh(page, rows=CHECKPOINTS, delay_ms=150, run=RUN):
    """Arm the model select and rebuild the steps in one evaluate.

    One evaluate, not two: the page's own model load rewrites this select, and
    split across two calls that rewrite lands in the gap and wipes the armed
    option, leaving the step list never rebuilt.
    """
    page.evaluate(
        """async ([run, rows, delay]) => {
            const sel = document.getElementById('run-policy-checkpoint');
            sel.innerHTML = `<option value="${run}" data-run-path="${run}">demo</option>`;
            sel.value = run;
            window.__origFetch = window.fetch;
            window.fetch = (url, ...rest) => {
                if (String(url).includes('/checkpoints')) {
                    return new Promise(res => setTimeout(
                        () => res({ ok: true, json: async () => rows }), delay));
                }
                return window.__origFetch(url, ...rest);
            };
            // The cache would skip the fetch entirely on a second call.
            if (window._policyStepCache) for (const k of Object.keys(window._policyStepCache)) delete window._policyStepCache[k];
            await _refreshPolicyStepOptions();
        }""",
        [run, rows, delay_ms],
    )


def _refresh(page, rows=CHECKPOINTS, run=RUN):
    """Arm the model select and rebuild the step options as one operation."""
    _arm_and_refresh(page, rows=rows, run=run)
    page.wait_for_function(
        "() => document.getElementById('run-policy-step').options.length >= 2", timeout=10_000
    )


def _step_value(page):
    return page.evaluate("() => document.getElementById('run-policy-step').value")


def test_a_step_chosen_during_the_fetch_is_not_replaced_by_the_latest(gui_page):
    page = gui_page
    _open_policy_form(page)

    _refresh(page)
    page.evaluate("v => { document.getElementById('run-policy-step').value = v; }", AT_10K)
    assert _step_value(page) == AT_10K

    # Rebuild while that choice stands -- the situation the bug lived in.
    _refresh(page)

    assert _step_value(page) == AT_10K, "the chosen step was replaced by the run's latest"


def test_a_step_the_new_run_does_not_offer_falls_back_to_its_latest(gui_page):
    """The complement. Preserving unconditionally would carry a checkpoint path
    from one run into another, which is worse than defaulting."""
    page = gui_page
    _open_policy_form(page)

    _refresh(page)
    page.evaluate("v => { document.getElementById('run-policy-step').value = v; }", AT_10K)

    other = [
        {"step": 200, "policy_path": "/runs/other/checkpoints/000200/pretrained_model", "is_last": False},
        {"step": 900, "policy_path": "/runs/other/checkpoints/000900/pretrained_model", "is_last": True},
    ]
    # A different run path as well as different rows: checkpoints are cached
    # per run, so reusing the path would replay the first run's list.
    _refresh(page, rows=other, run="/runs/other")

    assert _step_value(page) == other[1]["policy_path"], "should fall back to the new run's latest"


def test_the_newest_checkpoint_is_offered_first(gui_page):
    """Not touching this control must reproduce the old behaviour exactly --
    the latest, which is what every launch used before it existed."""
    page = gui_page
    _open_policy_form(page)
    _refresh(page)

    labels = page.evaluate("() => [...document.getElementById('run-policy-step').options].map(o => o.text)")
    assert "50,000" in labels[0] and "latest" in labels[0], labels
    assert _step_value(page) == AT_50K


def _arm_hvla_form(page):
    """Make the armed model an HVLA one and fill what its launch path needs.

    Patches the option in place rather than rebuilding the select, because
    _arm_and_refresh owns that markup and rebuilding it here would drop the
    step list under test. The option carries what the server's scan emits:
    ``value`` is ``default_policy_path``, always the newest checkpoint, while
    ``data-run-path`` is the run dir the steps came from. That difference is
    the point — a launch reading the option's value cannot see the operator's
    step.
    """
    page.evaluate(
        """([run, latest]) => {
            const robot = document.getElementById('run-policy-robot');
            robot.innerHTML = '<option value="arm">arm</option>';
            robot.value = 'arm';
            const sel = document.getElementById('run-policy-checkpoint');
            const opt = sel.options[0];
            opt.value = latest;
            opt.dataset.runPath = run;
            opt.dataset.policyType = 'hvla_flow_s1';
            sel.value = latest;
            document.getElementById('run-hvla-task').value = 'pick up the ball';
        }""",
        [RUN, AT_50K],
    )


def _launch_and_capture(page):
    """Drive the real launch path and return the body that goes on the wire."""
    sent: dict = {}

    def _profile(route):
        route.fulfill(json={"type": "so101_follower", "cameras": {}})

    def _hvla(route):
        sent.update(route.request.post_data_json)
        route.fulfill(json={"command": "hvla", "pid": 4242})

    page.route("**/api/robot/profiles/*", _profile)
    page.route("**/api/run/hvla", _hvla)
    page.evaluate("() => launchRun()")
    return sent


def test_an_hvla_launch_sends_the_chosen_step_not_the_run_s_latest(gui_page):
    """The step control is shared by every policy type, but only the standard
    launch path read it. HVLA runs went to the newest checkpoint whatever was
    picked, and the Run tab then reported the path it had launched, so the
    substitution left no trace."""
    page = gui_page
    _open_policy_form(page)
    _refresh(page)
    page.evaluate("v => { document.getElementById('run-policy-step').value = v; }", AT_10K)
    _arm_hvla_form(page)

    body = _launch_and_capture(page)

    assert body.get("s1_checkpoint") == AT_10K, body


def test_an_hvla_launch_with_no_steps_falls_back_to_the_model_path(gui_page):
    """The complement, and why the fallback in _selectedPolicyPath has to stay:
    flat layouts (HVLA-S2-VLM) expose no step dirs, so the model dropdown's own
    value is the policy path. Reading the step control alone would send ''."""
    page = gui_page
    _open_policy_form(page)
    _arm_and_refresh(page, rows=[])
    page.wait_for_function("() => document.getElementById('run-policy-step').value === ''", timeout=10_000)
    _arm_hvla_form(page)

    body = _launch_and_capture(page)

    assert body.get("s1_checkpoint") == AT_50K, body
