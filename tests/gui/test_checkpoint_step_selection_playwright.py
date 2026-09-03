# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A chosen checkpoint step survives the async rebuild of its own options.

The failure this pins ran a rollout against checkpoint 50000 that was meant for
10000. Nothing errored: the option list is rebuilt after an async fetch, the
rebuild reselects the newest, and a value chosen before the fetch returned was
replaced by a different one that is equally valid-looking. The run then reports
the checkpoint it actually used, which is not the one that was picked.
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
    """The step select is rendered by renderRunForm, not present in index.html."""
    page.evaluate("switchTab('run')")
    page.evaluate("() => { if (typeof switchRunMode === 'function') switchRunMode('policy'); }")
    page.evaluate("() => renderRunForm()")
    page.wait_for_selector("#run-policy-step", state="attached", timeout=10_000)


def _arm(page, rows=CHECKPOINTS, delay_ms=150, run=RUN):
    """A model select pointing at a run, and a checkpoint fetch that takes time.

    The delay is the point: the defect only appears when a value is chosen
    while the fetch is still outstanding, which is the ordinary case on a real
    run directory.
    """
    page.evaluate(
        """([run, rows, delay]) => {
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
        }""",
        [run, rows, delay_ms],
    )


def _refresh(page, rows=CHECKPOINTS, run=RUN):
    """Arm the model select, then rebuild the step options.

    Arming immediately before each refresh rather than once: the page runs its
    own asynchronous model load, which rewrites that select, and a refresh that
    finds no run path returns early and would test nothing.
    """
    _arm(page, rows=rows, run=run)
    page.evaluate("() => _refreshPolicyStepOptions()")
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
