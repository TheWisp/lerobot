# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Choosing a checkpoint step is not an HVLA feature.

The step list comes from the standard checkpoint layout -- ``config.json`` plus
``training_state/training_step.json`` -- so it works for any policy that writes
one. Worth pinning rather than assuming: the control was built while working on
HVLA, and a reader could reasonably expect it to be HVLA-shaped.
"""

from __future__ import annotations

import asyncio
import json

import pytest


def _run_dir(root, policy_type, steps=(10000, 50000)):
    """A training run in the layout every lerobot trainer writes."""
    for step in steps:
        pretrained = root / "checkpoints" / f"{step:06d}" / "pretrained_model"
        pretrained.mkdir(parents=True)
        (pretrained / "config.json").write_text(json.dumps({"type": policy_type}))
        state = root / "checkpoints" / f"{step:06d}" / "training_state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "training_step.json").write_text(json.dumps({"step": step}))
    (root / "checkpoints" / "last").symlink_to(root / "checkpoints" / f"{max(steps):06d}")
    return root


@pytest.mark.parametrize("policy_type", ["act", "diffusion", "smolvla", "hvla_flow_s1"])
def test_every_policy_type_lists_its_checkpoints(tmp_path, policy_type):
    from lerobot.gui.api.models import list_checkpoints

    root = _run_dir(tmp_path / policy_type, policy_type)
    rows = [
        r.model_dump() if hasattr(r, "model_dump") else r for r in asyncio.run(list_checkpoints(str(root)))
    ]

    assert [r["step"] for r in rows] == [10000, 50000], f"{policy_type}: {rows}"
    # Each carries its own path, which is what the selector puts in the launch.
    assert all(r["policy_path"] for r in rows)
    assert len({r["policy_path"] for r in rows}) == 2, "two checkpoints must not share a path"
    assert [r["is_last"] for r in rows] == [False, True]


def test_a_run_with_one_checkpoint_still_lists_it(tmp_path):
    """The common case early in a run. Nothing should require two."""
    from lerobot.gui.api.models import list_checkpoints

    root = _run_dir(tmp_path / "single", "act", steps=(500,))
    rows = asyncio.run(list_checkpoints(str(root)))

    assert len(rows) == 1


def test_a_flat_layout_refuses_rather_than_inventing_a_step(tmp_path):
    """Some runs (HVLA-S2-VLM) have no step directories.

    The endpoint 404s rather than returning a single synthetic entry, and the
    frontend turns that into an inert "n/a" control -- the model dropdown's own
    value is already the policy path there. Pinned because returning a made-up
    step would put a path in the launch that no checkpoint corresponds to.
    """
    from fastapi import HTTPException

    from lerobot.gui.api.models import list_checkpoints

    flat = tmp_path / "flat"
    (flat / "pretrained_model").mkdir(parents=True)
    (flat / "pretrained_model" / "config.json").write_text(json.dumps({"type": "act"}))

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(list_checkpoints(str(flat)))
    assert excinfo.value.status_code == 404
