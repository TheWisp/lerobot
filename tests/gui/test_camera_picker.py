# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The camera picker reaches every policy, and both trainers receive it.

Three things have to line up for a selection made in the form to arrive at a
trainer, and each has failed silently in this codebase before: the scan has to
report the names, the response model has to declare the field or FastAPI drops
it, and the recipe builder has to spell the flag the way that trainer's parser
expects. Each is pinned separately below.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from lerobot.gui.api.datasets import SourceDatasetInfo, _scan_source
from lerobot.gui.api.training import list_policies
from lerobot.gui.training.recipes import HVLA_FLOW_S1_RECIPE, build_lerobot_train_command
from lerobot.gui.training.runs import Run, RunPaths, RunState, new_run_id


def _make_run(args: dict) -> Run:
    return Run(
        run_id=new_run_id(),
        host_id="this-server",
        recipe_name="cameras",
        dataset_id="local/dummy",
        args=args,
        state=RunState.PENDING,
        created_at=time.time(),
    )


def _write_dataset(root: Path, cameras: dict[str, str]) -> None:
    """Write the minimum meta/info.json the scan reads, with the given cameras."""
    features: dict[str, dict] = {
        "action": {"dtype": "float32", "shape": [6], "names": ["a"] * 6},
        "observation.state": {"dtype": "float32", "shape": [6], "names": ["a"] * 6},
    }
    for key, dtype in cameras.items():
        features[key] = {"dtype": dtype, "shape": [3, 64, 96], "names": ["c", "h", "w"]}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 1, "fps": 30, "features": features})
    )


# --------------------------------------------------------------------------
# The catalog
# --------------------------------------------------------------------------


def test_every_policy_offers_the_camera_picker():
    """Not a per-policy opt-in: a policy registered upstream gets it for free."""
    catalog = list_policies()
    assert catalog, "expected at least one policy in the catalog"

    for entry in catalog:
        fields = {f["name"]: f for f in entry["fields"]}
        assert "cameras" in fields, f"{entry['type_name']} has no camera picker"
        assert fields["cameras"]["type"] == "cameras"


def test_the_picker_writes_a_dataset_key_for_draccus_recipes():
    """policy.cameras would be wrong: the selection restricts the dataset."""
    draccus = [e for e in list_policies() if e["recipe"] is None]
    assert draccus, "expected draccus-backed policies in the catalog"

    for entry in draccus:
        field = next(f for f in entry["fields"] if f["name"] == "cameras")
        assert field["arg_key"] == "dataset.cameras"
        # The prefix that would otherwise apply is exactly what arg_key overrides.
        assert entry["arg_key_prefix"] == "policy."


def test_the_picker_writes_a_bare_key_for_hvla():
    hvla = next(e for e in list_policies() if e["recipe"] == HVLA_FLOW_S1_RECIPE)
    field = next(f for f in hvla["fields"] if f["name"] == "cameras")
    # HVLA's prefix is empty, so the bare name is already the args-dict key and
    # an override would be noise.
    assert field.get("arg_key") is None
    assert hvla["arg_key_prefix"] == ""


def test_the_picker_declares_no_choices():
    """The choices come from the dataset chosen in the same form, not from here."""
    entry = list_policies()[0]
    field = next(f for f in entry["fields"] if f["name"] == "cameras")
    assert field["default"] is None
    assert "choices" not in field


# --------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------


def test_the_scan_reports_camera_names(tmp_path: Path):
    _write_dataset(
        tmp_path / "ds",
        {"observation.images.top_l": "video", "observation.images.top_r": "video"},
    )
    found = _scan_source(str(tmp_path))

    assert len(found) == 1
    assert found[0]["cameras"] == ["top_l", "top_r"]


def test_the_scan_names_a_camera_the_way_the_trainers_do(tmp_path: Path):
    """A prefix-less key has no short form, and must be offered whole."""
    _write_dataset(tmp_path / "ds", {"observation.image": "image"})
    assert _scan_source(str(tmp_path))[0]["cameras"] == ["observation.image"]


def test_a_dataset_without_cameras_reports_an_empty_list(tmp_path: Path):
    _write_dataset(tmp_path / "ds", {})
    assert _scan_source(str(tmp_path))[0]["cameras"] == []


def test_the_response_model_carries_cameras():
    """An undeclared field is stripped by FastAPI on the way out, silently."""
    info = SourceDatasetInfo(
        name="d", root="/tmp/d", total_episodes=1, total_frames=1, fps=30, cameras=["top_l"]
    )
    assert info.cameras == ["top_l"]
    assert "cameras" in SourceDatasetInfo.model_fields


# --------------------------------------------------------------------------
# The recipe builders
# --------------------------------------------------------------------------


def test_hvla_receives_a_comma_separated_selection(tmp_path: Path):
    paths = RunPaths.for_run("hvla-cams", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run(
        {
            "__recipe__": HVLA_FLOW_S1_RECIPE,
            "dataset_repo_id": "local/dummy",
            "cameras": ["top_l", "top_r"],
        }
    )
    cmd, _ = build_lerobot_train_command(run, paths)

    assert "--cameras" in cmd
    # One token, comma separated: HVLA's argparse splits on commas and would
    # read draccus's "[top_l,top_r]" as a camera named "[top_l".
    assert cmd[cmd.index("--cameras") + 1] == "top_l,top_r"


def test_lerobot_train_receives_the_draccus_list_form(tmp_path: Path):
    paths = RunPaths.for_run("draccus-cams", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run(
        {
            "policy.type": "act",
            "dataset.repo_id": "local/dummy",
            "dataset.cameras": ["top_l", "top_r"],
        }
    )
    cmd, _ = build_lerobot_train_command(run, paths)

    assert "--dataset.cameras=[top_l,top_r]" in cmd


@pytest.mark.parametrize("recipe_args", [{}, {"cameras": None}])
def test_no_selection_sends_no_flag(tmp_path: Path, recipe_args: dict):
    """All ticked submits nothing, so recipes recorded before this field replay unchanged."""
    paths = RunPaths.for_run("no-cams", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run({"__recipe__": HVLA_FLOW_S1_RECIPE, "dataset_repo_id": "local/dummy", **recipe_args})
    cmd, _ = build_lerobot_train_command(run, paths)

    assert "--cameras" not in cmd


def test_no_recipe_offers_the_same_field_twice():
    """A recipe that declares a picker main also appends renders it twice.

    `list_policies` appends `_cameras_field()` and `_flags_field()` to every
    `_NON_DRACCUS_RECIPES` entry, so a recipe that also spells the field out
    inline gets two of them. Nothing downstream deduplicates: the form renders
    both, and the second overwrites the first's value on submit.

    This is the shape a rebase between two branches that both implemented the
    picker produces — each side's definition survives — and it is invisible to a
    duplicate-key check, because the two definitions are separate dicts in a
    list rather than a repeated key in one dict.
    """
    import collections

    from lerobot.gui.api.training import list_policies

    offenders = {}
    for schema in list_policies():
        names = [f.get("name") for f in schema.get("fields", [])]
        repeated = sorted({n for n, c in collections.Counter(names).items() if c > 1})
        if repeated:
            offenders[schema.get("policy") or schema.get("id") or str(schema)[:40]] = repeated

    assert not offenders, f"recipes declaring a field twice: {offenders}"
