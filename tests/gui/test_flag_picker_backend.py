# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The three server-side halves of the flag picker.

The frontend is covered by ``flag_picker.test.js``; this covers what it talks
to. Each half fails silently on its own: without the scan payload the picker is
permanently empty, without the catalog field it is never rendered, and without
the CLI translation the selection is dropped on the way to the trainer and the
run trains on every frame while the form says otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from lerobot.gui.api.datasets import _declared_flags, _scan_recursive
from lerobot.gui.training.recipes import HVLA_FLOW_S1_FIELD_TO_FLAG, _fmt_arg

FLAGS = {"dtype": "int64", "shape": [1], "names": None, "flags": ["blurry", "fumble"]}
EPISODE_FLAGS = {"dtype": "int64", "shape": [1], "names": None, "flags": ["fumble", "wrong_task"]}
SCALAR = {"dtype": "float32", "shape": [6], "names": ["a", "b", "c", "d", "e", "f"]}
CATEGORICAL = {"dtype": "int64", "shape": [1], "names": ["phase"]}


# ── What the picker offers ────────────────────────────────────────────────────


def test_declared_flags_is_the_union_across_columns_in_declaration_order():
    """A flag is resolved by name across every flags column, so the picker must
    not make the operator choose a column first."""
    flags = _declared_flags({"quality": FLAGS, "take_quality": EPISODE_FLAGS, "action": SCALAR})
    assert flags == ["blurry", "fumble", "wrong_task"]


def test_a_flag_declared_twice_is_offered_once():
    """`fumble` above is in both columns. Offering it twice would let a user tick
    one box and leave an identical-looking one unticked."""
    flags = _declared_flags({"a": FLAGS, "b": dict(FLAGS)})
    assert flags == ["blurry", "fumble"]


def test_a_categorical_column_is_not_a_flags_column():
    """`names` and `flags` are different contracts on the same dtype; offering a
    class name as a flag would produce a selection the trainer refuses."""
    assert _declared_flags({"phase": CATEGORICAL, "action": SCALAR}) == []


def test_a_dataset_with_no_flags_column_offers_none():
    assert _declared_flags({"action": SCALAR}) == []


def test_a_malformed_features_dict_does_not_abort_the_sweep():
    """This runs inside a directory scan over datasets the user did not write."""
    assert _declared_flags({"quality": None, "other": "not a dict"}) == []


# ── The scan payload the picker reads ─────────────────────────────────────────


def _write_dataset(root: Path, name: str, features: dict) -> None:
    meta = root / name / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 10, "fps": 30, "features": features})
    )


def test_the_http_response_carries_the_flags(tmp_path: Path):
    """Through the route, not the scanner.

    The scanner built the list correctly while ``SourceDatasetInfo`` did not
    declare the field, so FastAPI dropped it on the way out and every dataset
    reached the picker declaring no flags. Nothing below the HTTP boundary can
    see that -- which is why this test is here and not beside the one above.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lerobot.gui.api import datasets as datasets_module

    # The source list lands in a tmp file: tests/gui/conftest.py redirects
    # SOURCES_FILE suite-wide, because the constant binds at import and the env
    # var alone would be too late.
    _write_dataset(tmp_path, "flagged", {"action": SCALAR, "quality": FLAGS})
    _write_dataset(tmp_path, "plain", {"action": SCALAR})

    app = FastAPI()
    app.include_router(datasets_module.router)
    client = TestClient(app)
    # The route serves only registered sources, so register through the API
    # rather than reaching into the config file it keeps them in.
    assert client.post("/api/datasets/sources", json={"path": str(tmp_path)}).status_code < 400

    encoded = quote(str(tmp_path), safe="")
    body = client.get(f"/api/datasets/sources/{encoded}/datasets")
    assert body.status_code == 200, body.text

    by_name = {d["name"]: d for d in body.json()}
    assert by_name["flagged"]["flags"] == ["blurry", "fumble"]
    assert by_name["plain"]["flags"] == []


def test_the_scan_reports_declared_flags(tmp_path: Path):
    _write_dataset(tmp_path, "flagged", {"action": SCALAR, "quality": FLAGS})
    _write_dataset(tmp_path, "plain", {"action": SCALAR})
    found: list[dict] = []
    _scan_recursive(tmp_path, tmp_path, found, max_depth=2, depth=0)

    by_name = {d["name"]: d for d in found}
    assert by_name["flagged"]["flags"] == ["blurry", "fumble"]
    # Present and empty, not absent: the frontend copies the key explicitly and
    # an absent one would leave the picker empty with no error anywhere.
    assert by_name["plain"]["flags"] == []


# ── The selection reaching each trainer ───────────────────────────────────────


def test_draccus_gets_the_bracketed_list_it_parses():
    assert _fmt_arg(["blurry", "fumble"]) == "[blurry,fumble]"


def test_hvla_has_a_flag_for_the_form_key():
    """An unmapped key is silently skipped by the HVLA recipe builder, so the run
    would train on every frame with the form showing a selection."""
    assert HVLA_FLOW_S1_FIELD_TO_FLAG["exclude_flags"] == "--exclude-flags"


def test_hvla_receives_one_comma_separated_token():
    """HVLA's argparse splits on commas; draccus's bracketed form would arrive as
    a single flag literally named "[blurry,fumble]"."""
    value = ",".join(str(x) for x in ["blurry", "fumble"])
    assert value == "blurry,fumble"


# ── The field both recipes carry ──────────────────────────────────────────────


def test_every_recipe_offers_the_picker_with_the_key_its_trainer_reads():
    """The two trainers name the same thing differently, and the wrong key is
    dropped rather than refused."""
    from lerobot.gui.api.training import list_policies

    for entry in list_policies():
        fields = {f["name"]: f for f in entry["fields"]}
        assert "exclude_flags" in fields, f"{entry['type_name']} has no flag picker"
        field = fields["exclude_flags"]
        assert field["type"] == "flags"
        # Unset, not [] -- DatasetConfig refuses an empty selection outright.
        assert field["default"] is None
        expected = "dataset.exclude_flags" if entry["recipe"] is None else None
        assert field.get("arg_key") == expected, entry["type_name"]


def test_the_default_selection_is_one_a_trainer_accepts():
    """The form's default must not be a value the config layer refuses."""
    from lerobot.configs.default import DatasetConfig

    DatasetConfig(repo_id="x/y", exclude_flags=None)  # the picker's default
    with pytest.raises(ValueError, match="at least one flag"):
        DatasetConfig(repo_id="x/y", exclude_flags=[])
