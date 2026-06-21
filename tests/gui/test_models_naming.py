# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The Models tab should identify a run by its human name (the GUI run.json's
recipe_name), not the random run_id directory — round-5 "which output is which
model?". The run_id stays available so the dir is still locatable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sft = pytest.importorskip("safetensors.torch")

from lerobot.gui.api.models import _scan_training_run  # noqa: E402


def _make_checkpoint(run_dir: Path, step: str = "000030") -> None:
    pretrained = run_dir / "output" / "checkpoints" / step / "pretrained_model"
    pretrained.mkdir(parents=True)
    sft.save_file({"w": torch.zeros(1)}, str(pretrained / "model.safetensors"))
    (pretrained / "config.json").write_text(json.dumps({"type": "act"}))


def test_scan_uses_run_name_from_run_json(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"recipe_name": "my-cool-run", "created_at": 123.0}))

    result = _scan_training_run(tmp_path)
    assert result is not None
    assert result["name"] == "my-cool-run"  # human name, not the hash dir
    assert result["run_id"] == tmp_path.name  # id still available
    assert result["created_at"] == 123.0


def test_scan_falls_back_to_dir_name_without_run_json(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, step="000010")  # converted / non-GUI: no run.json

    result = _scan_training_run(tmp_path)
    assert result is not None
    assert result["name"] == tmp_path.name
