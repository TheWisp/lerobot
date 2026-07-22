#!/usr/bin/env python

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

"""Tests for the `no_setup_hardcode` pre-commit lint hook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "lint" / "no_setup_hardcode.py"
_spec = importlib.util.spec_from_file_location("no_setup_hardcode", _MODULE_PATH)
lint = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve the module (importlib quirk).
sys.modules["no_setup_hardcode"] = lint
_spec.loader.exec_module(lint)


def _write(tmp_path: Path, body: str, name: str = "modeling_thing.py") -> Path:
    # Land under a policies/ path so scope/exclusion logic behaves realistically.
    p = tmp_path / "src" / "lerobot" / "policies" / "thing" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def _names(path: Path) -> set[str]:
    """Names of unannotated high-tier rules that fired on the file."""
    return {h.name for h in lint.check_file(path) if not h.annotated and h.tier == "high"}


# --------------------------- true positives --------------------------------- #


def test_qualified_image_key_literal_flagged(tmp_path):
    p = _write(tmp_path, 'X = batch["observation.images.top"]\n')
    assert "qualified-image-key" in _names(p)


def test_internal_cam_key_flagged(tmp_path):
    p = _write(tmp_path, 'KEYS = ["base_0_rgb", "left_wrist_0_rgb"]\n')
    assert "internal-cam-key" in _names(p)


def test_joint_feature_key_flagged(tmp_path):
    p = _write(tmp_path, 'J = ["left_shoulder_pan.pos", "right_gripper.pos"]\n')
    assert "joint-name-key" in _names(p)


def test_camera_key_default_field_flagged(tmp_path):
    p = _write(
        tmp_path, "import dataclasses\n\n@dataclasses.dataclass\nclass S:\n    camera_key: str = 'top'\n"
    )
    assert "camera-key-default" in _names(p)


def test_camera_key_default_binop_flagged(tmp_path):
    p = _write(tmp_path, "OBS_IMAGES = 'observation.images'\nimage_key = OBS_IMAGES + '.top'\n")
    assert "camera-key-default" in _names(p)


# ------------------------ precision / false positives ----------------------- #


def test_bare_gripper_word_not_flagged(tmp_path):
    # "gripper" appears in a log string and a generic EE interface key.
    p = _write(tmp_path, 'log = "Chunk R gripper[0:15]: %s"\nd = {}\nd["gripper"] = 1\n')
    assert _names(p) == set()


def test_matplotlib_va_top_not_flagged_high_tier(tmp_path):
    # bare "top" is medium tier only (matplotlib vertical-alignment collision).
    p = _write(tmp_path, 'ax.text(0, 0, "hi", va="top")\n')
    assert _names(p) == set()


def test_prose_mentioning_key_not_flagged(tmp_path):
    # A help/error message that merely mentions a key is not a hardcoded key.
    p = _write(tmp_path, 'msg = "rename e.g. observation.images.top to your camera name"\n')
    assert "qualified-image-key" not in _names(p)


def test_docstring_not_matched(tmp_path):
    p = _write(tmp_path, '"""Reads observation.images.top and base_0_rgb from the batch."""\n')
    assert _names(p) == set()


def test_config_driven_none_default_not_flagged(tmp_path):
    p = _write(
        tmp_path,
        "import dataclasses\n\n@dataclasses.dataclass\nclass S:\n    camera_key: str | None = None\n",
    )
    assert _names(p) == set()


# ------------------------------ suppression --------------------------------- #


def test_hardcode_ok_same_line_suppresses(tmp_path):
    p = _write(tmp_path, 'X = "observation.images.top"  # hardcode-ok: single-cam demo\n')
    assert _names(p) == set()


def test_hardcode_ok_previous_line_suppresses(tmp_path):
    p = _write(tmp_path, '# hardcode-ok: single-cam demo\nX = "observation.images.top"\n')
    assert _names(p) == set()


def test_annotation_on_declaration_covers_multiline_literal(tmp_path):
    # One annotation above a multi-line list covers every element inside it.
    body = "# hardcode-ok: fallback default\nJOINT_NAMES = [\n    'left_shoulder_pan.pos',\n    'right_gripper.pos',\n]\n"
    p = _write(tmp_path, body)
    assert _names(p) == set()


def test_annotation_above_multiline_call_covers_default(tmp_path):
    # Annotation above a multi-line call covers a literal in its keyword default.
    body = (
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "# hardcode-ok: default order\n"
        "p.add_argument(\n"
        "    '--keys',\n"
        "    default=['base_0_rgb', 'left_wrist_0_rgb'],\n"
        ")\n"
    )
    pth = _write(tmp_path, body)
    assert _names(pth) == set()


def test_file_level_ignore_suppresses_everything(tmp_path):
    body = "# hardcode-lint: ignore-file - single-setup module\nJ = ['left_shoulder_pan.pos']\nK = ['base_0_rgb']\n"
    p = _write(tmp_path, body)
    assert lint.check_file(p) == []


# ------------------------------ scope / cli --------------------------------- #


def test_excluded_dirs_skipped(tmp_path):
    robots = tmp_path / "src" / "lerobot" / "robots" / "arm.py"
    robots.parent.mkdir(parents=True, exist_ok=True)
    robots.write_text('J = ["left_shoulder_pan.pos"]\n')
    assert lint._excluded(robots, include_scripts=False)


def test_scripts_skipped_by_default(tmp_path):
    s = tmp_path / "src" / "lerobot" / "policies" / "hvla" / "scripts" / "x.py"
    s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text('K = ["base_0_rgb"]\n')
    assert lint._excluded(s, include_scripts=False)
    assert not lint._excluded(s, include_scripts=True)


def test_main_exit_codes(tmp_path):
    bad = _write(tmp_path, 'X = "observation.images.top"\n', name="bad.py")
    good = _write(
        tmp_path, 'X = "observation.images"  # the constant prefix, no concrete cam\n', name="good.py"
    )
    assert lint.main(["--include-upstream", str(bad)]) == 1
    assert lint.main(["--include-upstream", str(good)]) == 0


def test_upstream_lookup_graceful_when_ref_missing(monkeypatch, tmp_path):
    # A bogus ref must not raise; it falls back to "scan everything" (empty set).
    monkeypatch.setattr(lint, "UPSTREAM_REF", "definitely/nonexistent-ref")
    assert lint._upstream_paths([Path("src/lerobot/policies/thing/x.py")]) == set()
