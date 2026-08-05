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

"""Tests for the `no_network_in_tests` pre-commit lint hook.

The hook's value rests entirely on it being right about FALSE POSITIVES. A lint that
flags `requests.get(f"{base_url}/x")` against a local fixture server, or a helper the
test file defines itself, is a lint someone disables — at which point the real findings
stop being caught too. So the no-flag cases below matter at least as much as the
detections, and each names the shape it protects.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "lint" / "no_network_in_tests.py"
_spec = importlib.util.spec_from_file_location("no_network_in_tests", _MODULE_PATH)
lint = importlib.util.module_from_spec(_spec)
sys.modules["no_network_in_tests"] = lint
_spec.loader.exec_module(lint)


def _check(tmp_path: Path, body: str) -> list[str]:
    p = tmp_path / "test_sample.py"
    p.write_text(body)
    return lint.check(p)


# --- detections: the failure modes that motivated the hook --------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        ('LeRobotDataset("lerobot/pusht")', "Hub"),
        ('LeRobotDatasetMetadata(repo_id="lerobot/pusht")', "Hub"),
        ('snapshot_download("lerobot/pusht", repo_type="dataset")', "snapshot_download"),
        ('hf_hub_download("lerobot/pusht", "meta/info.json")', "hf_hub_download"),
        ('Model.from_pretrained("lerobot/smolvla_base")', "from_pretrained"),
        ('requests.get("https://huggingface.co/api/models")', "HTTP"),
        ('httpx.post("https://example.com/x", json={})', "HTTP"),
        ('urlopen("https://example.com/x")', "HTTP"),
    ],
)
def test_flags_calls_that_reach_a_remote_service(tmp_path, body, expected):
    problems = _check(tmp_path, body + "\n")
    assert len(problems) == 1, f"expected exactly one finding for {body!r}, got {problems}"
    assert expected in problems[0]


def test_finding_names_the_file_line_and_source(tmp_path):
    """The message has to be actionable on its own — a bare count sends people hunting."""
    p = tmp_path / "test_sample.py"
    p.write_text('x = 1\nLeRobotDataset("lerobot/pusht")\n')
    (problem,) = lint.check(p)
    assert "test_sample.py:2:" in problem
    assert "lerobot/pusht" in problem


# --- no-flag: local usage that must never be reported -------------------------


def test_local_root_is_not_a_download(tmp_path):
    """The sanctioned pattern: a committed fixture read from disk."""
    assert not _check(tmp_path, 'LeRobotDataset("lerobot/pusht", root=tmp_path)\n')


@pytest.mark.parametrize(
    "body,why",
    [
        ('LeRobotDataset("/home/me/datasets/pusht")', "a filesystem path is not a repo id"),
        ("LeRobotDataset(some_variable)", "a non-constant arg proves nothing"),
        ("Model.from_pretrained(checkpoint_dir)", "a local checkpoint dir"),
        ('Model.from_pretrained("./outputs/train/checkpoint")', "a relative path, not a repo id"),
        ('requests.get(f"{base_url}/api/status")', "the fixture-server shape this must not bury"),
        ('requests.get("http://127.0.0.1:8476/api/x")', "loopback is local"),
        ('requests.get("http://localhost:8080/x")', "localhost is local"),
        ('client.get("/api/overlays/models")', "a TestClient relative path"),
    ],
)
def test_does_not_flag_local_usage(tmp_path, body, why):
    assert not _check(tmp_path, body + "\n"), why


def test_a_locally_defined_helper_shadows_the_flagged_name(tmp_path):
    """tests/datasets defines its own load_dataset-style helpers; a name match alone is
    not evidence, which is why the hook resolves definitions in the file."""
    body = 'def snapshot_download(repo):\n    return repo\n\nsnapshot_download("lerobot/pusht")\n'
    assert not _check(tmp_path, body)


def test_mentioning_a_downloader_in_a_docstring_is_not_a_call(tmp_path):
    """Walking the AST rather than the text is the whole point."""
    body = '"""This test avoids snapshot_download and hf_hub_download entirely."""\nx = 1\n'
    assert not _check(tmp_path, body)


def test_monkeypatching_the_downloader_suppresses_hub_findings(tmp_path):
    """A file that patches the downloader never reaches the network through it."""
    body = 'monkeypatch.setattr(mod, "snapshot_download", fake)\nModel.from_pretrained("lerobot/x")\n'
    assert not _check(tmp_path, body)


# --- the three sanctioned escapes --------------------------------------------


def test_hub_live_marked_function_is_exempt(tmp_path):
    body = '@pytest.mark.hub_live\ndef test_downloads():\n    LeRobotDataset("lerobot/pusht")\n'
    assert not _check(tmp_path, body)


def test_hub_live_module_marker_exempts_the_file(tmp_path):
    body = 'pytestmark = pytest.mark.hub_live\n\nLeRobotDataset("lerobot/pusht")\n'
    assert not _check(tmp_path, body)


def test_external_ok_annotation_on_the_line_and_the_line_above(tmp_path):
    same_line = 'LeRobotDataset("lerobot/pusht")  # external-ok: the reference dataset IS the subject\n'
    assert not _check(tmp_path, same_line)
    line_above = '# external-ok: the reference dataset IS the subject\nLeRobotDataset("lerobot/pusht")\n'
    assert not _check(tmp_path, line_above)


def test_annotation_requires_a_reason(tmp_path):
    """A bare `# external-ok:` is a shrug; the escape must cost a sentence."""
    assert _check(tmp_path, 'LeRobotDataset("lerobot/pusht")  # external-ok:\n')


def test_ignore_file_directive_exempts_the_module(tmp_path):
    body = '# network-lint: ignore-file\nLeRobotDataset("lerobot/pusht")\n'
    assert not _check(tmp_path, body)


# --- robustness: a lint must never be the thing that breaks the commit --------


@pytest.mark.parametrize("body", ["def broken(:\n", "\x00\x01binary\n"])
def test_unparsable_files_are_skipped_not_crashed(tmp_path, body):
    assert lint.check(_written(tmp_path, body)) == []


def _written(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "test_sample.py"
    p.write_text(body, errors="ignore")
    return p


def test_missing_file_is_skipped(tmp_path):
    assert lint.check(tmp_path / "does_not_exist.py") == []


def test_main_returns_nonzero_only_when_something_is_flagged(tmp_path, capsys):
    clean = tmp_path / "test_clean.py"
    clean.write_text("x = 1\n")
    assert lint.main([str(clean)]) == 0

    dirty = tmp_path / "test_dirty.py"
    dirty.write_text('snapshot_download("lerobot/pusht")\n')
    assert lint.main([str(dirty)]) == 1
    out = capsys.readouterr().out
    assert "hub_live" in out and "external-ok" in out, "the failure must state the way out"
