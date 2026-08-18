# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""``Robot.send_action`` must survive an ``ActionChunk``.

HVLA S1 sends a chunk by default (``--send-action-shape=chunk``). Robots
without a lookahead controller are supposed to collapse it to ``frames[0]`` via
:func:`lerobot.types.action_first_frame`, but nothing enforced that, and the
OpenArm drivers went straight to ``action.items()``. Running S1 on OpenArm2
therefore died on the first policy action with::

    AttributeError: 'ActionChunk' object has no attribute 'items'

raised from inside the driver, naming neither the chunk nor the flag that
produced it, after policy load, dataset preload and robot connect had all
succeeded.

These tests exercise the unpacking in isolation — constructing a real driver
needs CAN hardware — plus the structural rule that keeps the next robot from
reintroducing it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.types import ActionChunk, action_first_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT = REPO_ROOT / "scripts" / "lint" / "robot_handles_action_chunk.py"

ARM_ACTION = {"joint_1.pos": 1.0, "joint_2.pos": 2.0}


class TestTheUnpackingItself:
    def test_a_chunk_collapses_to_its_first_frame(self):
        """frames[0] is the action for the receiver's "now"."""
        chunk = ActionChunk(fps=30.0, frames=[ARM_ACTION, {"joint_1.pos": 9.0}])
        assert action_first_frame(chunk) == ARM_ACTION

    def test_a_plain_dict_passes_through_untouched(self):
        """The historical shape has to keep working — every existing caller
        sends one, and a robot calling this unconditionally must not break."""
        assert action_first_frame(dict(ARM_ACTION)) == ARM_ACTION

    def test_the_result_supports_the_dict_protocol_drivers_use(self):
        """The defect was an AttributeError on `.items()`, so that is what a
        driver needs back — asserting equality alone would not have caught a
        fallback that returned some other mapping-like object."""
        out = action_first_frame(ActionChunk(fps=30.0, frames=[ARM_ACTION]))
        assert dict(out.items()) == ARM_ACTION


class TestTheStructuralRule:
    """The fix that generalises: a check no future robot can quietly skip."""

    def _run(self, *paths: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(LINT), *[str(p) for p in paths]],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

    def test_the_openarm_drivers_now_pass(self):
        paths = [
            REPO_ROOT / "src/lerobot/robots/openarm_follower/openarm_follower.py",
            REPO_ROOT / "src/lerobot/robots/bi_openarm_follower/bi_openarm_follower.py",
        ]
        result = self._run(*paths)
        assert result.returncode == 0, result.stdout

    def test_a_robot_that_skips_the_fallback_is_caught(self, tmp_path):
        """Put the bug back: without this the rule could pass vacuously."""
        offender = tmp_path / "bad_robot.py"
        offender.write_text(
            "class BadFollower:\n"
            "    def send_action(self, action):\n"
            "        return {k: v for k, v in action.items()}\n"
        )
        result = self._run(offender)
        assert result.returncode == 1
        assert "ignores the ActionChunk half" in result.stdout

    @pytest.mark.parametrize(
        "body",
        [
            "        action = action_first_frame(action)\n        return dict(action)\n",
            "        return dict(action.frames[0])\n",
        ],
        ids=["collapses-the-horizon", "reads-the-horizon"],
    )
    def test_both_sides_of_the_contract_satisfy_it(self, tmp_path, body):
        """A robot may honour the contract either way — the rule must not
        force the fallback on a robot that genuinely uses the lookahead."""
        f = tmp_path / "ok_robot.py"
        f.write_text(f"class OkFollower:\n    def send_action(self, action):\n{body}")
        assert self._run(f).returncode == 0

    def test_an_explicit_annotation_is_honoured(self, tmp_path):
        f = tmp_path / "annotated_robot.py"
        f.write_text(
            "class HelperArm:\n"
            "    # chunk-ok: internal helper, never addressed by a policy\n"
            "    def send_action(self, action):\n"
            "        return dict(action.items())\n"
        )
        assert self._run(f).returncode == 0

    def test_the_baseline_only_covers_files_that_still_need_it(self):
        """A baselined file that has since been fixed should be removed from
        the list, or the count stops meaning anything."""
        baseline = LINT.with_name("robot_handles_action_chunk_baseline.txt")
        listed = [
            ln.strip() for ln in baseline.read_text().splitlines() if ln.strip() and not ln.startswith("#")
        ]
        assert listed, "an empty baseline should be deleted, not kept"
        for rel in listed:
            path = REPO_ROOT / rel
            assert path.exists(), f"baseline names a file that no longer exists: {rel}"
            result = self._run(path)
            assert result.returncode == 1, f"{rel} no longer violates the rule — remove it from the baseline"
