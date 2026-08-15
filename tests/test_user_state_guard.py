# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The guard that fails tests writing to the developer's real state.

A guard nobody has watched fail is a guard nobody knows works. These run pytest
in a subprocess against a fabricated "home", so the assertions exercise the real
fixture without this file itself having to write anywhere real.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_pytest(tmp_path: Path, body: str, extra_args: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """Run one generated test under a HOME we control, with the guard active."""
    fake_home = tmp_path / "home"
    (fake_home / ".config" / "lerobot").mkdir(parents=True)
    (fake_home / ".config" / "lerobot" / "existing.json").write_text("{}")

    test_file = tmp_path / "test_generated.py"
    test_file.write_text(textwrap.dedent(body))

    env = {
        "HOME": str(fake_home),
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO_ROOT),
        # Keep the run hermetic: the guard reads Path.home() at call time, and
        # the suite's own conftest imports lerobot.
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "-p",
            "tests.fixtures.user_state_guard",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


class TestTheGuardCatchesWhatItPromises:
    def test_a_test_that_creates_a_file_under_config_fails(self, tmp_path):
        result = _run_pytest(
            tmp_path,
            """
            from pathlib import Path

            def test_writes():
                (Path.home() / ".config" / "lerobot" / "leaked.json").write_text("{}")
            """,
        )
        assert result.returncode != 0
        assert "wrote to the developer's real state" in result.stdout
        assert "leaked.json" in result.stdout

    def test_appending_to_an_existing_file_is_caught(self, tmp_path):
        """The hub-history case: the file already existed, the test added a line.

        Creation-only detection would have missed it, which is the shape that
        put 105 fabricated transfers in front of the developer.
        """
        result = _run_pytest(
            tmp_path,
            """
            from pathlib import Path

            def test_appends():
                p = Path.home() / ".config" / "lerobot" / "existing.json"
                p.write_text(p.read_text() + " ")
            """,
        )
        assert result.returncode != 0
        assert "existing.json" in result.stdout

    def test_deleting_real_state_is_caught_too(self, tmp_path):
        """Destroying the developer's config is worse than adding to it."""
        result = _run_pytest(
            tmp_path,
            """
            from pathlib import Path

            def test_deletes():
                (Path.home() / ".config" / "lerobot" / "existing.json").unlink()
            """,
        )
        assert result.returncode != 0
        assert "removed" in result.stdout

    def test_a_test_that_stays_in_tmp_path_passes(self, tmp_path):
        """The guard must not fire on what every well-behaved test does."""
        result = _run_pytest(
            tmp_path,
            """
            def test_behaves(tmp_path):
                (tmp_path / "fine.json").write_text("{}")
            """,
        )
        assert result.returncode == 0, result.stdout

    def test_the_marker_opts_a_test_out(self, tmp_path):
        """Some tests exercise real-path resolution deliberately."""
        result = _run_pytest(
            tmp_path,
            """
            from pathlib import Path
            import pytest

            @pytest.mark.touches_user_state
            def test_deliberate():
                (Path.home() / ".config" / "lerobot" / "on_purpose.json").write_text("{}")
            """,
            extra_args=("-m", "touches_user_state or not touches_user_state"),
        )
        assert result.returncode == 0, result.stdout


class TestTheGuardIsCheapEnoughToRunEverywhere:
    def test_a_snapshot_costs_little(self):
        """It runs per test, thousands of times. If a snapshot were expensive
        the guard would be removed rather than fixed."""
        import time

        from tests.fixtures.user_state_guard import _snapshot

        _snapshot()  # warm the directory cache; the first call is not typical
        start = time.perf_counter()
        for _ in range(20):
            _snapshot()
        per_call_ms = (time.perf_counter() - start) / 20 * 1000
        assert per_call_ms < 50, f"snapshot took {per_call_ms:.1f}ms — too slow to run per test"


class TestTheRealSuiteIsClean:
    def test_no_marker_is_currently_needed(self):
        """Nothing in the suite opts out today. When the first opt-out lands,
        this should be updated with the reason rather than deleted — an
        unexplained exemption is how the rule erodes."""
        hits = subprocess.run(
            ["git", "grep", "-l", "touches_user_state", "--", "tests/"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.split()
        allowed = {"tests/fixtures/user_state_guard.py", "tests/test_user_state_guard.py"}
        assert set(hits) <= allowed, f"new opt-outs: {set(hits) - allowed}"
