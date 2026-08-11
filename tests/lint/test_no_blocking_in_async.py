# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The event-loop lint has to fail on a real regression, or it is decoration.

It was verified once by hand — inject a `requests.get` into an async handler,
watch the exit code — and that check was never committed, so nothing would
notice if a future edit made the checker blind. These pin the behaviours that
matter, especially the transitive case: the Hub freeze that motivated the lint
was ``return get_auth_status()`` with ``whoami()`` one level down, and the first
version of the checker reported the whole codebase clean.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lint" / "no_blocking_in_async.py"


def _load():
    spec = importlib.util.spec_from_file_location("no_blocking_in_async", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules[cls.__module__]; without
    # registering it first, defining Hit raises inside exec_module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lint = _load()


def _check(tmp_path: Path, source: str):
    path = tmp_path / "mod.py"
    path.write_text(source)
    return lint.check_file(path)


class TestItFindsBlockingWork:
    def test_a_direct_blocking_call_in_an_async_def(self, tmp_path):
        hits = _check(tmp_path, "import requests\nasync def h():\n    return requests.get('x')\n")

        assert [h.kind for h in hits] == ["blocking-call"]

    def test_the_transitive_case_through_a_sync_helper(self, tmp_path):
        """The shape that actually caused the outage, and that v1 missed."""
        hits = _check(
            tmp_path,
            "def get_auth_status():\n    return whoami()\n\nasync def endpoint():\n    return get_auth_status()\n",
        )

        kinds = [h.kind for h in hits]
        assert "blocking-via-helper" in kinds
        assert any("get_auth_status" in h.detail and "whoami" in h.detail for h in hits)

    def test_a_chain_two_helpers_deep(self, tmp_path):
        hits = _check(
            tmp_path,
            "import subprocess\n"
            "def inner():\n    return subprocess.run(['ls'])\n"
            "def outer():\n    return inner()\n"
            "async def endpoint():\n    return outer()\n",
        )

        assert any(h.kind == "blocking-via-helper" for h in hits)

    def test_the_shared_default_executor(self, tmp_path):
        hits = _check(tmp_path, "async def h():\n    return await loop.run_in_executor(None, f)\n")

        assert [h.kind for h in hits] == ["shared-pool"]

    def test_to_thread_counts_as_the_shared_pool(self, tmp_path):
        hits = _check(tmp_path, "import asyncio\nasync def h():\n    return await asyncio.to_thread(f)\n")

        assert [h.kind for h in hits] == ["shared-pool"]


class TestItDoesNotCryWolf:
    """False positives get a lint disabled, which is worse than not having it."""

    def test_blocking_in_a_plain_sync_function_is_fine(self, tmp_path):
        """A CLI is allowed to block; only the event loop is not."""
        hits = _check(tmp_path, "import requests\ndef main():\n    return requests.get('x')\n")

        assert hits == []

    def test_a_nested_sync_def_inside_an_async_def_is_fine(self, tmp_path):
        """That is the shape of correctly offloaded work."""
        hits = _check(
            tmp_path,
            "import subprocess\n"
            "async def h():\n"
            "    def work():\n"
            "        return subprocess.run(['ls'])\n"
            "    return await loop.run_in_executor(pool, work)\n",
        )

        assert [h.kind for h in hits] == []

    def test_a_named_executor_is_fine(self, tmp_path):
        hits = _check(tmp_path, "async def h():\n    return await loop.run_in_executor(_decode_pool, f)\n")

        assert hits == []


class TestTheEscapeHatches:
    def test_a_line_annotation_marks_the_hit_annotated(self, tmp_path):
        hits = _check(
            tmp_path,
            "import requests\nasync def h():\n    # blocking-ok: startup only\n    return requests.get('x')\n",
        )

        assert len(hits) == 1
        assert hits[0].annotated is True

    def test_a_file_annotation_silences_the_module(self, tmp_path):
        hits = _check(
            tmp_path,
            "# blocking-lint: ignore-file - startup script\nimport requests\n"
            "async def h():\n    return requests.get('x')\n",
        )

        assert hits == []

    def test_a_syntax_error_is_left_to_ruff(self, tmp_path):
        assert _check(tmp_path, "async def h(:\n") == []


class TestTheRatchet:
    """Counts may only go down. Proven by driving main(), not by reading it."""

    def _scoped(self, tmp_path, monkeypatch, source: str):
        scope = tmp_path / "src" / "lerobot" / "gui"
        scope.mkdir(parents=True)
        (scope / "api.py").write_text(source)
        monkeypatch.setattr(lint, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(lint, "BASELINE_PATH", tmp_path / "baseline.txt")
        return scope

    CLEAN = "async def h():\n    return 1\n"
    ONE = "import requests\nasync def h():\n    return requests.get('x')\n"
    TWO = "import requests\nasync def h():\n    requests.get('x')\n    return requests.post('y')\n"

    def test_a_new_violation_fails(self, tmp_path, monkeypatch, capsys):
        scope = self._scoped(tmp_path, monkeypatch, self.ONE)
        lint.main(["--update-baseline"])
        (scope / "api.py").write_text(self.TWO)

        assert lint.main([]) == 1
        assert "went from 1 to 2" in capsys.readouterr().out

    def test_the_baselined_debt_passes(self, tmp_path, monkeypatch):
        self._scoped(tmp_path, monkeypatch, self.ONE)
        lint.main(["--update-baseline"])

        assert lint.main([]) == 0

    def test_removing_a_violation_still_passes(self, tmp_path, monkeypatch):
        """Shrinking below the baseline must not be reported as drift."""
        scope = self._scoped(tmp_path, monkeypatch, self.TWO)
        lint.main(["--update-baseline"])
        (scope / "api.py").write_text(self.ONE)

        assert lint.main([]) == 0

    def test_warn_only_reports_but_does_not_fail(self, tmp_path, monkeypatch):
        scope = self._scoped(tmp_path, monkeypatch, self.ONE)
        lint.main(["--update-baseline"])
        (scope / "api.py").write_text(self.TWO)

        assert lint.main(["--warn-only"]) == 0

    def test_report_mode_never_fails(self, tmp_path, monkeypatch):
        self._scoped(tmp_path, monkeypatch, self.TWO)

        assert lint.main(["--report"]) == 0

    def test_the_committed_baseline_matches_the_tree(self):
        """A stale baseline silently accepts new debt on the next edit."""
        counts: dict[tuple[str, str], int] = {}
        for path in sorted((Path(lint.REPO_ROOT) / lint.DEFAULT_SCOPE).rglob("*.py")):
            for hit in lint.check_file(path):
                if not hit.annotated:
                    counts[(hit.path, hit.kind)] = counts.get((hit.path, hit.kind), 0) + 1

        baseline = lint._read_baseline()

        drifted = {k: (v, baseline.get(k, 0)) for k, v in counts.items() if v > baseline.get(k, 0)}
        assert not drifted, f"baseline is stale; run --update-baseline: {drifted}"


def test_the_hook_is_executable():
    """pre-commit reports `is not executable` and fails the push, as it did once."""
    import os

    assert os.access(_SCRIPT, os.X_OK)
