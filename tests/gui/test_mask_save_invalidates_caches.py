# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Saving masks must invalidate the frame cache, whatever else is missing.

The rebind handler cleared two things after an in-place mask save: the frame
cache, and any composited transcodes. The transcode cleanup needs
`_playback_cache_dir`, which only exists where the playback rewrite is — and
the import sat ABOVE the frame-cache call inside one `try`, so on a branch
without it the ImportError aborted the handler before the cache was cleared.
The `except` then logged "rebind failed", which is not what had happened: the
rebind was already done, and what was lost was the invalidation.

The visible symptom is a mask save that changes nothing on screen, because the
editor keeps serving frames decoded before it.

So the order is the contract: whatever is absent, the frame cache is cleared.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src/lerobot/gui/api/process.py"


def _rebind_handler() -> ast.AST:
    """The function that reopens a dataset after an in-place mask save."""
    tree = ast.parse(SOURCE.read_text())
    fns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "invalidate_caches"
            for c in ast.walk(n)
        )
    ]
    assert len(fns) == 1, f"expected one handler calling invalidate_caches, found {len(fns)}"
    return fns[0]


def test_nothing_optional_is_imported_before_the_cache_is_cleared():
    """An import that can fail, placed above the call, silently skips it."""
    fn = _rebind_handler()
    call_line = min(
        c.lineno
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "invalidate_caches"
    )
    playback_imports = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.ImportFrom)
        and any(a.name.startswith("_playback") or a.name.startswith("_transcode") for a in n.names)
    ]
    assert not playback_imports or min(playback_imports) > call_line, (
        "a playback-only symbol is imported before invalidate_caches; where that symbol is "
        "absent the ImportError skips the invalidation and the editor serves stale frames"
    )


def test_the_transcode_cleanup_tolerates_the_symbol_being_absent():
    """The cleanup is real work where playback exists, and a no-op where it does not.

    Asserting only the import order would pass for code that never cleans up at
    all, so this pins that the cleanup is still attempted -- guarded rather than
    deleted.
    """
    src = ast.get_source_segment(SOURCE.read_text(), _rebind_handler()) or ""
    assert "_playback_cache_dir" in src, "the composited-transcode cleanup was dropped, not guarded"
    assert "getattr" in src, "the cleanup must tolerate the symbol being absent rather than import it"
