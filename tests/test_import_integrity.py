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

"""Every ``from lerobot... import NAME`` in the tree must still resolve.

Pytest collection only executes module-level imports. Imports written *inside*
a function or method — used throughout this codebase to keep optional heavy
dependencies lazy — are never exercised until that code path runs, so a symbol
can be deleted and its callers left dangling with the whole suite still green.

That is not hypothetical: an upstream sync removed
``lerobot.datasets.io_utils.load_subtasks``, which the GUI feature editor
imported from inside a helper. Nothing conflicted, nothing failed to collect,
and the breakage only surfaced when those specific tests ran.

This module walks the AST instead, so deletions and renames are caught wherever
the import is written.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCAN_ROOTS = (SRC_ROOT / "lerobot", REPO_ROOT / "tests")

# Dangling imports that already existed when this check was introduced
# (2026-07-31). Each is dead code, reachable only through a branch that would
# raise anyway, so none is user-facing — but none is fixed here either, because
# deleting robot and teleoperator modules is not this test's business.
#
# Add to this list only to record pre-existing debt. A NEW dangling import must
# fail the test; that is the entire point.
KNOWN_DANGLING: dict[str, str] = {
    # Orphaned layout: the live SO-107 follower is robots/so_follower.py, wired
    # up in robots/utils.py. The robots/so_follower/so107_follower/ package is a
    # leftover of an earlier split and imports bases that no longer exist, so it
    # cannot be imported at all. Delete the package.
    "lerobot.robots.so_follower.so_follower_config_base": "orphaned so107_follower package",
    "lerobot.robots.so_follower.so_follower_base": "orphaned so107_follower package",
    # Dead branches in make_teleoperator_from_config: these teleoperators have
    # no module in this fork, so `--teleop.type=stretch3` raises
    # ModuleNotFoundError instead of an unsupported-type error. Remove the
    # branches or restore the modules.
    "lerobot.teleoperators.stretch3_gamepad": "dead factory branch (type=stretch3)",
    "lerobot.teleoperators.widowx": "dead factory branch (type=widowx)",
}


def _containing_package(path: Path) -> list[str] | None:
    """Dotted package parts a file belongs to, or None if outside ``src/``.

    This is the file's *directory*, for both regular modules and package
    ``__init__.py`` files — ``lerobot/cameras/__init__.py`` and
    ``lerobot/cameras/opencv/camera_opencv.py`` belong to ``lerobot.cameras``
    and ``lerobot.cameras.opencv`` respectively. Deriving it from the module
    name instead and stripping the last component resolves one level too high
    for packages, which turns ``from .camera import ...`` into
    ``lerobot.camera``.
    """
    try:
        rel = path.resolve().relative_to(SRC_ROOT)
    except ValueError:
        return None
    return list(rel.parent.parts)


def _resolve(node: ast.ImportFrom, path: Path) -> str | None:
    """Absolute module a ``from ... import`` targets, or None if unresolvable.

    ``level`` counts packages to ascend: level 1 is the containing package,
    level 2 its parent, and so on.
    """
    if not node.level:
        return node.module if node.module and node.module.startswith("lerobot") else None

    pkg_parts = _containing_package(path)
    if pkg_parts is None:
        return None
    ascend = node.level - 1
    if ascend:
        if ascend > len(pkg_parts):
            return None
        pkg_parts = pkg_parts[:-ascend]
    if not pkg_parts:
        return None
    base = ".".join(pkg_parts)
    return f"{base}.{node.module}" if node.module else base


def _iter_imports() -> list[tuple[Path, int, str, str]]:
    """Every ``(file, lineno, absolute_module, name)`` importing from lerobot."""
    found: list[tuple[Path, int, str, str]] = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = _resolve(node, path)
                if module is None:
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        found.append((path, node.lineno, module, alias.name))
    return found


def _module_source_exists(module: str) -> bool:
    """True when ``module``'s source file is still present in this checkout.

    Discriminates "the env lacks an optional extra" from "the symbol is gone".
    Deliberately filesystem-based rather than exception-based: ``require_package``
    raises a bare ``ImportError`` with no ``name`` attribute, and packages such as
    ``lerobot.datasets`` call it at import time — so keying off the exception
    would flag a lean environment as breakage.
    """
    rel = Path(*module.split("."))
    return (SRC_ROOT / rel).with_suffix(".py").is_file() or (SRC_ROOT / rel / "__init__.py").is_file()


def test_every_lerobot_import_resolves():
    """No ``from lerobot... import y`` may name something that no longer exists.

    Covers absolute and relative imports, at any nesting depth.

    Pre: the working tree is importable with base dependencies installed.
    Post: on failure, every dangling (module, name) is listed with its call site.
    """
    references = _iter_imports()
    assert references, "found no lerobot imports to check — scan roots are wrong"

    module_cache: dict[str, object | None] = {}
    dangling: list[str] = []

    for path, lineno, module, name in references:
        if module in KNOWN_DANGLING:
            continue
        if module not in module_cache:
            try:
                module_cache[module] = importlib.import_module(module)
            except ImportError as exc:
                module_cache[module] = None
                if not _module_source_exists(module):
                    dangling.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: module '{module}' no longer exists ({exc})"
                    )
            except Exception:
                # Import-time side effects (hardware probes, GPU init) are not
                # what this guard is about.
                module_cache[module] = None

        mod = module_cache[module]
        if mod is None:
            continue
        if hasattr(mod, name):
            continue
        # `from pkg import submodule` is legitimate even when the submodule is
        # not yet an attribute of the parent package.
        try:
            importlib.import_module(f"{module}.{name}")
            continue
        except ImportError:
            pass
        dangling.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: '{module}' has no attribute '{name}'")

    assert not dangling, (
        "Dangling lerobot imports — a symbol was removed but its callers were not.\n"
        "If this appeared after an upstream merge, upstream deleted something the\n"
        "fork still uses; restore it or port the callers.\n\n" + "\n".join(dangling)
    )


def test_known_dangling_list_has_no_stale_entries():
    """Every allowlisted module must still be missing.

    Without this, a fixed entry lingers in ``KNOWN_DANGLING`` and silently
    suppresses a future regression at the same path — the allowlist would decay
    from "recorded debt" into "blind spot".

    Post: an entry whose module now imports is reported so it can be deleted.
    """
    resurrected = []
    for module, reason in KNOWN_DANGLING.items():
        if _module_source_exists(module):
            resurrected.append(f"{module} ({reason}) — source exists again; drop it from KNOWN_DANGLING")
    assert not resurrected, "Stale KNOWN_DANGLING entries:\n" + "\n".join(resurrected)
