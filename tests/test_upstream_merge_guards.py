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

"""Structural guards against silent breakage when merging upstream lerobot.

A fork merge has a failure mode that neither conflict markers nor a green
collection run detects: upstream *deletes* something that existed at the merge
base and that only fork code still calls. Git applies the deletion without a
conflict, because the fork never edited those exact lines, and the fork's
callers are left dangling.

That is not hypothetical. The 2026-07 sync silently dropped
``lerobot.datasets.io_utils.load_subtasks`` — present at the merge base, removed
by upstream's rewrite of that module, still imported by fork tests. It cost 19
test failures that were only noticed because tests happened to reference it;
had the callers been in untested fork code, it would have shipped.

These guards are deliberately structural rather than behavioural. They cost a
few seconds, they need no hardware or network, and they turn a whole class of
merge damage into a named failure instead of a mystery.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (REPO_ROOT / "src" / "lerobot", REPO_ROOT / "tests")


def _iter_lerobot_imports() -> list[tuple[Path, int, str, str]]:
    """Every ``from lerobot... import NAME`` in the tree, at any nesting depth.

    Returns ``(file, lineno, module, name)`` tuples.

    Deliberately walks the whole AST rather than just ``tree.body``: imports
    written *inside* a function or method are exactly the ones a pytest
    collection pass never executes, so they are where dangling references
    survive undetected. ``load_subtasks`` was imported inside a helper.
    """
    found: list[tuple[Path, int, str, str]] = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            # This module's own imports are assertions about structure, not
            # usage — and some of them deliberately name post-merge locations
            # that do not exist yet. Scanning ourselves would make the guard
            # fail on its own forward-looking references.
            if path == Path(__file__).resolve():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                # Relative imports (level > 0) resolve against the containing
                # package; skipping them keeps this guard simple and they are
                # already covered by module-level import at collection time.
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                if not node.module or not node.module.startswith("lerobot"):
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    found.append((path, node.lineno, node.module, alias.name))
    return found


def _module_source_exists(module: str) -> bool:
    """True when ``module``'s source file is still present in this checkout.

    This is the discriminator between "the env lacks an optional extra" and
    "the merge deleted the module". Deliberately filesystem-based rather than
    exception-based: ``require_package`` raises a bare ``ImportError`` with no
    ``name`` attribute, and several packages (``lerobot.datasets``,
    ``lerobot.async_inference``) call it at import time. Keying off the
    exception would therefore flag a lean CI env — where those extras are
    legitimately absent — as merge damage, failing the build for no reason.

    A module whose file is on disk but won't import is an environment problem
    and not this guard's business. A module whose file is gone is exactly what
    this guard exists to catch.
    """
    rel = Path(*module.split("."))
    src = REPO_ROOT / "src"
    return (src / rel).with_suffix(".py").is_file() or (src / rel / "__init__.py").is_file()


def test_every_lerobot_import_resolves():
    """No ``from lerobot.x import y`` may reference a name that no longer exists.

    Pre: the working tree is importable (base deps installed).
    Post: on failure, every dangling (module, name) is listed with its call site.
    """
    references = _iter_lerobot_imports()
    assert references, "found no lerobot imports to check — scan roots are wrong"

    module_cache: dict[str, object | None] = {}
    dangling: list[str] = []

    for path, lineno, module, name in references:
        if module not in module_cache:
            try:
                module_cache[module] = importlib.import_module(module)
            except ImportError as exc:
                # Source still on disk → the env lacks an optional extra, which
                # is expected in a base test env. Source gone → merge damage.
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
        # A submodule is a legitimate `from pkg import submod` target even when
        # it is not an attribute of the parent package until first imported.
        if hasattr(mod, name):
            continue
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


def test_record_config_keeps_fork_only_fields():
    """``lerobot-record``'s config must stay a superset of upstream's.

    The fork subclasses upstream's ``DatasetRecordConfig`` to add fields that
    ``record()`` reads unconditionally. If a future merge flattens the subclass
    back to upstream's, ``record()`` dies at ``AttributeError`` deep inside the
    episode loop rather than at config-parse time — which is how it failed
    during the 2026-07 sync.

    Pre: ``lerobot.scripts.lerobot_record`` is importable. The subclass check
    is skipped before the 2026-07 upstream sync, which is what introduces
    ``lerobot.configs.dataset`` — so this guard arms itself automatically once
    the merge lands rather than needing to be enabled by hand.
    Post: the fork-only fields are present and the base class is still the
    upstream one (so upstream additions keep flowing in by inheritance).
    """
    from lerobot.scripts.lerobot_record import DatasetRecordConfig as ForkConfig, RecordConfig

    base_mod = pytest.importorskip(
        "lerobot.configs.dataset",
        reason="pre-sync tree still defines DatasetRecordConfig in lerobot_record",
    )
    assert issubclass(ForkConfig, base_mod.DatasetRecordConfig), (
        "lerobot_record.DatasetRecordConfig must subclass lerobot.configs.dataset."
        "DatasetRecordConfig so upstream field additions are inherited, not forked."
    )

    fields = {f.name for f in dataclasses.fields(ForkConfig)}
    for fork_only in ("record_images", "rename_map"):
        assert fork_only in fields, (
            f"fork-only field '{fork_only}' is gone from DatasetRecordConfig, but "
            f"record() still reads it — this is a merge regression, not a cleanup."
        )

    annotation = RecordConfig.__dataclass_fields__["dataset"].type
    # draccus/dataclasses may hand back the string form under postponed
    # evaluation; accept either spelling of the same contract.
    assert annotation in (ForkConfig, "DatasetRecordConfig"), (
        f"RecordConfig.dataset must be annotated with the fork's DatasetRecordConfig, got {annotation!r}"
    )


@pytest.mark.parametrize("name", ["load_subtasks", "load_info", "write_info"])
def test_dataset_io_helper_survives_somewhere(name: str):
    """The dataset IO helpers the GUI's feature editor depends on must still exist.

    Deliberately does NOT pin the module: upstream relocates these between
    ``io_utils`` and ``utils`` legitimately, and a guard that hard-codes today's
    address would cry wolf on every reshuffle. What must never happen is the
    symbol vanishing entirely — that is the ``load_subtasks`` failure, where the
    fork's callers import it from inside functions and a collection pass sees
    nothing wrong.

    Post: on failure the symbol is gone from every dataset module and fork
    callers are dangling; restore it or port them.
    """
    candidates = ("lerobot.datasets.io_utils", "lerobot.datasets.utils", "lerobot.datasets")
    any_imported = False
    for module in candidates:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        any_imported = True
        if hasattr(mod, name):
            return
    # None of the candidates would import at all — the env is missing a dataset
    # extra, not the symbol. Skip rather than fail: a lean CI runner must not be
    # told the merge deleted something when it simply cannot load the package.
    if not any_imported:
        pytest.skip(f"none of {candidates} importable in this env; cannot judge '{name}'")
    pytest.fail(f"'{name}' no longer exists in any of {candidates} — fork callers depend on it")
