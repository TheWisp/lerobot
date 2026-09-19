"""Host-side spawns must name their own interpreter, not trust PATH.

`python` is not guaranteed to exist on POSIX (PEP 394); Debian and Ubuntu have
shipped only `python3` for years, so a bare `"python"` argv[0] depends on the
optional `python-is-python3` package. The GUI server is long-lived and inherits
whatever PATH launched it — systemd, cron, `sg docker -c`, a desktop file — none
of which we control. On the SO-107 rig this is not hypothetical: the server is
started by absolute path into a venv whose `bin` is never prepended, so PATH
carries no `python` at all and the spawn dies with FileNotFoundError.

`sys.executable` needs no lookup and cannot drift from the parent.

The rule is about *whose environment defines the interpreter*, so it is scoped
rather than absolute: a command that runs inside an image we build should use a
bare name, because that image's Dockerfile sets PATH. Those sites are named in
CONTAINER_ENTRYPOINTS below, which is the assertion that actually carries this
test — an ordering or spelling check would pass while a new host-side spawn
quietly reintroduced the bug.
"""

from __future__ import annotations

import ast
from pathlib import Path

GUI_ROOT = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui"

BARE_INTERPRETERS = {"python", "python3"}

# Assignments whose argv runs inside a container we build, where PATH is set by
# the image (docker/Dockerfile.training exports PATH=/lerobot/.venv/bin:$PATH)
# and the *host's* sys.executable would be meaningless. Adding a name here must
# be a deliberate act, which is the point of listing them.
CONTAINER_ENTRYPOINTS = {
    "HVLA_FLOW_S1_ENTRYPOINT",  # gui/training/recipes.py — runs in lerobot-training:dev
}


def _enclosing_assignment(tree: ast.Module, node: ast.AST) -> str | None:
    """Name the assignment target a list literal belongs to, if any."""
    for parent in ast.walk(tree):
        if not isinstance(parent, (ast.Assign, ast.AnnAssign)):
            continue
        if parent.value is not node:
            continue
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        for t in targets:
            if isinstance(t, ast.Name):
                return t.id
    return None


def _bare_interpreter_spawns() -> list[str]:
    """Every list literal in the GUI package whose first element is a bare interpreter."""
    offenders: list[str] = []
    for path in sorted(GUI_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
                continue
            first = node.elts[0]
            if not isinstance(first, ast.Constant) or first.value not in BARE_INTERPRETERS:
                continue
            if _enclosing_assignment(tree, node) in CONTAINER_ENTRYPOINTS:
                continue
            rel = path.relative_to(GUI_ROOT.parents[2])
            offenders.append(f"{rel}:{node.lineno} starts with {first.value!r}")
    return offenders


def test_host_side_spawns_use_sys_executable() -> None:
    offenders = _bare_interpreter_spawns()
    assert not offenders, (
        "These argv lists resolve the interpreter through PATH, which carries no "
        "`python` on a stock Ubuntu or on the rig. Use sys.executable, or — if the "
        "command runs inside an image we build — name its constant in "
        "CONTAINER_ENTRYPOINTS:\n  " + "\n  ".join(offenders)
    )


def test_container_entrypoints_still_exist() -> None:
    """A stale allowlist silently widens the rule, so pin each entry to real code."""
    sources = "\n".join(p.read_text() for p in GUI_ROOT.rglob("*.py"))
    for name in CONTAINER_ENTRYPOINTS:
        assert f"{name} =" in sources, (
            f"{name} is allowlisted as a container entrypoint but no longer exists. "
            "Remove it from CONTAINER_ENTRYPOINTS rather than leaving the exemption."
        )
