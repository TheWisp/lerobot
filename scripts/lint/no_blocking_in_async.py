#!/usr/bin/env python
"""Lint check: flag blocking work reachable from the GUI server's event loop.

Rationale: the GUI is one asyncio process. A synchronous network or subprocess
call inside an ``async def`` blocks *everything* — static files, websockets,
SSE, other users' requests — for as long as that call takes. This is not
theoretical: a blackholed Hub made the whole GUI unusable until kernel TCP
timeouts fired, because ``whoami()`` was awaited inline. ``/training/image-status``
still shells out to ``git`` and ``docker`` — including a ``git fetch`` — on every
page load.

The three rules this enforces:

1. **No blocking call directly inside an ``async def``.** Offload it.
2. **No offloading to the shared default executor** (``run_in_executor(None, …)``
   or ``asyncio.to_thread``) for anything that can touch the network. That pool
   is shared with video decode, camera teardown and FastAPI's own sync handlers,
   so one stalled Hub probe starves work that has nothing to do with it. Use a
   named, bounded ``ThreadPoolExecutor`` per class of work.
3. **No network I/O on a passive GET.** A GET returns cached state and its
   freshness; refreshing happens in a background task or an explicit POST.

Note that a timeout on the *await* does not bound the work: cancelling
``asyncio.wait_for(asyncio.to_thread(f))`` abandons the await while ``f`` keeps
running and keeps holding its slot in the pool. The timeout has to be passed to
the client itself — ``HfApi(timeout=…)``, ``requests(…, timeout=…)``,
``subprocess.run(…, timeout=…)``.

How to silence a legitimate case:
  - a single line: add ``# blocking-ok: <reason>`` on that line or the one above.
    Legitimate cases exist — startup paths that run before the loop serves
    traffic, and CPU work already confined to its own bounded executor.
  - a whole module: ``# blocking-lint: ignore-file - <reason>``.

Scope: the GUI server only. Everything else in the tree is either a CLI (where
blocking is correct) or a subprocess the GUI supervises.

Run:
    python scripts/lint/no_blocking_in_async.py path/to/file.py [...]
    python scripts/lint/no_blocking_in_async.py --report          # inventory, exit 0
    python scripts/lint/no_blocking_in_async.py --report --warn-only

Exit 0 if no unannotated violations; non-zero otherwise. ``--warn-only`` always
exits 0, for staging the migration before the rule is enforced.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE = "src/lerobot/gui"

LINE_ESCAPE = "blocking-ok:"
FILE_ESCAPE = "blocking-lint: ignore-file"

# Dotted call names that block on the network or on another process. Matched
# against the tail of the dotted path, so `requests.get` catches
# `foo.requests.get` too.
BLOCKING_CALLS: dict[str, str] = {
    "subprocess.run": "await asyncio.create_subprocess_exec, or offload to a named executor",
    "subprocess.check_output": "await asyncio.create_subprocess_exec",
    "subprocess.check_call": "await asyncio.create_subprocess_exec",
    "subprocess.call": "await asyncio.create_subprocess_exec",
    "requests.get": "use an async client, or offload with a client-side timeout",
    "requests.post": "use an async client, or offload with a client-side timeout",
    "requests.put": "use an async client, or offload with a client-side timeout",
    "requests.head": "use an async client, or offload with a client-side timeout",
    "urlopen": "use an async client, or offload with a client-side timeout",
    "whoami": "route through the probe helper (cached, single-flight, bounded)",
    "snapshot_download": "route through hub_worker — transfers never run in-process",
    "upload_large_folder": "route through hub_worker — transfers never run in-process",
    "hf_hub_download": "route through hub_worker — transfers never run in-process",
    "dataset_info": "route through the probe helper (cached, single-flight, bounded)",
    "list_repo_files": "route through the probe helper (cached, single-flight, bounded)",
    "time.sleep": "await asyncio.sleep",
}

SHARED_POOL_HINT = (
    "the shared default executor is contended with decode/camera/FastAPI work; "
    "use a named bounded ThreadPoolExecutor for this class of work"
)


@dataclass
class Hit:
    path: str
    line: int
    kind: str
    detail: str
    hint: str
    annotated: bool


def _dotted(node: ast.AST) -> str:
    """Reconstruct a dotted call name; '' when it is not a plain attribute path."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _match_blocking(dotted: str) -> tuple[str, str] | None:
    if not dotted:
        return None
    for name, hint in BLOCKING_CALLS.items():
        if dotted == name or dotted.endswith("." + name) or dotted.split(".")[-1] == name:
            return name, hint
    return None


def _blocking_sync_helpers(tree: ast.Module) -> dict[str, str]:
    """Module-level sync functions that block, directly or via another such helper.

    The freeze that motivated this lint was exactly this shape: an async endpoint
    whose body is ``return get_auth_status()``, with ``whoami()`` one level down.
    Checking only direct calls inside ``async def`` would report the codebase
    clean while the bug sat in plain sight, so resolve calls within the module to
    a fixed point. Cross-module chains are still missed — a known limit, not a
    claim of completeness.
    """
    bodies: dict[str, ast.FunctionDef] = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    blocking: dict[str, str] = {}

    def direct_hit(fn: ast.FunctionDef) -> str | None:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                match = _match_blocking(_dotted(node.func))
                if match:
                    return match[0]
        return None

    for name, fn in bodies.items():
        hit = direct_hit(fn)
        if hit:
            blocking[name] = hit

    changed = True
    while changed:  # propagate along call edges until stable
        changed = False
        for name, fn in bodies.items():
            if name in blocking:
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    callee = _dotted(node.func).split(".")[-1]
                    if callee in blocking and callee != name:
                        blocking[name] = f"{callee}() -> {blocking[callee]}"
                        changed = True
                        break
    return blocking


class _Visitor(ast.NodeVisitor):
    """Collect blocking calls, tracking whether we are inside an ``async def``."""

    def __init__(self, path: str, lines: list[str], blocking_helpers: dict[str, str] | None = None):
        self.path = path
        self.lines = lines
        self.helpers = blocking_helpers or {}
        self.hits: list[Hit] = []
        self._async_depth = 0

    # A nested sync def inside an async def runs wherever it is called from --
    # usually handed to an executor -- so it is not itself a violation.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        outer, self._async_depth = self._async_depth, 0
        self.generic_visit(node)
        self._async_depth = outer

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._async_depth += 1
        self.generic_visit(node)
        self._async_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted(node.func)

        if self._async_depth:
            match = _match_blocking(dotted)
            if match:
                name, hint = match
                self._record(node.lineno, "blocking-call", f"{dotted}() inside async def", hint)
            else:
                callee = dotted.split(".")[-1]
                if callee in self.helpers:
                    self._record(
                        node.lineno,
                        "blocking-via-helper",
                        f"{callee}() inside async def, which reaches {self.helpers[callee]}",
                        "offload the helper to a named bounded executor, or make the endpoint sync",
                    )

        # Shared-pool offload is a violation wherever it appears: the pool is
        # process-global, so the damage does not depend on the caller.
        if dotted.endswith("run_in_executor") and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value is None:
                self._record(node.lineno, "shared-pool", "run_in_executor(None, ...)", SHARED_POOL_HINT)
        if dotted.endswith("asyncio.to_thread") or dotted == "to_thread":
            self._record(node.lineno, "shared-pool", "asyncio.to_thread(...)", SHARED_POOL_HINT)

        self.generic_visit(node)

    def _record(self, lineno: int, kind: str, detail: str, hint: str) -> None:
        self.hits.append(Hit(self.path, lineno, kind, detail, hint, annotated=self._is_annotated(lineno)))

    def _is_annotated(self, lineno: int) -> bool:
        for idx in (lineno - 1, lineno - 2):
            if 0 <= idx < len(self.lines) and LINE_ESCAPE in self.lines[idx]:
                return True
        return False


def check_file(path: Path) -> list[Hit]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if FILE_ESCAPE in source:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # ruff owns syntax; do not double-report

    try:
        rel = str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
    except ValueError:
        rel = str(path)  # outside the repo (a test tmpdir, a sibling checkout)
    visitor = _Visitor(rel, source.splitlines(), _blocking_sync_helpers(tree))
    visitor.visit(tree)
    return visitor.hits


def _scope_files() -> list[Path]:
    return sorted((REPO_ROOT / DEFAULT_SCOPE).rglob("*.py"))


BASELINE_PATH = Path(__file__).with_name("no_blocking_in_async_baseline.txt")


def _read_baseline() -> dict[tuple[str, str], int]:
    if not BASELINE_PATH.is_file():
        return {}
    out: dict[tuple[str, str], int] = {}
    for line in BASELINE_PATH.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        path, kind, count = line.rsplit(" ", 2)
        out[(path.strip(), kind)] = int(count)
    return out


def _write_baseline(hits: list[Hit]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for h in hits:
        counts[(h.path, h.kind)] = counts.get((h.path, h.kind), 0) + 1
    lines = [
        "# Accepted event-loop blocking, per (file, kind). Regenerate with",
        "#   python scripts/lint/no_blocking_in_async.py --update-baseline",
        "# These numbers should only ever go DOWN.",
    ]
    lines += [f"{path} {kind} {n}" for (path, kind), n in sorted(counts.items())]
    BASELINE_PATH.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="files to check (default: the GUI package)")
    ap.add_argument("--report", action="store_true", help="inventory mode: group by kind, exit 0")
    ap.add_argument("--warn-only", action="store_true", help="print violations but always exit 0")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from current state")
    args = ap.parse_args(argv)

    paths = [Path(f) for f in args.files] if args.files else _scope_files()
    paths = [p for p in paths if p.suffix == ".py" and DEFAULT_SCOPE in str(p).replace("\\", "/")]

    hits: list[Hit] = []
    for path in paths:
        hits.extend(check_file(path))

    if args.report:
        by_kind: dict[str, list[Hit]] = {}
        for h in hits:
            by_kind.setdefault(h.kind, []).append(h)
        for kind, group in sorted(by_kind.items()):
            print(f"\n=== {kind}: {len(group)} ===")
            for h in sorted(group, key=lambda x: (x.path, x.line)):
                flag = "  [annotated]" if h.annotated else ""
                print(f"  {h.path}:{h.line}: {h.detail}{flag}")
        unannotated = [h for h in hits if not h.annotated]
        print(
            f"\nTOTAL: {len(hits)} hits ({len(unannotated)} unannotated) across "
            f"{len({h.path for h in hits})} files"
        )
        return 0

    unannotated = [h for h in hits if not h.annotated]

    # Ratchet: the debt below is accepted, anything new is not. Counted per
    # (file, kind) rather than by line so ordinary edits above a violation do
    # not trip it, while adding one does. Shrink the baseline as areas migrate.
    if args.update_baseline:
        _write_baseline(unannotated)
        print(f"baseline updated: {len(unannotated)} accepted violations")
        return 0

    baseline = _read_baseline()
    if baseline and not args.files:
        current: dict[tuple[str, str], int] = {}
        for h in unannotated:
            current[(h.path, h.kind)] = current.get((h.path, h.kind), 0) + 1
        regressions = [
            (key, n, baseline.get(key, 0)) for key, n in current.items() if n > baseline.get(key, 0)
        ]
        if not regressions:
            return 0
        for (path, kind), now, was in sorted(regressions):
            print(f"{path}: {kind} went from {was} to {now} accepted-blocking sites")
        print("\nNew blocking work on the GUI event loop. Offload it, annotate it with")
        print(f"    # {LINE_ESCAPE} <reason>")
        print("or, if you deliberately accept it, run --update-baseline and say why in the commit.")
        return 0 if args.warn_only else 1

    for h in unannotated:
        print(f"{h.path}:{h.line}: blocking work on the event loop [{h.kind}]: {h.detail}")
        print(f"    fix: {h.hint}")
    if unannotated:
        print("\nA timeout on the await does not bound the work — cancelling wait_for leaves")
        print("the thread running. Pass the timeout to the client itself.")
        print("If a case is genuinely safe (startup path, already-bounded executor), annotate it:")
        print(f"    # {LINE_ESCAPE} <one-line reason>")
        print(f"    # {FILE_ESCAPE} - <reason>   (whole module)")

    return 0 if args.warn_only else (1 if unannotated else 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
