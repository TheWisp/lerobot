#!/usr/bin/env python
"""Lint check: flag tests that call an external service.

Rationale: a test that reaches the network fails for reasons unrelated to the code
under test. This suite has hit both failure modes for real — `429 Too Many Requests`
when HuggingFace throttled CI, and `401 Unauthorized` for a dataset that existed only
on the developer's disk, because ``LeRobotDataset`` silently falls back to a Hub
download whenever a local read fails. Both produced a red build that said nothing
about the change being tested.

Use a committed fixture instead (see ``tests/artifacts/datasets/`` and
``tests.fixtures.local_artifacts.local_dataset``). When a test's *purpose* is the
remote service, say so:

  - ``@pytest.mark.hub_live`` — the marker this repo already deselects by default
  - ``# external-ok: <reason>`` on the offending line or the line above
  - ``# network-lint: ignore-file`` for a module that is entirely about the service

The check walks the AST rather than matching text, so defining a stub named
``snapshot_download`` or mentioning a call in a docstring is not a finding.

Run:
    python scripts/lint/no_network_in_tests.py tests/path/to/test.py [...]
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ANNOTATION = re.compile(r"#\s*external-ok:\s*\S+", re.IGNORECASE)
IGNORE_FILE = re.compile(r"#\s*network-lint:\s*ignore-file", re.IGNORECASE)
REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
LOCAL_URL = re.compile(r"^https?://(127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\])")
HUB_DOWNLOADERS = {"snapshot_download", "hf_hub_download"}
HTTP_VERBS = {"get", "post", "put", "delete", "head", "patch", "request"}
HTTP_LIBS = {"requests", "httpx", "urllib3", "aiohttp"}
DATASET_CTORS = {"LeRobotDataset", "LeRobotDatasetMetadata"}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _str_arg(node: ast.Call, index: int = 0) -> str | None:
    if len(node.args) > index and isinstance(node.args[index], ast.Constant):
        value = node.args[index].value
        return value if isinstance(value, str) else None
    return None


def _kw(node: ast.Call, name: str):
    return next((k for k in node.keywords if k.arg == name), None)


def _finding(node: ast.Call) -> str | None:
    """The reason this call reaches a remote service, or None."""
    name = _dotted(node.func)
    tail = name.rsplit(".", 1)[-1]

    if tail in DATASET_CTORS:
        repo = _str_arg(node) or (
            _kw(node, "repo_id").value.value
            if _kw(node, "repo_id") and isinstance(_kw(node, "repo_id").value, ast.Constant)
            else None
        )
        if isinstance(repo, str) and REPO_ID.match(repo) and not _kw(node, "root"):
            return f"loads {repo!r} from the Hub (no local root=)"
        return None

    if tail in HUB_DOWNLOADERS:
        return f"calls {tail}() — downloads from a remote hub"

    if tail == "from_pretrained":
        repo = _str_arg(node)
        if isinstance(repo, str) and REPO_ID.match(repo):
            return f"from_pretrained({repo!r}) — downloads from the Hub"
        return None

    parts = name.split(".")
    is_http = (len(parts) >= 2 and parts[0] in HTTP_LIBS and parts[-1] in HTTP_VERBS) or tail == "urlopen"
    if is_http and _remote_url(node):
        return f"{name or tail}() — HTTP request to a non-local host"
    return None


def _remote_url(node: ast.Call) -> bool:
    """True only when the URL is PROVABLY remote.

    A URL assembled from a variable is nearly always a local fixture server
    (`f"{base_url}/api/..."`), and flagging those would bury the real findings.
    """
    if not node.args:
        return False
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value.startswith(("http://", "https://")) and not LOCAL_URL.match(arg.value)
    if isinstance(arg, ast.JoinedStr) and arg.values:
        head = arg.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value.startswith(("http://", "https://")) and not LOCAL_URL.match(head.value)
    return False


def _hub_live_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Line ranges of functions/classes marked hub_live — the sanctioned escape."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            if "hub_live" in _dotted(dec) or (isinstance(dec, ast.Call) and "hub_live" in _dotted(dec.func)):
                out.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
    return out


def check(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    if IGNORE_FILE.search(source) or re.search(r"^pytestmark\s*=.*hub_live", source, re.M):
        return []

    lines = source.splitlines()
    live = _hub_live_ranges(tree)
    # A helper defined in this file shadows the flagged name (tests/datasets has its
    # own load_dataset); and a file that monkeypatches the downloader never reaches
    # the network through it.
    local_defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    mocks_network = any(
        isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value in HUB_DOWNLOADERS | {"from_pretrained"}
        for n in ast.walk(tree)
    )
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _dotted(node.func).rsplit(".", 1)[-1] in local_defs:
            continue
        why = _finding(node)
        if not why:
            continue
        if mocks_network and "Hub" in why:
            continue
        ln = node.lineno
        if any(start <= ln <= end for start, end in live):
            continue
        if ANNOTATION.search(lines[ln - 1]) or (ln > 1 and ANNOTATION.search(lines[ln - 2])):
            continue
        problems.append(f"{path}:{ln}: {why}\n    {lines[ln - 1].strip()}")
    return problems


def main(argv: list[str]) -> int:
    problems: list[str] = []
    for arg in argv:
        problems.extend(check(Path(arg)))
    if problems:
        print("Tests must not call external services:\n")
        print("\n".join(sorted(set(problems))))
        print(
            "\nUse a committed fixture (tests/artifacts/datasets/, "
            "tests.fixtures.local_artifacts.local_dataset).\n"
            "If the remote service IS what the test exercises, mark it "
            "@pytest.mark.hub_live or annotate `# external-ok: <reason>`."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
