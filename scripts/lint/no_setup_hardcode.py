#!/usr/bin/env python
"""Lint check: flag setup-specific identifiers hardcoded into general-purpose code.

Rationale: AI-assisted edits tend to "overfit" the code to one robot setup by
baking a concrete camera name, joint name, or robot identifier directly into
policy / processor code, e.g. ``batch["observation.images.top"]`` or a
``DEFAULT_S2_CAM_KEY_MAP`` of SO107 cameras. This silently breaks on any other
robot, camera layout, or dataset, and is very hard to catch in review.

Correct code derives these from config/features instead:
  - camera/image keys  -> config.image_features / dataset.meta.camera_keys
  - the key prefix      -> constants OBS_IMAGES / OBS_STATE / ACTION
  - motor/joint names   -> robot.action_features / robot.observation_features

How to silence a legitimate case:
  - a single line: add ``# hardcode-ok: <reason>`` on that line or the one above.
  - a whole module that is intentionally single-setup (e.g. an SO107-only
    subsystem): add ``# hardcode-lint: ignore-file - <reason>`` anywhere in it.

Scope: only fork-authored general-purpose code (policies/, processor/). Robot/
teleoperator definitions, env adapters, configs, tests, examples, one-off
scripts, AND files that exist in upstream lerobot are all excluded (upstream is
its own responsibility, and annotating it would churn on every rebase).

Run:
    python scripts/lint/no_setup_hardcode.py path/to/file.py [...]
    python scripts/lint/no_setup_hardcode.py --report            # audit (default scope)
    python scripts/lint/no_setup_hardcode.py --report --all --include-scripts

Exit 0 if no unannotated violations; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# String-literal patterns. Each is matched against the *value* of a string
# constant in the AST, so comments and docstrings are never matched.
#
# mode:
#   "search"      token may appear anywhere in the literal (token is unambiguous)
#   "key"         the whole literal must be the key, or a comma/space-joined list
#                 of keys (rejects prose that merely mentions a key)
#   "feature-key" the whole literal must be a single motor feature key
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    mode: str
    hint: str
    tier: str  # "high" = near-zero false positives; "medium" = real but noisier


# Specific arm joints are unambiguous tokens; "gripper" is generic (appears in
# log strings and end-effector interfaces) so it is only treated as a motor
# feature key when arm-prefixed or .pos/.vel-suffixed.
_ARM_JOINT = r"shoulder_pan|shoulder_lift|elbow_flex|forearm_roll|wrist_flex|wrist_roll"
_FEATURE_KEY = (
    rf"(?:left_|right_)?(?:{_ARM_JOINT})(?:\.pos|\.vel)?"  # arm joints, affixes optional
    r"|(?:left_|right_)gripper(?:\.pos|\.vel)?"  # prefixed gripper
    r"|gripper(?:\.pos|\.vel)"  # suffixed gripper (not bare)
)

PATTERNS: list[Pattern] = [
    Pattern(
        name="qualified-image-key",
        regex=re.compile(r"observation\.images\.[A-Za-z0-9_]+"),
        mode="key",
        hint="iterate config.image_features / dataset.meta.camera_keys; build keys with f'{OBS_IMAGES}.{name}'",
        tier="high",
    ),
    Pattern(
        name="internal-cam-key",
        regex=re.compile(r"\b(?:base|left_wrist|right_wrist|wrist)_\d+_rgb\b"),
        mode="search",
        hint="map keys from the policy/robot camera config, not a baked-in name",
        tier="high",
    ),
    Pattern(
        name="joint-name-key",
        regex=re.compile(rf"^(?:{_FEATURE_KEY})$"),
        mode="feature-key",
        hint="derive motor names from robot.action_features / robot.observation_features",
        tier="high",
    ),
    Pattern(
        name="bare-camera-name",
        regex=re.compile(
            r"^(?:top|front|side|wrist|left_wrist|right_wrist|wrist_left|wrist_right|"
            r"laptop|phone|overhead|egocentric|third_person)$"
        ),
        mode="feature-key",
        hint="don't assume a camera is named this; select from the available camera keys",
        tier="medium",
    ),
]

# Structural rule: a config field / variable named *_key assigned a concrete
# camera/image literal. Keys on the *target name*, so no value-collision FPs.
KEY_FIELD_RE = re.compile(r"(?:camera|image|cam|img|obs)_key$")

ANNOTATION = re.compile(r"#\s*hardcode-ok:\s*\S+", re.IGNORECASE)
FILE_IGNORE = re.compile(r"#\s*hardcode-lint:\s*ignore-file\b", re.IGNORECASE)

# Path segments where these identifiers legitimately live.
EXCLUDED_SEGMENTS = (
    "/robots/",
    "/teleoperators/",
    "/motors/",
    "/cameras/",
    "/envs/",
    "/tests/",
    "/examples/",
)

# Upstream code the fork relocated under a new path (upstream keeps it elsewhere),
# so the exact-path upstream-skip below misses it — but it still isn't ours to lint.
EXCLUDED_RELOCATED = ("/policies/sarm/",)  # upstream: src/lerobot/rewards/sarm/

DEFAULT_ROOTS = ("src/lerobot/policies", "src/lerobot/processor")

# We lint only fork-authored code. Files present in upstream lerobot are its
# responsibility, so they are skipped unless --include-upstream is passed.
UPSTREAM_REF = "upstream/main"


@dataclass
class Hit:
    path: Path
    line: int
    literal: str
    name: str
    hint: str
    tier: str
    annotated: bool


def _matches(pat: Pattern, value: str) -> bool:
    if pat.mode == "search":
        return pat.regex.search(value) is not None
    if pat.mode == "feature-key":
        return pat.regex.fullmatch(value) is not None
    if pat.mode == "key":
        # whole literal is a key or a comma/space-separated list of keys
        if not pat.regex.search(value):
            return False
        parts = re.split(r"[,\s]+", value.strip())
        return all(pat.regex.fullmatch(p) for p in parts if p)
    raise ValueError(pat.mode)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _key_field_targets(node: ast.AST) -> list[str]:
    """Names assigned by this node (AnnAssign/Assign) that look like *_key fields."""
    names: list[str] = []
    targets: list[ast.AST] = []
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = list(node.targets)
    for t in targets:
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, ast.Attribute):
            names.append(t.attr)
    return [n for n in names if KEY_FIELD_RE.search(n)]


def _concrete_camera_literal(value: ast.AST | None) -> str | None:
    """Return a concrete camera/image literal from a *_key field value, else None.

    Handles `"top"`, `OBS_IMAGES + ".top"`, and `field(default="top")`-style.
    A value that is None/""/a plain variable (config-driven) returns None.
    """
    if value is None:
        return None
    if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
        return value.value
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        for side in (value.left, value.right):
            # a concrete dotted suffix like ".top" appended to a key constant
            if (
                isinstance(side, ast.Constant)
                and isinstance(side.value, str)
                and re.fullmatch(r"\.[A-Za-z0-9_]+", side.value)
            ):
                return side.value
    if isinstance(value, ast.Call):  # field(default=..., default_factory=lambda: ...)
        for kw in value.keywords:
            if kw.arg == "default":
                return _concrete_camera_literal(kw.value)
    return None


def check_file(path: Path) -> list[Hit]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if FILE_IGNORE.search(text):
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    lines = text.splitlines()
    skip = _docstring_nodes(tree)
    parent = _parent_map(tree)
    hits: list[Hit] = []

    for node in ast.walk(tree):
        # string-literal patterns
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            for pat in PATTERNS:
                if _matches(pat, node.value):
                    hits.append(
                        _make_hit(path, node, node.value, pat.name, pat.hint, pat.tier, lines, parent)
                    )
                    break  # first (most specific) pattern wins

        # structural rule: *_key field assigned a concrete camera literal
        if isinstance(node, (ast.AnnAssign, ast.Assign)):
            fields = _key_field_targets(node)
            if fields:
                lit = _concrete_camera_literal(node.value)
                if lit is not None:
                    hits.append(
                        _make_hit(
                            path,
                            node,
                            f"{fields[0]} = ...{lit!r}",
                            "camera-key-default",
                            "make the key config-driven (default None -> resolve from features) instead of a fixed camera",
                            "high",
                            lines,
                            parent,
                        )
                    )
    return hits


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    return parent


def _enclosing_stmt(node: ast.AST, parent: dict[int, ast.AST]) -> ast.AST | None:
    """Nearest enclosing statement, so one annotation on a declaration line (e.g.
    the line of/above ``JOINT_NAMES = [``) covers every literal inside it."""
    cur = parent.get(id(node))
    while cur is not None and not isinstance(cur, ast.stmt):
        cur = parent.get(id(cur))
    return cur


def _make_hit(
    path: Path,
    node: ast.AST,
    literal: str,
    name: str,
    hint: str,
    tier: str,
    lines: list[str],
    parent: dict[int, ast.AST],
) -> Hit:
    line_no = getattr(node, "lineno", 0)
    annotated = _is_annotated(lines, line_no)
    if not annotated:
        stmt = _enclosing_stmt(node, parent)
        if stmt is not None and getattr(stmt, "lineno", None):
            annotated = _is_annotated(lines, stmt.lineno)
    return Hit(path, line_no, literal, name, hint, tier, annotated)


def _is_annotated(lines: list[str], line_no: int) -> bool:
    idx = line_no - 1
    if 0 <= idx < len(lines) and ANNOTATION.search(lines[idx]):
        return True
    j = idx - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    return j >= 0 and ANNOTATION.search(lines[j]) is not None


def _excluded(path: Path, include_scripts: bool) -> bool:
    p = "/" + str(path).replace("\\", "/").lstrip("/")
    if any(seg in p for seg in EXCLUDED_SEGMENTS):
        return True
    if any(seg in p for seg in EXCLUDED_RELOCATED):
        return True
    if not include_scripts and "/scripts/" in p:
        return True
    return path.name.startswith("test_")


def _upstream_paths(paths: list[Path]) -> set[str]:
    """Repo-relative paths (of `paths`) that exist in upstream lerobot.

    Best-effort: one batched `git ls-tree`. If the upstream ref or git is
    unavailable (e.g. CI without the remote fetched), returns empty set so we
    fall back to scanning everything (upstream code is clean under this ruleset).
    """
    if not paths:
        return set()
    try:
        out = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", UPSTREAM_REF, "--", *[str(p) for p in paths]],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _gather(argv_files: list[str]) -> list[Path]:
    if argv_files:
        return [Path(a) for a in argv_files if a.endswith(".py")]
    files: list[Path] = []
    for root in DEFAULT_ROOTS:
        rp = Path(root)
        if rp.is_dir():
            files.extend(rp.rglob("*.py"))
    return files


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--report", action="store_true", help="audit mode: list all hits grouped by tier")
    ap.add_argument("--all", action="store_true", help="also enforce medium tier (bare camera names)")
    ap.add_argument("--include-scripts", action="store_true", help="also scan */scripts/* utilities")
    ap.add_argument(
        "--include-upstream",
        action="store_true",
        help="also scan files that exist in upstream lerobot (default: skip them)",
    )
    args = ap.parse_args(argv)

    tiers_on = {"high", "medium"} if args.all else {"high"}

    candidates = [p for p in _gather(args.files) if p.is_file() and not _excluded(p, args.include_scripts)]
    if not args.include_upstream:
        upstream = _upstream_paths(candidates)
        candidates = [p for p in candidates if str(p) not in upstream]

    all_hits: list[Hit] = []
    for path in candidates:
        all_hits.extend(check_file(path))

    if args.report:
        return _report(all_hits)

    violations = [h for h in all_hits if not h.annotated and h.tier in tiers_on]
    for h in violations:
        print(f"{h.path}:{h.line}: hardcoded setup identifier [{h.name}]: {h.literal!r}")
        print(f"    fix: {h.hint}")
    if violations:
        print("\nIf a value is genuinely setup-specific and intentional, annotate it:")
        print("    # hardcode-ok: <one-line reason>   (single line)")
        print("    # hardcode-lint: ignore-file - <reason>   (whole single-setup module)")
        return 1
    return 0


def _report(hits: list[Hit]) -> int:
    for tier in ("high", "medium"):
        group = [h for h in hits if h.tier == tier]
        print(
            f"\n===== tier={tier}  ({len(group)} hits, "
            f"{sum(not h.annotated for h in group)} unannotated) ====="
        )
        for h in sorted(group, key=lambda x: (str(x.path), x.line)):
            flag = "  [annotated]" if h.annotated else ""
            print(f"{h.path}:{h.line}: [{h.name}] {h.literal!r}{flag}")
    print(f"\nTOTAL: {len(hits)} hits across {len({h.path for h in hits})} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
