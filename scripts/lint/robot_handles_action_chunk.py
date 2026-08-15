#!/usr/bin/env python3
"""Every ``Robot.send_action`` must accept an ``ActionChunk``.

``Robot.send_action`` is typed ``RobotAction | ActionChunk`` and its docstring
states the contract: chunk-aware robots use the horizon, chunk-unaware robots
call :func:`lerobot.types.action_first_frame` to fall back to ``frames[0]``.

The second half was an unenforced convention, and it was not honoured. HVLA S1
sends a chunk by default, so running it on a robot that skipped the call raised
``AttributeError: 'ActionChunk' object has no attribute 'items'`` from inside
the driver — a message naming neither the chunk, the flag that produced it, nor
the robots that support it. Three of the tree's robots implemented the fallback;
the rest inherited the failure.

A subclass that overrides ``send_action`` therefore has to declare which side of
the contract it is on:

* call ``action_first_frame`` — collapse the horizon and use frame 0; or
* accept ``ActionChunk`` in the annotation and read ``.frames`` — chunk-aware; or
* annotate ``# chunk-ok: <reason>`` when neither applies.

Checked structurally rather than by running it, because constructing these
classes needs hardware.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ANNOTATION = re.compile(r"#\s*chunk-ok:")


def _handles_chunk(fn: ast.FunctionDef) -> bool:
    """True if this ``send_action`` deals with a horizon in some deliberate way."""
    for node in ast.walk(fn):
        # The fallback: action = action_first_frame(action)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "action_first_frame"
        ):
            return True
        # Chunk-aware: reads .frames / .fps off the argument.
        if isinstance(node, ast.Attribute) and node.attr in {"frames", "fps"}:
            return True
        # Delegates wholesale to another send_action (wrappers, bimanual splits
        # that forward the untouched argument) — the callee is checked on its
        # own, so flagging here would report the same defect twice.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "send_action"
            and _forwards_arg(node, fn)
        ):
            return True
    return False


def _forwards_arg(call: ast.Call, fn: ast.FunctionDef) -> bool:
    """True if `call` passes this function's action parameter through untouched."""
    names = {a.arg for a in fn.args.args[1:2]}  # the first non-self parameter
    return any(isinstance(a, ast.Name) and a.id in names for a in call.args)


def check(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    problems: list[str] = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name != "send_action" or not fn.body:
                continue
            # Abstract declarations carry no behaviour to get wrong.
            body = [s for s in fn.body if not isinstance(s, ast.Expr)]
            if not body or all(isinstance(s, ast.Pass | ast.Raise) for s in body):
                continue
            ln = fn.lineno
            # Walk up over decorators, comments and blanks: the annotation is a
            # comment block that may sit several lines above the `def`, and a
            # fixed-size window silently misses a long justification — which is
            # exactly the kind of near-miss this rule exists to catch.
            i = ln - 2
            annotated = False
            while i >= 0:
                stripped = lines[i].strip()
                if not stripped or stripped.startswith(("#", "@")):
                    if ANNOTATION.search(stripped):
                        annotated = True
                        break
                    i -= 1
                    continue
                break
            if annotated:
                continue
            if _handles_chunk(fn):
                continue
            problems.append(
                f"{path}:{ln}: {cls.name}.send_action ignores the ActionChunk half of its contract"
            )
    return problems


BASELINE_PATH = Path(__file__).with_name("robot_handles_action_chunk_baseline.txt")


def _read_baseline() -> set[str]:
    """Files whose existing violations are accepted. This list only shrinks."""
    if not BASELINE_PATH.exists():
        return set()
    out: set[str] = set()
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def main(argv: list[str]) -> int:
    baseline = _read_baseline()
    problems: list[str] = []
    for arg in argv:
        # Baselined files are upstream robots carrying the same gap. Fixing them
        # here would mean editing a dozen upstream-owned drivers for a defect
        # that only fires when a chunk-sending policy is pointed at them, and
        # every such edit is rebase conflict surface. Recorded instead, so the
        # count can only go down and a new robot cannot join them silently.
        if str(Path(arg)) in baseline:
            continue
        problems.extend(check(Path(arg)))
    if problems:
        print("Robot.send_action must accept an ActionChunk:\n")
        print("\n".join(sorted(set(problems))))
        print(
            "\nRobot.send_action is typed `RobotAction | ActionChunk`. A policy "
            "may send either.\n"
            "  - no lookahead controller: `action = action_first_frame(action)` "
            "at the top of the method\n"
            "  - chunk-aware: accept ActionChunk in the annotation and read "
            "`.frames`\n"
            "  - neither applies: annotate `# chunk-ok: <reason>` above the def\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
