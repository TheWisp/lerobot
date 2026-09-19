#!/usr/bin/env python
"""Lint check: flag issue references that GitHub will read as a closing instruction.

GitHub closes an issue when a commit message merged to the default branch — or a
pull request body — contains one of `close, closes, closed, fix, fixes, fixed,
resolve, resolves, resolved` followed by an issue reference. The match is plain
pattern matching. It has no notion of grammar, and there is no repository or
organisation setting that turns it off.

In particular it does not understand negation. Issue #98 was closed one second
after PR #108 merged, by a sentence written to say the opposite:

    "It does not close #98. The option naming ... are untouched"

Both the PR body and a commit body carried that sentence, so it would have fired
twice. The issue then read as fixed while the defect it described was untouched,
which is worse than never having filed it.

The rule here is about placement, because placement is what separates the
deliberate case from the accident:

  - A closing keyword on its own line, in the conventional trailer form, is
    deliberate and allowed:      Closes #109
  - The same keyword mid-sentence is flagged, whether the sentence is negated
    ("does not close #98") or merely descriptive ("the fix #98 proposed").

Mid-prose, refer to an issue without a closing keyword — `Refs #98`, `see #98`,
`the problem #98 describes`. That reads the same to a human and is inert to
GitHub.

Run on a commit message (this is the commit-msg hook), or on any file holding a
body you are about to post:

    python scripts/lint/no_accidental_issue_close.py .git/COMMIT_EDITMSG
    python scripts/lint/no_accidental_issue_close.py pr_body.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

KEYWORDS = "close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved"

# An issue reference GitHub will link: #12, owner/repo#12, or a full issue URL.
REFERENCE = r"(?:[\w.-]+/[\w.-]+)?#\d+|https?://github\.com/[\w.-]+/[\w.-]+/issues/\d+"

# Any keyword followed by a reference, wherever it appears.
CLOSING = re.compile(rf"\b({KEYWORDS})\b[:\s]+({REFERENCE})", re.IGNORECASE)

# The deliberate form: the whole line is the trailer, nothing else on it.
TRAILER = re.compile(rf"^\s*({KEYWORDS})\b[:\s]+({REFERENCE})\s*$", re.IGNORECASE)


def findings(text: str) -> list[tuple[int, str, str]]:
    """Return (line number, matched phrase, whole line) for each risky reference.

    Pre: ``text`` is a commit message or a PR/issue body.
    Post: trailer-form lines are absent from the result; every other keyword and
    reference pair is present exactly once per match.
    """
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        # git strips comment lines from a commit message before using it.
        if line.lstrip().startswith("#"):
            continue
        if TRAILER.match(line):
            continue
        for m in CLOSING.finditer(line):
            out.append((lineno, m.group(0), line.strip()))
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: no_accidental_issue_close.py <file> [...]", file=sys.stderr)
        return 2

    failed = False
    for name in argv:
        path = Path(name)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, phrase, line in findings(text):
            failed = True
            print(f"{path}:{lineno}: GitHub will close an issue on {phrase!r}")
            print(f"    {line}")

    if failed:
        print()
        print(
            "GitHub matches these keywords anywhere in the text and ignores negation:\n"
            '  "does not close #98" closes #98.\n'
            "\n"
            "Mid-sentence, drop the keyword — `Refs #98`, `see #98`, `the problem\n"
            "#98 describes` all read the same and are inert. If you do mean to close\n"
            "the issue, put it on its own line as a trailer: `Closes #98`."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
