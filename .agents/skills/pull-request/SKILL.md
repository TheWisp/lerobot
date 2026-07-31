---
name: pull-request
description: How to write a pull request for this repository — structure, the evidence each change type owes a reviewer, and the mechanics of getting images and links to render. Use when opening a PR, rewriting a PR body, or reviewing whether a PR description is complete.
---

# Pull requests

A PR body is read once, by someone deciding whether to trust the change. It has
one job: make that decision cheap. Everything below serves that.

Read [references/templates.md](references/templates.md) for the per-type
skeletons and [references/mechanics.md](references/mechanics.md) before
embedding an image or editing a PR via the CLI — both have failure modes that
waste a round trip.

## Non-negotiables

**Lead with the problem, not the solution.** The first paragraph says what was
broken or missing and why it mattered. A reviewer who disagrees with the
problem statement should be able to stop reading there. Never open with a list
of files changed.

**Every claim owes evidence, and the evidence type follows the change type.**

| Change   | What the reviewer needs to see                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| Feature  | It working. Screenshot, GIF, or short video for anything with a UI; a rendered artefact or captured transcript otherwise |
| Bug fix  | The bug is gone. Ideally the failing case before/after; at minimum, the specific scenario now exercised by a test        |
| Refactor | Behaviour is unchanged. Name the tests that pin the old behaviour and still pass; if you had to change a test, say why   |
| Perf     | A measurement, with the setup stated. "Faster" without numbers is not a claim                                            |

**Name guardrails, do not count them.** State _what_ is now protected and
_where_ the test lives — `tests/gui/test_overlays.py` covers the publisher
no-op contract and the lifecycle state machine. Do **not** write "45 tests
pass" or "10 passing". Counts go stale the moment anyone adds a case, they
invite pointless body edits, and they measure volume rather than coverage. If a
guardrail was added _because_ of a specific defect, say which defect — that is
the sentence that stops the bug coming back.

**Be self-contained.** The reviewer should not need to open a linked design doc,
a Slack thread, or a prior PR to understand what this one does. Link those for
depth, but the body must stand alone. Assume the reader has not been following
the work.

**Say what is deliberately not in scope.** A PR that lands a framework with one
example step should say so, and name the follow-ups. Scope stated up front reads
as judgment; scope discovered in review reads as an omission.

## Length

Succinct, not thin. Most PRs here land between roughly 2,000 and 4,000
characters of body. Use tables to enumerate states, behaviours, or
before/after pairs — they compress well and are skimmable. Prose is for the
problem statement and the reasoning; tables are for the surface area.

Cut: file-by-file walkthroughs, restatements of the diff, changelog-style
bullet dumps, and anything the reader can see in the Files tab.

## Honesty

State known limitations, deliberate shortcuts, and anything left broken. If a
test is failing for a pre-existing reason, say so and say why it is out of
scope — a reviewer who finds it themselves has to wonder what else you did not
mention. If a fix is a workaround rather than a root-cause fix, label it and
record the follow-up. A PR body that admits one weakness is more trustworthy
than one that admits none.

Never describe a behaviour you have not observed. "Should now work" means it
has not been run.

## Checklist before opening

- Problem stated before solution, in the first paragraph
- Evidence matching the change type is present and actually shows the claim
- Guardrails named by file and contract, with no test counts
- Out-of-scope work and follow-ups called out
- Known limitations stated
- Every link absolute; every image commit-pinned (see references/mechanics.md)
- Body makes sense to someone who has not read the branch
