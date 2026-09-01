---
name: pull-request
description: How to write a pull request for this repository — structure, the evidence each change type owes a reviewer, the pre-review audit of the branch diff, and the mechanics of getting images and links to render. Use when opening a PR, rewriting a PR body, preparing a branch for review or merge, or reviewing whether a PR description is complete.
---

# Pull requests

A PR body is read once, by someone deciding whether to trust the change. It has
one job: make that decision cheap. Everything below serves that.

Read [references/templates.md](references/templates.md) for the per-type
skeletons and [references/mechanics.md](references/mechanics.md) before
embedding an image, editing a PR via the CLI, **squashing the branch into
logical commits**, or **rebasing a branch that sits on another branch** — all
have failure modes that waste a round trip, and the stacked rebase silently
flattens the stack when done wrong.

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

**Prefer a diagram to a paragraph.** If you are explaining in prose _why the
design is shaped this way_ — which of three paths is taken, what talks to what,
what order things happen in — that is a diagram, and it will be shorter and
clearer. GitHub renders Mermaid directly in a PR body, so a diagram costs no
image hosting, no commit pinning, and stays diffable. Keep it to five to eight
nodes: a diagram that needs study is worse than the paragraph it replaced.

**Screenshots show one state each, cropped to the component.** Not a full
desktop, not a whole browser window with unrelated panels — the thing you
changed, with just enough surroundings to locate it. Give each shot a heading
naming the state it shows. Several small, labelled shots beat one busy one.

**Say what you could not capture.** If a state needs hardware, a long training
run, or a live subprocess you do not have, show what you can and state plainly
which state is missing. Never synthesise a screenshot by injecting fake state
into the UI — a fabricated shot of a state you never observed is worse than no
shot, because it looks like proof.

See [references/evidence.md](references/evidence.md) for how to produce both.

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

Budget the **main text**, not the whole body. The main text is everything
before the first appendix heading, and it should stay short enough to read in
one pass — a screen or so. Appendices can be as long as the change deserves,
because a reader chooses whether to open them.

What that buys: a reviewer who trusts you reads the summary and approves; a
reviewer who wants the receipts scrolls. Neither is punished for the other's
needs.

Use tables to enumerate states, behaviours, or before/after pairs — they
compress well and skim well. Prose is for the problem and the reasoning.

Cut: file-by-file walkthroughs, restatements of the diff, changelog-style
bullet dumps, and anything the reader can see in the Files tab.

## Shape: problem → solution → confidence, then appendices

The main text answers three questions in this order, and nothing else:

1. **Problem** — why this change exists. What was broken, missing, or costly.
2. **Solution** — what the change actually is. The shape of it, not its history.
3. **Confidence** — why the reviewer can believe it works. Evidence and
   guardrails, stated as conclusions.

Everything else is an appendix: per-regression detail, known limitations,
follow-ups, links to related branches, reproduction notes. Put it under clear
headings after the summary so it can be skipped or skimmed.

Do **not** write "how it went". A PR is not a lab notebook. The order you
discovered things, the hypotheses you discarded, the environment that misled
you — none of that belongs in the main text, and most of it belongs nowhere.
The reviewer wants the current state of the code, not the path that produced
it. Where a discovery genuinely informs future work — a defect class that will
recur, a trap the next person will hit — state it once as a finding in an
appendix, not as a narrative.

## Tone

Plain and factual. The change has to be judged on what it does, so let the
facts carry it.

Avoid drama. No "silently", "entirely", "outright", "took N tests with it", no
bolded warnings, no build-up before a reveal. Emphasis used everywhere is
emphasis nowhere, and a body that sounds urgent invites a reviewer to discount
it. A serious defect reads as more serious in flat prose.

Titles describe the change, not its excitement. "sync upstream through
`<sha>`" beats "sync upstream and fix what it hid".

## Honesty

State known limitations, deliberate shortcuts, and anything left broken. If a
test is failing for a pre-existing reason, say so and say why it is out of
scope — a reviewer who finds it themselves has to wonder what else you did not
mention. If a fix is a workaround rather than a root-cause fix, label it and
record the follow-up. A PR body that admits one weakness is more trustworthy
than one that admits none.

Never describe a behaviour you have not observed. "Should now work" means it
has not been run.

That bar applies to inherited prose too. A body describing a codec fix as
repairing rejected hardware codecs and stalled camera reads was quoting an
earlier PR — whose own words were "could reject" and "could stall". The rig logs
showed the codec selected 864 times as `libsvtav1` and once as `h264_nvenc`, in
a capability probe: nothing had ever failed. **A previous author's "could" is
not your "did".** Repeating an unverified claim launders it into a fact.

## Audit the branch before you describe it

Writing the body is not a substitute for reading the code. Before opening or
marking ready, read the whole branch diff at its final scope — **Green suite,
then read the diff** in the `verifying-changes` skill lists what that pass looks
for, none of which a passing suite can see.

What it finds is fixed in the branch, not confessed in the body. What belongs in
the body is the limitation you chose to keep.

**Search the repository's own backlog for the problem first** — `gh issue list`,
and the `TODO.md` files still draining. Someone may have already written the
problem down, with measurements you would otherwise re-derive or miss: the
Docker layer-order entry carried an 8.4 GB figure that made the change's real
benefit republish and re-pull cost, not just local rebuild time, which the body
had under-claimed. Finding it is also what obliges you to delete it, so the work
does not land leaving a second record of itself behind.

## Checklist before opening

- The branch diff has been audited at final scope, not just tested
- Every fix-up to this branch's own work is folded into the commit it fixes
- The backlog was searched for the problem; any entry it closes is deleted here
- Main text is problem → solution → confidence, and stops there
- Main text readable in one pass; everything else moved under appendix headings
- No narrative of how the work went — no discovery order, no discarded theories
- Tone is flat; no dramatic verbs, no bolded alarm
- Evidence matching the change type is present and actually shows the claim
- Anything with a UI has an image; anything explaining a shape has a diagram
- Screenshots cropped to the component, one state each, nothing else on screen
- States you could not capture are named as gaps, not faked
- Guardrails named by file and contract, with no test counts
- Out-of-scope work and follow-ups called out
- Known limitations stated
- Every link absolute; every image commit-pinned (see references/mechanics.md)
- Body makes sense to someone who has not read the branch
