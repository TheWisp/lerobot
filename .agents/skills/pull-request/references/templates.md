# PR skeletons by change type

Adapt rather than fill in mechanically. Headings are load-bearing only where
they help a reviewer skip; a short PR needs fewer of them.

## Feature

```markdown
## <What it is, in one line>

<One paragraph: what a user can now do, and where. Concrete, not abstract —
name the tab / CLI / API surface.>

### Goal & scope

<What this PR deliberately does and does not land. If it is a framework plus
one representative example, say that, and name the follow-ups that plug into
the same seam. Scope stated here reads as judgment.>

### Landed

| Area | What |
| ---- | ---- |
| ...  | ...  |

### Proof

<Screenshots / GIF / video showing it working. One per meaningful state, each
under a heading that says what the state is. For non-visual features, a
rendered artefact or a captured transcript.>

### Guardrails

<Which tests pin which contract, by file. No counts. If a test exists because
of a specific defect, name the defect.>
```

## Bug fix

```markdown
## Problem

<What went wrong, concretely. Name the real case that hit it — the actual
dataset, the actual config, the actual sequence. Abstract descriptions of bugs
are unconvincing; a reviewer wants to recognise it.>

<Why the previous behaviour was wrong, not just different.>

## Fix

<What changed, and why this is the root cause rather than a symptom. If it is
a workaround, say so and link the follow-up.>

| Case | Before | After |
| ---- | ------ | ----- |
| ...  | ...    | ...   |

### Evidence

<The bug not happening: before/after screenshots, or the scenario now covered
by a test. For anything visual, show it.>

### Guardrails

<The regression test, by file, and the exact scenario it pins.>
```

## Refactor

```markdown
## Why

<What the old structure made hard or unsafe. A refactor with no forcing
function is hard to justify — name the thing that got painful.>

## What changed

<The shape change, at the level of modules and seams. Not a file list.>

## Behaviour is unchanged

<How you know. Which existing tests pin the old behaviour and still pass.
If any test had to change, name it and justify the change — a changed test in
a refactor is the one thing a reviewer will look for.>
```

## Sync / merge

Main text stays short; the per-defect detail is what appendices are for.

```markdown
<Problem: how far behind, and what that was costing. One short paragraph.>

<Solution: what the sync brings, and the scope boundary — what was split out
and why. A few lines.>

<Confidence: the audit performed and its result, as a table of check → result.
Then, in one or two lines: how many defects were found, that each has a
guardrail, and that nothing outstanding is attributable to the merge. Detail
goes below.>

---

## Regressions found and fixed

<One entry per defect: what broke, why it survived the merge, how it was
fixed, and which test now pins it. A sync with no findings usually means
nobody looked — but this belongs here, not in the summary.>

## Known limitations

<Deferred work, coexisting designs, environment constraints the reviewer will
hit.>

## Follow-ups (not in this PR)

<Dependent branches, deferred migrations, remaining upstream distance.>
```
