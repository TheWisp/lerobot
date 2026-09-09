---
name: design-doc
description: How to write and maintain a design document in this repository — what belongs in the document and what belongs in the progress ledger beside it, how to state requirements with priorities and measured targets, and the language faults that keep recurring. Use when writing, restructuring or reviewing a design document, when a document mixes design with status, or when deciding where a fact belongs.
---

# Design documents

A design document says what the system is. It is read by whoever implements it,
and again months later by whoever changes it. It has to still be correct then,
which is what most of the rules below are for.

## The document holds no status

No `Status: proposed`, no `built / not started`, no build-status checklist, no
"the branch implements these in this order". Those are true for a week and then
quietly wrong, and a reader cannot tell which sentences have rotted.

Three artefacts, three jobs:

|                 | Holds                                                | Lives                                              |
| --------------- | ---------------------------------------------------- | -------------------------------------------------- |
| Design document | what the system is, and why                          | next to the code, permanently                      |
| Progress ledger | stages, the test pinning each, measurements as taken | `docs/proofs/<topic>/PROGRESS.md`, drains and goes |
| PR              | the conversation, the current state of the branch    | GitHub                                             |

The ledger links to the design. The design does not link back — a permanent
document should not point at a temporary one.

This does not contradict the repository rule that a document describing
behaviour that does not exist is annotated **NOT IMPLEMENTED.** at the claim.
That rule is for _reference_ documentation, which purports to describe what
runs. A design document's whole subject is the intended system; annotating every
claim would annotate the document. Keep the two kinds separate, and never let a
design document drift into being read as a reference for what shipped.

## Requirements: numbered, prioritized, sorted

One table. Every row has a number (`R1`), a priority (`P0`/`P1`/`P2`), the
requirement, its target, and why that target. Sorted by priority, so the reader
learns what matters most first and an implementer under time pressure cuts from
the bottom.

State the ordering principle above the table in a sentence or two — _this is an
observation view, so smoothness beats fidelity_ — because the priorities are a
judgement and the reader deserves the reason.

**Decisions are requirements.** A separate "Decisions" list at the end repeats
the requirements in different words and then disagrees with them. Fold each
decision into the requirement it constrains.

**Do not invent requirements that restate another.** "It fits the link" and
"playback keeps time" are one requirement with two wordings; "a window builds in
under 300 ms" is implied by the latency targets, not a separate demand. Each row
must be falsifiable on its own.

## Targets are numbers against named conditions

A latency target without a link is not a target. Name the conditions once —
`Local`, `Link`, the workload — with the measurement's date, and state every
target against one of them. Network numbers move; date them and say to
re-measure before quoting.

Say why the number is that number. "≤ 2 s" is arbitrary; "≤ 2 s, because the
operator scans episodes one after another, and 13.8 s is what unusable looked
like" is a target someone can argue with.

## State the constraint; don't rule out the others

Write what was measured in the context it was measured in. Do not write "the
link is the bottleneck, not the CPU, not the disk" — over a slow link the link
usually dominates, on a LAN it does not, and the sentence tells an implementer
not to look at the things they will need to look at. Give the constraint that
shapes the design and let the evidence appendix say what everything else cost.

## Plain language

Cut the scaffolding. These all appeared in one draft and all say nothing:

- "…does not work either, **and the reason is not opinion**"
- "A path that fits the link plays; one that does not, does not, **and no amount
  of buffering rescues it**"
- "**This must not be blurred:** …"
- "Resizing is not an optimisation, **it is the difference between working and
  not**"

Say the fact and stop. Emphasis used everywhere is emphasis nowhere, and a
document that argues with an imagined objector reads as unsure.

## Terms of art must not be common words

A glossary entry called **Size** is unusable: the word appears fifty times in the
document meaning ordinary things. Pick a term that is wrong as ordinary English
in this context, and check it against the domain's existing vocabulary before
adopting it — `segment` is the obvious name for a few seconds of video and is
already taken by SAM segmentation here.

Watch for the term that hides a plurality. "The size" implied one output
resolution when every camera has its own; the term had to become "the profile"
(one setting) plus "encoded resolution" (per camera). If a term is singular and
the thing is not, the document will be wrong wherever it is used.

When terminology is unsettled, say so in the glossary and make it an open
question, rather than picking silently.

## Evidence in an appendix

Main text carries the argument; the appendix carries the numbers, with dates,
conditions and where they were taken. Cite from the text into the appendix so a
reader can check any claim without the numbers interrupting the argument.

Numbers taken on a superseded branch stay useful. Say which branch, so nobody
re-measures what is already known — and so nobody mistakes them for a measurement
of the current design.

## Open questions carry what each side costs

An open question is a fork the reader can settle. Give it a number, the choice,
what each side costs in facts already in the document, where the facts are, and
your leaning. A question without a leaning is work handed back; a leaning without
the cost is a decision disguised as a question.

Do not raise as a question something a measurement already settled. Check the
evidence before writing one — a "question" that the appendix already answers
tells the reader the author did not read their own document.

## Checklist

- No status, progress, branch state or build order anywhere in the document
- Requirements numbered, each with P0/P1/P2, sorted by priority
- Ordering principle stated in a sentence
- No requirement that restates or is implied by another
- Every target has a number, a named condition and a reason
- Constraints stated in context, without ruling out other constraints
- No rhetorical scaffolding; every sentence carries a fact
- Every glossary term is distinctive, collision-checked, and singular only if the
  thing is
- Evidence in an appendix, dated, attributed to where it was taken
- Open questions carry costs and a leaning, and none is already answered
