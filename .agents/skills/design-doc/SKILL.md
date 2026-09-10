---
name: design-doc
description: How to write and maintain a design document in this repository — fixing the problem before drafting, requirements with priorities and measured targets, where status goes instead of the document, how to amend one when reality contradicts it, and the language faults that keep recurring. Use when asked to design, propose, spec or plan a system before building it (design doc, RFC, ADR, architecture proposal), when writing or restructuring a document under `docs/plans/` or `src/**/docs/`, when reviewing one, when a document mixes design with status, or when recording progress against one. Not for reference documentation describing what already runs, not for PR bodies (`pull-request`), not for issue bodies.
---

# Design documents

A design document says what the system is, and why. It is read by whoever
implements it, and again months later by whoever changes it. It has to still be
correct then, which is what most of the rules below are for.

Read [references/template.md](references/template.md) for the section order and
what each section costs when it moves.

## Fix the problem before drafting

Write the problem in one short paragraph, in the words the reader would use, and
make each later section answer one clause of it. Ask the questions that decide
the shape before writing rather than after: most of a design is forced once the
requirement is fixed, and a document that enumerates every trade-off the
requirement already settles reads as not knowing which one matters.

Then state the proposal in one paragraph, and only then argue for it. Main text
carries the load-bearing facts; everything else goes to an appendix the argument
links to.

## The document holds no build status

No build-status checklist, no `not started`, no "the branch implements these in
this order", no sentence that is true for a week. Months later a reader cannot
tell which of those have rotted, and has no way to check.

One narrow exception: a single field naming where the **document** stands —
`proposed`, `accepted`, or `superseded by <link>`. Closed value set, one line, no
build words in it. It changes at most twice in a document's life, and it is what
tells a reader whether they are holding a live design or an abandoned one.
`Status: proposed. None of this is built.` fails the rule because its second
sentence is build state; `Status: proposed` alone does not. Every process that
survives contact keeps such a field and keeps progress out of the prose — ADR
statuses, Kubernetes' `kep.yaml`, Oxide's RFD states.

Four artefacts, four jobs:

|                 | Holds                                             | Lives                                      |
| --------------- | ------------------------------------------------- | ------------------------------------------ |
| Design document | what the system is, and why                       | next to the code it describes, permanently |
| Tracking issue  | the stages, what is next, what is done            | GitHub, closed by the PR that finishes it  |
| Evidence        | measurements and captures, as taken, dated        | `docs/proofs/<topic>/EVIDENCE.md`          |
| PR              | the conversation, the current state of the branch | GitHub                                     |

Stages with a completion state go in the issue, not in a markdown checklist
beside the document. A checklist cannot close, assign, or link the PR that
resolves it, which is the failure `CLAUDE.md` records for `TODO.md`; a file named
`PROGRESS.md` is that failure with a new name. `docs/proofs/` is for captured
evidence — what was observed, on which revision — and its `README.md` says so;
intent does not go there.

**Link to the durable thing.** One line in the document — `State of the work:
<tracking issue>` — so a reader can find out whether the design is live without
guessing. An issue outlives the branch; a scratch file does not, and a permanent
document must not point at something that gets deleted.

## When reality contradicts the document

While the design is unbuilt, update it — the shortcomings that surface during
implementation are the design changing, and a document that stopped tracking its
own subject is worse than none.

Once behaviour has shipped, do not quietly rewrite the part that was wrong.
Supersede it: state what changed and why, and leave the superseded claim
reachable. A silent correction destroys the evidence of how the wrong thing
survived, which is the same reason `CLAUDE.md` requires **NOT IMPLEMENTED.**
annotations rather than quiet fixes.

That annotation rule is for _reference_ documentation, which purports to describe
what runs. A design document's whole subject is the intended system; annotating
every claim would annotate the document. Keep the two kinds separate, and never
let a design document drift into being read as a reference for what shipped.

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

## The argument is numbered, and cited

Requirements (`R1`) → observations about the system as it is (`O1`, each sourced
to a commit, script or module, and each closing with the conclusion it forces) →
the constraints and freedoms those conclusions add up to (`C1`) → the
architecture, where every element cites the requirement, observation or
constraint it comes from. Every citation is a link, and every anchor resolves.

State the fact before the contract. "`hvla_img_*` is the same array as
`lerobot_obs_img_*`; what differs is the owner" is checkable; quoting a
docstring's rule the reader cannot evaluate is not.

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

A term in a budget is a fact, not a defect. Present each term, what it buys, and
whether it is free to take back; whether it matters is the reader's call against
the baseline they accept.

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

The text must also stand on its own. "Is kept", "fixes", "the branch's path"
make sense only in the author's context; name the thing and say what happens to
it.

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

Do not invent a term the document could do without. Every term of art is either
in the glossary and linked at first use, or is meant in plain English. When
terminology is unsettled, say so in the glossary and make it an open question
rather than picking silently.

## Evidence in an appendix

Main text carries the argument; the appendix carries the numbers, with dates,
conditions and where they were taken. Cite from the text into the appendix so a
reader can check any claim without the numbers interrupting the argument.
Captures that live as files — screenshots, transcripts, recordings — go in
`docs/proofs/<topic>/EVIDENCE.md`, and the appendix links them.

Numbers taken on a superseded branch stay useful. Say which branch, so nobody
re-measures what is already known — and so nobody mistakes them for a measurement
of the current design.

## Open questions carry what each side costs

An open question is a fork the reader can settle. Give it a number, the choice,
what each side costs in facts already in the document, where the facts are, and
your leaning. A question without a leaning is work handed back; a leaning without
the cost is a decision disguised as a question.

Write it for the reader, in everyday words — not in the document's own shorthand.
A question the reader cannot parse is not a question.

If a measurement would settle it, it is not a question for the reader: put it
under "to measure" and go and measure it. And do not raise as a question
something a measurement already settled — a "question" the appendix answers tells
the reader the author did not read their own document.

Where a conclusion is not yet forced, leave the decision open and say what would
force it. An asserted conclusion that turns out wrong costs more than an open
question.

## Where the document lives

Next to the code it describes, under that package's `docs/` — for example
`src/lerobot/gui/docs/dataset_playback.md`. Older documents sit in `docs/plans/`
or as a `DESIGN.md` beside their module; leave them where they are, and put new
ones next to their code.

## Checklist

- Problem stated in one paragraph, in the reader's words, before the argument
- Proposal in one paragraph before the case for it
- No build status, progress, branch state or build order anywhere; at most one
  standing field, from the closed set
- One line linking the tracking issue that holds the state of the work
- Requirements numbered, each with P0/P1/P2, sorted by priority
- Ordering principle stated in a sentence
- No requirement that restates or is implied by another; no separate decisions
  list
- Observations, conclusions and constraints numbered, and every architecture
  element cites one — as a link that resolves
- Every target has a number, a named condition and a reason
- Constraints stated in context, without ruling out other constraints
- No rhetorical scaffolding; no sentence that needs the author's context
- Every glossary term is distinctive, collision-checked, linked at first use, and
  singular only if the thing is; unsettled terms are open questions
- Evidence in an appendix, dated, attributed to where and on which branch it was
  taken
- Open questions in plain words, with costs and a leaning; none already answered
  by the appendix, and none that a measurement should settle

Before review, audit the document the way `verifying-changes` audits a diff —
prose the branch adds is where staleness hides, because nothing compiles it. The
PR that carries the document is written per `pull-request`.
