# Design document skeleton

Section order, not a form to fill in. A short design needs fewer headings; the
order is what matters, because each section is only readable once the one above
it is settled.

```markdown
# <The system, named as the reader would name it>

Status: proposed <!-- proposed | accepted | superseded by <link> -->
State of the work: <link to the tracking issue>

<The problem, one paragraph, in the reader's words. What goes wrong now, for
whom, and what it costs. Every section below answers one clause of this.>

**Proposal.** <One paragraph. What the system does, in the shape you are
arguing for. No justification here — that is the rest of the document.>

## Scope

In: <what this design covers.>

Out, and why: <each exclusion with the reason it is excluded. "Out" without a
reason reads as an oversight.>

## Requirements

<The ordering principle, one or two sentences: why P0 is P0.>

| #   | Pri | Requirement | Target | Why that target |
| --- | --- | ----------- | ------ | --------------- |
| R1  | P0  | ...         | ...    | ...             |

<Conditions named once — `Local`, `Link`, the workload — with the date they
were measured. Every target above is stated against one of them.>

## Observations

<O1..On. Each sourced to a commit, script or module, each closing with the
conclusion it forces. This is what makes the architecture below follow rather
than appear.>

## Constraints and freedoms

<C1..Cn. What the observations fix, and what they leave free.>

## Architecture

<Each element cites the R/O/C it comes from, as a link. If an element cites
nothing, either it is unmotivated or a requirement is missing.>

## Open questions

<Q1..Qn. The choice in everyday words, what each side costs (citing facts
already above), and your leaning. Separately, "to measure": the questions a
measurement settles, which are not questions for the reader.>

## Glossary

<Every term of art, linked from its first use. Terms that are ordinary English
in this document do not belong here — pick a different term instead.>

## Appendix: evidence

<E1..En. Each with the number, the date, the conditions, the machine or link,
and which branch it was taken on. Captured files live in
`docs/proofs/<topic>/EVIDENCE.md` and are linked from here.>
```

## What the sections cost if reordered

- Requirements before observations: the reader cannot tell an invented
  requirement from a forced one.
- Architecture before constraints: every element reads as a preference.
- Open questions at the end but written first: the questions that decide the
  shape have to be answered before the shape is drawn, or the document argues
  for something the answer may delete.
