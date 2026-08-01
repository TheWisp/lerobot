---
name: verifying-changes
description: How to know a change actually works in this repository — which tests are worth writing, why green suites still ship broken products, and how to verify by driving the GUI rather than reasoning about it. Use when writing tests for a fix, auditing an upstream merge, or deciding whether a change is proven.
---

# Verifying changes

Every rule below was bought with a bug that shipped to the user's hands with a
green test suite behind it. They are written in the order they tend to fail.

## The one that keeps happening

**A test that asserts what your own code produces proves nothing.**

Recording broke because upstream re-nested a config field and the GUI kept
emitting the old flag name. There _was_ a test:

```python
assert "--dataset.vcodec=libsvtav1" in captured_args   # green. product broken.
```

It compared the GUI's output to a constant written by the same person, in the
same repo, from the same misunderstanding. It could never have failed for the
reason that mattered. The fix is to hand the output to its **real consumer**:

```python
argv = capture_what_the_endpoint_emits(...)
cfg = draccus.parse(config_class=RecordConfig, args=argv[1:])   # the CLI's own parser
assert cfg.dataset.rgb_encoder.vcodec == "libsvtav1"           # and it landed where it's read
```

Ask of every assertion: _what would have to change in the world for this to go
red?_ If the only answer is "someone edits the line right above it", delete it
and write one that faces outward.

## Bugs live at boundaries; mocks are where you stop looking

The GUI reaches the `lerobot-*` scripts across a **process boundary**. No import
links them, so no type checker, no linter and no collection pass connects the
flag the GUI emits to the config that receives it. Every GUI test mocked
`_launch_subprocess` away — which is to say, every test stopped exactly where
the bugs were.

Wherever a boundary is untyped — subprocess argv, HTTP payloads, file formats
written by one component and read by another, config keys crossing a registry —
assume nothing checks it and write the test that crosses it.

## Mirror the production path, or you build a liar

The first version of the contract test failed on `--policy.path`, a flag the
real CLI accepts. It was the harness that was wrong: `parser.wrap` strips
`.path` args before parsing, and config `__post_init__` hooks re-read `sys.argv`
to recover them. The harness did neither.

Two ways that costs you: false positives burn a debugging cycle chasing a bug
that isn't there, and the "fix" is usually to loosen the check until it passes —
at which point it no longer catches the real thing either. **A guardrail that
does not match the production path is worse than none.** When a check disagrees
with reality, suspect the check first.

## Guard the guard

Two habits, both cheap:

- **Prove the test fails without the fix.** Revert the fix, run it, watch it go
  red, restore. Untested tests are as suspect as untested code — a regression
  test that never demonstrated its own failure mode may be pinning nothing.
- **Include a case that must fail.** The contract suite deliberately feeds a
  bogus flag and asserts the parse rejects it. Without that, a parser that
  silently tolerated extras would make every other assertion in the file vacuous
  — which is precisely how the original break went unnoticed.

## Pin the invariant, not the enumeration

`lerobot-replay` didn't import `virtual_bi_so107`, so it rejected the fork's own
robot. The tempting test is a hardcoded list of expected robots — which someone
must remember to update, and which is silently wrong the moment they don't.

The invariant is **parity**: all three launch scripts must register the same
robot set. A robot added to one and forgotten in another fails immediately, and
there is no list to maintain. Prefer "these two things agree" over "this thing
equals a constant I typed."

Beware of assertions weakened by shared global state: robot registration is
cumulative, so importing all three scripts and reading the registry would let
whichever imported first register on the others' behalf and hide the asymmetry.
Read each side independently.

## A merge that applies cleanly is not a merge that works

Textual conflict detection finds none of this. Upstream renames a field the fork
never touched: no conflict, clean apply, broken product. When merging upstream,
audit **semantic** conflicts too — every symbol the fork calls that upstream
moved, renamed, re-nested or retyped, whether or not the files overlap.

The cheap systematic version: enumerate what the fork _consumes_ from upstream
(imported symbols, config field paths, CLI flags) and check each still resolves.

## Keep a hardware-free path through every flow

`virtual_bi_so107` — no buses, no cameras — is what makes real end-to-end tests
affordable: a full record→replay round trip on any machine, nothing plugged in,
nothing that can move, ~30s.

Its value is also its warning. Replay's missing import meant replay was the one
flow with no hardware-free path, so it was the one flow nobody could verify —
and that is where a bug sat. **If a flow can't be exercised without hardware,
treat that as a gap to close, not a fact of life.** Coverage the testbench can't
reach is coverage you don't have.

## Drive the product; don't reason about it

Claims about a UI flow are worth very little until the flow has been run. The
GUI is drivable headlessly — POST to `/api/run/*`, or Playwright/CDP for the
frontend (see the screenshot tooling in `scripts/gui/`). Doing so found the
codec break, the replay break, and a stale-state toast, in one pass.

Two rules when reporting what you find:

- **Read the logs; don't infer from symptoms.** When teleop failed mid-session,
  the application traceback said "read thread is not running" — but `journalctl
-k` showed the USB device physically disconnecting (`error -71`, _"Cannot
  enable. Maybe the USB cable is bad?"_). Same symptom, entirely different layer.
  Verify at the layer that has the evidence.
- **Say plainly which parts you could not verify.** "Verified on the virtual
  robot; not verified against real hardware" is useful. Implying otherwise is
  the failure mode that makes all the work above worthless.

## Tests must not touch the user's real state

A pytest fixture wrote a `tmp_path` dataset into the user's `opened_datasets.json`,
so the GUI opened with a "Failed to open dataset" toast pointing at a
long-deleted pytest directory. Tests write to `tmp_path` and throwaway repos —
never the real cache, never real config, never a real dataset.
