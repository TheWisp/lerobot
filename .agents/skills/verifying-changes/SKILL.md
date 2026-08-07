---
name: verifying-changes
description: How to know a change actually works in this repository — which tests are worth writing, why green suites still ship broken products, how to audit a branch diff before review, and how to verify by driving the GUI rather than reasoning about it. Use when writing tests for a fix, auditing a branch or an upstream merge, deciding whether a change is proven or ready to merge, or refactoring — moving, replacing or reimplementing existing behaviour, where the claim is that the new code is equivalent to the old.
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

## Prove the refactor equivalent

Moving, replacing or reimplementing behaviour? **Write a test that enumerates
both versions over real inputs and asserts they match.** The whole set —
parameter names, emitted flags, config keys — not a spot check of a few cases.

Build the real object and read it. Do not reason about what it should contain;
that is the step that fails.

```python
# Write this. When the old code is gone, reconstruct its rule as the oracle:
legacy = {n for n, _ in model.named_parameters()          # what the old hack froze
          if not any(k in n for k in LEGACY_ALLOWLIST)}
now = {n for n, p in model.named_parameters() if not p.requires_grad}  # the new flag
assert now == legacy
```

That test caught a config flag that claimed to replace a hack and left one
parameter trainable. Reading the code had said equivalent.

**Ask what your number would read if you had not made the change.** A CSS fix
compared element-box centres: 0.0px before _and_ after. If the answer is "the
same", measure something else.

**Commit the check.** A lint proven once by hand, never committed, shipped with
a crash nobody caught.

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

## Green suite, then read the diff

Read `git diff origin/main...HEAD` whole, once the scope stops moving. **Every
file** — an audit scoped to "the code" missed a 142-line design doc the same
branch had already disproven, which the author then found by opening the PR at
random. Docs are where staleness hides, because nothing compiles them.

Each question below caught a defect the suite could not, because nothing was
broken — something was absent, stale, or duplicated.

- **Who consumes this?** For every field or flag the branch adds, find the line
  that reads it. One was declared on a request model, sent by the client, honoured
  by the process that received it — and dropped by the endpoint in between, so the
  feature quietly ran its default.

- **Who clears this?** Caches, latches, sequence counters and derived UI flags need
  a lifetime. One cache outlived its subprocess and was replayed into the
  replacement, which had reset the counter that would have rejected it.

- **Is this rule enforced twice?** One limit, both sides of an API, two
  definitions: the client offered what the server refused. Pick the authoritative
  copy, delete the other.

- **Does anything use this export?** A new public symbol with no caller is a
  contract someone will honour later. Worst is state exported for a test that
  never reaches it.

- **Does a default hide a mistake?** `getattr(self, "_attr", fallback)` on an
  attribute the constructor sets turns a typo into a plausible value forever. If
  partial construction needs tolerating, give it one initializer.

- **Do the docs still say something true?** Check prose the branch _adds_ against
  what it later concluded, not just prose it edits. The disproven doc above still
  argued for the mechanism and against the two approaches that survived.

Some of this cannot be found earlier: splitting a commit out to `main` and
rebasing left a duplicated call and CSS rule that neither commit showed alone.

## A device's behaviour is not in the repository

An audit of the OpenArm CAN driver reported `set_control_mode` as a blocking
bug: it writes to the broadcast parameter channel (`0x7FF`) and then waits for
the acknowledgement on the motor's master ID — the ID that everywhere else in
that driver carries state feedback. The reasoning was careful and the conclusion
was wrong. Damiao firmware multiplexes parameter acks onto the master ID. One
line from the rig settled it:

```
CAN_CTRL_MODE_ACK motor=gripper rx_id=0x18 mode=TORQUE_POS data=0800550a04000000
```

No amount of reading could have produced that, because the fact lives in the
motor. The same review flagged a second "bug" resting on whether the gripper
replies to a command at all — also unanswerable from source, also wrong.

So: **before reporting a defect, ask what evidence would settle it.** If the
answer is "what the hardware does", it is a hypothesis, not a finding. Go read
rig logs (`~/projects/lerobot-*/logs/*.log`) first. Watch for the tell — a claim
that rests on a convention the code follows _elsewhere_ rather than on something
this code states.

Then close it permanently: capture the real frame into a committed fixture with
its provenance, so the next reader gets the answer from the test suite instead
of from a robot (`tests/motors/test_damiao_protocol.py`). A device fact encoded
as a golden fixture does not rot when the driver is refactored — and it is the
only form of the fact that a reviewer can find.

## The environment can manufacture a failure

Three red suites in one session, none of them real:

- `tests/motors/test_damiao*` "passing" in CI for months — actually skipping at
  import, because `python-can` lives in an extra CI did not install.
- Five GUI e2e tests failing with "No active process to stop" — pytest ran from
  one virtualenv while `shutil.which("lerobot-record")` resolved to **another**,
  so the subprocess was a different install and died instantly.
- A checkpoint test failing because the operator's `outputs/` held a checkpoint
  awaiting a migration — a stale local precondition, asserted as if it were code.

Before believing a red test, reproduce CI's environment exactly — the extras
list in `.github/workflows/fork_tests.yml`, and that venv **first on `PATH`**,
not merely its `python`. Then run the same test on the base commit in the same
environment. Identical failure means environment; only a difference means code.

Two corollaries. A failure in files the branch never touched
(`git diff main...HEAD --name-only`) is almost always flake or environment — a
wall-clock invariant on a loaded runner, say. And a test whose precondition is
"a file on this machine is in the right state" will eventually fail for everyone
who has not done that migration; assert the precondition in the skip, or the
test reports someone's pending chore as your bug.

## Tests must not touch the user's real state

A pytest fixture wrote a `tmp_path` dataset into the user's `opened_datasets.json`,
so the GUI opened with a "Failed to open dataset" toast pointing at a
long-deleted pytest directory. Tests write to `tmp_path` and throwaway repos —
never the real cache, never real config, never a real dataset.
