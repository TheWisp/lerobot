# FLASH consolidation — 2026-11-14

This records the live-reviewed surface-refresh milestone and its consolidation.
It does **not** claim that all poses are correct or that the full test suite passes.
The starting commit was `245a92612`; the pending averaging experiments were not
part of that commit. No editor worktree or camera recording was changed.

## Landed behavior

- ICP fits the full confirmed map. Positive overlap also uses that full map.
- Only coordinates refreshed from the latest accepted reliable measurement,
  plus that accepted view, supply negative geometry evidence. Eligibility is
  fixed before the candidate exists. No eligible old points does not erase
  positive agreement with the full map (recorded ring/dowel regression).
- Both the proposed and Kalman-fused poses pass checks against the unchanged map
  before mutation. A rejection cannot refresh, remove, confirm or promote points.
- On acceptance, replace overlapping visible regions with actual measured
  points, remove reliably contradicted individual points, preserve hidden,
  missing-depth and boundary regions, then apply the existing stricter growth
  and later-confirmation rules to new patches. No averaging or confidence weight.
- Fresh shared defaults enable refresh, with 4 px inside / 2 px outside depth
  uncertainty bands. The silhouette remains intact. small_gicp and LK remain
  the shared defaults. Surface refresh does not alter Kalman's motion source.
- A click sends no stale text prompt after a named selection.

The factual architecture remains in `src/lerobot/gui/static/flash_architecture.html`;
algorithm details remain in `src/lerobot/policies/flash/docs/tracking_pipeline.md`.

## Review corrections and coverage

Removed the uncommitted weighted-averaging implementation and its unrelated
LK-to-Kalman prediction branch. Saved disabled averaging settings are normalized
away; enabled historical trials produce an explicit migration error. Kept one
explicit refresh switch and the
append-only control. The pre-consolidation tracked diff is preserved beside this
file for interpreting historical experiments; it is not executable production.

The retained working patch could exceed the combined cloud's memory bound.
The new dense-observation regression first failed (560 points against a 100-point
budget), then passed after applying the existing `max_points` limit to each
working patch. Original measurements retain their `points_per_patch` bound.
Sampling retains measured coordinates; it does not average or invent geometry.

New policy tests instantiate production classes, rather than monkeypatching the
check. Coverage includes arbitrary coordinate transforms, full-map solver input,
expiry and renewal of veto eligibility, zero/one/two eligible points, centimeter
wrong-pose controls, rotations that protrude, pending-point cleanup, hidden-point
preservation, repeated frames, final Kalman-result rejection, and shared GUI settings.
The real session tests verify refresh can happen when the new-growth gate fails.

The architecture test previously assumed the whole map was unchanged while new
points were pending. It now runs both policies: accepted overlap may refresh, but
only active patches enter fitting; pending patches still cannot confirm themselves.

An uncommitted research assertion expected the old cube fixture to be rejected
by _all_ stored points after refresh. Its three deciding old points had no reliable
source depth and are removed by accepted maintenance. That assertion contradicts
the requested policy, rather than independently establishing a pose error. It was
not turned into an xfail. The existing cube publication test and independent
protrusion controls remain. Texture-poor cube rotation ambiguity remains open;
this policy does not prove every plausible fit is the correct rotation.

## Verification

- **261 passed**, focused policy, geometry, recorded fixtures, frame architecture,
  live session/history, GUI controls and review axes suites (`focused.txt`).
- Full post-cleanup audit before updating the stale architecture assertion:
  1187 passed / 83 failed / 30 skipped / 5 xfailed. Of those failures, **82 also
  fail at HEAD**; the remaining stale assertion now passes for both policies.
- The isolated HEAD archive initially had six additional failures because CLI
  snapshot tests require Git metadata. After initializing metadata, all 56 tests
  in the affected three modules passed. Full failure IDs are in `test-audit.json`.
- No committed failure was hidden with a skip, xfail, or relaxed numerical limit.
- Deliberately restoring permanent veto eligibility fails the actual acceptance
  regression. Disabling replacement fails the real accepted-refresh test. The
  click-label bug was separately restored in an isolated browser and failed the
  name→click→click regression. Mutation logs accompany this record.

## Whole-clip checks and limits

| Replay                              | Before cleanup                                 | After cleanup                     |
| ----------------------------------- | ---------------------------------------------- | --------------------------------- |
| Green toy, accepted of 215          | 175 and 194 on fresh repeats (earlier run 195) | 192 with diagnostics; 193 without |
| Green second loss, frames 2873–3057 | 23/24 and 24/24                                | 23/24 and 24/24                   |
| Cube, accepted of 223               | earlier shared run 223/223                     | 223/223                           |

This exposes existing replay nondeterminism; it does not establish identical
trajectories or a statistically meaningful availability change. The cube and green
clips have no independent pose labels. Every published pose in the audited runs
passed its frozen pre-update production check. Full results and status sequences
are in `replay-audit.json`; raw maps are local cache artifacts.

Replay timing is not camera-to-result latency: depth and masks are recorded, and
audits add work. Green diagnostic mean/p99 was 121/297 ms, cube 77/91 ms. Do not
present these as live FPS or compare small differences while other tests run.

## Remaining branch work

Tracked in **LOCAL_FLASH_TRACKER_TODO.md**, not external issues:

1. Clicked-object mask recovery after occlusion. The stopped dowel run loses SAM's
   mask; the fitter cannot update without it. This is separate from the fixed name.
2. Stale retained prediction and its misleading display after mask loss.
3. Repeated replay determinism, then strict correctness/performance comparisons.
4. Existing full-suite failures and known texture-poor cube rotation ambiguity.
   Clock correctness remains parked as requested.

Historical experiment reports and hosted review packets remain locally available.
They are not refreshed results from this commit. The current implementation can be
replayed without policy monkeypatches:

```sh
export PYTHONPATH=src:.:tests/policies/flash OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python -m benchmarks.flash_surface_refresh /path/to/saved-clip \
  --arm replace --out /path/to/new-audit-directory
# --arm off runs the append-only comparison with the same 4/2 boundary settings.
```

Local consolidation artifacts:
`/home/feit/.cache/flash-tracker-restart/consolidation-20261114/`.
The `pre-live` directory preserves the complete pre-cleanup source used for repeats.
