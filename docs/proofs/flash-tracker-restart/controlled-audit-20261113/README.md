# Controlled surface and pose audit — 2026-11-13

No tracking-quality improvement from the uncommitted surface-maintenance bundle
is established yet. The independently useful change is the diagnostic and its
guards: it tests the same labeled pose against independently varied map and
retained-view placements without changing the running tracker. No production
acceptance rule, matcher, Kalman input, or map-maintenance behavior is changed
by this audit commit.

## Frozen comparison

Base revision: `e85ec5972`. The working tree additionally contains the earlier
refinement, retirement, LK-to-Kalman and retained-view experiments. The complete
tracked diff is saved in `working.patch` under the artifact root below. Each
clock audit has a `manifest.json` with source hashes, label hash, case, and full
settings. The replay input hash is
`1e395c9078ea1819b600b06004d1c37a392353247f936b693b7f5b950fa2f6aa`.

The clock uses LK + small_gicp, the normal geometry checks and default iteration
budget. Its control has averaging, retirement, and the extra LK-to-Kalman input
off. A separate replay runs on a clean checkout of the base revision.

Labels are read for detached diagnostic checks only. They never become the
running tracker's seed, published pose, map, retained view or group motion.
The audit-on and audit-off controls have identical statuses and reasons on all
203 frames; the largest pose-matrix element difference is `2.4e-14`. Their
surface scores also agree to floating-point precision.

## 1. The clock is primarily blocked by the retained view's placement in the first loss

At each frame, keep the tested labeled pose, current RGB-D/mask, geometry policy
and input patch samples fixed. Change only where those past measurements were
placed. Retained views are the saved RGB-D observation plus its estimated pose;
they are distinct from the accumulated point map.

| Frames  | Actual map + actual view pose | Actual map + labeled view pose | Labeled patch placement + actual view pose | Both placements labeled |
| ------- | ----------------------------: | -----------------------------: | -----------------------------------------: | ----------------------: |
| 1–77    |                         68/77 |                          73/77 |                                      66/77 |                   71/77 |
| 78–120  |                          1/43 |                          36/43 |                                       1/43 |                   36/43 |
| 121–202 |                         40/82 |                          65/82 |                                      38/82 |                   62/82 |

These are **labeled-pose acceptance counts**, not poses recovered by the tracker.
The actual tracker publishes 77 poses and then stays lost for 125 frames.
Repositioning the stored patches alone does not explain the first loss. Correcting
only the pose attached to the saved view removes most of that rejection.

The same table and tracker output reproduce on the clean committed baseline.
This failure therefore predates the uncommitted surface-refinement work.

At frame 78, the labeled pose has zero forward map-to-current depth conflict.
The reverse current-to-retained check reports 56.5% conflict with the actual
frame-77 view pose, versus zero with its labeled pose. Frame 78 still fails an
overlap/other check with labeled placements, so correcting placement is not
a complete solution. At frame 108 even labeled view placement leaves 30.5%
reverse conflict. The remaining seven failures require depth/mask/label review.

Later, 40/82 labeled poses already pass with the **unchanged** current map and
view. Continued loss there includes a proposal failure, separate from the
first interval's check inconsistency.

Controls and limits:

- Both raw-patch maps use exactly the same sample identities and counts, with
  no new points or resampling. The tracked raw-patch map reproduces the actual
  map's acceptance counts, controlling for the map's normal voxel selection.
- The initial-only map also goes from 1/43 to 36/43 when the view pose changes.
- A no-retained-view result is recorded, but removing views also changes the
  overlap policy from partial-view to all-current-point coverage. Its overall
  acceptance must **not** be interpreted as a pure reverse-veto ablation.
- Label poses and depth are imperfect. Even labeled placements fail some
  healthy frames. This isolates an important error source, not a new oracle.
- Offline tolerance sweeps at 3/5/8/10/15 mm do not establish a safe global
  tolerance. Frame 108 still fails at 15 mm; changing one scalar is insufficient.

## 2. The refinement factor results are not stable enough for a positive claim

The independent review's factor script was rerun without production edits.
R = averaging, T = retirement, K = extra object LK motion into Kalman prediction.
The controls are runtime patches for research, not new product options.

| R   | T   | K   | Available | Within 20 mm / 20° | Outside threshold | Surface median / p90 (mm) |
| --- | --- | --- | --------: | -----------------: | ----------------: | ------------------------: |
| off | off | off |        77 |                 77 |                 0 |              4.07 / 10.49 |
| off | off | on  |        96 |                 76 |                20 |              4.43 / 11.67 |
| off | on  | off |        75 |                 75 |                 0 |              4.04 / 10.44 |
| on  | off | off |        93 |                 77 |                16 |              4.46 / 12.17 |
| on  | on  | off |        76 |                 76 |                 0 |              3.82 / 10.11 |
| on  | on  | on  |       124 |                 76 |                48 |              4.28 / 13.23 |

Full results, including other combinations, are in `results.json`. The
symmetry-aware surface metric is distance from the partial reconstruction to
the full labeled model. It supplements pose identity error; neither availability
nor a low residual against the tracker's own map proves correctness.

Two same-settings repeat runs invalidate a simple harmless-averaging conclusion:

| Variant                                              | Earlier reviewer | This run | Fresh-process repeat |
| ---------------------------------------------------- | ---------------- | -------- | -------------------- |
| Averaging only: available / within threshold         | 76 / 76          | 93 / 77  | 118 / 76             |
| Averaging + retirement: available / within threshold | 117 / 77         | 76 / 76  | 96 / 78              |

Inputs have the same hash. Control outputs repeat to numerical precision. The
first pose divergences versus the earlier reviewer are frame 6 for averaging
alone and frame 52 for averaging + retirement. The source of this sensitivity
has not been isolated; it must not be attributed to a particular library or
threshold yet. The extra LK-only and full-bundle runs reproduce the review's
96/76 and 124/76 endpoints.

Thus we cannot say averaging alone is harmless, nor that exactly one additional
correct pose in a repeat establishes an improvement. The bundled LK prediction
change should be separated from map maintenance in further work.

## 3. Green toy: a better proposal source alone does not bridge frame 2882

Replay the saved green-toy clip from its first retained frame using weight 8.
At frame 2882, freeze the accumulated map, the frame-2873 accepted observation,
current observation, LK seed and acceptance policy. Extra solves are detached:
the real replay always receives its original solve result.

| Solver's source    | Iterations per level | Conflict against the same untouched map | Accepted |
| ------------------ | -------------------: | --------------------------------------: | -------- |
| Accumulated map    |                   20 |                                  15.59% | no       |
| Recent observation |                   20 |                                   3.06% | no       |
| Accumulated map    |                  100 |                                  15.61% | no       |
| Recent observation |                  100 |                                   3.06% | no       |

The recent-source candidate also fails the check against its own recent source
(1.09% conflict with a coherent protrusion). The comparison proves improvement
in that geometry residual, not correctness: this clip has no independent labels.
All four candidates are rejected before temporal fusion. More iterations do not
resolve this frame. No point is removed to make a candidate pass.

## 4. The changed green control was a real confound

Full green replay, maintenance off:

| Code                                                          | Observed | Predicted | Lost |
| ------------------------------------------------------------- | -------: | --------: | ---: |
| Clean committed base                                          |       98 |         1 |  116 |
| Working tree                                                  |      115 |         2 |   98 |
| Working tree with only committed `scene_depth_check` restored |       98 |         1 |  116 |

The ungated retained-view exception, not refinement, explains those extra
observed frames. It lets the latest measured surface refute a saved view's
depth even before that surface is confirmed in the map. This has not yet been
shown to improve pose accuracy. Clock counts and labeled-pose gate results are
unchanged by it. Do not compare 98 versus a refinement result as if only
refinement changed.

## Verification and next controlled steps

- 91 focused tests passed, including the three new audit tests.
- The new tests also pass on a clean `e85ec5972` checkout without experimental
  production code. Deliberately ignoring patch placement makes the placement
  regression test fail. A wrong map must fail while a correct map passes.
- The clean checkout runs the full clock audit and reproduces the findings.
- Diagnostic timing includes extra checks; it is not an inference performance
  claim. No server restart or default change was made for this audit.

Next: inspect the remaining label/depth conflicts, then compare a principled
retained-view pose-uncertainty treatment against both true and false poses on
frozen snapshots. Separately isolate refinement's run-to-run sensitivity before
using its replay totals as evidence. For the green toy, inspect the remaining
contradicting samples at the recent-source candidate. These are different
failure mechanisms; do not try to solve them by jointly changing Kalman,
proposal sources, map updates and gate tolerances.

## Reproduction and artifacts

Interpreter used:
`/home/feit/.cache/flash-tracker-restart/coarse-open3d-env/bin/python`.
Set `PYTHONPATH=src:.:tests/policies/flash OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2`.
Run from the research repository.

```sh
python -m benchmarks.flash_surface_gate_audit --cases docs/proofs/flash-tracker-restart/controlled-audit-20261113/clock-case.json --out /tmp/clock-audit
python -m benchmarks.flash_surface_gate_audit --cases docs/proofs/flash-tracker-restart/controlled-audit-20261113/clock-case.json --out /tmp/clock-control --no-audit
python docs/proofs/flash-tracker-restart/controlled-audit-20261113/factor_ablation.py --cases docs/proofs/flash-tracker-restart/controlled-audit-20261113/clock-case.json --out /tmp/clock-factors
python docs/proofs/flash-tracker-restart/controlled-audit-20261113/frozen_fit_source.py /home/feit/.cache/flash-tracker-restart/live-history/saved-4710ff450ed54995bfb2cbca12d049f4 --out /tmp/green-source --weight 8
```

The two scripts in this proof directory specifically audit the uncommitted
refinement experiment and require that implementation. The general benchmark
and its tests work on the committed baseline as well.

Full local evidence:
`/home/feit/.cache/flash-tracker-restart/controlled-gate-20261113/`:
`clock-audit-final/`, `clock-control/`, `committed-clock-audit/`, `factors/`,
`factors-repeat/`, `green-source/`, and the three `*green*.json` comparisons.
Original frame snapshots and per-frame checks are retained there; raw clips
and labels are referenced by the saved case/settings rather than duplicated.
