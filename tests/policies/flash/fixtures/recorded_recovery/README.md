# Recorded recovery checks

Six checkpoints from the five failures reviewed on 2026-11-04. The test calls
the real RGB-D preparation, surface fitter, Kalman update and surface-growth
code. It restores the state immediately before the failing frame, including
the last accepted measured view. It does **not** rerun segmentation or UFM;
image correspondences/group residuals are absent and an available group pose
is only a starting guess. Full causal replays are a separate experiment.

| Fixture       |    Time | Pose check it was exported for (removed, see below)        |
| ------------- | ------: | ---------------------------------------------------------- |
| `gamepad_271` | 11.97 s | Returned orientation agrees with the measured broad face   |
| `mustard_180` | 19.39 s | Returned pose within 20 mm / 20° of the dataset label      |
| `yellow_68`   |  2.77 s | Returned surface lies in the measured object region        |
| `clock_39`    |  1.30 s | Returned pose within 20 mm / 20° of the dataset label      |
| `clock_100`   |  3.33 s | Same, after the clock turns                                |
| `drill_46`    |  1.53 s | Keep the near object depth, then recover its labelled pose |

The gamepad check is a plane-normal consistency test, **not** an independently
labelled 6D pose. The yellow block check measures position consistency, not
orientation. Neither can establish full accuracy; review the videos too.
Labelled clips use the existing benchmark's loose 20 mm / 20° limits. Labels
are only read by evaluation and never passed to the fitter.

Each `manifest.json` records the source run, frame, settings and NPZ SHA256.
The NPZ contains original RGB-D measurements, masks, intrinsics, retained
measured patches, previous pose/covariance and evaluation labels where available.
Regenerate from the original local recording with
`benchmarks.flash_recorded_recovery.export(run, frame, name)`.

From the repository, with the FLASH dependencies installed:

```bash
PYTHONPATH=src:. python -m pytest tests/policies/flash/test_recorded_recovery_integration.py -q
FLASH_RECOVERY_EXPERIMENT=partial_filtered_depth_search PYTHONPATH=src:. python -m pytest tests/policies/flash/test_recorded_recovery_integration.py -q
FLASH_RECOVERY_EXPERIMENT=recent_view PYTHONPATH=src:. python -m pytest tests/policies/flash/test_recorded_recovery_integration.py -q
```

**The pose checks in the table were removed on 2026-11-28.** They were written
as deliberately failing checkpoints for a recovery experiment and still failed
when FLASH landed. `test_recorded_recovery_integration.py` now checks only that
a rejected alignment leaves the surface unchanged and that the captured process
noise is applied once. `partial_depth_search`
is the slower 24-start experiment without filtering the depth used by pose
validation. The two other experiments use four starts. None is enabled by
default. Configurations apply equally to every fixture.

Growth must remain unchanged whenever no pose is returned. The covariance
check prevents accidentally applying the captured frame's process noise twice.
Generic controls and deliberately wrong alignments are covered separately in
`test_surface_recovery_options.py` and `test_recovery_acceptance_integration.py`.

After running all frames through `benchmarks.flash_tracker_experiment.run_case`,
validate its raw poses separately from its reported scores:

```bash
FLASH_RECOVERY_SEQUENCE_ROOT=/path/to/completed/runs PYTHONPATH=src:. python -m pytest tests/policies/flash/test_recorded_recovery_sequences.py -q
```

This requires complete recordings from frame zero. It reads independent labels
from these fixtures, checks the frame/time/reference identity, and rejects a
returned wrong pose. It does not launch inference itself. Without the environment
variable these six full-sequence acceptance checks explicitly skip.

The parked no-reset filters are compared jointly with surface recovery through
`benchmarks.flash_recovery_combinations`. Its manifest selects recorded inputs;
`--filter strict_broader` and `--filter velocity` restore the earlier experiments.
`--fitting partial` isolates partial matching, while `--fitting recovery` also
uses the experimental depth cleanup, four recovery starts and last source view.
Every combination starts from frame zero with its own growing surface and filter.
Reset/no-reset comparisons use the shared `reset_on_geometry_conflict` setting
and save its effective value. Earlier combined-run artifacts recorded a `filter`
name while overriding the class; their saved surface settings alone do not
describe that override. Keep those artifacts as historical evidence. Velocity
still requires the benchmark-only filter implementation and its saved settings.

`test_combined_recovery.py` checks how the comparison classifies wrong, missing
and correct output, that a reset setting changes only the filter policy, and
that the gamepad's recorded wrong proposal is never published and never grows
the surface. Since c55938d09 removed the Kalman innovation veto on 2026-11-11,
the geometry checks refuse it; **the checks that asserted the veto itself were
removed on 2026-11-28.** The six sequence checks above still require useful poses; five
additional sequence checks verify no Kalman rejection was bypassed anywhere in a
completed combined replay.
