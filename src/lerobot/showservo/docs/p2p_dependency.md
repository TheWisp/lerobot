# Reproducible Point2Pose experiments

Point2Pose runs in its own Python environment. Its source comes from
[TheWisp/point-to-pose](https://github.com/TheWisp/point-to-pose), a fork of
[tzuyuan/point-to-pose](https://github.com/tzuyuan/point-to-pose).

The fork's `main` follows upstream. Small fix branches retain independent bug
fixes and regression tests. `showservo-configurable` integrates the experiments
without importing unrelated upstream runtime changes. The selected immutable
commit is `95bd7a8808fc1aaf015d510dffcccaa3c6971a9c`.

The fork defaults its correctness fixes on. The rig explicitly records local
exceptions in `benchmarks/p2p_rig.yaml`, following the user's preference to fix
problems unless our setup regresses:

| Option                                             | Rig default | Local evidence                                                                                                                                   |
| -------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pipeline.params.preserve_delayed_keyframe_motion` | **true**    | Corrects historical-pose assignment; two complete local replays add no lost frames or disagreement in the checked visible-surface windows.       |
| `register.params.consensus_refit`                  | false       | Better early clear views, but worse late normal agreement. Earlier loss is diagnosed separately below; lost count alone does not veto promotion. |
| `register.params.return_refined_pose`              | false       | Returning SDF refinement worsens the recorded local trajectory. This override deliberately retains the known pose/statistics inconsistency.      |
| `pipeline.params.tracked_depth_mode`               | legacy      | Nearest and edge modes remain selectable; their integrated local replays worsen later poses. Only one depth mode can be active.                  |

`pipeline.params.pose_jump_guard_object_center` is also available (fork default
true, rig override false). It corrects the origin-dependent translation check,
but paired local replays show worse reported poses after its additional accepts.
Both distances and the reference point are now retained in
`pose_jump_guard_info` through the worker and act recording. This is an accuracy
exception, not a rule against becoming uncertain.

`enable_sdf_refine` remains true: the SDF stage still participates in candidate
selection/statistics. Disabling that stage was separately tested and did not
remove the observed local regressions. The dense-register configuration shares
the explicit pipeline/depth policy; the new complete local comparisons use the
normal sparse rig configuration, not dense mode.

These are measured local exceptions, not a requirement that every reference
video improve. Detailed results and limitations are in
[regression evidence](p2p_regression.md#october-13-integrated-rig-selection).

## Install the selected source

From this LeRobot checkout, with the existing Point2Pose environment and model
checkpoints already installed:

```sh
python benchmarks/setup_p2p.py
python benchmarks/setup_p2p.py --check
```

`benchmarks/p2p.lock.json` records the fork URL, immutable commit and upstream
base. Setup clones into `~/.cache/point2pose/checkouts/<commit>` and checks out
that commit detached. It links the existing checkpoint directory; use
`--assets-from /path/to/checkpoints` to select another. The existing source
checkout, environment and running processes are untouched. Setup refuses to
overwrite a modified checkout. No model downloads or dependency installs run.

The worker and groups view use this checkout by default. `LEROBOT_P2P_REPO` can
select another location, but it must contain the same clean pinned commit.
Changing an experiment requires changing the committed pin. The bridge checks
the commit and source cleanliness before model loading; it never silently runs
an unknown source version.

## Identify a recorded experiment

Each P2P result carries `software`: P2P commit, LeRobot commit and dirty flag,
configuration SHA-256, bridge SHA-256, and environment SHA-256. After initialization,
`software.p2p_options` also records the actual implementation settings, including
defaults and local overrides; unsupported options are absent. Act recordings
retain it alongside each frame and in `tracker_at_start`.

`benchmarks/p2p.environment.json` snapshots the environment used for the initial
migration: Python/package versions plus editable dependencies' source commits
and local patches. It is an audit snapshot, not an automatically resolved
installer lock. In particular it preserves the existing SAM2 build-only patch.
Each bridge startup writes its actual environment snapshot to
`~/.cache/point2pose/environments/<environment_sha256>.json`. Archive that file
with a run when sharing results or changing machines. The source pin is
enforced; environment differences are recorded rather than silently assumed
identical. Regenerate a reviewed baseline snapshot with:

```sh
~/.cache/point2pose/venv/bin/python benchmarks/setup_p2p.py --check \
  --snapshot-environment benchmarks/p2p.environment.json
```

## Update deliberately

1. Fetch upstream in the fork's development checkout. Prepare an update branch
   from the experimental integration branch; merge the intended upstream
   revision there. Keep published experiment commits available for old pins.
2. Resolve conflicts and remove local fixes superseded by upstream. Run the
   fork's regression tests, recorded tracking comparisons and relevant moving
   object benchmarks. A passing unit test does not establish placement quality.
   Follow the [regression protocol](p2p_regression.md), including sequence-level
   tail errors and availability; aggregate accuracy alone cannot approve a change.
3. Push the reviewed integration commit, update LeRobot's manifest, and run
   setup. Commit the manifest/configuration changes and their validation.
4. Switch the live process only at a safe restart; never restart the GUI while
   it owns a connected arm. Preserve the previous pin for rollback.

Each correction is implemented once in the fork, with configuration switches
for controlled comparisons. Bridge patches remain historical performance and
integration changes with source checks. The configurable RANSAC and depth
implementations have audited source hashes; all switches off reproduce both
objects' old poses bit for bit over the original 129-frame comparison window.

To reproduce the previous behavior on this pin, set the delayed-keyframe,
consensus-refit, refined-return and object-center guard switches false and use
depth mode `legacy`.
The original pin `c3d2068d7aa12945c146206dea315bf1379a561c` is also retained.
Do not switch pins or configurations inside an active robot run.
