# Reproducible Point2Pose experiments

Point2Pose runs in its own Python environment. Its source comes from
[TheWisp/point-to-pose](https://github.com/TheWisp/point-to-pose), a fork of
[tzuyuan/point-to-pose](https://github.com/tzuyuan/point-to-pose).

The fork's `main` follows upstream. Small fix branches hold independent bug
fixes and their regression tests. `showservo-experiments` combines the changes
selected for this workstream. Its initial integration preserves the two
existing dense-registration commits and adds the SDF return-value correction
without importing upstream's September runtime changes.

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
an unknown source version without its correctness fix.

## Identify a recorded experiment

Each P2P result carries `software`: P2P commit, LeRobot commit and dirty flag,
configuration SHA-256, bridge SHA-256, and environment SHA-256. Act recordings
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
3. Push the reviewed integration commit, update LeRobot's manifest, and run
   setup. Commit the manifest/configuration changes and their validation.
4. Switch the live process only at a safe restart; never restart the GUI while
   it owns a connected arm. Preserve the previous pin for rollback.

The SDF correction is implemented once in the fork. Remaining bridge patches
are historical performance/integration changes with source checks; migrating
one requires separate equivalence checks. Do not combine dependency upgrades,
new estimator designs and a small correctness fix into an untraceable update.
