# Offline demo tracker review

Open **Run → FLASH → Demos**, play a prepared demo, then choose **Edit demo → Trackers**.
Select a tracker (all annotated objects by default) and choose **Process demo and preview**. Processing uses the
selected version's per-object reference masks and the recorded RGB-D frames. It does not connect to
the robot. The result appears as a pink outline alongside the saved skill pose; hovering
the candidate card isolates that outline.

**Apply as new demo version** replaces the processed objects’ reference/held poses at
covered skill samples and opens the resulting version. Missing estimates retain their
previous pose and evidence. The report records changed and preserved sample counts.
The previous prepared version and raw episode remain available. No smoothing or
interpolation fabricates observations between sparse tracker outputs.

## Local runners

Runner installations live in `~/.config/lerobot/flash_trackers.json`, or the file named
by `LEROBOT_FLASH_TRACKERS_CONFIG`. The GUI only offers processing when a runner is
configured. Saved NPZ results can also be imported without installing that model.

TrackEverything has a built-in adapter:

```json
{
  "trackeverything": {
    "python": "/absolute/path/to/environment/bin/python",
    "repository": "/absolute/path/to/trackeverything",
    "checkpoint": "/absolute/path/to/model.pth",
    "query_count": 128,
    "timeout_s": 3600
  }
}
```

The query count is a configurable spatial sampling budget, not a pose acceptance rule.
Rigid fitting uses the selected demo's registration threshold, minimum inliers and
trial count. TrackEverything receives measured depth and camera intrinsics and runs
with future context within temporal windows. This is an offline demo extractor; this
integration does not make it a causal live perception source. The current adapter
requires one appearance reference per object and the existing fixed-camera RGB-D recording layout.

All selected objects share one TrackEverything inference call, including objects seeded at
different reference frames. Each object gets its own rigid fit. It tracks a fixed reference
query set; adding points on newly revealed surfaces is not implemented.

Other trackers register an argv template under their tracker ID:

```json
{
  "point2pose": {
    "command": [
      "/absolute/path/to/python",
      "/absolute/path/to/adapter.py",
      "--recording",
      "{recording}",
      "--reference-mask",
      "{reference_mask}",
      "--first-frame",
      "{first_frame}",
      "--last-frame",
      "{last_frame}",
      "--output",
      "{output}"
    ]
  }
}
```

Available substitutions are `dataset_root`, `recording`, `bundle_path`, `reference_mask`,
`role`, `first_frame`, `last_frame`, `output`, and `run_dir`. Arguments are passed without
a shell. Single-object external adapters run once per selected object, sequentially,
then publish one candidate containing all results. Each run has its own dataset sidecar directory and log. The GUI offloads the
subprocess wait to a dedicated executor, separate from replay reads and robot control.
The configurable timeout bounds an offline job; closing the editor does not cancel it.

Point2Pose's existing experiment runner accepts arbitrary role names with `--initial-mask`
and an exact `--output-file`, so it can be registered directly. The registry also accepts
BundleSDF, TAPNext++, CoTracker3, LiteTracker, PANY and SAM3/DINO outputs. Those entries
do not imply that a model environment or conversion adapter has been installed.

## Pose trace contract

The output NPZ contains integer `frames[N]`, rigid `poses[N,4,4]`, and boolean `lost[N]`.
Frame indices refer to the original camera recording. The reference frame must have an
available pose. Transforms map reference-camera coordinates into the current camera;
the importer normalizes the reference self-pose and converts into the prepared object's
calibrated base frame. An object-to-camera output from another tracker must be converted
to this relative convention by its adapter before import.

The candidate manifest records its source bundle and trace hashes. A candidate cannot
be applied to a different or changed bundle. Preview coverage is not an accuracy score:
even a high-coverage pose trace may contradict the image/depth evidence.

## TrackEverything access

The [Hugging Face model page](https://huggingface.co/katefgroup/TrackEverything) gates
the checkpoint behind accepting its conditions. Download with the account that accepted
them. Code and weights are published under CC BY-NC 4.0; accepting the download gate
does not grant commercial rights. Separate permission must come from the rights holder;
the [repository license](https://github.com/ayushjain1144/trackeverything/blob/main/LICENSE)
names Meta Platforms and affiliates. No commercial-license application form is provided
in those materials.
