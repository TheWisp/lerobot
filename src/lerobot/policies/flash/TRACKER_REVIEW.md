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
    "mode": "dense",
    "timeout_s": 3600
  }
}
```

The [upstream integration guide](https://github.com/ayushjain1144/trackeverything#inference-demopy--demo_hfpy)
uses dense mode and `output.tracks()` to expose persistent 3D scene points, their birth
frames and the pixel-to-track map. It documents sparse mode as an optional way to track
specified query points; query points are not merged with scene points and may retain
higher resolution. The model outputs point trajectories, **not object identities or
6D object poses**. FLASH must supply segmentation and fit or otherwise infer poses.

This adapter presently fits a rigid pose independently per frame from points sampled
inside one reference mask per object. Dense mode is the default because it also runs
the model's persistent scene representation while preserving those high-resolution
queries. It does **not** use `output.tracks()` for pose fitting or associate newly
born scene points with an object. Thus dense mode does not by itself address a rotating
object whose original visible surface disappears. Sparse mode remains available for
comparison. The raw per-query tracks and visibility logits are retained for diagnosis.

On the evaluated 613-frame insertion demo, switching this same query fit from sparse
to dense context took 27.1 s versus 27.9 s and improved median held-dowel RGB-D
surface discrepancy over frames 400–500 from about 27 mm to 18 mm. The dowel fit still
had only 18 inliers from a median 83.5 visible queries, and the native scene output
represented its reference mask with just 9 distinct tracks after deduplication at the
upstream default 20 mm voxel size. The candidate is therefore still an inaccurate
pose estimate, not a corrected skill. Use segmentation across later views to associate
new points with an object, then check temporal rigidity and RGB-D surface agreement
before accepting a pose. Reducing voxel size may retain more scene detail but costs
memory/compute; it should be measured rather than assumed to fix correspondence.
The review card reports fits whose frame-to-frame pose motion disagrees with the
motion of tracked points that supported the previous fit. This uses the saved
registration tolerance rather than a fixed speed limit. It is a warning, not
proof that other frames are accurate.
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
be applied to a different or changed bundle. Preview coverage and rigid-fit support are not accuracy scores:
even a high-coverage pose trace may contradict the image/depth evidence.

## TrackEverything access

The [Hugging Face model page](https://huggingface.co/katefgroup/TrackEverything) gates
the checkpoint behind accepting its conditions. Download with the account that accepted
them. Code and weights are published under CC BY-NC 4.0; accepting the download gate
does not grant commercial rights. Separate permission must come from the rights holder;
the [repository license](https://github.com/ayushjain1144/trackeverything/blob/main/LICENSE)
names Meta Platforms and affiliates. No commercial-license application form is provided
in those materials.
