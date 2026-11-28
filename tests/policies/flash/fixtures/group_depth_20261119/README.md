# Recorded group-depth regression

The archive contains 38 sequential measured scene-point snapshots from frames
5118–5381 and the original Foundation Stereo depth image for frame 5381.
It is 832,785 bytes (813 KiB); it contains no RGB video, fitted object pose, or
serialized internal group state. Calibration, timestamps, source settings and
SHA-256 are in [manifest.json](manifest.json).

The shared LK tracker produced these inputs from the saved recording named in
the manifest. The regression replays all measured snapshots through the real
frame owner and group tracker. Earlier depth is unknown to reconstruct the
historical stale membership; endpoint depth must remove all 15 specified
members while retaining over 400 measured scene points. This is a group-depth
regression, not an independent test of LK or learned depth accuracy.

The full-pipeline performance comparison separately runs every frame with its
own original RGB, mask and depth; see the
[dated evidence](https://github.com/TheWisp/lerobot/blob/29a575fe8a8621ebf38990a8af578d3ed02f5ede/docs/proofs/flash-recovery-20261119/README.md#depth-rejection-of-carried-group-points).

Re-export from the original recording:

```sh
PYTHONPATH=src:. python -m benchmarks.flash_group_depth /path/to/saved-recording --export
```
