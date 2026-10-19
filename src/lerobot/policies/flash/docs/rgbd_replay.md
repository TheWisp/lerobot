# Initial RGB-D pose replay

This is an offline diagnostic of the first SAM3 mask → DINOv2 patch match →
aligned-depth rigid-fit backend. It tests whether a recorded view supplies
correspondences and a self-consistent fit; it is **not** an independently
measured pose-accuracy or hardware-rollout result. The recording is the sealed
insertion demo on this rig, under
`~/.cache/huggingface/lerobot/demos/insertion/perception/9f96bc840f73/stage/`.
Its JPEG RGB and PNG depth are aligned at 848×480, with `cam_K.txt` intrinsics
and millimetre depth units. The recording is not stored in git.

Reproduce using `benchmarks/flash_rgbd_replay.py` from the repository root:

```bash
PYTHONPATH=src python benchmarks/flash_rgbd_replay.py \
  ~/.cache/huggingface/lerobot/demos/insertion/perception/9f96bc840f73/stage \
  --prompt 'green ring' --role ring --reference-frame 100 \
  --start 100 --stop 130 --output /tmp/flash-ring.csv

PYTHONPATH=src python benchmarks/flash_rgbd_replay.py \
  ~/.cache/huggingface/lerobot/demos/insertion/perception/9f96bc840f73/stage \
  --prompt 'wooden dowel' --role dowel --reference-frame 100 \
  --start 100 --stop 320 --stride 5 --output /tmp/flash-dowel.csv
```

Both runs used cached SAM3 and `facebook/dinov2-small` on CUDA, cosine floor
0.75, 5 mm registration tolerance, 12 minimum depth-backed inliers and 100
RANSAC trials. These are explicit benchmark settings, not object-specific
defaults in the pose backend. The CPU loop remained separate from any motor
loop. Warm pose-update time includes segmentation, DINO, matching and fit,
but excludes initial model loading.

| Sequence                                | Poses accepted | Median pose-update time | Observation                                                                                                     |
| --------------------------------------- | -------------: | ----------------------: | --------------------------------------------------------------------------------------------------------------- |
| Ring, static frames 100–130             |          31/31 |                 82.4 ms | Fitted reference-centre displacement: median 0.22 mm, maximum 0.43 mm. No independent target pose is available. |
| Dowel, frames 100–320 every fifth frame |          19/45 |                 77.2 ms | The tray view tracks until occlusion near lift; it does not reacquire reliably after the object turns.          |

For the dowel, 17 frames had too few pixel correspondences, five had no mask,
two had enough pixel matches but too little aligned depth, and two failed
geometric consistency. A few frames during the pickup still fit, but without
independent pose truth they are not evidence that the held pose is correct.
The fallback gripper attachment must remain marked as a **prediction**, never
silently upgraded to an observation. A later demo view of the held object can
extend the appearance bank only if registered into the same stable object
frame; that must be evaluated on held frames and across rollouts.

The original fitter rejected frame 101 of the static ring despite 323 inliers
and 1.8 mm median surface residual. It refit a second time after admitting
boundary points and then rejected the shifted boundary. The fit now retains
the first consensus refit and reports its measured inliers. A compact
correspondence fixture from frames 100→101 protects this correction.
