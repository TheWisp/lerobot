# Sparse motion and reacquisition regression

Eleven lossless camera observations (6.4 MiB, stored in Git LFS) extracted from
`saved-747dd29b27a745418524e2744ac29cb5`, recorded by the 9103 inspector on
2026-11-17 at commit `efdc380a5`. The archive stores PNG RGB bytes, original
float32 FoundationStereo depth, saved SAM masks, camera intrinsics and camera
transforms. Source timestamps and settings are in `manifest.json`. Checksums
verify both the archive and decoded RGB/depth. No video, robot commands,
estimated poses, fitted correspondences or serialized estimator objects are
test inputs.

| Sequence     | Frames                 | Reproduced baseline failure                         |
| ------------ | ---------------------- | --------------------------------------------------- |
| Cube         | 2735, 2741, 2747, 2752 | No compatible local alignment after motion          |
| Pointer head | 2992, 3000, 3008       | No compatible local alignment after motion          |
| Ring         | 4740, 4750, 4759, 4765 | Compatible fit rejected after Kalman fusion at 4759 |

The tracker initializes freshly at each sequence's first frame. These compact
sequences reproduce the failure mechanisms, not the original accumulated map
or its complete background-group association history. In particular, the
stale-group screenshot comes from a separate full-history replay; it is not
claimed as exact group-history coverage by this fixture.

`review_regions.json` contains broad pixel regions manually reviewed in the
camera images for the transported model centre. They are evaluation only and
exclude the stale cube location. A separate raw-depth check bounds the model
centre's distance from the observed object. These establish visible location, not precise
6D pose or correctness around object symmetries. Neither those annotations nor
later frames enter the estimator.

The complete-loss test replaces one intermediate observation with black RGB,
zero depth and an empty mask. This is a controlled input intervention, not a
claim that the recorded frame was black. It destroys the image-identity chain,
checks that the map/reference are not updated without measurements, and then
tests reacquisition and its continuation from real subsequent observations.
A negative control initializes on the cube, then deliberately supplies the
ring's actual observations: recovery must not accept it as the cube.

Run with an environment containing the repository dependencies and `small_gicp`:

```sh
git lfs pull --include='tests/policies/flash/fixtures/motion_recovery_20261117/*.npz'
PYTHONPATH=src:. python -m pytest tests/policies/flash/test_recorded_motion_recovery.py -q -rx
```

The production fitter now initializes translation from the measured surface
centres when that improves geometry agreement. Cube and pointer recovery,
including complete loss of image tracks, are ordinary passing regressions.
One **strict expected failure** pins the separate ring Kalman-fusion defect.
Passing it raises XPASS and requires removing its marker.
The same assertions run against opt-in policies in
`benchmarks/flash_motion_recovery.py`; these do not change production defaults.
For an unmarked red demonstration of that remaining defect, pass
`--runxfail -k 'baseline and ring'`.

Real UFM tests require CUDA, the installed UFM implementation, and the cached
checkpoint revision `980825b59331413bb51e5773f9272bfce525bce2`:

```sh
HF_HUB_OFFLINE=1 FLASH_RUN_UFM_RECOVERY=1 PYTHONPATH=src:. \
  python -m pytest tests/policies/flash/test_recorded_motion_recovery.py -q -k real_ufm
```

Re-export requires the pinned source recording; ordinary tests do not:

```sh
PYTHONPATH=src:. python -m benchmarks.flash_motion_recovery --export /path/to/saved-recording
```

Only observation data and provenance are regenerated. Review regions remain
separate so a replay or exporter cannot rewrite its own evaluation labels.
