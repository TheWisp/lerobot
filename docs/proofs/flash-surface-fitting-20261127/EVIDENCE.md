# FLASH surface fitting — evidence

Rendered on 2026-11-27 from two recorded fixtures that the surface fitting PR
ships, using that PR's own functions. These record one run at one revision;
they are not design documents.

| File                                           | What it shows                                                                                                                                                                                                                                                      | Inputs                                                                                                                                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01-controller-fit-and-publish.png`            | Frame 807 of a recorded controller turn. The stored surface drawn over the camera image at three poses: frame 800's published pose, the motion proposal from this frame's image matches, and the pose published after the local surface fit and Kalman pose update | `controller_motion_recovery_20261110.npz`; one `GrowingSurfaceSession.update` with the fresh-session settings (small_gicp solver), as in `test_fresh_session_recipe_recovers_the_recorded_controller_turn` |
| `02-geometry-check-rejects-published-cube.png` | Frame 3011 of a recorded cube session. The stored surface at the pose an earlier revision of FLASH published there in a 2026-11-10 replay, with the points that would sit in measured empty space                                                                  | `cube_published_geometry_20261110.npz`; `WholeSurfaceFit.check_pose` with the recording's settings, as in `test_final_published_cube_must_fail_geometry`                                                   |

Measured on these inputs:

- Controller, frames 800 → 807: 68 image matches give a motion proposal that
  places the 807-point stored surface a median 3.2 mm from frame 800's pose.
  The local surface fit ends a median 3.0 mm from that starting estimate, and
  the Kalman pose update moves it 0.55 mm more. The published pose is 5.2 mm
  from frame 800's. Against frame 807's measured surface it has a 1.2 mm
  median distance and 100% overlap, and its status is an observed pose.
- Cube, frame 3011: 13 of the 242 stored-surface points lie more than the
  5 mm depth tolerance in front of the surface the camera measured, a median
  9.7 mm. They lie in several separate patches, which the check counts as
  independent evidence of measured empty space rather than one noisy reading
  (outlier fraction 0.14 under its weighting), and it rejects the pose.

Re-render with `render_287.py` from the landing-build tooling:
`python render_287.py <tree with src/ and tests/> <out>`.
