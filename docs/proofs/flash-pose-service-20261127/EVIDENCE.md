# FLASH pose service — evidence

Rendered on 2026-11-27 from three recorded fixtures that the pose service PR
ships, using that PR's own functions. These record one run at one revision;
they are not design documents.

| File                                       | What it shows                                                                                                                                                                                          | Inputs                                                                                                                                                                                                                               |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `01-direct-versus-point-motion.png`        | Frame 498 of a recorded dowel insertion. For each object, the pose carried by its point motion and a late direct pose for the same frame, and which one `CausalPoseService` publishes                  | `direct_point_contact_20261021.npz`; one `CausalPoseService._anchor` per object, set up as in `test_real_contact_conflict_keeps_point_prediction_and_records_direct_candidate` and `test_same_rule_accepts_recorded_ring_correction` |
| `02-direct-pose-in-empty-space.png`        | Frame 400 of a recorded dowel insertion. A published group prediction, a direct pose that would put the dowel in measured empty space, and a direct pose at the predicted position                     | `free_space_insertion_20261021.npz`; two `CausalPoseService._anchor` calls, as in `test_recorded_pose_in_measured_empty_space_cannot_replace_group_prediction`                                                                       |
| `03-rejection-leaves-groups-unchanged.png` | Frames 1866 and 1871 of a recorded ring on a table. The scene points of the motion group, the ring's pose hint from that group, its rejection by the surface check, and the groups after the rejection | `ring_group_rejection_20261107.npz`; shared LK tracking, `SceneMotionGroups` and `ObjectGroupAssociation`, as in `test_recorded_ring_geometry_rejection_cannot_erase_measured_scene_support`                                         |

Measured on these inputs:

- Insertion, frame 498, dowel: the point-motion pose and the direct pose are
  11 mm and 15° apart. Against this frame's depth inside the dowel's mask, 90%
  of the point-motion pose's points are within 9.6 mm of the measured surface
  and 90% of the direct pose's within 75 mm. The verdict is `prior_favored`;
  the point-motion pose stays published as a prediction and the direct pose is
  recorded as `direct_point_unresolved`.
- Insertion, frame 498, ring: the two poses are 1.0 mm and 3.0° apart, and 90%
  of either pose's points are within 3.0 mm of the measured surface. The
  verdict is `agree`, and the direct pose is published as an observed pose.
- Insertion, frame 400: the refused direct pose is a median 60 mm from the
  prediction. Its median stored point lies 107 mm in front of the measured
  surface, against a 10 mm limit, so it is `contradicted` and the prediction
  stays published. A direct pose at the predicted position has its median
  point 3.4 mm behind the measured surface, is `not_contradicted`, and is
  published as an observed pose.
- Ring, frames 1866 → 1871: the hint comes from 39 ring points and 662 scene
  points in one motion group. The surface check fails it: 20% of its visible
  points disagree with measured depth, over the 10% allowed with the test's
  settings (the fresh-session setting is 25%); overlap 80%, median distance
  2.3 mm.
  After the rejection is recorded, the group has the same point IDs and
  positions, and the next query returns the same hint from the same 662 points.

Re-render with `render_288.py` from the landing-build tooling, from a tree
with `src/` and `tests/`: `python render_288.py <out>`.
