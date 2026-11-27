# FLASH per-frame tracking — evidence

Rendered on 2026-11-27 from a recorded fixture that the per-frame tracking PR
ships, using that PR's own functions. It records one run at one revision; it is
not a design document.

| File                                       | What it shows                                                                                                                                                                                            | Inputs                                                                                                                                                                                       |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `03-rejection-leaves-groups-unchanged.png` | Frames 1866 and 1871 of a recorded ring on a table. The scene points of the motion group, the ring's pose hint from that group, its rejection by the geometry checks, and the groups after the rejection | `ring_group_rejection_20261107.npz`; shared LK tracking, `SceneMotionGroups` and `ObjectGroupAssociation`, as in `test_recorded_ring_geometry_rejection_cannot_erase_measured_scene_support` |

Measured on these inputs:

- The hint comes from 39 ring points and 662 scene points in one motion group.
- The geometry checks fail it: 20% of its visible points disagree with measured
  depth, over the 10% allowed with the test's settings; overlap 80%, median
  distance 2.3 mm.
- After the rejection is recorded, the group has the same point IDs and
  positions, and the next query returns the same hint from the same 662 points.

The same values come out at b2a7399ff, before the older pose service was
retired.

Re-render with `render_288.py` from the retirement build tooling, from a tree
with `src/` and `tests/`: `python render_288.py <out>`.
