# URDF State Visualization

An interactive 3D view of the robot — orbit and zoom with the mouse —
shown beside the camera feeds during teleop, recording, replay, and policy
runs. It mirrors the robot's **observed** joint pose; it is a viewer, not a
controller, so the operator can read the arm's state at a glance even when
the physical robot is out of view or the camera angle is unhelpful.

![The visualizer tile (bottom-centre) beside the camera feeds during a run.](images/urdf_viz.png)

It is **on by default**: whenever the running robot has a vendored URDF, the
viewer appears automatically as one tile in the observation grid.

## Data flow

```
observation stream (shared mem)
        │  {motor}.pos values
        ▼
resolve_robot(obs_keys)            urdf_viz.py — matches the motor set to a
        │                          vendored *_description package
        ▼
compute_joint_angles(spec, obs)    urdf_viz.py — motor pos -> URDF rad
        │  {prefix: {urdf_joint: radians}}
        ▼
GET /api/run/urdf-viz              run.py — thin JSON glue
        │
        ▼
static/urdf_viz.html               three.js + urdf-loader, in an iframe tile
```

All kinematics lives in **Python** (`urdf_viz.py`), not JavaScript. The
motor->URDF conversion is the correctness-critical part, so it sits under
pytest (`tests/gui/test_urdf_viz.py`) and carries asserts. The browser page
only fetches angles and draws them — it does no math.

## The two-layer motor->URDF mapping

A robot's motor readings rarely line up with its URDF's joint frame. Two
layers bridge the gap:

- **Layer 1 — the URDF itself.** A well-authored URDF (the vendored SO-101)
  is built to match the motor calibration: joint zero = motor zero, axes
  agree. Nothing else is needed.
- **Layer 2 — a per-joint `(sign, offset_deg)` delta.** A URDF exported
  straight from CAD (the SO-107, with generic `S1..S7` joint names) has a
  joint-zero that does not match the motors and some inverted axes. The
  delta realigns it:

  ```
  urdf_rad = deg2rad(sign * motor_pos + offset_deg)
  ```

Layer-2 alignment is **calibration data, not a model constant** — a
different physical build measures slightly different values, and the two
arms of a bimanual robot differ (some axes are mirror-mounted). It ships in
the description package as the _default_; an editable per-robot calibration
layer is future work.

## The `VIZ_SPEC` contract

`urdf_viz.py` names no robot. Each robot ships a `lerobot.robots.*_description`
package that declares a module-level `VIZ_SPEC` dict, and `resolve_robot`
discovers them at runtime by walking `lerobot.robots`:

```python
VIZ_SPEC = {
    "name":        str,                # human-facing display name
    "motors":      tuple[str, ...],    # motor names this robot exposes
    "urdf_joints": tuple[str, ...],    # URDF joint name per motor (parallel)
    "urdf_file":   str,                # URDF filename within urdf/
    "alignment":   {prefix: {motor: (sign, offset_deg)}} | None,
}
```

`alignment` is layer 2; `None` means identity. `prefix` is `"left_"` /
`"right_"` for a bimanual robot; a unimanual robot reuses the `"right_"`
entry.

`resolve_robot` matches a live robot by its **motor set**: the description
whose `motors` are all present wins, and the **most specific** (largest
motor set) wins a tie. That is why a 7-motor SO-107 resolves to SO-107 even
though its motors are a superset of the 6-motor SO-101.

## Adding a new robot

The GUI hardcodes no robot. A URDF is, of course, robot-specific — what is
generic is the GUI code that consumes it. Everything robot-specific (the
URDF, its meshes, any calibration) lives in a self-contained package under
`src/lerobot/robots/`, discovered at runtime:

```
src/lerobot/robots/so107_description/
├── __init__.py          # get_urdf_path(), get_meshes_dir(), VIZ_SPEC
├── joint_alignment.py   # optional — layer-2 (sign, offset_deg) deltas
├── urdf/
│   └── SO107.urdf
├── meshes/
│   └── *.stl            # link meshes the URDF references
└── PROVENANCE.md        # where the URDF + meshes came from
```

To onboard a robot:

1. Create `<name>_description/` with the `urdf/` and `meshes/` files and a
   `PROVENANCE.md` recording the upstream source. Meshes are git-LFS
   tracked — a `.gitignore` negation keeps these packages tracked despite
   the global `*.urdf` / `*.stl` ignore.
2. Add `__init__.py` exposing `get_urdf_path()`, `get_meshes_dir()`, and a
   module-level `VIZ_SPEC` (the contract above).
3. Choose the calibration layer. A URDF authored to match the motor
   calibration uses `"alignment": None`. A CAD-exported URDF needs the
   per-joint `(sign, offset_deg)` deltas measured against the real robot —
   conventionally in a `joint_alignment.py` that the `VIZ_SPEC` imports.
4. That is all. `server.py` mounts every `*_description` under
   `/urdf-assets/`, `resolve_robot` discovers the package by motor set, and
   `tests/gui/test_urdf_viz.py` parametrizes over it automatically —
   asserting the URDF parses, its `VIZ_SPEC` joints exist, and its meshes
   are on disk.

## Vendored browser libraries

`static/vendor/` holds three.js (r169) and urdf-loader, vendored so the
viewer runs offline with no CDN dependency. See `static/vendor/PROVENANCE.md`
for sources and the one local modification. `urdf-loader` clamps
`setJointValue` to the URDF `<limit>` by default; the viewer sets
`ignoreLimits = true` because layer-2 offsets legitimately exceed the CAD
limits, and the goal is to show the _true_ observed pose.
