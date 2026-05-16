# SO-107 description / experiments

Research scripts that accumulated during the so107 teleop development. Kept around
for reproducibility and for picking up specific tools as needed. None of these are
production-ready or part of the lerobot public API; for actual Cartesian teleop see
`../teleop_keyboard_dls.py` at the package root.

## Calibration / setup

- **`motor_to_viewer.py`** — Hand-move the right arm; URDF in MeshCat tracks live.
  Used to discover the per-joint sign in `RIGHT_ARM_MAP`.
- **`calibrate_offsets.py`** — Interactive Tk slider UI to capture each joint's
  offset by visually aligning the URDF to the real arm. Output: a `RIGHT_ARM_MAP`
  dict literal printed to stdout.
- **`motor_health_check.py`** — Bypass IK entirely; nudge each motor ±N° and
  print compliance %. Useful when a servo seems wedged.

## Visualization / diagnostics

- **`fk_live.py`** — Hand-move arm; print live FK xyz at 20Hz. Verifies
  URDF base-frame alignment matches physical reality.
- **`explore_workspace.py`** — Autonomous probe through a grid of (shoulder_lift,
  elbow_flex) poses, measuring per-direction EE response. Outputs a heatmap.

## Tabulated IK pipeline (deprecated, kept for reference)

- **`record_ik_table.py`** — Record (motor_pos, FK_EE) pairs while hand-moving.
- **`teleop_keyboard_lookup.py`** — KD-tree IK from a recorded table.

## Teleop variants

- **`teleop_keyboard.py`** — placo-based IK + JacobianVelocityController.
  First attempt; superseded.
- **`teleop_keyboard_ikpy.py`** — ikpy SLSQP IK. Branch-hops between solution
  configs (the "weird pose" issue).
- **`teleop_sim.py`** — Headless sim version with motor-dynamics modelling.
- **`sim_bench.py`** / **`sim_min_motion_bench.py`** — Parameter sweep harnesses
  for tuning IK in sim without involving real hardware.
