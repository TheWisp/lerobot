# Click-target teleop — shelved experiment

**Status: archived.** This branch (`experiment/click-target`) carries a
self-contained experiment in click-to-go-to on a top-mounted depth camera.
It works end-to-end on the SO-107 bimanual rig but the failure modes are
fundamental to single-camera pixel-clicking, not to the implementation.

## The idea

Mount an Intel RealSense above the workspace, calibrate its 6-DoF pose
in robot base frame, and let the operator drive the right arm by
clicking on a target point in the camera view. The system unprojects
`(pixel, depth)` through the camera intrinsics, applies
`T_base_camera`, and pushes the resulting world XYZ to the IK as the
goto target. Scroll-wheel adjusts a hover offset above the clicked
surface so the gripper doesn't dive into the desk.

## What was built (in this branch)

| Piece                                                                                                                                                | File                                       |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `ClickTargetBimanualEETeleop` — thin Cartesian-VR shape teleop with `set_world_target`                                                               | [teleop](click_target_teleop.py)           |
| `ClickCalibrationService` — robot-owned mailbox handler for calibration + goto                                                                       | [service](service.py)                      |
| File-based JSON mailbox (single-host IPC GUI ↔ teleop subprocess)                                                                                   | [mailbox](mailbox.py)                      |
| Kabsch SE(3) fit + pinhole unprojection + extrinsics JSON I/O                                                                                        | [calibration](calibration.py)              |
| New action keys `use_world_target` / `target_world_x/y/z` / `world_target_top_down` on `CartesianIKController`, plus relaxed `_GOTO_MAX_*` caps      | `robots/so107_description/cartesian_ik.py` |
| `RealSenseCamera.get_color_intrinsics()` + depth caching even when a `post_grab_processor` is installed                                              | `cameras/realsense/camera_realsense.py`    |
| Click service init / start / stop / goto-callback wiring on both follower variants                                                                   | `robots/bi_so107_follower*/...py`          |
| GUI: right-side docked panel, two-step calibration (click surface → drive gripper → confirm), hover preview with surface_z + target_z + colored ring | `gui/api/run.py`, `gui/static/run.js`      |
| New endpoints: `/click-target/{sample-pixel, sample-fk, goto, finalize, clear, status}`                                                              | `gui/api/run.py`                           |
| `click_target` added to the registry-loading import block                                                                                            | `scripts/lerobot_{teleoperate,record}.py`  |

## What worked

- Two-step calibration (click on a known fixture point, drive gripper there with any teleop, confirm) hit RMSE ≈ 13 mm with 4 points. 6 well-spread points would likely drop to <5 mm.
- The mailbox-based IPC was solid after the start race was fixed (drain on start, defer to `attach_teleop` end).
- Service architecture is clean: robot owns the calibration service; click_target teleop is a thin Cartesian-VR-shape wrapper that the service can push targets into. Calibration works with any teleop attached.
- The absolute-world branch on `CartesianIKController` is a useful primitive in general — any teleop emitting `use_world_target=1.0` gets an IK-clipped, rate-limited walk-in to the target XYZ.

## Why it's shelved

Two problems neither code nor calibration can solve from a **single** top-down camera:

1. **Heavy occlusion.** The arm is regularly in front of the workspace.
   A click that "looks like" the desk often lands on the gripper, the
   leader arm, a fixture, or a part being grasped — and the depth at
   that pixel is whatever the visible surface is. The unprojected
   surface_z then jumps wildly (≈ 0 mm for a desk click, 200+ mm for a
   gripper-top click), and the goto target follows it. We added a hover
   preview showing `surface_z` + `target_z` in real time so the operator
   sees the gotcha before clicking — it helps, but it doesn't fix the
   underlying ambiguity of "what does this pixel mean?".

2. **Safety boundary is hard to draw.** With a 13 mm calibration RMSE
   the floor on `z_offset` has to be ≥ 30 mm to keep the gripper clear
   of the desk. That's enough to dive a few mm into a part you just
   placed if you click slightly off. There's no clean signal for "you're
   about to hit something" that the operator can act on in time:
   - `target_z` is the only number you have, and it conflates "object is
     tall" with "you scrolled the offset up";
   - the IK's per-tick caps catch IK singularities, not surface contact;
   - real proximity sensing or a calibrated workspace floor (z_world =
     some constant) would help, but the current rig doesn't have either.

The combination of (1) and (2) means **every click is one mental error
away from a crash**, and the operator can't easily build the right
mental model from what the UI shows. That's a usability ceiling, not a
bug list — fixing it needs either multi-view depth (so occlusion is
recoverable), or a known fixed workspace plane (so all clicks resolve
to a known z), or both. Neither is in scope for the SO-107 rig.

The pose preview / hover ring UI is the right design pattern; the
**input modality** is what doesn't work.

## When to revive

This branch is worth coming back to **if**:

- The workspace gets a second top camera at a different angle (resolves
  occlusion: two views constrain the click to a real 3D point even when
  one view is blocked).
- The rig moves to a known-flat workspace where `z_workspace` is a
  hardcoded constant and any depth read below that gets clamped (resolves
  safety boundary: gripper never goes below the surface).
- Someone wants to use the absolute-world IK keys for a non-click teleop
  (scripted demos, RL replay, etc.) — the keys + caps are general and
  cherry-pick cleanly.

To revive, branch off this one rather than re-merging: the diff against
`feat/quest-vr` (or `main`) is ~10 files and a few hundred lines.

## What works for further teleop UX experiments

The architecture pieces here that survive their first contact with users:

- **Robot-owned background service** ([service.py](service.py)) +
  file-based mailbox ([mailbox.py](mailbox.py)). Works for any
  teleop-agnostic auxiliary service the GUI wants to talk to. Robust
  start (drains stale state) and clean lifecycle (started in
  `attach_teleop` after motor reads, stopped in `disconnect`).
- **`set_world_target` callback pattern** on the click_target teleop +
  `set_goto_target_callback` on the service. Clean way for a service to
  push state into a teleop without the teleop knowing the service
  exists.
- **Right-side docked panel pattern in the GUI**
  ([run.js](../../gui/static/run.js)) matching the dataset Inspector
  visual style. Works for any teleop-paired UI that needs persistent
  state alongside the live camera view.
- **Two-step capture** (click pixel → user does something → click
  confirm). Lets the GUI mediate any "user-in-the-loop" workflow that
  needs an action between two backend samples.

## Original feature request → final reality

| Original ask                            | What we ended up with                                                                                                                                     |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Click in cam view, gripper goes there   | Works mechanically but operator can't reliably tell where "there" is going to be                                                                          |
| Calibrate the camera pose interactively | Two-step Kabsch fit, teleop-agnostic. Solid but RMSE leaves a safety margin we couldn't tighten on a single camera                                        |
| Scroll to raise Z above clicked surface | Implemented; preview + numeric label make it legible but don't fix the surface_z ambiguity                                                                |
| Safety / not-scary motion               | Tightened EE/joint caps to 1.5 cm/tick + 25°/tick (peak ~1.4 m/s), STOP button. Still felt fast in operation because of the ambiguity above, not the rate |
