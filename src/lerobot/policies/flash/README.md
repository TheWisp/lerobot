# FLASH experiments

**Run → FLASH** offers three views:

- **Video examples:** select an RGB-D recording, play or scrub it, compare saved
  methods and scores, inspect the 3D scene, or rerun one/all examples.
- **Live camera:** open an RGB-D camera, name an object or use Pick object,
  then inspect the mask, fitted surface, axes and supporting groups in image/3D mode.
  This inspector never moves the robot.
- **Demo editor:** review a recorded robot episode and marked keyframes, then
  prepare a live trial using the same camera and tracking components.

Video examples and live camera use `SurfaceFrameSession.step`; the demo editor's
`DemoRunTracker` also uses that shared tracking session. Each run records its
actual image matcher, surface matcher, depth source, and other settings.

Geometry checks reject poses that contradict reliable measured depth. Texture
and outline coverage rank compatible poses; they cannot excuse protrusion into
measured empty space. Missing depth and occlusion are unknown evidence.
Tracking remains imperfect, especially through occlusion and recovery. The demo
editor's action path is in `demo_execution.py` and `demo_live_worker.py`: it
builds Cartesian tool targets from marked demo frames and current object poses,
converts them to joint actions through the robot's `CartesianModel`, and sends
the actions through `ActionQueue` and `MotorLoop`. `Robot.send_action` receives
joint and gripper values, not a Cartesian transform. The current `CartesianModel.inverse`
can return an approximate, joint-clipped solution without checking that the tool
reaches the requested Cartesian pose. Queue completion means actions were sent;
it does not establish that the tool reached the goal or the manipulation succeeded.

The remaining demo-action work is tracked in [demo action follow-ups](docs/demo_action_todo.md).

Each demo action run first moves to the rest position saved in the selected robot
profile, using the shared three-second rest-position ramp. The live status shows
`resetting to rest position`; Stop interrupts the ramp and prevents demo dispatch.
Planning starts from joint readback after the ramp. Robot runs require a saved
rest position. Test runs simulate the same reset when one is saved, without
connecting to robot motors. Stopping or completing a run does not reset the arm;
an explicit return-to-rest action after a run is tracked in
[#273](https://github.com/TheWisp/lerobot/issues/273).

Start with:

- [Pipeline and limitations](docs/tracking_pipeline.md): data flow, geometry,
  groups, uncertainty and surface growth.
- [Configuration reference](docs/pipeline_settings.md): settings shared by
  recorded benchmarks and live camera inspection, including iteration limits.
- [Research evidence](../../../../docs/proofs/flash-tracker-restart/RESEARCH.md):
  dated experiments, rejected approaches, comparisons and measured trade-offs.

Old saved playback retains its actual solver/settings and never reruns inference.
Rerunning an old SciPy configuration selects the maintained fitter and records
that change; the previous result is unchanged.

The [sealed handover](docs/SEALED_20261031.md) describes the previous demo compiler,
robot engine and editor design. That code, and the pose service it ran on, were removed
on 2026-11-27; [b2a7399ff](https://github.com/TheWisp/lerobot/tree/b2a7399ffa0cefee408f741b1d45ad3118954c63/src/lerobot/policies/flash) is the last commit that has it. The [old implementation map](https://github.com/TheWisp/lerobot/blob/18fa9d9f40d230ca2fa8d8bead75d24b573a8316/src/lerobot/policies/flash/README.md)
is historical, not the current UI specification.
