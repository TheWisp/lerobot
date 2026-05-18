# SO-107 live URDF visualization

Browser-rendered 3D view of the robot's observed joint state, served alongside
camera feeds. Works for any run workflow (teleop, record, replay, policy) and
for either the unimanual or bimanual SO-107 follower.

## Quick start

### From the GUI

1. In the **Run** tab, pick your workflow (Teleop / Replay / Policy).
2. Check **URDF viz** in the form.
3. Launch. The camera grid appears with an extra tile labeled
   _"urdf (commanded joints)"_ — that's an iframe of the MeshCat scene.

### From the CLI

Same flag on every script:

```bash
# Teleop
uv run lerobot-teleoperate --display_urdf=true \
    --teleop.type=bimanual_quest_vr \
    --robot.type=bi_so107_follower --robot.id=white \
    --robot.left_arm_port=/dev/ttyACM0 --robot.right_arm_port=/dev/ttyACM2

# Replay an episode through the real arms
uv run lerobot-replay --display_urdf=true \
    --robot.type=bi_so107_follower --robot.id=white \
    --robot.left_arm_port=/dev/ttyACM0 --robot.right_arm_port=/dev/ttyACM2 \
    --dataset.repo_id=thewisp/intervention_cylinder_ring_assembly \
    --dataset.episode=0

# Record (collect a dataset under teleop control)
uv run lerobot-record --display_urdf=true ...

# HVLA policy rollout
python -m lerobot.policies.hvla.launch --display-urdf ...
```

(Note: the HVLA launcher uses `--display-urdf` with a dash; everything else
uses the snake-case `--display_urdf=true` that draccus expects.)

Standalone MeshCat URL: `http://127.0.0.1:7000/static/`.

## Architecture

```
   robot.get_observation()            <-- per tick
            │
            ▼
   robot_observation_processor
            │
       ┌────┴────┐
       ▼         ▼
   stream     UrdfVizMirrorStep
   writer     │
              ▼
       set_arm_joints_deg("left", q_deg_7)
       set_arm_joints_deg("right", q_deg_7)
              │
              ▼
       MeshCat HTTP server
       (port 7000, three.js viewer)
              │
              ▼
       iframe inside the GUI Run panel
```

The viz **only** reads observations (the actual robot state). Commanded
joints are NOT displayed — for that, the same `--display_urdf` flag also
attaches a `CommandedJointsLogStep` to the action pipeline that prints
post-IK joint targets at INFO (~1 Hz) so you can grep the file log.

## Implementation notes

- The MeshCat scene loads the SO-107 URDF twice (one per arm) under
  separate root nodes, offset along world ±X by ~30 cm so they don't
  overlap. URDFs map joint angles 1:1 to SO-107 motors S1..S7 (yes,
  including the gripper — it's a URDF joint, not a controller-only DOF).
- `UrdfVizMirrorStep` is observation-driven and returns its input
  unchanged. Any render failure is warn-once + DEBUG-after so the run
  loop is never broken by a viz hiccup.
- `BimanualUrdfViz` keeps the camera position low + close to fit the
  ~50 cm arms. Drag in the viewer to orbit / zoom freely; defaults
  reset on next run.
- The URDF base sits at world Z=0. The physical SO-107 is usually
  mounted on a riser above the desk, so when the arm reaches down for
  a task, the gripper renders below the grid floor — that's a viz
  framing choice, not a kinematics error.

## Calibrating the motor → URDF map per arm

The SO-107's motor calibration (homing_offset) zeroes each motor at a
specific encoder position, but that physical zero often doesn't coincide
with the URDF's "joint angle 0". Both FK and IK run in URDF space, so we
apply a per-motor `urdf_deg = sign * motor_deg + offset_deg` map at the
pipeline boundaries.

- `RIGHT_ARM_MAP` in
  [kinematics.py](kinematics.py) was discovered for the right arm via
  `experiments/motor_to_viewer.py` (slider-based visual alignment).
- `LEFT_ARM_MAP` is currently identity. **The left arm is rendered with
  whatever offsets its physical mounting needs, NOT corrected.** Run the
  same discovery process for the left arm to fill in its sign/offset
  table.

The same map is applied throughout the production Cartesian IK pipeline
(`EEReferenceAndDelta` for FK, `PinkInverseKinematicsEEToJoints` for IK)
— this is what `So107PinkKinematics` from the experimental learned-IK
work did, just plumbed through the modern `lerobot.processor` pipeline.

## Limitations

- SO-107 only. The viz module hardcodes the SO-107 URDF. Extending to
  another robot is straightforward: add a sibling
  `your_robot/urdf_viz.py` with a `maybe_attach_urdf_viz` of its own,
  and update the run scripts to look up the right one by robot type
  (or move the registry into `lerobot.robots.urdf_viz`).
- Port 7000 hardcoded (MeshCat default). Two simultaneous runs on the
  same host will collide; this hasn't been an issue in practice
  because the run scripts serialize through the GUI's `_launch_lock`.
- iframe is `http://127.0.0.1:7000/static/`. If you open the GUI from
  a different host (e.g. `http://lerobot.local:8000/`), the iframe
  will try to load localhost on the user's machine, not the teleop
  machine — local-host setups work; remote access needs MeshCat bound
  to LAN + iframe to use the LAN IP.
