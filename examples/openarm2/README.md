# OpenArm 2.0 — Data Collection Examples

> For what this branch changes and why (dora→lerobot port map, VR teleop
> solution choice, validation status), see [PORTING.md](PORTING.md).

`lerobot-record` command snippets for a bimanual OpenArm 2.0 setup
(`bi_openarm_follower`), with either the joint-space leader arms
(`bi_openarm_leader`) or a Meta Quest 3 (`quest_vr`, Cartesian IK) as the
teleoperator. Adjust CAN ports, camera indices, and `HF_USERNAME` to your
setup.

```bash
export HF_USERNAME=<your_hf_username>
export ROBOT_CAMERAS="{ left_wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30} }"
```

## Setup

- Follower arms on `can0` (left) and `can1` (right); leader arms on `can2` /
  `can3`. CAN FD is on by default (1 Mbps nominal / 5 Mbps data).
- Gravity feedforward (`gravity_ff_gain=0.9`, validated) needs the
  `openarm-ff` extra (MuJoCo inverse dynamics on the OpenArm 2.0 MJCF):

  ```bash
  uv sync --locked --extra openarm-ff
  ```

- The Quest VR path additionally needs the `quest-vr` extra and `pin-pink`
  (Cartesian IK):

  ```bash
  uv sync --locked --extra quest-vr --extra openarm-ff
  ```

## Option A — Leader-follower (joint space)

`bi_openarm_leader` already emits joint positions, so its action keys match
the follower directly.

```bash
lerobot-record \
    --robot.type=bi_openarm_follower \
    --robot.id=bi_openarm_follower \
    --robot.left_arm_config.port=can0 \
    --robot.left_arm_config.side=left \
    --robot.left_arm_config.gravity_ff_gain=0.9 \
    --robot.left_arm_config.use_velocity_and_torque=true \
    --robot.right_arm_config.port=can1 \
    --robot.right_arm_config.side=right \
    --robot.right_arm_config.gravity_ff_gain=0.9 \
    --robot.right_arm_config.use_velocity_and_torque=true \
    --robot.cameras="$ROBOT_CAMERAS" \
    --teleop.type=bi_openarm_leader \
    --teleop.id=bi_openarm_leader \
    --teleop.left_arm_config.port=can2 \
    --teleop.right_arm_config.port=can3 \
    --dataset.repo_id=$HF_USERNAME/openarm2_leader_pick_place \
    --dataset.single_task="Pick the cube and place it in the bin." \
    --dataset.fps=30 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=true \
    --display_data=true
```

## Option B — Quest 3 VR (Cartesian IK)

`quest_vr` streams per-arm end-effector deltas; the follower's
`attach_teleop` installs the OpenArm Cartesian-IK transform (per-arm URDFs in
`lerobot.robots.openarm_description`) that turns them into joint commands.
OpenArm 2.0 specifics vs the SO-107 defaults:

- `robot_forward_in_urdf=[1,0,0]` — the OpenArm base frame reaches in +X
  (up stays +Z).
- Gripper motor ranges (motor degrees): left arm 0 = open .. +45 = closed;
  right arm 0 = open .. -45 = closed.

```bash
lerobot-record \
    --robot.type=bi_openarm_follower \
    --robot.id=bi_openarm_follower \
    --robot.left_arm_config.port=can0 \
    --robot.left_arm_config.side=left \
    --robot.left_arm_config.gravity_ff_gain=0.9 \
    --robot.left_arm_config.use_velocity_and_torque=true \
    --robot.right_arm_config.port=can1 \
    --robot.right_arm_config.side=right \
    --robot.right_arm_config.gravity_ff_gain=0.9 \
    --robot.right_arm_config.use_velocity_and_torque=true \
    --robot.cameras="$ROBOT_CAMERAS" \
    --teleop.type=quest_vr \
    --teleop.id=quest_vr \
    --teleop.port=8443 \
    --teleop.robot_forward_in_urdf="[1,0,0]" \
    --teleop.robot_up_in_urdf="[0,0,1]" \
    --teleop.left_gripper_open_motor=0 \
    --teleop.left_gripper_closed_motor=45 \
    --teleop.right_gripper_open_motor=0 \
    --teleop.right_gripper_closed_motor=-45 \
    --dataset.repo_id=$HF_USERNAME/openarm2_quest_vr_pick_place \
    --dataset.single_task="Pick the cube and place it in the bin." \
    --dataset.fps=30 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=true \
    --display_data=true
```

Open `https://<your-LAN-IP>:8443/` in the Quest 3 browser (self-signed cert
warning once per device), enter the immersive session, and use the side grip
button to engage arm tracking and the index trigger for the gripper.

## Control-regime rule: one FF/ramp regime per dataset

The recorded actions are the _commands sent to the motors_, so the control
regime (gravity FF, velocity FF, alignment ramp) is part of what a policy
learns from. Keep exactly one regime per dataset:

- Pick `gravity_ff_gain` / `velocity_ff_gain` / `align_step_limit` before
  recording and do not change them between episodes of the same dataset.
- Changing any of these settings = a new control regime = record into a new
  `dataset.repo_id` so the regime is captured in that dataset's metadata
  instead of silently mixed into an existing one.
- The validated regime is `gravity_ff_gain=0.9` with velocity FF off and the
  alignment ramp off (`align_step_limit=null`). If you enable the ramp, the
  validated value is `align_step_limit=0.003` (rad/step, gripper excluded).
