# OpenArm2 sources

## Official OpenArm sources

- [OpenArm 2.0 overview](https://docs.openarm.dev/overview/whats-new-in-2.0/)
- [OpenArm hardware documentation](https://docs.openarm.dev/hardware/)
- [OpenArm software guide](https://docs.openarm.dev/api-reference/)
- [OpenArm CAN CLI reference](https://docs.openarm.dev/api-reference/can/cli/)
- [OpenArm SocketCAN setup](https://docs.openarm.dev/setup/openarm-setup/can-setup/)
- [Official OpenArm repository](https://github.com/enactic/openarm)

Use these sources for the current hardware, assembly, wiring, motor, and
software specifications. OpenArm is actively developed; avoid copying mutable
configuration values into this skill.

## LeRobot integration in this repository

Read the linked files instead of assuming a fixed robot layout:

- Damiao CAN bus and volatile control-mode handling:
  [damiao.py](../../../../src/lerobot/motors/damiao/damiao.py)
- Damiao registers, modes, and limits:
  [tables.py](../../../../src/lerobot/motors/damiao/tables.py)
- OpenArm follower configuration:
  [config_openarm_follower.py](../../../../src/lerobot/robots/openarm_follower/config_openarm_follower.py)
- OpenArm follower implementation:
  [openarm_follower.py](../../../../src/lerobot/robots/openarm_follower/openarm_follower.py)
- Bimanual follower configuration:
  [config_bi_openarm_follower.py](../../../../src/lerobot/robots/bi_openarm_follower/config_bi_openarm_follower.py)
- Bimanual follower composition:
  [bi_openarm_follower.py](../../../../src/lerobot/robots/bi_openarm_follower/bi_openarm_follower.py)
- OpenArm leader configuration and implementation:
  [config_openarm_leader.py](../../../../src/lerobot/teleoperators/openarm_leader/config_openarm_leader.py),
  [openarm_leader.py](../../../../src/lerobot/teleoperators/openarm_leader/openarm_leader.py)
- Bimanual leader configuration and implementation:
  [config_bi_openarm_leader.py](../../../../src/lerobot/teleoperators/bi_openarm_leader/config_bi_openarm_leader.py),
  [bi_openarm_leader.py](../../../../src/lerobot/teleoperators/bi_openarm_leader/bi_openarm_leader.py)
- Damiao and OpenArm control tests:
  [test_damiao_mit.py](../../../../tests/motors/test_damiao_mit.py),
  [test_openarm_follower_control.py](../../../../tests/robots/test_openarm_follower_control.py)

The current follower configuration selects the gripper control path. During
connection, the follower restores that configured volatile Damiao control mode
before enabling torque. Verify the current behavior in the linked configuration,
follower, Damiao, and test files rather than relying on copied mode numbers.

Read the active robot profile for motor and camera configuration. Read the
target dataset metadata for feature names, ordering, dimensions, and visual
keys.
