# FLASH implementation status

FLASH lives at **Run → FLASH**. The current Run panel validates a versioned
spatial skill annotation. It deliberately does not command a robot yet: the
live tracker must register each object into the _same stable object frame_ used
by the demonstration before the spatial equation can produce safe goals.

The executable core already separates:

- demonstrated contact intervals, sampled on their own 1× local clock;
- generated connections, timed from live path length and configured speed;
- milestone verdicts and explicit, bounded recovery transitions;
- metric RGB-D and pose evidence with source and availability times;
- held-object world pose from the gripper-to-object attachment;
- kinematics-neutral short IK chunks and atomic future-queue replacement;
- a camera-independent motor read/dispatch tick through `Robot`;
- standard LeRobot episode frames, with a `flash/v1` trace joined by episode
  and frame index. The trace records intended and robot-accepted actions.

The annotation points to a source LeRobot dataset episode. Dense transform
samples are derived geometry; the original RGB-D, robot observations and
actions remain in the regular episode. Each interval identifies its object
role and reference-frame version. A held-to-target interval also declares the
held object's reference-frame version. Missing versions are shown in skill
inspection and rejected by the joint planner.

The next integration must provide a pose backend that consumes aligned RGB-D,
matches live objects to the demonstration's reference snippets, reports
supported pose dimensions and versioned object frames, and publishes snapshots
asynchronously. Then wire the producer, milestone verifier, recorder and
existing robot-specific kinematics into a session runner. A physical rollout
cannot be claimed from the core unit tests or skill-inspection UI alone.
