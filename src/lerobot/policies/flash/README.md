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
- skill compilation from ordinary LeRobot state/action/timestamp columns plus
  versioned object-pose annotations;
- metric cross-episode object-frame registration from RGB-D correspondences,
  with a checked binding from a live tracker's frame to the demo frame;
- a matcher-neutral RGB-D pose backend that fits visible matched points from
  any number of annotated demo views and reports loss when depth or geometry
  cannot support a pose;
- kinematics-neutral short IK chunks and atomic future-queue replacement;
- a camera-independent motor read/dispatch tick through `Robot`;
- the separately configured SO-107 gravity-compensated robot controller used
  in the sealed rig measurements, with its measured gains and URDF inertias;
- a session state machine that consumes asynchronous evidence before each tick
  and gates semantic actions on current milestone verdicts;
- standard LeRobot episode frames, with a `flash/v1` trace joined by episode
  and frame index. The trace records intended and robot-accepted actions.

The annotation points to a source LeRobot dataset episode. Dense transform
samples are derived geometry; the original RGB-D, robot observations and
actions remain in the regular episode. Each interval identifies its object
role and reference-frame version. A held-to-target interval also declares the
held object's reference-frame version. Missing versions are shown in skill
inspection and rejected by the joint planner.

The next integration must supply a pixel matcher that finds correspondences
across the demonstration and live view (or accepts human-selected points),
and publish registered pose snapshots asynchronously. The RGB-D backend fits
a transform once correspondences are given; it does not discover them.
The planner worker, robot-specific kinematics, milestone verifiers and
recorder then need wiring into a hardware launch path.
A physical rollout cannot be claimed from the core unit tests or
skill-inspection UI alone.
