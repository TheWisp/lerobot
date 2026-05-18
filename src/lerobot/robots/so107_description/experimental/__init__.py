"""Superseded SO-107 prototype subpackages, kept for reference.

The production paths now live in upstream LeRobot:

- ``teleop_quest/`` predates :class:`lerobot.teleoperators.quest_vr.QuestVRTeleop`
  and :class:`BimanualQuestVRTeleop`. Use those instead of the standalone
  WebSocket receivers here.
- ``learned_ik/`` was a small NN-IK experiment; the production stack uses
  :class:`lerobot.model.pink_kinematics.PinkKinematics` (QP-based, with
  posture regularization) via the
  :class:`lerobot.robots.so_follower.pink_kinematic_processor.PinkInverseKinematicsEEToJoints`
  ProcessorStep.
- ``trajectory_replay.py`` is a one-off replay script not part of any
  shipped pipeline.

These modules are not imported by anything in the production path, but
they still work standalone (``python -m`` invocation strings in their
docstrings have been updated to reflect the new path).
"""
