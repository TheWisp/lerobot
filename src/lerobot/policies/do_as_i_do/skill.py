# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""A pick distilled from ONE human demo: the grasp pose in the OBJECT's frame.

Storing the grasp relative to the object (not the base) is what makes the single demo
generalize and run closed-loop: at run time we localize the object live and recover the
grasp pose as ``object_pose @ grasp_in_object``, so the target tracks the object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PickSkill:
    """Grasp pose in the object frame + the open-loop schedule around it (heights, gripper)."""

    grasp_in_object: np.ndarray  # 4x4 robot-EE grasp pose, expressed in the object frame
    pregrasp_h: float = 0.08  # metres above the grasp for the pre-grasp and the lift
    lift_h: float = 0.12
    gripper_open: float = 100.0  # RANGE_0_100 (100 = open)
    gripper_closed: float = 12.0

    @staticmethod
    def from_demo(object_pose: np.ndarray, ee_grasp: np.ndarray, **kw) -> PickSkill:
        """Build from the demo: ``object_pose`` and the robot-EE grasp pose ``ee_grasp``
        (= ``calibrate`` ∘ human-hand pose), both 4x4 in the base frame at the grasp."""
        return PickSkill(grasp_in_object=np.linalg.inv(object_pose) @ ee_grasp, **kw)

    def grasp_in_base(self, object_pose: np.ndarray) -> np.ndarray:
        """The grasp EE pose in the base frame given the object's live pose."""
        return object_pose @ self.grasp_in_object
