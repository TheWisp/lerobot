#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE


@dataclass
@ProcessorStepRegistry.register("bi_gripper_force_processor")
class BiGripperForceProcessorStep(ObservationProcessorStep):
    """
    Reads gripper force feedback from a bimanual robot and appends to observation state.

    This step queries both left and right arm hardware interfaces to get the present current
    for the gripper motors and concatenates this information to the existing state vector.
    Current values are filtered with an exponential moving average (EMA) for stability.

    Attributes:
        robot: An instance of a bimanual robot (e.g., BiSO107Follower) that provides access
               to left_arm and right_arm hardware buses.
        filter_alpha: EMA filter coefficient (0-1). Higher = more responsive but noisier.
                      Lower = smoother but more latency.
        deadband: Minimum change threshold to update output. Values below this are ignored,
                  giving flat lines when force is stable. Set to 0 to disable.
    """

    robot: Any | None = None
    filter_alpha: float = 0.3
    deadband: float = 3.0  # Minimum change to update output (in raw current units)

    # Internal filter state (excluded from init/repr)
    _accepted_left: float | None = field(default=None, init=False, repr=False)
    _accepted_right: float | None = field(default=None, init=False, repr=False)
    _output_left: float | None = field(default=None, init=False, repr=False)
    _output_right: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Reset the filter state."""
        self._accepted_left = None
        self._accepted_right = None
        self._output_left = None
        self._output_right = None

    def observation(self, observation: dict) -> dict:
        """
        Fetches gripper current from both arms and adds them to the observation state.

        Args:
            observation: The input observation dictionary.

        Returns:
            A new observation dictionary with the `observation.state` tensor
            extended to include filtered gripper current from both arms.

        Raises:
            ValueError: If the `robot` attribute has not been set.
        """
        if self.robot is None:
            raise ValueError("Robot is not set")

        try:
            left_current_dict = self.robot.left_arm.bus.sync_read("Present_Current")
            right_current_dict = self.robot.right_arm.bus.sync_read("Present_Current")
        except Exception as e:
            import logging

            logging.warning(f"Failed to read motor current: {e}")
            return observation

        # Get gripper currents only
        left_raw = float(left_current_dict.get("gripper", 0))
        right_raw = float(right_current_dict.get("gripper", 0))

        # Apply deadband first: only accept raw values that differ significantly
        # This prevents noise from entering the EMA filter
        if self._accepted_left is None:
            self._accepted_left = left_raw
            self._accepted_right = right_raw
            self._output_left = left_raw
            self._output_right = right_raw
        else:
            # Only update accepted value if change exceeds deadband
            if abs(left_raw - self._accepted_left) > self.deadband:
                self._accepted_left = left_raw
            if abs(right_raw - self._accepted_right) > self.deadband:
                self._accepted_right = right_raw

            # Apply EMA filter on deadbanded values
            self._output_left = self.filter_alpha * self._accepted_left + (1 - self.filter_alpha) * self._output_left
            self._output_right = self.filter_alpha * self._accepted_right + (1 - self.filter_alpha) * self._output_right

        gripper_force = torch.tensor(
            [self._output_left, self._output_right],
            dtype=torch.float32,
        ).unsqueeze(0)

        new_observation = dict(observation)

        # Add output values for visualization
        new_observation["left_gripper.force"] = self._output_left
        new_observation["right_gripper.force"] = self._output_right

        # Extend the state tensor for policy input (uses filtered values)
        current_state = observation.get(OBS_STATE)
        if current_state is not None:
            extended_state = torch.cat([current_state, gripper_force], dim=-1)
            new_observation[OBS_STATE] = extended_state

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the `observation.state` feature to reflect the added gripper force.

        This method increases the size of the first dimension of the `observation.state`
        shape by 2 (left and right gripper force values).

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION] and self.robot is not None:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # Add 2 dimensions for left and right gripper force
            new_shape = (original_feature.shape[0] + 2,) + original_feature.shape[1:]
            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
        return features
