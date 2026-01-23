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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE


def _load_to_percent(raw: int) -> float:
    """Convert Feetech Present_Load to signed percentage.

    The raw value uses sign-magnitude encoding:
    - Bits 0-9 (0x03FF): magnitude (0-1000)
    - Bit 10 (0x0400): direction (0=CW/positive, 1=CCW/negative)
    - Result is percentage of max torque (-100 to +100)
    """
    magnitude = raw & 0x03FF
    is_negative = (raw >> 10) & 1
    return (magnitude * -1 if is_negative else magnitude) / 10.0


@dataclass
@ProcessorStepRegistry.register("bi_gripper_load_processor")
class BiGripperLoadProcessorStep(ObservationProcessorStep):
    """
    Reads gripper load (torque) from a bimanual robot and appends them to the observation state.

    This step queries both left and right arm hardware interfaces to get the present load
    for the gripper motors only and concatenates this information to the existing state vector.
    Load values are converted from raw sign-magnitude encoding to percentage (-100 to +100).

    Attributes:
        robot: An instance of a bimanual robot (e.g., BiSO107Follower) that provides access
               to left_arm and right_arm hardware buses.
    """

    robot: Any | None = None

    def observation(self, observation: dict) -> dict:
        """
        Fetches gripper load from both arms and adds them to the observation state.

        Args:
            observation: The input observation dictionary.

        Returns:
            A new observation dictionary with the `observation.state` tensor
            extended to include gripper load from both arms.

        Raises:
            ValueError: If the `robot` attribute has not been set.
        """
        if self.robot is None:
            raise ValueError("Robot is not set")

        try:
            left_load_dict = self.robot.left_arm.bus.sync_read("Present_Load")
            right_load_dict = self.robot.right_arm.bus.sync_read("Present_Load")
        except Exception as e:
            import logging

            logging.warning(f"Failed to read motor load: {e}")
            return observation

        # Get gripper loads only, converted to signed percentages (-100 to +100)
        left_gripper_load = _load_to_percent(left_load_dict.get("gripper", 0))
        right_gripper_load = _load_to_percent(right_load_dict.get("gripper", 0))
        gripper_loads = torch.tensor(
            [left_gripper_load, right_gripper_load],
            dtype=torch.float32,
        ).unsqueeze(0)

        new_observation = dict(observation)

        # Add gripper load values for rerun visualization
        new_observation["left_gripper.load"] = left_gripper_load
        new_observation["right_gripper.load"] = right_gripper_load

        # Extend the state tensor for policy input
        current_state = observation.get(OBS_STATE)
        if current_state is not None:
            extended_state = torch.cat([current_state, gripper_loads], dim=-1)
            new_observation[OBS_STATE] = extended_state

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the `observation.state` feature to reflect the added gripper load.

        This method increases the size of the first dimension of the `observation.state`
        shape by 2 (left and right gripper loads).

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION] and self.robot is not None:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # Add 2 dimensions for left and right gripper loads
            new_shape = (original_feature.shape[0] + 2,) + original_feature.shape[1:]
            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
        return features
