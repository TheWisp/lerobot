#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""A teleoperator that commands nothing.

Every other teleoperator in this package needs something: a leader arm on a
bus, a headset on the network, an input listener that needs a display. That
leaves no way to run a real robot's own loop — its cameras, its control
rate, its processors, its observation tap — without also moving it, and no
way at all on a machine with no display.

This one connects to nothing and returns an empty action on every cycle.
The action processors are identities and a follower builds its motor
commands from the keys the action has, so an empty action reaches the bus
as no command at all: the arm holds under torque and the loop runs. That is
what makes it safe to point at real hardware, and it is pinned by tests on
both sides — the action stays empty through the processors, and the
follower writes nothing when it arrives.

It is a testbench, not a workflow: a run driven by it records observations
of a robot nobody is moving.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.types import RobotAction

from ..teleoperator import Teleoperator
from .configuration_no_input import NoInputTeleopConfig

logger = logging.getLogger(__name__)


class NoInputTeleop(Teleoperator):
    """A teleoperator with no device behind it, whose every action is empty."""

    config_class = NoInputTeleopConfig
    name = "no_input"

    def __init__(self, config: NoInputTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # nothing to calibrate, accepted for the interface
        self._connected = True
        logger.info("%s connected: it will command nothing", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        assert self._connected, "NoInputTeleop.get_action requires connect()"
        return {}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Accepted and dropped, so a robot that sends feedback still runs."""

    def disconnect(self) -> None:
        self._connected = False
