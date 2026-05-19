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

"""Single concept the teleop loop driver consumes: ``MotorActionBinding``.

Returned by ``Robot.bind_teleop(teleop)``. Encapsulates two facts:

1. **How to get the motor-space action for THIS tick.** Hides whether it
   comes from polling a joint-space leader, running a Cartesian IK
   pipeline, reading an adapter's cached output, or some future scheme.
   The loop driver just calls ``binding.get_action(obs)``.

2. **Whether the robot's own controller drives motors.** When
   ``pull_path_active`` is true, the predictive controller is already
   polling its bound teleop at 200 Hz and writing to motors itself —
   the loop driver MUST NOT also call ``robot.send_action(joints)``
   with the dict, because doing so would set a redundant intent that
   the controller would ignore (the "secret no-op" that confused us).
   When false, the loop driver's ``robot.send_action(joints)`` is what
   actually writes to motors.

This abstraction lets the script remain embodiment-agnostic: it knows
nothing about Cartesian-vs-joint teleops, IK pipelines, adapter threads,
or chunk-aware controllers. The robot owns those decisions.

Action chunks (``ActionChunk``) are orthogonal to this binding: chunks
are sent via ``robot.send_action(chunk)`` regardless of
``pull_path_active`` because the predictive controller's exact-lookup
path uses them. Only the *per-tick dict* form of ``send_action`` is
gated on ``pull_path_active``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.teleoperators.teleoperator import Teleoperator


@dataclass(frozen=True)
class MotorActionBinding:
    """Result of ``Robot.bind_teleop(teleop)``. See module docstring."""

    get_action: Callable[[dict[str, Any]], dict[str, float]]
    """Returns motor-space joint dict for the current tick.

    Argument is the most recent observation (used by sources that need
    FK; ignored by sources that read a separately-cached value).
    Returns an empty dict if no action is available yet (e.g., adapter
    not yet warm). The caller treats empty as "hold last command."
    """

    pull_path_active: bool
    """True when the robot's controller drives motors itself.

    In that case the loop driver should NOT call ``send_action`` with
    the per-tick dict form (the controller would ignore it). It should
    still call ``send_action`` with action chunks if the teleop
    publishes them (chunk path is independent).
    """


def make_direct_binding(teleop: Teleoperator, *, pull_path_active: bool = False) -> MotorActionBinding:
    """The default binding: poll the teleop, return its output as-is.

    Use this when the teleop already emits motor-space joint dicts
    (joint-space leaders) and either:
      * ``pull_path_active=False``: a base ``Robot`` whose ``send_action``
        writes to motors directly each tick.
      * ``pull_path_active=True``: a predictive robot whose controller
        polls the teleop itself at 200 Hz; the script's ``send_action``
        is unnecessary for motor control.
    """

    def _get(_obs: dict[str, Any]) -> dict[str, float]:
        return teleop.get_action()

    return MotorActionBinding(get_action=_get, pull_path_active=pull_path_active)


def make_pipeline_binding(
    teleop: Teleoperator,
    pipeline: Any,
    *,
    pull_path_active: bool = False,
) -> MotorActionBinding:
    """Binding for teleops whose output needs conversion via a pipeline.

    Each tick: poll ``teleop.get_action()``, push through ``pipeline``
    along with the observation, return the resulting joint dict.

    Used for non-predictive Cartesian-teleop scenarios where the script's
    ``send_action`` is what drives motors. For predictive bimanual +
    Cartesian, the adapter handles conversion off the main loop instead
    (see :func:`make_adapter_binding`).
    """

    def _get(obs: dict[str, Any]) -> dict[str, float]:
        return pipeline((teleop.get_action(), obs))

    return MotorActionBinding(get_action=_get, pull_path_active=pull_path_active)


def make_adapter_binding(adapter: Any) -> MotorActionBinding:
    """Binding that reads from a Cartesian→joint IK adapter's cache.

    Used by predictive bimanual + Cartesian teleop: the adapter
    converts at 90 Hz in a background thread, the per-arm controllers
    poll the adapter's cached joints at 200 Hz to drive motors. The
    loop driver also reads from the same cache here so the recorded
    action matches what reached the motors.
    """

    def _get(_obs: dict[str, Any]) -> dict[str, float]:
        return adapter.get_full_joint_action() or {}

    return MotorActionBinding(get_action=_get, pull_path_active=True)
