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

"""Gravity feedforward for OpenArm follower arms (MIT torque slot).

The stock MIT command sends (kp, kd, pos, 0, 0) — a pure PD loop with zero
feedforward, so the steady-state error under load is tau_g/kp. This module
computes the gravity torque tau_g(q) at the measured pose via MuJoCo
inverse dynamics (mj_forward with zero velocity/acceleration; qfrc_bias at
zero velocity is exactly tau_g) and feeds it into the MIT torque slot.

Best practices applied (standard PD+FF):

- gain <= 1.0: under-compensation is benign, over-compensation pushes
  past the target. Validated value: 0.9.
- fade-in: the feedforward ramps up over `fade_secs` when enabled, so
  there is no torque step on enable.
- low-pass: tau_g is first-order filtered (`lpf_hz`) to smooth encoder
  noise and model transients.
- clamped: per-joint |tff| <= torque_frac * joint actuatorfrcrange.
- sanity: non-finite torques or model/state garbage -> tff = 0.

Units and conventions: joint angles are in radians, zero = arms hanging
straight down (the OpenArm calibration zero), which matches the
openarm_bimanual.xml qpos convention.

Requires the `openarm-ff` extra (mujoco + openarm-mujoco model files):
`pip install 'lerobot[openarm-ff]'`.
"""

import logging
import time
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from lerobot.utils.import_utils import (
    _mujoco_available,
    _openarm_mujoco_available,
    require_package,
)

logger = logging.getLogger(__name__)

# dof addresses of J1..J7 in the bimanual model (gripper dofs 7,8,16,17
# excluded — the gripper runs POS_FORCE, not MIT).
DOFS = {"left": list(range(0, 7)), "right": list(range(9, 16))}


def default_bimanual_xml() -> str:
    """Resolve the openarm_bimanual.xml shipped by the `openarm-mujoco` package."""
    require_package("openarm-mujoco", extra="openarm-ff", import_name="openarm_mujoco")
    import openarm_mujoco.v2 as model_module

    model_path = Path(model_module.openarm_bimanual_xml()).resolve()
    if model_path.is_file():
        return str(model_path)

    # ``pip/uv --target`` keeps package data beside the imported distribution,
    # while openarm-mujoco resolves its default through the interpreter prefix.
    # Use only that distribution-local share directory as a fallback.
    distribution_root = Path(model_module.__file__).resolve().parents[2]
    target_local = distribution_root / "share" / "openarm_mujoco" / "v2" / "openarm_bimanual.xml"
    if target_local.is_file():
        return str(target_local)
    raise FileNotFoundError(f"openarm-mujoco returned a missing model path: {model_path}")


class GravityFF:
    """Computes, fades in, low-passes and clamps gravity torque for one arm."""

    def __init__(
        self,
        side: str,
        xml: str | None = None,
        gain: float = 0.9,
        torque_frac: float = 0.5,
        lpf_hz: float = 20.0,
        fade_secs: float = 1.0,
    ) -> None:
        require_package("mujoco", extra="openarm-ff")
        import mujoco

        if side not in DOFS:
            raise ValueError(f"side must be one of {list(DOFS)}, got {side!r}")
        if not 0.0 <= gain <= 1.0:
            raise ValueError(f"gain must be in [0, 1], got {gain}")
        if not 0.0 < torque_frac <= 1.0:
            raise ValueError(f"torque_frac must be in (0, 1], got {torque_frac}")
        self.side = side
        self.gain = gain
        self.torque_frac = torque_frac
        self.lpf_hz = lpf_hz
        self.fade_secs = fade_secs

        self._model = mujoco.MjModel.from_xml_path(xml or default_bimanual_xml())
        self._data = mujoco.MjData(self._model)
        self._dofs = DOFS[side]
        self._qposadr = [int(self._model.jnt_qposadr[self._model.dof_jntid[d]]) for d in self._dofs]
        self._limits = torque_frac * np.array(
            [abs(float(self._model.jnt_actfrcrange[self._model.dof_jntid[d]][1])) for d in self._dofs]
        )

        self._tau_lp: np.ndarray | None = None  # low-pass state
        self._t_enable: float | None = None  # fade-in start
        self._t_prev: float | None = None  # last update (LPF dt)
        self._last_warn = 0.0

    @staticmethod
    def is_available() -> bool:
        """Whether the optional mujoco/model dependencies are installed."""
        return _mujoco_available and _openarm_mujoco_available

    def raw_tau(self, q7: ArrayLike) -> np.ndarray:
        """Unfiltered gravity torque tau_g(q) at the given joint angles [Nm]."""
        import mujoco

        q7 = np.asarray(q7, dtype=np.float64)
        d = self._data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        for adr, q in zip(self._qposadr, q7, strict=True):
            d.qpos[adr] = q
        # mj_forward runs the full pipeline; qfrc_bias at zero velocity is tau_g.
        mujoco.mj_forward(self._model, d)
        return np.array([d.qfrc_bias[dof] for dof in self._dofs])

    def torque(self, q7: ArrayLike, now: float | None = None) -> np.ndarray:
        """Feedforward torque [Nm] for the MIT torque slot: faded, filtered, clamped."""
        now = time.monotonic() if now is None else now
        if self._t_enable is None:
            self._t_enable = now
            self._t_prev = now

        tau = self.raw_tau(q7)
        if not np.all(np.isfinite(tau)):
            if now - self._last_warn > 2.0:
                self._last_warn = now
                logger.warning(f"[gravity_ff:{self.side}] non-finite tau_g, sending zero")
            return np.zeros(7)

        # Low-pass (first-order IIR). dt clamped to [1ms, 0.5s].
        if self._tau_lp is None:
            self._tau_lp = tau
        else:
            dt = min(0.5, max(1e-3, now - (self._t_prev or now)))
            rc = 1.0 / (2.0 * np.pi * self.lpf_hz)
            alpha = dt / (dt + rc)
            self._tau_lp = self._tau_lp + alpha * (tau - self._tau_lp)
        self._t_prev = now

        # Fade-in after enable to avoid a torque step.
        fade = 1.0 if self.fade_secs <= 0 else min(1.0, (now - self._t_enable) / self.fade_secs)
        return np.clip(self.gain * fade * self._tau_lp, -self._limits, self._limits)
