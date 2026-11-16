# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from lerobot.robots.so107_description.gravity import GravityFeedForward
from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT

_SPEC = importlib.util.spec_from_file_location(
    "so107_gravity_sag", Path(__file__).resolve().parents[2] / "benchmarks" / "so107_gravity_sag.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
fit_compliance = _MODULE.fit_compliance


def test_tick_logs_the_goal_actually_sent_after_the_robot_clamps_it(tmp_path):
    class Robot:
        def send_action(self, action):
            return {key: value + 0.25 for key, value in action.items()}

        def get_observation(self):
            return {f"{motor}.pos": 0.0 for motor in _MODULE.MOTOR_NAMES}

    class Controller:
        is_holding = False

        def __call__(self, _target):
            return {f"{motor}.pos": 0.0 for motor in _MODULE.MOTOR_NAMES}

    run = _MODULE.Run(Robot(), Controller(), None, np.zeros(3), 0.0, tmp_path)
    run.tick(np.zeros(3))
    assert run.rows[-1][-len(_MODULE.MOTOR_NAMES) :] == [0.25] * len(_MODULE.MOTOR_NAMES)


def test_held_leg_remembers_its_last_requested_point_for_the_return(tmp_path, monkeypatch):
    run = _MODULE.Run(None, None, None, np.zeros(3), 0.0, tmp_path)
    run.wait_for_arm = lambda _pos, _label: None
    run.tick = lambda _pos: (None, None, True)
    monkeypatch.setattr(_MODULE.time, "sleep", lambda _: None)

    target = np.array([0.02, 0.0, 0.0])
    assert run.goto(target, "held") is False
    assert 0.0 < run.pos_cmd[0] < target[0]


def test_station_fit_recovers_uncompensated_joint_compliance(tmp_path):
    ff = GravityFeedForward(1.0, LEFT_ARM_ALIGNMENT)
    poses = [
        (0, -45, 74, 2, -41, -12, 93),
        (15, -30, 60, 0, -25, 5, 70),
        (-10, -55, 90, 0, -35, 0, 45),
    ]
    alpha = 3.2
    stations = []
    for pose in poses:
        q_cmd = np.asarray(pose, dtype=float)
        torque = ff.gravity_torque_nm(q_cmd)
        q_now = q_cmd - alpha * torque
        stations.append({"q_cmd_deg": q_cmd.tolist(), "q_present_deg": q_now.tolist()})
    (tmp_path / "stations.json").write_text(json.dumps(stations))
    (tmp_path / "meta.json").write_text(json.dumps({"ff_alpha": 0, "ki": 0, "servo_i": 0, "servo_p": 32}))

    result = fit_compliance(tmp_path)
    assert result["alpha_deg_per_nm"] == pytest.approx(alpha, abs=1e-10)
    assert result["servo_p"] == 32
    assert result["stations"] == 3
    for motor in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        assert result[f"{motor}_alpha_deg_per_nm"] == pytest.approx(alpha, abs=1e-10)

    (tmp_path / "meta.json").write_text(json.dumps({"ff_alpha": alpha, "ki": 0, "servo_i": 0, "servo_p": 32}))
    with pytest.raises(ValueError, match="--ff-alpha 0"):
        fit_compliance(tmp_path)

    (tmp_path / "meta.json").write_text(json.dumps({"ff_alpha": 0, "ki": 0, "servo_i": 1, "servo_p": 32}))
    with pytest.raises(ValueError, match="--i 0"):
        fit_compliance(tmp_path)
