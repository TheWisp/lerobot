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

"""Tests for the SO-107 Cartesian-IK teleop layer.

Most tests use a stub kinematics and need no optional dependency — they
exercise the controller state machine (hold / gripper / clip), the
motor<->URDF joint map, the bimanual split/merge, the teleop->IK key
contract, and ``BiSO107Follower.attach_teleop``'s branching. The single
real-IK test builds ``PinkKinematics`` from the vendored SO-107 URDF and
is skipped without ``pin-pink``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.so107_description.cartesian_ik import (
    CartesianIKController,
    JointMappedKinematics,
    make_bimanual_ik_transform,
)
from lerobot.robots.so107_description.joint_alignment import (
    MOTOR_NAMES,
    RIGHT_ARM_ALIGNMENT,
    JointAlignment,
)
from lerobot.utils.import_utils import _pin_pink_available

# The eight keys a Quest controller emits and a CartesianIKController reads.
_IK_INPUT_KEYS = {
    "enabled",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_pos",
}
_WS_MIN = (-0.20, -0.35, 0.03)
_WS_MAX = (0.25, 0.05, 0.36)


class _StubKinematics:
    """Deterministic FK/IK stand-in for controller-logic tests.

    FK drops the first three joint values into the translation; IK copies
    the target translation back into the first three joints of the seed.
    Enough to exercise the reference-latch / hold / clip logic without a
    real solver.
    """

    def forward_kinematics(self, q):
        t = np.eye(4)
        t[:3, 3] = np.asarray(q, dtype=float)[:3]
        return t

    def inverse_kinematics(self, seed, target):
        q = np.asarray(seed, dtype=float).copy()
        q[:3] = target[:3, 3]
        return q


def _ee_action(enabled=0.0, gripper=50.0, **deltas):
    """Build a full eight-key EE-delta action; unset deltas default to 0."""
    action = dict.fromkeys(_IK_INPUT_KEYS, 0.0)
    action["enabled"] = enabled
    action["gripper_pos"] = gripper
    action.update(deltas)
    return action


def _make_controller(q_init, *, max_ee_step_m=0.10):
    return CartesianIKController(
        kinematics=_StubKinematics(),
        motor_names=list(MOTOR_NAMES),
        q_init=q_init,
        workspace_min=_WS_MIN,
        workspace_max=_WS_MAX,
        max_ee_step_m=max_ee_step_m,
    )


# ── CartesianIKController state machine (stub kinematics) ─────────────────


def test_disabled_holds_arm_joints():
    q_init = np.array([0.0, -0.2, 0.15, 1.0, 2.0, 3.0, 50.0])
    ctrl = _make_controller(q_init)

    # A non-zero target while disabled must not move the arm.
    out = ctrl(_ee_action(enabled=0.0, target_x=0.1, target_y=0.1))

    for i, motor in enumerate(MOTOR_NAMES):
        if motor != "gripper":
            assert out[f"{motor}.pos"] == pytest.approx(q_init[i])


def test_gripper_passes_through_engaged_or_not():
    ctrl = _make_controller(np.array([0.0, -0.2, 0.15, 0.0, 0.0, 0.0, 50.0]))
    assert ctrl(_ee_action(enabled=0.0, gripper=77.0))["gripper.pos"] == pytest.approx(77.0)
    assert ctrl(_ee_action(enabled=1.0, gripper=12.0))["gripper.pos"] == pytest.approx(12.0)


def test_enabled_tracks_position_delta():
    # Rising edge latches reference = FK(q_init); the stub puts q_init[:3]
    # in the translation, so a +x delta should land in shoulder_pan.
    q_init = np.array([0.0, -0.2, 0.15, 0.0, 0.0, 0.0, 50.0])
    ctrl = _make_controller(q_init)

    out = ctrl(_ee_action(enabled=1.0, target_x=0.05))

    assert out["shoulder_pan.pos"] == pytest.approx(0.05)  # 0.0 + 0.05
    assert out["shoulder_lift.pos"] == pytest.approx(-0.2)  # unchanged
    assert out["elbow_flex.pos"] == pytest.approx(0.15)  # unchanged


def test_target_clipped_to_workspace_box():
    # max_ee_step_m huge so the per-tick cap never binds — isolates the clip.
    q_init = np.array([0.0, -0.2, 0.15, 0.0, 0.0, 0.0, 50.0])
    ctrl = _make_controller(q_init, max_ee_step_m=1e6)

    far = ctrl(_ee_action(enabled=1.0, target_x=5.0))
    assert far["shoulder_pan.pos"] == pytest.approx(_WS_MAX[0])  # clipped to +0.25

    # Reference stays latched, so the opposite target clips to the low bound.
    near = ctrl(_ee_action(enabled=1.0, target_x=-5.0))
    assert near["shoulder_pan.pos"] == pytest.approx(_WS_MIN[0])  # clipped to -0.20


def test_per_tick_jump_is_capped():
    q_init = np.array([0.0, -0.2, 0.15, 0.0, 0.0, 0.0, 50.0])
    ctrl = _make_controller(q_init, max_ee_step_m=0.10)

    # target_y would move the reference -0.15 m in one tick; cap is 0.10.
    out = ctrl(_ee_action(enabled=1.0, target_y=-0.15))
    moved = abs(out["shoulder_lift.pos"] - (-0.2))
    assert moved == pytest.approx(0.10, abs=1e-6)


# ── JointMappedKinematics (pure numpy) ────────────────────────────────────


def test_joint_map_round_trips():
    jmk = JointMappedKinematics(_StubKinematics(), list(MOTOR_NAMES), RIGHT_ARM_ALIGNMENT)
    q = np.array([10.0, 20.0, -30.0, 5.0, -15.0, 45.0, 60.0])
    np.testing.assert_allclose(jmk._urdf_to_motor(jmk._motor_to_urdf(q)), q, atol=1e-9)


def test_joint_map_applies_sign_and_offset():
    jmk = JointMappedKinematics(_StubKinematics(), list(MOTOR_NAMES), RIGHT_ARM_ALIGNMENT)
    q = np.zeros(len(MOTOR_NAMES))
    urdf = jmk._motor_to_urdf(q)
    # shoulder_lift: sign=+1, offset=-90 -> urdf angle at motor 0 is -90.
    assert urdf[MOTOR_NAMES.index("shoulder_lift")] == pytest.approx(-90.0)


def test_joint_map_identity_is_passthrough():
    identity = {m: JointAlignment(sign=1.0, offset_deg=0.0) for m in MOTOR_NAMES}
    jmk = JointMappedKinematics(_StubKinematics(), list(MOTOR_NAMES), identity)
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(jmk._motor_to_urdf(q), q, atol=1e-9)


# ── Bimanual split / merge ────────────────────────────────────────────────


def test_bimanual_transform_splits_and_merges():
    def _echo(action):
        # A fake per-arm controller: surface what it received so routing is visible.
        return {"shoulder_pan.pos": action["target_x"], "gripper.pos": action["gripper_pos"]}

    transform = make_bimanual_ik_transform(_echo, _echo)
    out = transform(
        {
            "left_target_x": 1.0,
            "left_gripper_pos": 5.0,
            "right_target_x": 2.0,
            "right_gripper_pos": 6.0,
        }
    )
    assert out["left_shoulder_pan.pos"] == 1.0
    assert out["right_shoulder_pan.pos"] == 2.0
    assert out["left_gripper.pos"] == 5.0
    assert out["right_gripper.pos"] == 6.0


# ── Teleop -> IK key contract ─────────────────────────────────────────────


def test_quest_arm_controller_output_drives_ik_controller():
    """The Quest teleop's action keys must be exactly what the IK controller reads.

    Guards the cross-layer seam: a rename on either side breaks this test
    instead of silently producing a KeyError at teleop time.
    """
    from lerobot.teleoperators.quest_vr.arm_controller import QuestArmController

    arm = QuestArmController(
        clutch_button_index=1,
        gripper_button_index=0,
        position_scale=1.0,
        max_rot_step_rad_per_tick=3.14,
        gripper_open_motor=50.0,
        gripper_closed_motor=90.0,
        key_prefix="",
    )
    # Idle and engaged poses must both carry exactly the IK input keys.
    assert set(arm.idle_action()) == _IK_INPUT_KEYS
    engaged = arm.process_pose({"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0, 1.0], "buttons": [1.0, 1.0]})
    assert set(engaged) == _IK_INPUT_KEYS

    # The teleop's actual output must flow through the controller without error.
    ctrl = _make_controller(np.array([0.0, -0.2, 0.15, 0.0, 0.0, 0.0, 50.0]))
    out = ctrl(engaged)
    assert set(out) == {f"{m}.pos" for m in MOTOR_NAMES}


# ── BiSO107Follower.attach_teleop branching ───────────────────────────────


class _FakeTeleop:
    def __init__(self, names: dict):
        self.action_features = {"names": names}
        self.installed = "<unset>"

    def set_action_transform(self, transform):
        self.installed = transform


def _bare_follower():
    """A BiSO107Follower with no __init__ run — enough to call attach_teleop."""
    from lerobot.robots.bi_so107_follower.bi_so107_follower import BiSO107Follower

    return BiSO107Follower.__new__(BiSO107Follower)


def test_attach_teleop_ignores_joint_teleop():
    robot = _bare_follower()
    leader = _FakeTeleop(names={f"{m}.pos": i for i, m in enumerate(MOTOR_NAMES)})
    robot.attach_teleop(leader)
    # A joint-space leader already emits joint dicts: no transform installed.
    assert leader.installed == "<unset>"


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_attach_teleop_installs_ik_for_cartesian_teleop():
    class _FakeArm:
        def __init__(self):
            self.bus = SimpleNamespace(is_connected=True, motors=dict.fromkeys(MOTOR_NAMES))

        def get_observation(self):
            seed = [0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0]
            return {f"{m}.pos": v for m, v in zip(MOTOR_NAMES, seed, strict=True)}

    robot = _bare_follower()
    robot.left_arm = _FakeArm()
    robot.right_arm = _FakeArm()
    robot.cameras = {}

    teleop = _FakeTeleop(names={"left_target_x": 0, "right_target_x": 1})
    robot.attach_teleop(teleop)

    assert callable(teleop.installed)
    bimanual = {
        f"{side}_{k}": (50.0 if k == "gripper_pos" else 0.0)
        for side in ("left", "right")
        for k in _IK_INPUT_KEYS
    }
    joints = teleop.installed(bimanual)
    assert set(joints) == {f"{side}_{m}.pos" for side in ("left", "right") for m in MOTOR_NAMES}


# ── Real IK round-trip (needs pin-pink) ───────────────────────────────────


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_real_ik_controller_traces_a_position_offset():
    """Hold a +3 cm EE target; the IK joints must FK back to that offset.

    The Pink IK is an iterative QP solver seeded from the previous tick's
    solution — exactly how a Quest streams continuous motion. So the
    controller is polled repeatedly with the (held) target and converges,
    rather than expected to land a 3 cm jump in a single call.
    """
    from lerobot.model.pink_kinematics import PinkKinematics
    from lerobot.robots.so107_description import get_urdf_path
    from lerobot.robots.so107_description.joint_alignment import URDF_JOINT_NAMES, URDF_TIP_FRAME

    inner = PinkKinematics(
        urdf_path=str(get_urdf_path()),
        target_frame_name=URDF_TIP_FRAME,
        joint_names=list(URDF_JOINT_NAMES),
    )
    kin = JointMappedKinematics(inner, list(MOTOR_NAMES), RIGHT_ARM_ALIGNMENT)
    q_init = np.array([0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0])
    # Wide bounds so neither the workspace clip nor the jump cap interferes.
    ctrl = CartesianIKController(
        kinematics=kin,
        motor_names=list(MOTOR_NAMES),
        q_init=q_init,
        workspace_min=(-10.0, -10.0, -10.0),
        workspace_max=(10.0, 10.0, 10.0),
        max_ee_step_m=10.0,
    )
    reference = kin.forward_kinematics(q_init)

    out = None
    for _ in range(120):
        out = ctrl(_ee_action(enabled=1.0, target_x=0.03, gripper=0.0))
    q_out = np.array([out[f"{m}.pos"] for m in MOTOR_NAMES])
    achieved = kin.forward_kinematics(q_out)

    np.testing.assert_allclose(achieved[:3, 3], reference[:3, 3] + np.array([0.03, 0.0, 0.0]), atol=3e-3)
