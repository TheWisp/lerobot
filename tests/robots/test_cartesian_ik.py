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

import logging
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
    pytest.importorskip(
        "lerobot.teleoperators.quest_vr.arm_controller",
        reason="quest_vr layer not in this PR; the test re-enables when Quest VR lands",
    )
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
    # __init__ is bypassed here, so pre-build the kinematics it normally
    # would (see BiSO107Follower.__init__).
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT

    robot._ik_kinematics = {
        "left": make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT),
        "right": make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT),
    }

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


# ── Factory plumbing for the IK config knobs ──────────────────────────────


def test_make_so107_arm_kinematics_passes_tuning_kwargs_to_pink():
    """The SO-107 factory forwards the config-surfaced tuning knobs
    (``posture_cost`` / ``max_iters``) to ``PinkKinematics``. Guards the
    plumbing that lets a robot profile fine-tune the IK feel from the
    GUI without changing code.
    """
    from unittest.mock import patch

    from lerobot.robots.so107_description import cartesian_ik

    with patch("lerobot.model.pink_kinematics.PinkKinematics") as mock_pk:
        cartesian_ik.make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT, posture_cost=0.42, max_iters=17)
    kwargs = mock_pk.call_args.kwargs
    assert kwargs["posture_cost"] == pytest.approx(0.42)
    assert kwargs["max_iters"] == 17


def test_make_so107_arm_kinematics_defaults_match_so107_factory_choice():
    """The defaults are visible config (the robot config's
    ``ik_posture_cost`` / ``ik_max_iters`` defaults mirror these), so if
    you change the factory default you must also change those config
    defaults — otherwise the GUI shows a value different from the actual
    behavior. Pins both sides of the seam.
    """
    from unittest.mock import patch

    from lerobot.robots.bi_so107_follower.config_bi_so107_follower import (
        BiSO107FollowerConfig,
    )
    from lerobot.robots.so107_description import cartesian_ik

    with patch("lerobot.model.pink_kinematics.PinkKinematics") as mock_pk:
        cartesian_ik.make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT)
    factory_defaults = mock_pk.call_args.kwargs

    config_default = BiSO107FollowerConfig(left_arm_port="/dev/ttyACM0", right_arm_port="/dev/ttyACM1")
    assert factory_defaults["posture_cost"] == config_default.ik_posture_cost
    assert factory_defaults["max_iters"] == config_default.ik_max_iters


# ── Real IK round-trip (needs pin-pink) ───────────────────────────────────


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_real_ik_controller_traces_a_position_offset():
    """Hold a +3 cm EE target; the IK joints must FK back to that offset.

    The Pink IK is an iterative QP solver seeded from the previous tick's
    solution — exactly how a Quest streams continuous motion. So the
    controller is polled repeatedly with the (held) target and converges,
    rather than expected to land a 3 cm jump in a single call.
    """
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics

    kin = make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT)
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


# ── Glitch / dropout safety ───────────────────────────────────────────────


def test_implausible_ik_jump_is_held():
    """An IK solve that swings the joints wildly is rejected, not slewed into."""

    class _WildIK:
        def forward_kinematics(self, q):
            t = np.eye(4)
            t[:3, 3] = np.asarray(q, dtype=float)[:3]
            return t

        def inverse_kinematics(self, seed, target):
            # A solver gone wrong: every joint flung 999 deg off the seed.
            return np.asarray(seed, dtype=float) + 999.0

    q_init = np.array([0.0, -0.2, 0.15, 0.0, 0.0, 0.0, 50.0])
    ctrl = CartesianIKController(
        kinematics=_WildIK(),
        motor_names=list(MOTOR_NAMES),
        q_init=q_init,
        workspace_min=_WS_MIN,
        workspace_max=_WS_MAX,
    )
    out = ctrl(_ee_action(enabled=1.0, target_x=0.01))
    # The wild solution is rejected; the arm holds at its seed configuration.
    for i, motor in enumerate(MOTOR_NAMES):
        if motor != "gripper":
            assert out[f"{motor}.pos"] == pytest.approx(q_init[i])


def test_quest_controller_reanchors_after_tracking_dropout():
    """A controller moved while untracked must not emit a phantom target delta.

    The bimanual-teleop bug: a tracking dropout left the engage snapshot
    stale, so motion during the dropout slewed the arm on re-acquire.
    ``on_tracking_lost`` must drop the engage state so re-acquire re-anchors.
    """
    pytest.importorskip(
        "lerobot.teleoperators.quest_vr.arm_controller",
        reason="quest_vr layer not in this PR; the test re-enables when Quest VR lands",
    )
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

    def _pose(xyz, clutch):
        return {"pos": list(xyz), "rot": [0.0, 0.0, 0.0, 1.0], "buttons": [0.0, clutch]}

    # Engage at the origin, clutch held.
    arm.process_pose(_pose((0.0, 0.0, 0.0), clutch=1.0))

    # Tracking drops out, then re-acquires with the hand moved 0.9 m away —
    # the clutch was held the whole time.
    lost = arm.on_tracking_lost()
    assert lost["enabled"] == 0.0  # a dropout disengages the arm

    reacquired = arm.process_pose(_pose((0.9, 0.0, 0.0), clutch=1.0))

    # The 0.9 m of motion-while-untracked must NOT become a teleop delta:
    # re-acquire re-anchors the engage snapshot, so target_* is back near zero.
    assert abs(reacquired["target_x"]) < 0.05
    assert abs(reacquired["target_y"]) < 0.05
    assert abs(reacquired["target_z"]) < 0.05


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_full_stack_no_slew_on_tracking_dropout():
    """The dropout fix must prevent joint slew through the full stack.

    Wires QuestArmController + make_bimanual_ik_transform + CartesianIKController
    with the real Pink IK. Engages the right arm, simulates a tracking dropout
    while the hand "moves" 0.5 m, then re-acquires with the clutch still held.
    The right arm must hold near its baseline pose (no phantom slew through
    the IK). A regression of the re-anchor logic anywhere in the stack fails
    this test.
    """
    from lerobot.robots.so107_description.cartesian_ik import (
        make_bimanual_ik_transform,
        make_so107_arm_ik_controller,
        make_so107_arm_kinematics,
    )
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT

    pytest.importorskip(
        "lerobot.teleoperators.quest_vr.arm_controller",
        reason="quest_vr layer not in this PR; the test re-enables when Quest VR lands",
    )
    from lerobot.teleoperators.quest_vr.arm_controller import QuestArmController

    q_init = np.array([0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0])
    left_kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    right_kin = make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT)
    transform = make_bimanual_ik_transform(
        make_so107_arm_ik_controller(left_kin, q_init),
        make_so107_arm_ik_controller(right_kin, q_init),
    )

    def _arm(prefix: str) -> QuestArmController:
        # max_pos_step_m_per_tick=0 disables the per-frame glitch cap so the
        # dropout-re-anchor logic is the only thing standing between a
        # phantom delta and the IK — isolating what this test guards.
        return QuestArmController(
            clutch_button_index=1,
            gripper_button_index=0,
            position_scale=1.0,
            max_rot_step_rad_per_tick=3.14,
            gripper_open_motor=50.0,
            gripper_closed_motor=90.0,
            max_pos_step_m_per_tick=0.0,
            key_prefix=prefix,
        )

    left, right = _arm("left_"), _arm("right_")

    def _pose(xyz, clutch):
        return {"pos": list(xyz), "rot": [0.0, 0.0, 0.0, 1.0], "buttons": [0.0, clutch]}

    def _tick(left_p, right_p):
        action: dict = {}
        action.update(left.process_pose(left_p) if left_p is not None else left.on_tracking_lost())
        action.update(right.process_pose(right_p) if right_p is not None else right.on_tracking_lost())
        return transform(action)

    # Engage the right arm at the origin; let the IK settle for a few ticks.
    out: dict = {}
    for _ in range(5):
        out = _tick(None, _pose((0.0, 0.0, 0.0), clutch=1.0))
    baseline = np.array([out[f"right_{m}.pos"] for m in MOTOR_NAMES])

    # Both arms drop tracking, then right re-acquires 0.5 m away with the
    # clutch still held — the old bug trigger.
    _tick(None, None)
    for _ in range(10):
        out = _tick(None, _pose((0.5, 0.0, 0.0), clutch=1.0))
    after = np.array([out[f"right_{m}.pos"] for m in MOTOR_NAMES])

    # With the fix the right arm holds within IK drift; without it the EE
    # would have walked toward (baseline + 0.5 m) under the per-tick cap and
    # joints would have slewed dozens of degrees.
    max_delta = float(np.max(np.abs(after - baseline)))
    assert max_delta < 5.0, f"right arm slewed {max_delta:.1f} deg — re-anchor regressed"


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_joint_step_guard_does_not_false_trigger_on_smooth_motion(caplog):
    """The _MAX_JOINT_STEP_DEG backstop must not hold ticks during normal teleop.

    A too-tight threshold would silently throttle the arm. This test drives
    a smooth 5 cm EE ramp through the real IK and asserts the guard does
    not log a single "implausible IK jump" — i.e. it never engages on
    motion that looks like normal teleop.
    """
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics

    kin = make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT)
    q_init = np.array([0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0])
    ctrl = CartesianIKController(
        kinematics=kin,
        motor_names=list(MOTOR_NAMES),
        q_init=q_init,
        workspace_min=(-10.0, -10.0, -10.0),
        workspace_max=(10.0, 10.0, 10.0),
        max_ee_step_m=10.0,
    )

    with caplog.at_level(logging.WARNING, logger="lerobot.robots.so107_description.cartesian_ik"):
        # Smooth ramp: target_x climbs from 0 to 5 cm over 100 ticks.
        for i in range(100):
            target_x = 0.05 * (i + 1) / 100
            ctrl(_ee_action(enabled=1.0, target_x=target_x, gripper=0.0))

    assert "implausible IK jump" not in caplog.text, (
        f"joint-step guard false-triggered during smooth motion: {caplog.text}"
    )


# ── Trajectory shapes (mirrors PR #9 test_pink_ik_trajectory) ────────────


def _shape_deltas(ref_pose: np.ndarray, shape: str, size_m: float, n: int) -> list[np.ndarray]:
    """Generate ``n`` world-frame EE deltas tracing a planar shape.

    Mirrors PR #9's ``_shape_targets`` (test_pink_ik_trajectory): the
    shape lives in the arm's *natural* plane — ``inward`` toward the base
    in the horizontal plane, ``perp`` perpendicular — so the IK can move
    the EE smoothly around it. The first delta is zero; the path returns
    to zero. ``size_m`` is the radius for ``circle`` and the side length
    for ``square``.
    """
    p = ref_pose[:3, 3]
    flat = np.array([p[0], p[1], 0.0])
    norm = float(np.linalg.norm(flat))
    inward = -flat / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])
    perp = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    perp /= np.linalg.norm(perp)

    positions: list[np.ndarray] = []
    if shape == "circle":
        r = size_m
        center = p + r * inward  # The seed EE sits on the circle at theta=0.
        for i in range(n):
            theta = 2 * np.pi * i / n
            positions.append(center + r * (np.cos(theta) * -inward + np.sin(theta) * perp))
    elif shape == "square":
        s = size_m
        corners = [p, p + s * inward, p + s * inward + s * perp, p + s * perp]
        per_edge = n // 4
        for e in range(4):
            a, b = corners[e], corners[(e + 1) % 4]
            for k in range(per_edge):
                positions.append(a + (b - a) * (k / per_edge))
    else:
        raise ValueError(f"unknown shape {shape!r}")
    return [pos - p for pos in positions]


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
@pytest.mark.parametrize(
    ("shape", "size_m", "pos_tol_mm", "rot_tol_deg"),
    [
        # Position tolerances reflect PinkKinematics's per-call QP convergence
        # on a moving target — lag scales with target speed, so the 60 mm
        # circle (2x faster per tick than the 30 mm one) needs more headroom.
        # Rotation is tight: with the L6 anchor + TIP_OFFSET fix the IK is
        # now S7-independent, so orientation tracks to the IK floor — a
        # regression in the structural fix would re-introduce multi-degree
        # rot drift on 2D paths. See quest_vr/TODO.md.
        ("circle", 0.030, 2.0, 0.2),
        ("circle", 0.060, 8.0, 0.2),
        ("square", 0.050, 2.0, 0.2),
    ],
)
def test_bimanual_tracks_shape(shape, size_m, pos_tol_mm, rot_tol_deg):
    """Both arms trace a PR #9 shape through the full bimanual stack.

    Mirrors test_pink_ik_trajectory.py — 30 mm / 60 mm circles + 50 mm
    square — but driven through ``CartesianIKController`` + the bimanual
    transform, so a regression in any layer (controller, transform,
    joint-map) breaks the EE trace. Both arms get the same shape; the
    orientation is held at the latched reference (``target_w=0``).

    Tolerances reflect ``PinkKinematics`` per-call QP convergence on a
    moving target — bigger circle ⇒ faster waypoint speed ⇒ more lag;
    square corners spike the lag at direction changes. The IK-tuning
    follow-up in quest_vr/TODO.md would let these tolerances tighten.
    """
    from lerobot.robots.so107_description.cartesian_ik import (
        make_bimanual_ik_transform,
        make_so107_arm_kinematics,
    )
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT

    # PR #9's seed is in URDF space; with the per-arm motor->URDF alignment
    # applied here, the same motor values land at very different URDF poses
    # per arm (the alignment offsets differ). Reverse the alignment so each
    # arm starts at the same well-conditioned URDF pose PR #9 used — far
    # from the URDF joint limits (±π) so the IK can manoeuvre freely and
    # the test measures the bimanual stack, not joint-limit struggle.
    urdf_seed = np.array([0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0])

    def _motor_seed(alignment) -> np.ndarray:
        sign = np.array([alignment[m].sign for m in MOTOR_NAMES])
        offset = np.array([alignment[m].offset_deg for m in MOTOR_NAMES])
        return (urdf_seed - offset) / sign

    q_left = _motor_seed(LEFT_ARM_ALIGNMENT)
    q_right = _motor_seed(RIGHT_ARM_ALIGNMENT)
    # The gripper slot has different motor values per arm (the alignment
    # offsets differ); using the same gripper_pos for both would force the
    # IK to move the gripper joint every tick, perturbing the EE frame.
    gripper_idx = MOTOR_NAMES.index("gripper")
    grip_left = float(q_left[gripper_idx])
    grip_right = float(q_right[gripper_idx])
    left_kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    right_kin = make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT)

    # Wide bounds + cap so the test exercises tracking, not the clip / cap.
    def _ctrl(kin, q_init: np.ndarray):
        return CartesianIKController(
            kinematics=kin,
            motor_names=list(MOTOR_NAMES),
            q_init=q_init,
            workspace_min=(-10.0, -10.0, -10.0),
            workspace_max=(10.0, 10.0, 10.0),
            max_ee_step_m=10.0,
        )

    transform = make_bimanual_ik_transform(_ctrl(left_kin, q_left), _ctrl(right_kin, q_right))
    ref_left = left_kin.forward_kinematics(q_left)
    ref_right = right_kin.forward_kinematics(q_right)

    deltas_left = _shape_deltas(ref_left, shape, size_m, n=256)
    deltas_right = _shape_deltas(ref_right, shape, size_m, n=256)
    pos_err_left: list[float] = []
    pos_err_right: list[float] = []
    rot_err_left: list[float] = []
    rot_err_right: list[float] = []

    for dl, dr in zip(deltas_left, deltas_right, strict=True):
        action = {
            "left_enabled": 1.0,
            "left_target_x": float(dl[0]),
            "left_target_y": float(dl[1]),
            "left_target_z": float(dl[2]),
            "left_target_wx": 0.0,
            "left_target_wy": 0.0,
            "left_target_wz": 0.0,
            "left_gripper_pos": grip_left,
            "right_enabled": 1.0,
            "right_target_x": float(dr[0]),
            "right_target_y": float(dr[1]),
            "right_target_z": float(dr[2]),
            "right_target_wx": 0.0,
            "right_target_wy": 0.0,
            "right_target_wz": 0.0,
            "right_gripper_pos": grip_right,
        }
        out = transform(action)
        q_left_out = np.array([out[f"left_{m}.pos"] for m in MOTOR_NAMES])
        q_right_out = np.array([out[f"right_{m}.pos"] for m in MOTOR_NAMES])
        fk_left = left_kin.forward_kinematics(q_left_out)
        fk_right = right_kin.forward_kinematics(q_right_out)

        exp_left = ref_left[:3, 3] + dl
        exp_right = ref_right[:3, 3] + dr
        pos_err_left.append(float(np.linalg.norm(fk_left[:3, 3] - exp_left)))
        pos_err_right.append(float(np.linalg.norm(fk_right[:3, 3] - exp_right)))

        # Orientation held at latched reference: error = angle of (ref.T @ fk).
        for ref, fk, errs in (
            (ref_left, fk_left, rot_err_left),
            (ref_right, fk_right, rot_err_right),
        ):
            r_err = ref[:3, :3].T @ fk[:3, :3]
            cos_a = max(-1.0, min(1.0, (float(np.trace(r_err)) - 1.0) * 0.5))
            errs.append(float(np.degrees(np.arccos(cos_a))))

    # Warmup: let the iterative IK converge from the q_init seed.
    warmup = 20
    pos_tol_m = pos_tol_mm * 1e-3
    assert max(pos_err_left[warmup:]) < pos_tol_m, (
        f"left position drift {max(pos_err_left[warmup:]) * 1000:.2f} mm "
        f"> {pos_tol_mm} mm on {shape} {size_m * 1000:.0f}mm"
    )
    assert max(pos_err_right[warmup:]) < pos_tol_m, (
        f"right position drift {max(pos_err_right[warmup:]) * 1000:.2f} mm "
        f"> {pos_tol_mm} mm on {shape} {size_m * 1000:.0f}mm"
    )
    assert max(rot_err_left[warmup:]) < rot_tol_deg, (
        f"left rotation drift {max(rot_err_left[warmup:]):.2f} deg "
        f"> {rot_tol_deg} deg on {shape} {size_m * 1000:.0f}mm"
    )
    assert max(rot_err_right[warmup:]) < rot_tol_deg, (
        f"right rotation drift {max(rot_err_right[warmup:]):.2f} deg "
        f"> {rot_tol_deg} deg on {shape} {size_m * 1000:.0f}mm"
    )
