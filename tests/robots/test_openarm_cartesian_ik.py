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

"""Tests for the OpenArm 2.0 Cartesian-IK teleop layer.

Mirrors tests/robots/test_cartesian_ik.py (SO-107). Most tests use a stub
kinematics and need no optional dependency — they exercise the controller
wiring with OpenArm motor names, the bimanual split/merge, the teleop->IK
key contract, and ``BiOpenArmFollower.attach_teleop``'s branching. The
real-IK tests build ``PinkKinematics`` from the vendored per-arm OpenArm
URDFs and are skipped without ``pin-pink``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.openarm_description.cartesian_ik import (
    MOTOR_NAMES,
    OPENARM_WORKSPACE_MAX,
    OPENARM_WORKSPACE_MIN,
    build_openarm_bimanual_ik_transform,
    is_openarm_bimanual_cartesian_teleop,
    make_bimanual_ik_transform,
    make_openarm_arm_ik_controller,
)
from lerobot.robots.so107_description.cartesian_ik import CartesianIKController
from lerobot.utils.import_utils import _pin_pink_available

# The nine keys a Quest controller emits and a CartesianIKController reads.
_IK_INPUT_KEYS = {
    "enabled",
    "reset",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_pos",
}

# Bent-elbow "ready" poses (motor degrees, joint_1..joint_7 + gripper) —
# the arms-down zero is fully stretched and a poor IK seed. Left/right are
# mirror builds: joint_2's valid range is (-190, +10) on the left arm and
# (-10, +190) on the right, so the two seeds differ by sign on joint_2.
LEFT_READY_DEG = np.array([0.0, -45.0, 0.0, 90.0, 0.0, 30.0, 0.0, 0.0])
RIGHT_READY_DEG = np.array([0.0, +45.0, 0.0, 90.0, 0.0, 30.0, 0.0, 0.0])


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


def _ee_action(enabled=0.0, gripper=0.0, **deltas):
    """Build a full nine-key EE-delta action; unset deltas default to 0."""
    action = dict.fromkeys(_IK_INPUT_KEYS, 0.0)
    action["enabled"] = enabled
    action["gripper_pos"] = gripper
    action.update(deltas)
    return action


def _make_controller(q_init, *, max_ee_step_m=0.10):
    return make_openarm_arm_ik_controller(
        _StubKinematics(),
        q_init,
        workspace_min=OPENARM_WORKSPACE_MIN,
        workspace_max=OPENARM_WORKSPACE_MAX,
    )


# ── CartesianIKController wiring with OpenArm motor names (stub) ──────────


def test_disabled_holds_arm_joints():
    q_init = np.array([1.0, -20.0, 15.0, 45.0, 5.0, -10.0, 25.0, 0.0])
    ctrl = _make_controller(q_init)

    # A non-zero target while disabled must not move the arm.
    out = ctrl(_ee_action(enabled=0.0, target_x=0.1, target_y=0.1))

    for i, motor in enumerate(MOTOR_NAMES):
        if motor != "gripper":
            assert out[f"{motor}.pos"] == pytest.approx(q_init[i])


def test_gripper_passes_through_engaged_or_not():
    ctrl = _make_controller(np.array([0.0, -45.0, 0.0, 90.0, 0.0, 30.0, 0.0, 0.0]))
    # OpenArm gripper ranges: left 0 (open) .. +45 (closed); right 0 .. -45.
    assert ctrl(_ee_action(enabled=0.0, gripper=45.0))["gripper.pos"] == pytest.approx(45.0)
    assert ctrl(_ee_action(enabled=1.0, gripper=-45.0))["gripper.pos"] == pytest.approx(-45.0)


def test_enabled_tracks_position_delta():
    # Rising edge latches reference = FK(q_init); the stub puts q_init[:3]
    # in the translation, so a +x delta should land in joint_1. (Small
    # seed values so the stub "EE position" stays inside the workspace box.)
    q_init = np.array([0.0, -0.2, 0.15, 90.0, 0.0, 30.0, 0.0, 0.0])
    ctrl = _make_controller(q_init)

    out = ctrl(_ee_action(enabled=1.0, target_x=0.05))

    assert out["joint_1.pos"] == pytest.approx(0.05)  # 0.0 + 0.05
    assert out["joint_2.pos"] == pytest.approx(-0.2)  # unchanged
    assert out["joint_3.pos"] == pytest.approx(0.15)  # unchanged


def test_target_clipped_to_openarm_workspace_box():
    q_init = np.array([0.0, -0.2, 0.15, 90.0, 0.0, 30.0, 0.0, 0.0])
    ctrl = make_openarm_arm_ik_controller(_StubKinematics(), q_init)
    # Replace the per-tick cap so it never binds — isolates the clip.
    ctrl._max_step = 1e6

    far = ctrl(_ee_action(enabled=1.0, target_x=5.0))
    assert far["joint_1.pos"] == pytest.approx(OPENARM_WORKSPACE_MAX[0])  # +0.55

    # Reference stays latched, so the opposite target clips to the low bound.
    near = ctrl(_ee_action(enabled=1.0, target_x=-5.0))
    assert near["joint_1.pos"] == pytest.approx(OPENARM_WORKSPACE_MIN[0])  # -0.20


# ── Bimanual split / merge + send_action key contract ─────────────────────


def test_bimanual_transform_output_keys_match_biopenarm_send_action():
    """The transform must emit exactly the keys BiOpenArmFollower.send_action
    consumes: ``{left,right}_joint_{1..7}.pos`` + ``{left,right}_gripper.pos``
    (the per-arm OpenArmFollower strips the prefix and wants ``joint_1.pos``
    … ``gripper.pos``)."""
    left = make_openarm_arm_ik_controller(_StubKinematics(), LEFT_READY_DEG.copy(), label="left")
    right = make_openarm_arm_ik_controller(_StubKinematics(), RIGHT_READY_DEG.copy(), label="right")
    transform = make_bimanual_ik_transform(left, right)

    action = {f"{side}_{k}": v for side in ("left", "right") for k, v in _ee_action().items()}
    out = transform(action)

    expected = {f"{side}_{m}.pos" for side in ("left", "right") for m in MOTOR_NAMES}
    assert set(out) == expected
    assert "left_gripper.pos" in out and "right_gripper.pos" in out


def test_quest_arm_controller_output_drives_openarm_ik_controller():
    """The Quest teleop's action keys must be exactly what the IK controller reads."""
    from lerobot.teleoperators.quest_vr.arm_controller import QuestArmController

    arm = QuestArmController(
        clutch_button_index=1,
        gripper_button_index=0,
        position_scale=1.0,
        max_rot_step_rad_per_tick=3.14,
        gripper_open_motor=0.0,
        gripper_closed_motor=45.0,
        key_prefix="",
    )
    assert set(arm.idle_action()) == _IK_INPUT_KEYS
    engaged = arm.process_pose({"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0, 1.0], "buttons": [1.0, 1.0]})
    assert set(engaged) == _IK_INPUT_KEYS

    ctrl = _make_controller(LEFT_READY_DEG.copy())
    out = ctrl(engaged)
    assert set(out) == {f"{m}.pos" for m in MOTOR_NAMES}


# ── BiOpenArmFollower.attach_teleop branching ─────────────────────────────


class _FakeTeleop:
    def __init__(self, names: dict):
        self.action_features = {"names": names}
        self.installed = "<unset>"

    def set_action_transform(self, transform):
        self.installed = transform


def _bare_follower():
    """A BiOpenArmFollower with no __init__ run — enough to call attach_teleop."""
    from lerobot.robots.bi_openarm_follower.bi_openarm_follower import BiOpenArmFollower

    return BiOpenArmFollower.__new__(BiOpenArmFollower)


def test_attach_teleop_ignores_joint_teleop():
    robot = _bare_follower()
    leader = _FakeTeleop(names={f"{m}.pos": i for i, m in enumerate(MOTOR_NAMES)})
    robot.attach_teleop(leader)
    # A joint-space leader already emits joint dicts: no transform installed.
    assert leader.installed == "<unset>"


def test_attach_teleop_tolerates_none():
    """lerobot_teleoperate calls ``robot.attach_teleop(None)`` on teardown."""
    robot = _bare_follower()
    robot.attach_teleop(None)  # must not raise


def test_is_openarm_bimanual_cartesian_teleop_structural_check():
    assert is_openarm_bimanual_cartesian_teleop(_FakeTeleop(names={"left_target_x": 0, "right_target_x": 1}))
    assert not is_openarm_bimanual_cartesian_teleop(_FakeTeleop(names={"left_target_x": 0}))
    assert not is_openarm_bimanual_cartesian_teleop(None)
    assert not is_openarm_bimanual_cartesian_teleop(SimpleNamespace(action_features=None))


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_attach_teleop_installs_ik_for_cartesian_teleop():
    class _FakeArm:
        def __init__(self, seed):
            self.is_connected = True
            self._seed = seed

        def get_observation(self):
            return {f"{m}.pos": v for m, v in zip(MOTOR_NAMES, self._seed, strict=True)}

    robot = _bare_follower()
    robot.left_arm = _FakeArm(LEFT_READY_DEG)
    robot.right_arm = _FakeArm(RIGHT_READY_DEG)
    # __init__ is bypassed here, so pre-build the kinematics it normally
    # would (see BiOpenArmFollower.__init__).
    from lerobot.robots.openarm_description.cartesian_ik import make_openarm_arm_kinematics

    robot._ik_kinematics = {
        "left": make_openarm_arm_kinematics("left"),
        "right": make_openarm_arm_kinematics("right"),
    }

    teleop = _FakeTeleop(names={"left_target_x": 0, "right_target_x": 1})
    robot.attach_teleop(teleop)

    assert callable(teleop.installed)
    bimanual = {f"{side}_{k}": v for side in ("left", "right") for k, v in _ee_action().items()}
    joints = teleop.installed(bimanual)
    assert set(joints) == {f"{side}_{m}.pos" for side in ("left", "right") for m in MOTOR_NAMES}
    # Disengaged + zero deltas: both arms hold their seed poses.
    for side, seed in (("left", LEFT_READY_DEG), ("right", RIGHT_READY_DEG)):
        for i, m in enumerate(MOTOR_NAMES):
            assert joints[f"{side}_{m}.pos"] == pytest.approx(seed[i], abs=1e-6)


def test_attach_teleop_warns_and_noops_when_kinematics_unavailable(caplog):
    import logging

    robot = _bare_follower()
    robot.left_arm = SimpleNamespace(is_connected=True)
    robot.right_arm = SimpleNamespace(is_connected=True)
    robot._ik_kinematics = None  # e.g. pin-pink missing at __init__

    teleop = _FakeTeleop(names={"left_target_x": 0, "right_target_x": 1})
    with caplog.at_level(logging.WARNING, logger="lerobot.robots.bi_openarm_follower.bi_openarm_follower"):
        robot.attach_teleop(teleop)

    assert teleop.installed == "<unset>"
    assert "IK kinematics are unavailable" in caplog.text


# ── Factory plumbing for the IK config knobs ──────────────────────────────


def test_make_openarm_arm_kinematics_passes_tuning_kwargs_to_pink():
    from unittest.mock import patch

    from lerobot.robots.openarm_description import cartesian_ik

    with patch("lerobot.model.pink_kinematics.PinkKinematics") as mock_pk:
        cartesian_ik.make_openarm_arm_kinematics("left", posture_cost=0.42, max_iters=17)
    kwargs = mock_pk.call_args.kwargs
    assert kwargs["posture_cost"] == pytest.approx(0.42)
    assert kwargs["max_iters"] == 17
    assert kwargs["target_frame_name"] == "openarm_left_ee_base_link"
    assert kwargs["joint_names"] == [f"openarm_left_joint{i}" for i in range(1, 8)]


def test_make_openarm_arm_kinematics_defaults_match_biopenarm_config():
    """The factory defaults are visible config — BiOpenArmFollowerConfig's
    ``ik_posture_cost`` / ``ik_max_iters`` defaults must mirror them."""
    from unittest.mock import patch

    from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig
    from lerobot.robots.openarm_description import cartesian_ik
    from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase

    with patch("lerobot.model.pink_kinematics.PinkKinematics") as mock_pk:
        cartesian_ik.make_openarm_arm_kinematics("right")
    factory_defaults = mock_pk.call_args.kwargs

    config_default = BiOpenArmFollowerConfig(
        left_arm_config=OpenArmFollowerConfigBase(port="can0"),
        right_arm_config=OpenArmFollowerConfigBase(port="can1"),
    )
    assert factory_defaults["posture_cost"] == config_default.ik_posture_cost
    assert factory_defaults["max_iters"] == config_default.ik_max_iters


# ── Real kinematics: zero pose, FK/IK round-trip (needs pin-pink) ─────────


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
@pytest.mark.parametrize(
    ("side", "expected_y"),
    [("left", +0.1225), ("right", -0.1225)],
)
def test_zero_pose_is_arms_hanging_straight_down(side, expected_y):
    """FK at the all-zero pose must put the gripper flange directly below the
    base — the OpenArm 2.0 motor factory zero. Pins the "no motor<->URDF
    alignment needed" contract: (0, ±0.1225, -0.436) m with identity
    orientation matches the validated dora stack's _EE_ZERO exactly."""
    from lerobot.robots.openarm_description.cartesian_ik import make_openarm_arm_kinematics

    kin = make_openarm_arm_kinematics(side)
    fk = kin.forward_kinematics(np.zeros(8))

    np.testing.assert_allclose(fk[:3, 3], [0.0, expected_y, -0.436], atol=2e-3)
    # Rot atol 1e-3: PinkKinematics clamps q to joint limits with a 1e-4 rad
    # safety inset, and joint4's lower limit is exactly 0.
    np.testing.assert_allclose(fk[:3, :3], np.eye(3), atol=1e-3)


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
@pytest.mark.parametrize(
    ("side", "poses_deg"),
    [
        (
            "left",
            [
                [0.0] * 7,  # arms-down zero
                [0.0, -45.0, 0.0, 90.0, 0.0, 30.0, 0.0],  # ready pose
                [30.0, -60.0, 20.0, 100.0, -30.0, 20.0, -40.0],
                [-40.0, -30.0, -45.0, 60.0, 45.0, -20.0, 30.0],
            ],
        ),
        (
            "right",
            [
                [0.0] * 7,
                [0.0, +45.0, 0.0, 90.0, 0.0, 30.0, 0.0],
                [-30.0, +60.0, 20.0, 100.0, -30.0, 20.0, -40.0],
                [40.0, +30.0, -45.0, 60.0, 45.0, -20.0, 30.0],
            ],
        ),
    ],
)
def test_fk_ik_round_trip(side, poses_deg):
    """IK seeded near a pose must return a solution whose FK lands back on
    that pose's flange transform — for the arms-down zero and several
    bent configurations per arm."""
    from lerobot.robots.openarm_description.cartesian_ik import make_openarm_arm_kinematics

    kin = make_openarm_arm_kinematics(side)
    for pose in poses_deg:
        q = np.array(pose + [0.0])  # + gripper passthrough slot
        target = kin.forward_kinematics(q)
        q_ik = np.asarray(kin.inverse_kinematics(q, target), dtype=float)
        achieved = kin.forward_kinematics(q_ik)
        np.testing.assert_allclose(achieved[:3, 3], target[:3, 3], atol=2e-3, err_msg=f"pose {pose}")
        # Gripper slot passes through untouched.
        assert q_ik[7] == pytest.approx(0.0)


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_real_ik_controller_traces_a_position_offset():
    """Hold a +3 cm EE target; the IK joints must FK back to that offset.

    Seeded from the bent ready pose (the arms-down zero is fully stretched
    and near-singular). The iterative QP is polled repeatedly with the held
    target, exactly how a Quest streams continuous motion.
    """
    from lerobot.robots.openarm_description.cartesian_ik import make_openarm_arm_kinematics

    kin = make_openarm_arm_kinematics("left")
    q_init = LEFT_READY_DEG.copy()
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


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_full_stack_no_slew_on_tracking_dropout():
    """Dropout + re-acquire with the clutch still held must not slew the arm.

    Wires QuestArmController + the bimanual transform + the real OpenArm IK.
    ``settle_secs=0`` on the controllers isolates the re-anchor logic (the
    settle window has its own dedicated tests in test_quest_vr.py).
    """
    from lerobot.robots.openarm_description.cartesian_ik import make_openarm_arm_kinematics
    from lerobot.teleoperators.quest_vr.arm_controller import QuestArmController

    left_kin = make_openarm_arm_kinematics("left")
    right_kin = make_openarm_arm_kinematics("right")
    transform = build_openarm_bimanual_ik_transform(
        {"left": left_kin, "right": right_kin},
        left_arm=SimpleNamespace(
            get_observation=lambda: {f"{m}.pos": v for m, v in zip(MOTOR_NAMES, LEFT_READY_DEG, strict=True)}
        ),
        right_arm=SimpleNamespace(
            get_observation=lambda: {f"{m}.pos": v for m, v in zip(MOTOR_NAMES, RIGHT_READY_DEG, strict=True)}
        ),
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
            gripper_open_motor=0.0,
            gripper_closed_motor=45.0,
            max_pos_step_m_per_tick=0.0,
            key_prefix=prefix,
            settle_secs=0.0,
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

    max_delta = float(np.max(np.abs(after - baseline)))
    assert max_delta < 5.0, f"right arm slewed {max_delta:.1f} deg — re-anchor regressed"


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_bimanual_tracks_small_circle():
    """Both arms trace a 30 mm circle through the full bimanual stack.

    A regression in any layer (URDF, controller, transform) breaks the EE
    trace. Orientation is held at the latched reference (``target_w=0``).
    Tolerances mirror the SO-107 shape tests (PinkKinematics per-call QP
    convergence on a moving target).
    """
    from lerobot.robots.openarm_description.cartesian_ik import make_openarm_arm_kinematics

    def _ctrl(kin, q_init):
        return CartesianIKController(
            kinematics=kin,
            motor_names=list(MOTOR_NAMES),
            q_init=q_init,
            workspace_min=(-10.0, -10.0, -10.0),
            workspace_max=(10.0, 10.0, 10.0),
            max_ee_step_m=10.0,
        )

    left_kin = make_openarm_arm_kinematics("left")
    right_kin = make_openarm_arm_kinematics("right")
    transform = make_bimanual_ik_transform(
        _ctrl(left_kin, LEFT_READY_DEG.copy()), _ctrl(right_kin, RIGHT_READY_DEG.copy())
    )
    ref_left = left_kin.forward_kinematics(LEFT_READY_DEG)
    ref_right = right_kin.forward_kinematics(RIGHT_READY_DEG)

    def _circle_deltas(ref_pose, r, n):
        p = ref_pose[:3, 3]
        flat = np.array([p[0], p[1], 0.0])
        norm = float(np.linalg.norm(flat))
        inward = -flat / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])
        perp = np.cross(inward, np.array([0.0, 0.0, 1.0]))
        perp /= np.linalg.norm(perp)
        center = p + r * inward
        return [
            center + r * (np.cos(2 * np.pi * i / n) * -inward + np.sin(2 * np.pi * i / n) * perp) - p
            for i in range(n)
        ]

    deltas_left = _circle_deltas(ref_left, 0.030, 256)
    deltas_right = _circle_deltas(ref_right, 0.030, 256)
    pos_err: list[float] = []
    rot_err: list[float] = []

    for dl, dr in zip(deltas_left, deltas_right, strict=True):
        action = {}
        for side, d in (("left", dl), ("right", dr)):
            action[f"{side}_enabled"] = 1.0
            action[f"{side}_target_x"] = float(d[0])
            action[f"{side}_target_y"] = float(d[1])
            action[f"{side}_target_z"] = float(d[2])
            action[f"{side}_target_wx"] = 0.0
            action[f"{side}_target_wy"] = 0.0
            action[f"{side}_target_wz"] = 0.0
            action[f"{side}_gripper_pos"] = 0.0
        out = transform(action)
        for side, kin, ref, d in (
            ("left", left_kin, ref_left, dl),
            ("right", right_kin, ref_right, dr),
        ):
            q_out = np.array([out[f"{side}_{m}.pos"] for m in MOTOR_NAMES])
            fk = kin.forward_kinematics(q_out)
            pos_err.append(float(np.linalg.norm(fk[:3, 3] - (ref[:3, 3] + d))))
            r_err = ref[:3, :3].T @ fk[:3, :3]
            cos_a = max(-1.0, min(1.0, (float(np.trace(r_err)) - 1.0) * 0.5))
            rot_err.append(float(np.degrees(np.arccos(cos_a))))

    warmup = 20
    assert max(pos_err[warmup:]) < 3e-3, f"position drift {max(pos_err[warmup:]) * 1000:.2f} mm"
    assert max(rot_err[warmup:]) < 1.0, f"rotation drift {max(rot_err[warmup:]):.2f} deg"
