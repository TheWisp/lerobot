"""The sealed insertion demo was recorded at a calibrated physical fingertip."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT


def test_measured_tool_and_joint_zeros_recover_recorded_demo_fk():
    # Compact provenance fixture from insertion/showservo_demo.npz and the
    # white_left calibration used when that episode was recorded.
    calibration = {
        "tool_point": {"offset_m": [-0.00732108362056063, -0.08923142140583068, 0.004925306681653079]},
        "joint_zero_deg": {
            "shoulder_pan": 0.005593719865083844,
            "shoulder_lift": 5.279106382223435,
            "elbow_flex": 7.003265671991713,
            "forearm_roll": 0.8407507837451832,
            "wrist_flex": -2.759580730624803,
        },
    }
    joints = np.array(
        [3.2967033, 18.81318681, 13.58241758, -0.26373626, -58.59340659, -11.03296703, 93.55281207]
    )
    recorded_tip_m = np.array([-0.0155329, -0.19482794, -0.01398757])
    calibrated = make_calibrated_so107_kinematics(calibration, arm="left")
    nominal = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)

    np.testing.assert_allclose(calibrated.forward_kinematics(joints)[:3, 3], recorded_tip_m, atol=1e-6)
    assert np.linalg.norm(nominal.forward_kinematics(joints)[:3, 3] - recorded_tip_m) > 0.1


@pytest.mark.parametrize("arm", ["left", "right"])
def test_folded_pose_matches_unclamped_dynamics_frame_and_remains_solvable(arm):
    """The insertion recording starts with S2 below the CAD export's -pi limit.

    Compare the public calibrated FK/IK to raw Pinocchio frame placement so
    clipping the observation and then solving that clipped target cannot pass.
    """
    pin = pytest.importorskip("pinocchio")
    from lerobot.robots.so107_description import get_gravity_urdf_path
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES, RIGHT_ARM_ALIGNMENT

    alignment = LEFT_ARM_ALIGNMENT if arm == "left" else RIGHT_ARM_ALIGNMENT
    calibration = {
        "tool_point": {"offset_m": [-0.00732108362056063, -0.08923142140583068, 0.004925306681653079]},
        "joint_zero_deg": {"shoulder_lift": 5.279106382223435, "elbow_flex": 7.003265671991713},
    }
    # First-frame motor state from the regular LeRobot insertion recording.
    joints = np.array([-2.8571, -98.0220, 88.5714, 14.9451, -58.3297, -13.8462, 79.5610])
    q = np.deg2rad(
        [
            alignment[name].sign * (value + calibration["joint_zero_deg"].get(name, 0.0))
            + alignment[name].offset_deg
            for name, value in zip(MOTOR_NAMES, joints, strict=True)
        ]
    )
    assert q[1] < -np.pi
    dynamics = pin.buildModelFromUrdf(str(get_gravity_urdf_path()))
    data = dynamics.createData()
    pin.framesForwardKinematics(dynamics, data, q)
    tool = np.eye(4)
    tool[:3, 3] = calibration["tool_point"]["offset_m"]
    expected = data.oMf[dynamics.getFrameId("L6_1")].homogeneous @ tool
    kin = make_calibrated_so107_kinematics(calibration, arm=arm)
    np.testing.assert_allclose(kin.forward_kinematics(joints), expected, atol=1e-12)
    solved = kin.inverse_kinematics(joints, expected)
    np.testing.assert_allclose(solved[:6], joints[:6], atol=1e-5)


def test_recorded_stack_exit_reaches_target_without_exceeding_wrist_limit():
    """Frame 778 stalled after 339 keys: the unconstrained IK crossed the limit.

    Both the measured target and seed are legal. Supplying the same limit to
    the solver must let it move the other joints, without widening the limit.
    Values come from the 2026-11-14 gamepad-on-ring episode and calibrated rig.
    """
    calibration = {
        "tool_point": {"offset_m": [-0.00732108362056063, -0.08923142140583068, 0.004925306681653079]},
        "joint_zero_deg": {
            "shoulder_pan": 0.005593719865083844,
            "shoulder_lift": 5.279106382223435,
            "elbow_flex": 7.003265671991713,
            "forearm_roll": 0.8407507837451832,
            "wrist_flex": -2.759580730624803,
        },
    }
    lower = -93.31868131868131
    kin = make_calibrated_so107_kinematics(
        calibration, arm="left", joint_limits_deg={"wrist_flex": (lower, -lower)}
    )
    seed = np.array(
        [3.8191859677, -0.9870246565, -19.6926600293, 8.0682787935, -93.2810375670, 5.8028871252, 0]
    )
    measured = np.array(
        [3.6483516693, -2.7252748013, -18.5054950714, 8.0879125595, -93.2307662964, 4.9670329094, 0]
    )
    target = kin.forward_kinematics(measured)
    result = kin.inverse_kinematics(seed, target)
    assert lower <= result[4] <= -lower
    actual = kin.forward_kinematics(result)
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) < 0.003
    assert Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude() < np.deg2rad(3)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_planning_limits_preserve_measured_fk_and_motor_direction(arm):
    calibration = {"tool_point": {"offset_m": [0.02, -0.025, -0.017]}, "joint_zero_deg": {"wrist_flex": 5}}
    unrestricted = make_calibrated_so107_kinematics(calibration, arm=arm)
    limited = make_calibrated_so107_kinematics(
        calibration, arm=arm, joint_limits_deg={"wrist_flex": (-35, -25)}
    )
    seed = np.array([0, -30, 45, 0, -30, 0, 0], dtype=float)
    outside = seed.copy()
    outside[4] = -45
    # Planning limits cannot turn a measured -45 degrees into a fictional -35.
    np.testing.assert_allclose(limited.forward_kinematics(outside), unrestricted.forward_kinematics(outside))
    solved = limited.inverse_kinematics(seed, unrestricted.forward_kinematics(outside))
    assert -35 <= solved[4] <= -25


@pytest.mark.parametrize("bounds", [(0, 0), (10, -10), (float("nan"), 20), (1,)])
def test_invalid_motor_planning_limits_fail_at_construction(bounds):
    with pytest.raises(ValueError, match="Invalid motor joint limits"):
        make_calibrated_so107_kinematics(
            {"tool_point": {"offset_m": [0, 0, 0]}}, arm="left", joint_limits_deg={"wrist_flex": bounds}
        )
