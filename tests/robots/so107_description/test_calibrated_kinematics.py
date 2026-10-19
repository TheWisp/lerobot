"""The sealed insertion demo was recorded at a calibrated physical fingertip."""

import numpy as np

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
