import inspect
import sys

import pytest

mode = sys.argv[1]
if mode == "moving_reference":
    import lerobot.policies.flash.demo_execution as m

    source = inspect.getsource(m.compile_demo).replace(
        "if previous is None or key.frame != previous.frame + 1 or key.reference != previous.reference:",
        "if True:",
    )
    # Execute only inspected repository source with the deliberate test mutation.
    exec(source, m.__dict__)  # nosec B102
    target = "tests/policies/flash/test_demo_execution.py::test_held_object_motion_does_not_cancel_lift_and_later_occlusion_keeps_keys"
elif mode == "occlusion":
    import lerobot.policies.flash.demo_execution as m

    source = inspect.getsource(m.KeyframeExecutor).replace(
        'if self.reference_frame != key["reference_frame"]:', "if True:"
    )
    # Execute only inspected repository source with the deliberate test mutation.
    exec(source, m.__dict__)  # nosec B102
    target = "tests/policies/flash/test_demo_execution.py::test_range_finishes_through_occlusion_then_waits_for_new_reference"
elif mode == "stall":
    import lerobot.policies.flash.demo_execution as m

    source = inspect.getsource(m.KeyframeExecutor).replace(
        "if np.allclose(self.model.degrees(action), self.model.degrees(state)):", "if False:"
    )
    # Execute only inspected repository source with the deliberate test mutation.
    exec(source, m.__dict__)  # nosec B102
    target = "tests/policies/flash/test_demo_execution.py::test_stalled_ik_is_waiting_without_sending_unchanged_commands"
elif mode == "joint_limits":
    import lerobot.robots.so107_description.calibrated_kinematics as m

    original = m.make_so107_arm_kinematics

    def unbounded(*args, **kwargs):
        kwargs.pop("joint_limits_deg", None)
        return original(*args, **kwargs)

    m.make_so107_arm_kinematics = unbounded
    target = "tests/robots/so107_description/test_calibrated_kinematics.py::test_recorded_stack_exit_reaches_target_without_exceeding_wrist_limit"
else:
    raise ValueError(mode)
raise SystemExit(pytest.main(["-q", target]))
