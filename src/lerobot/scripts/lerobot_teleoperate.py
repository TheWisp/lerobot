# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Simple script to control a robot from teleoperation.

Requires: pip install 'lerobot[hardware]'

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so107_follower,
    bi_so107_follower_predictive,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so107_description,
    so107_follower_predictive,
    so_follower,
    so_follower_predictive,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so107_leader,
    bi_so107_leader_highrate,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    quest_vr,
    reachy2_teleoperator,
    so107_leader_highrate,
    so_leader,
    so_leader_highrate,
    trajectory_replay,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.latency import LatencySession
from lerobot.utils.latency.ik_debug import get_recorder as get_ik_recorder
from lerobot.utils.latency.motion import MotionLogger
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import move_cursor_up, setup_run_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Live URDF visualization of the IK-output joint commands. Opens a
    # MeshCat scene at http://127.0.0.1:7000/static/ rendering one (unimanual)
    # or two (bimanual) SO-107 arms driven by the post-IK <motor>.pos stream.
    # Pair with the robot config's dry_run=True for safe testing without
    # any motor traffic. Only the SO-107 description is wired today.
    display_urdf: bool = False
    # Verbose per-tick log of the Cartesian IK pipeline internals (EE
    # targets, bounds clipping, IK seed / output / position error).
    # Written to ``<latency_output_dir>/ik_debug_<ts>.npz`` alongside the
    # motion log. Use when chasing IK instability symptoms; off by default
    # because the per-tick allocations add a small overhead at 200 Hz.
    debug_teleop_log: bool = False
    # Latency monitoring: capture per-stage timing into an in-memory aggregator
    # and publish a JSON snapshot for the GUI to read.
    # See src/lerobot/gui/docs/latency_monitoring.md.
    latency_monitor: bool = False
    # Where to write latency_snapshot.json (when --latency_monitor=true).
    # The GUI reads from this fixed location to render the live overlays.
    latency_output_dir: str = "outputs/teleop"


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    motor_binding,  # MotorActionBinding; forward-ref to avoid a top-level import cycle
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    obs_stream_steps: list | None = None,
    latency_session: LatencySession | None = None,
    motion_logger: MotionLogger | None = None,
    commanded_joints_log: Any | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
        latency_session: Per-loop latency monitoring lifecycle. Pass
            ``LatencySession.disabled()`` (or omit) for no monitoring; the
            loop body is identical either way thanks to the no-op session.
    """

    display_len = max(len(key) for key in robot.action_features)
    if latency_session is None:
        latency_session = LatencySession.disabled()
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        with latency_session.iteration():
            # Get robot observation. We wrap the whole call in a single span:
            # this includes the follower's motor sync_read AND the (cached,
            # microseconds-fast) cam.read_latest() per camera. In practice
            # the span time is dominated by the motor sync_read; finer
            # breakdown (motor vs. cam consume) is V2.
            with latency_session.span("get_observation"):
                obs = robot.get_observation()

            # Per-camera staleness/period — read latest_timestamp from each
            # camera right after get_observation so we capture what was
            # just consumed.
            latency_session.cam_consume_all(getattr(robot, "cameras", None))

            # Run obs processors + stream writer (for GUI live viewer with overlays)
            with latency_session.span("process_obs"):
                if obs_stream_steps:
                    obs_for_stream = obs
                    for step in obs_stream_steps:
                        obs_for_stream = step.observation(obs_for_stream)

            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            with latency_session.span("process_action"):
                # Ask the robot for the motor-space action this tick. The
                # binding hides the conversion mechanism: it might be a
                # passthrough from a joint-space leader, the output of a
                # script-side IK pipeline (push path), or a read from the
                # robot's internal adapter cache (pull path). Either way,
                # ``joints`` is what we record / log / optionally send.
                joints = motor_binding.get_action(obs)

            # Send to motors. Two independent paths can publish:
            #   1. Action chunks (e.g. TrajectoryReplayTeleop, policy with
            #      horizon output). Chunks are honored even under pull-path
            #      because the predictive controller's exact-lookup uses
            #      them.
            #   2. Per-tick dict. Skipped when ``pull_path_active`` is True
            #      because the robot's own controller polls its teleop
            #      directly at control-rate and writes to motors itself;
            #      a redundant ``send_action`` would just set an intent
            #      the controller would ignore.
            with latency_session.span("action_send"):
                chunk = teleop.get_action_with_horizon()
                if chunk is not None:
                    _ = robot.send_action(chunk)
                elif not motor_binding.pull_path_active:
                    _ = robot.send_action(joints)

            # Per-tick motion logging (intent + state). MotionLogger
            # filters to .pos keys so non-position obs (cameras) are
            # silently skipped.
            if motion_logger is not None:
                motion_logger.tick(joints, obs)

            # End-of-tick marker for the verbose IK debug log. No-op when
            # the recorder isn't installed.
            _ik_rec = get_ik_recorder()
            if _ik_rec is not None:
                _ik_rec.tick_done()

            # Periodic commanded-joints log line. Debug-only; called here
            # so it runs regardless of which conversion path produced
            # ``joints``.
            if commanded_joints_log is not None and joints:
                commanded_joints_log.action(dict(joints))

            if display_data:
                log_rerun_data(
                    observation=robot_observation_processor(obs),
                    action=joints,
                    compress_images=display_compressed_images,
                )

                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                # Display the final robot action that was sent
                for motor, value in joints.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(joints) + 3)
            # iteration() commits BEFORE precise_sleep so loop_dt_ms reflects
            # iteration *work* time, not work + sleep. Otherwise overrun
            # fires every iteration — precise_sleep slightly overshoots
            # its target by design.

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start

        if not latency_session.enabled:
            # Legacy line-rewriting print, only when monitoring is off
            # (else the 1 Hz INFO log from LatencySession is cleaner UX).
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return
        # File-backed teleops (trajectory_replay) flip ``is_exhausted`` once
        # the recorded duration has elapsed. Treat that as a clean end-of-
        # session signal, the same as ``duration`` expiring.
        if getattr(teleop, "is_exhausted", False):
            logging.info("Teleop exhausted (no more frames) — ending session.")
            return


def _make_commanded_joints_log(robot) -> Any | None:
    """Return a :class:`CommandedJointsLogStep` configured for ``robot``, or None.

    Throttled INFO logs of post-IK joint commands so users can grep the
    file log to see what the teleop -> IK chain is producing. The teleop
    loop calls this step's ``action()`` directly on the per-tick joints
    — no longer attached to a pipeline since the pipeline-in-script
    construct is gone. None if the SO107 viz module isn't available.
    """
    try:
        from lerobot.robots.so107_description.urdf_viz import CommandedJointsLogStep
    except ImportError:
        return None
    try:
        obs_features = robot.observation_features
        bimanual = any(k.startswith("left_") and k.endswith(".pos") for k in obs_features)
    except Exception:
        bimanual = False
    return CommandedJointsLogStep(bimanual=bimanual)


def _attach_urdf_viz_obs_stream(obs_stream_steps, robot) -> None:
    """Wrapper that attaches the URDF viz step to the obs stream step list.

    The teleop loop iterates obs_stream_steps explicitly every tick (line
    ~210 in this file), independent of the display_data flag. Attaching
    here is what makes the viz reflect the live robot state.
    """
    try:
        from lerobot.robots.so107_description.urdf_viz import maybe_attach_urdf_viz
    except ImportError as e:
        logging.warning(
            f"display_urdf=True but URDF viz module failed to import "
            f"({type(e).__name__}: {e}). Skipping; check pinocchio + meshcat are installed."
        )
        return
    maybe_attach_urdf_viz(obs_stream_steps, robot, logging.getLogger())


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    setup_run_logging(cfg.latency_output_dir, "teleop")
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    _is_remote = cfg.display_ip is not None and cfg.display_ip not in ("127.0.0.1", "localhost", "::1")
    display_compressed_images = (
        True
        if (cfg.display_data and _is_remote and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    _, _, robot_observation_processor = make_default_processors()

    # Periodic commanded-joints log line (post-IK joints, INFO every ~1s).
    # Called explicitly in the teleop_loop on the per-tick joint dict so
    # it works the same regardless of whether the joints came from a
    # passthrough leader, a script-side pipeline, or a robot-owned
    # adapter cache. None when display_urdf is off.
    commanded_joints_log = _make_commanded_joints_log(robot) if cfg.display_urdf else None

    # Add custom observation processor steps from the robot
    custom_steps = robot.get_observation_processor_steps()
    if custom_steps:
        # Insert custom steps at the beginning of the pipeline
        robot_observation_processor.steps = custom_steps + robot_observation_processor.steps
        logging.info(f"Added {len(custom_steps)} custom observation processor step(s) from robot")

    # Build obs stream processor chain: robot processors + stream writer at the end
    # The stream writer handles both ObservationStream (GUI viewer) and optionally
    # SharedImageBuffer (S2 debug model) based on env vars.
    from lerobot.robots.obs_stream import make_obs_stream_writer_step

    obs_stream_steps = list(robot.get_observation_processor_steps() or [])
    obs_stream_writer = make_obs_stream_writer_step()
    if obs_stream_writer is not None:
        obs_stream_steps.append(obs_stream_writer)

    # URDF viz: state-based observation step that taps obs_stream_steps
    # (which the teleop loop iterates every tick). Attaching to
    # robot_observation_processor instead would only fire when
    # --display_data=true, which GUI launches don't set — that's why
    # an earlier wiring left the URDF stuck at zero pose.
    if cfg.display_urdf:
        _attach_urdf_viz_obs_stream(obs_stream_steps, robot)

    latency_session = LatencySession.from_config(
        enabled=cfg.latency_monitor,
        loop_kind="teleop",
        target_fps=float(cfg.fps),
        output_dir=cfg.latency_output_dir if cfg.latency_monitor else None,
    )
    if latency_session.enabled and latency_session.writer is not None:
        logging.info("Latency monitoring enabled; snapshots → %s", latency_session.writer.path)

    # Per-tick motion log (intent + state) — same output dir as the latency
    # snapshot, timestamped filename so back-to-back runs don't clobber.
    # Gated on latency_monitor so non-monitored runs have zero overhead.
    motion_logger: MotionLogger | None = None
    if cfg.latency_monitor:
        motion_logger = MotionLogger(cfg.latency_output_dir)
        logging.info("Motion logging enabled; trace → %s", motion_logger.path)

    # Verbose per-tick IK pipeline log. Installed as a module-level singleton
    # so the individual ProcessorSteps don't need explicit plumbing.
    ik_debug_recorder = None
    if cfg.debug_teleop_log:
        from lerobot.utils.latency.ik_debug import IKDebugRecorder, set_recorder

        ik_debug_recorder = IKDebugRecorder(cfg.latency_output_dir)
        set_recorder(ik_debug_recorder)
        logging.info("IK debug log enabled; trace → %s", ik_debug_recorder.path)

    teleop.connect()
    robot.connect()

    # Bind teleop to robot. The robot returns a MotorActionBinding that
    # the loop uses to (a) get the motor-space action per tick and (b)
    # decide whether to call send_action(joints) — see the binding's
    # docstring. ``bind_teleop`` also handles any controller-side wiring
    # (calls attach_teleop internally) so the script doesn't need to.
    motor_binding = robot.bind_teleop(teleop)

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            motor_binding=motor_binding,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            obs_stream_steps=obs_stream_steps,
            latency_session=latency_session,
            motion_logger=motion_logger,
            commanded_joints_log=commanded_joints_log,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if motion_logger is not None:
            motion_logger.close()
        if ik_debug_recorder is not None:
            from lerobot.utils.latency.ik_debug import set_recorder

            ik_debug_recorder.close()
            set_recorder(None)
        if cfg.display_data:
            shutdown_rerun()
        # Detach the teleop from the robot BEFORE disconnecting it. On a
        # chunk-aware / predictive robot, ``attach_teleop`` wires the
        # 200 Hz controller thread to ``teleop.get_action()``. If we
        # disconnect the teleop without detaching first, the controller
        # keeps polling for the ~40 ms it takes the subsequent
        # ``robot.disconnect()`` to stop the thread — every poll hits a
        # ``DeviceNotConnectedError`` from the closed teleop bus, logged
        # as a noisy ERROR per tick. Detaching first makes shutdown
        # silent. No-op for non-predictive robots (base ``attach_teleop``
        # is empty).
        robot.attach_teleop(None)
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
