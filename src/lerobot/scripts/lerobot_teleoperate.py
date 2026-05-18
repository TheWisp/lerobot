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

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
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
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    obs_stream_steps: list | None = None,
    latency_session: LatencySession | None = None,
    motion_logger: MotionLogger | None = None,
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
                # Get teleop action
                raw_action = teleop.get_action()
                # Process teleop action through pipeline
                teleop_action = teleop_action_processor((raw_action, obs))
                # Process action for robot through pipeline
                robot_action_to_send = robot_action_processor((teleop_action, obs))
                # Chunk-aware path: if the teleop publishes an upcoming
                # horizon, route the chunk to the robot so chunk-aware
                # robots (SO107FollowerPredictive et al.) can perform
                # exact-lookup lookahead at now + L instead of velocity
                # extrapolation. The processor pipeline is dict-only by
                # design (its converters validate isinstance(action, dict)),
                # so chunks bypass it — frames[0] is what the dict path
                # would have produced for "now" and the controller treats
                # the chunk's later frames as the authoritative future.
                # robot_action_to_send (the post-pipeline dict) is still
                # used for the display block below so per-tick UX is
                # unaffected.
                action_to_send = teleop.get_action_with_horizon() or robot_action_to_send

            # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
            with latency_session.span("action_send"):
                _ = robot.send_action(action_to_send)

            # Per-tick motion logging (intent + state). No-op when disabled.
            # Use the POST-pipeline dict (``robot_action_to_send``) as intent
            # so the trace works for any teleop type — leader arms (raw
            # action == joint commands, pipeline is identity), Cartesian
            # teleops (raw action == target_x/y/z, pipeline runs IK to
            # produce <motor>.pos), and bimanual variants (post-pipeline
            # carries left_<motor>.pos + right_<motor>.pos, matching the
            # bimanual observation's key prefix). MotionLogger filters to
            # .pos keys so non-position obs (cameras) are silently skipped.
            if motion_logger is not None:
                motion_logger.tick(robot_action_to_send, obs)

            if display_data:
                log_rerun_data(
                    observation=robot_observation_processor(obs),
                    action=teleop_action,
                    compress_images=display_compressed_images,
                )

                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                # Display the final robot action that was sent
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(robot_action_to_send) + 3)
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


def _attach_commanded_joints_log(action_pipeline, robot) -> None:
    """Append a CommandedJointsLogStep to the action pipeline.

    Throttled INFO logs of post-IK joint commands so users can grep the
    file log to see what the teleop -> IK chain is producing. No-op on
    the action stream itself.
    """
    try:
        from lerobot.robots.so107_description.urdf_viz import CommandedJointsLogStep
    except ImportError:
        return
    try:
        obs_features = robot.observation_features
        bimanual = any(k.startswith("left_") and k.endswith(".pos") for k in obs_features)
    except Exception:
        bimanual = False
    action_pipeline.steps.append(CommandedJointsLogStep(bimanual=bimanual))


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
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Auto-compose the Cartesian-IK pipeline when the teleop emits EE-delta
    # actions (quest_vr, keyboard_ee, phone, ...) and the robot has a
    # registered Cartesian config. Robots register their config at import
    # time via lerobot.processor.cartesian_ik_pipeline.register_cartesian_ik_robot.
    # Falls back to identity (with a clear log) when no config is available
    # — useful for joint-mode teleops, which don't need the IK chain.
    from lerobot.processor.cartesian_ik_pipeline import (
        is_cartesian_teleop,
        make_cartesian_ik_pipeline,
    )

    if is_cartesian_teleop(teleop):
        cartesian_pipeline = make_cartesian_ik_pipeline(robot)
        if cartesian_pipeline is not None:
            robot_action_processor = cartesian_pipeline
            logging.info(
                f"Auto-composed Cartesian IK pipeline for teleop={cfg.teleop.type} -> robot={cfg.robot.type}"
            )
        else:
            logging.warning(
                f"teleop {cfg.teleop.type!r} emits Cartesian actions but "
                f"robot {cfg.robot.type!r} has no registered Cartesian IK config "
                f"(see lerobot.processor.cartesian_ik_pipeline.register_cartesian_ik_robot). "
                f"Falling back to identity pipeline; the robot will likely fail to apply the action."
            )

    # Commanded-joints log on the action pipeline (post-IK targets at INFO).
    if cfg.display_urdf:
        _attach_commanded_joints_log(robot_action_processor, robot)

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

    teleop.connect()
    robot.connect()

    # Chunk-aware / predictive robots can poll the teleop directly at
    # their own control rate (e.g. 200 Hz) instead of waiting for
    # send_action pushes from the 30 Hz loop. Default Robot.attach_teleop
    # is a no-op for non-predictive robots, so this is unconditionally
    # safe to call. The loop's send_action path still runs for dataset
    # recording — when the teleop is bound, send_action's intent is
    # ignored by the controller, but the dict return value is still
    # what the dataset writer records.
    robot.attach_teleop(teleop)

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            obs_stream_steps=obs_stream_steps,
            latency_session=latency_session,
            motion_logger=motion_logger,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if motion_logger is not None:
            motion_logger.close()
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
