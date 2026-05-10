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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

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
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so107_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.latency import (
    LatencyAggregator,
    LatencySnapshotWriter,
    LatencyTracer,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
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
    # Latency monitoring: capture per-stage timing into an in-memory aggregator
    # and publish a JSON snapshot for the GUI to read.
    # See src/lerobot/gui/docs/latency_monitoring.md.
    latency_monitor: bool = False
    # Where to write latency_snapshot.json (when --latency_monitor=true).
    # The GUI reads from this fixed location to render the live overlays.
    latency_output_dir: str = "outputs/teleop"


def _maybe_span(tracer: LatencyTracer | None, name: str):
    """Context manager around a tracer span; nullcontext when tracer is None."""
    return tracer.span(name) if tracer is not None else nullcontext()


def _format_latency_summary(snap: dict[str, Any]) -> str:
    """One-line digest of an aggregator snapshot — used for stderr at 1 Hz."""
    stages = snap.get("stages", {})

    def p50_p95(key: str) -> str | None:
        s = stages.get(key)
        if not s:
            return None
        return f"{s.get('p50', 0):.1f}/{s.get('p95', 0):.1f}ms"

    parts: list[str] = []
    if (s := p50_p95("loop_dt_ms")) is not None:
        parts.append(f"loop {s}")
    if (gobs := stages.get("get_observation_ms")) is not None:
        parts.append(f"obs {gobs.get('p50', 0):.1f}ms")
    if (send := stages.get("action_send_ms")) is not None:
        parts.append(f"send {send.get('p50', 0):.1f}ms")
    cam_keys = sorted(k for k in stages if k.startswith("cam_") and k.endswith("_stale_ms"))
    for k in cam_keys:
        cam_name = k[len("cam_") : -len("_stale_ms")]
        parts.append(f"{cam_name} stale {stages[k].get('p50', 0):.0f}ms")
    parts.append(f"overrun {snap.get('overrun_ratio', 0) * 100:.0f}%")
    return " · ".join(parts)


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
    latency_aggregator: LatencyAggregator | None = None,
    latency_writer: LatencySnapshotWriter | None = None,
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
    """

    display_len = max(len(key) for key in robot.action_features)
    tracer = (
        LatencyTracer(latency_aggregator, loop_kind="teleop", target_fps=fps)
        if latency_aggregator is not None
        else None
    )
    last_summary_at: float = 0.0
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        if tracer is not None:
            tracer.start()

        # Get robot observation. We wrap the whole call in a single span:
        # this includes the follower's motor sync_read AND the (cached,
        # microseconds-fast) cam.read_latest() per camera. In practice the
        # span time is dominated by the motor sync_read; finer breakdown
        # (motor vs. cam consume) is V2.
        with _maybe_span(tracer, "get_observation"):
            obs = robot.get_observation()

        # Per-camera staleness/period — read latest_timestamp from each camera
        # after get_observation so we capture what was just consumed.
        if tracer is not None:
            cams = getattr(robot, "cameras", None) or {}
            for cam_key, cam in cams.items():
                ts = getattr(cam, "latest_timestamp", None)
                if ts is not None:
                    tracer.cam_consume(cam_key, ts)

        # Run obs processors + stream writer (for GUI live viewer with overlays)
        with _maybe_span(tracer, "process_obs"):
            if obs_stream_steps:
                obs_for_stream = obs
                for step in obs_stream_steps:
                    obs_for_stream = step.observation(obs_for_stream)

        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        with _maybe_span(tracer, "process_action"):
            # Get teleop action
            raw_action = teleop.get_action()
            # Process teleop action through pipeline
            teleop_action = teleop_action_processor((raw_action, obs))
            # Process action for robot through pipeline
            robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        with _maybe_span(tracer, "action_send"):
            _ = robot.send_action(robot_action_to_send)

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

        # Commit BEFORE precise_sleep so loop_dt_ms reflects iteration *work*
        # time, not work + sleep. Otherwise overrun fires every iteration —
        # precise_sleep slightly overshoots its target by design, so loop_dt
        # is always ε > target_period and overrun% reads 100. This way
        # overrun fires only when actual work exceeds 1000/fps.
        if tracer is not None:
            tracer.commit()

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start

        if tracer is not None:
            # Snapshot publish + 1 Hz stderr summary run on idle time after
            # sleep. Cost is bounded (~50–200 µs at 1 Hz amortized).
            if latency_writer is not None:
                latency_writer.maybe_write(latency_aggregator)
            now = time.time()
            if now - last_summary_at >= 1.0:
                last_summary_at = now
                snap = latency_aggregator.snapshot(percentiles=(50, 95))
                if snap["n_records"] > 0:
                    logging.info("[latency] %s", _format_latency_summary(snap))
        else:
            # Legacy line-rewriting print, only when monitoring is off (else
            # the 1 Hz INFO log is the cleaner UX).
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
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

    latency_aggregator: LatencyAggregator | None = None
    latency_writer: LatencySnapshotWriter | None = None
    if cfg.latency_monitor:
        latency_aggregator = LatencyAggregator()
        latency_writer = LatencySnapshotWriter(
            cfg.latency_output_dir,
            loop_kind="teleop",
            target_fps=float(cfg.fps),
        )
        logging.info("Latency monitoring enabled; snapshots → %s", latency_writer.path)

    teleop.connect()
    robot.connect()

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
            latency_aggregator=latency_aggregator,
            latency_writer=latency_writer,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            shutdown_rerun()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
