"""Editor execution adapter around the existing live camera/depth/tracker worker."""

import json
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np

from lerobot.gui.flash_live_scene import object_evidence
from lerobot.robots.rest_position import move_to_rest_position

from .appearance import Sam3MaskSource
from .demo_document import atomic_json
from .demo_execution import POSE_AVAILABLE, DemoRunTracker, load_reference, plan_demo_range
from .execution import ActionQueue, MotorLoop, TimedAction
from .geometry import transform
from .live_surface_session import LIVE_METHOD
from .live_surface_worker import run


class ExecutionLoop:
    """One worker owns the initial rest move and subsequent MotorLoop dispatch.

    A changed object target rebuilds and replaces the unissued queue. Motor
    dispatch never waits for IK, camera processing, or a new target.
    """

    def __init__(self, plan, robot, directory, stop, *, simulate, rest_position=None):
        self.robot, self.directory, self.stop_event = robot, directory, stop
        self.simulate = simulate
        self.rest_position = dict(rest_position or {})
        self.model = robot.cartesian_model()
        if (
            robot.geometry.to_dict() != plan["robot_geometry"]
            or self.model.description() != plan["robot_cartesian"]
        ):
            raise ValueError("Robot geometry or joint units changed since preparation")
        self.plan = plan
        self.state = dict(plan["keyframes"][0]["state"])
        self.observations = {}
        self.lock = Lock()
        self.thread = None
        self.error = None
        self.commands_sent = 0
        self.index = 0  # Next demonstration keyframe that has not been sent.
        self.status = "ready"
        self.evidence = None
        self.target = None
        self._planned = None
        self._range_start_index = None
        self._future = None
        self._planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-editor-ik")
        self._queue = ActionQueue()
        self._revision = 0
        self._motor = None
        self._awaiting_final_read = False
        self._measurements = deque(maxlen=30)
        self._completed_ranges = []
        self._range_first_command_s = None
        self._demo_first_command_s = None
        self._demo_last_command_s = None
        self._motor_trace = deque(maxlen=3600)
        self._motor_trace_total = 0
        self._frame_to_index = {key["frame"]: i for i, key in enumerate(plan["keyframes"])}

    class _SimulationRobot:
        def __init__(self, state):
            self.state = dict(state)
            self.action_features = dict.fromkeys(state, float)

        def get_state_observation(self):
            return self.state.copy()

        def get_observation(self):
            return self.get_state_observation()

        def send_action(self, action):
            self.state.update(action)
            return action

    def observe(self, frame, row):
        available = time.perf_counter()
        values = {
            role: {
                **object_evidence(value, LIVE_METHOD),
                "source_time_s": frame.source_time_s,
                "available_time_s": available,
            }
            for role, value in row["objects"].items()
        }
        with self.lock:
            self.observations = values

    def start(self):
        io = self._SimulationRobot(self.state) if self.simulate else self.robot
        self._motor = MotorLoop(
            io,
            self._queue,
            lambda receipt: None,
            on_tick=self._record_motor_tick,
            clock=time.perf_counter,
        )
        self.thread = Thread(target=self._run, name="flash-editor-motors", daemon=True)
        self.thread.start()

    def _record_motor_tick(self, tick):
        self._motor_trace_total += 1
        self._motor_trace.append(
            {
                "tick": tick.tick,
                "scheduled_s": tick.scheduled_s,
                "observed_at_s": tick.observed_at_s,
                "observed": dict(tick.observed),
                "requested": dict(tick.receipt.requested.action) if tick.receipt else None,
                "accepted": dict(tick.receipt.accepted) if tick.receipt and tick.receipt.accepted else None,
            }
        )

    def _current_observations(self):
        with self.lock:
            return dict(self.observations)

    def _target_changed(self, observations):
        if self._planned is None or self._future is not None or self.index >= self._planned.next_index:
            return False
        first = self.plan["keyframes"][self.index]
        if first["reference"] == "world":
            return False
        # Once a demo changes its gripper command, the range may carry a
        # grasped object. Keep the range-entry object frame through that motion.
        range_first = self.plan["keyframes"][self._range_start_index]
        if any(
            key["gripper"] != range_first["gripper"]
            for key in self.plan["keyframes"][self._range_start_index : self.index]
        ):
            return False
        observed = observations.get(first["reference"])
        if not observed or observed.get("pose") is None or observed["status"] not in POSE_AVAILABLE:
            return False
        return not np.array_equal(
            np.asarray(self.plan["base_from_camera"]) @ transform(observed["pose"]),
            self._planned.reference_pose,
        )

    def _submit(self):
        if self.index >= len(self.plan["keyframes"]):
            self.status = "complete"
            return
        observations = self._current_observations()
        first = self.plan["keyframes"][self.index]
        if first["reference"] != "world":
            observed = observations.get(first["reference"])
            if not observed or observed.get("pose") is None or observed["status"] not in POSE_AVAILABLE:
                if self._planned is None:
                    self.status = f"waiting: {first['reference']} needs a pose"
                return
        self._future = self._planner.submit(
            plan_demo_range,
            self.plan,
            self.model,
            self.index,
            dict(self.state),
            observations,
            time.perf_counter(),
        )
        if self._planned is None:
            self.status = "planning"

    def _publish_ready(self):
        if self._future is None or not self._future.done():
            return
        future, self._future = self._future, None
        planned = future.result()
        if self.index >= planned.next_index:
            return  # The motor finished this range during planning.
        actions = planned.actions
        if self._planned is not None:
            # The old queue kept moving while IK ran. Drop every keyframe it
            # already sent. Preserve its remaining entry time; otherwise a
            # new camera result on every frame would restart the approach.
            next_frame = self.plan["keyframes"][self.index]["frame"]
            remaining = [
                action
                for action in actions
                if action.interval.startswith("demo:")
                and int(action.interval.removeprefix("demo:")) >= next_frame
            ]
            if not remaining:
                return
            pending = self._queue.snapshot().pending
            entry_ticks = sum(action.interval.startswith("entry:") for action in pending)
            bridge = []
            if entry_ticks:
                start = {name: float(self.state[name]) for name in remaining[0].action}
                goal = remaining[0].action
                bridge = [
                    TimedAction(
                        tick=0,
                        action={
                            name: start[name] + step / entry_ticks * (goal[name] - start[name])
                            for name in goal
                        },
                        interval=f"entry:{next_frame}",
                    )
                    for step in range(1, entry_ticks + 1)
                ]
            actions = (*bridge, *remaining)
        first_tick = self._motor.metrics.ticks
        self._queue.publish(
            self._revision,
            [replace(action, tick=first_tick + offset) for offset, action in enumerate(actions)],
        )
        self._revision += 1
        if self._range_start_index is None:
            self._range_start_index = planned.start_index
        self._planned = planned
        self.evidence = planned.evidence
        self.target = planned.target
        self.status = "moving"
        if self._range_first_command_s is None:
            atomic_json(
                self.directory / "preflight.json",
                {
                    "keyframe": self.index,
                    "state": {name: float(value) for name, value in self.state.items()},
                    "evidence": self.evidence,
                    "commands": len(actions),
                },
            )

    def _record_status(self, now):
        tool = self.model.forward(self.state)
        self._measurements.append((now, tool[:3, 3].copy()))
        speed = None
        if len(self._measurements) > 1:
            elapsed = self._measurements[-1][0] - self._measurements[0][0]
            if elapsed > 0:
                positions = np.asarray([point for _, point in self._measurements])
                speed = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum() / elapsed)
        atomic_json(
            self.directory / "execution.json",
            {
                "mode": "simulation" if self.simulate else "robot",
                "state": self.status,
                "reason": str(self.error) if self.error else None,
                "keyframe": self.index,
                "total": len(self.plan["keyframes"]),
                "commands": self.commands_sent,
                "tool": tool.tolist(),
                "target": self.target.tolist() if self.target is not None else None,
                "measured_tool_speed_m_s": speed,
                "last_range": self._completed_ranges[-1] if self._completed_ranges else None,
                "completed_ranges": self._completed_ranges,
                "motor": {
                    "late_ticks": self._motor.metrics.late_ticks,
                    "skipped_ticks": self._motor.metrics.skipped_ticks,
                    "max_dispatch_gap_s": self._motor.metrics.max_dispatch_gap_s,
                },
                "updated_s": now,
                "evidence": self.evidence,
            },
        )

    def _finish_range(self):
        planned = self._planned
        self._completed_ranges.append(
            {
                "reference_frame": self.plan["keyframes"][planned.start_index]["reference_frame"],
                "commands": self.commands_sent,
                "recorded_demo_path_m": planned.demo_path_m,
                "recorded_demo_duration_s": planned.demo_duration_s,
                "commanded_demo_duration_s": (
                    self._demo_last_command_s - self._demo_first_command_s
                    if self._demo_first_command_s is not None
                    else 0.0
                ),
            }
        )
        self._planned = None
        self._range_start_index = None
        self._awaiting_final_read = False
        self._range_first_command_s = None
        self._demo_first_command_s = None
        self._demo_last_command_s = None
        self.target = None
        self.status = "complete" if self.index == len(self.plan["keyframes"]) else "ready"

    def _run(self):
        try:
            if self.stop_event.is_set():
                return
            if not self.simulate:
                if not self.rest_position:
                    raise ValueError("Save a rest position in the robot profile before running a demo")
                self.status = "connecting"
                self._record_status(time.perf_counter())
                self.robot.connect(calibrate=False)
            self.state = self._motor.robot.get_state_observation()
            if self.rest_position and not self.stop_event.is_set():
                self.status = "resetting to rest position"
                self._record_status(time.perf_counter())
                move_to_rest_position(self._motor.robot, self.rest_position, stop_event=self.stop_event)
                # Plan the approach from measured joints, including any lag or
                # clipping during reset, never from the requested rest target.
                self.state = self._motor.robot.get_state_observation()
            if self.stop_event.is_set():
                return
            self.status = "ready"
            self._motor.start()
            last_status_write = 0.0
            while not self.stop_event.is_set():
                if self._future is not None:
                    self._publish_ready()
                elif (
                    self.status != "complete"
                    and not self._awaiting_final_read
                    and (self._planned is None or self._target_changed(self._current_observations()))
                ):
                    self._submit()
                receipt = self._motor.tick()
                if self._motor.last_observed is not None:
                    self.state = self._motor.last_observed
                if receipt is not None:
                    self.commands_sent += 1
                    self._range_first_command_s = self._range_first_command_s or receipt.dispatched_s
                    interval = receipt.requested.interval
                    if interval.startswith("demo:"):
                        self._demo_first_command_s = self._demo_first_command_s or receipt.dispatched_s
                        self._demo_last_command_s = receipt.dispatched_s
                        self.index = self._frame_to_index[int(interval.removeprefix("demo:"))] + 1
                        if receipt.requested.interval_complete:
                            self._awaiting_final_read = True
                            self.status = "reading final state"
                if self._awaiting_final_read and receipt is None:
                    self._finish_range()
                if time.perf_counter() - last_status_write >= 0.2:
                    self._record_status(time.perf_counter())
                    last_status_write = time.perf_counter()
        except Exception as exc:
            self.error = exc
            self.status = "failed"
            self.stop_event.set()
        finally:
            self._planner.shutdown(wait=False, cancel_futures=True)
            try:
                if not self.simulate and self.robot.is_connected:
                    self.robot.disconnect()
            finally:
                if self.error is None:
                    self.status = "stopped"
                # Startup can fail before a valid joint read. Publishing the
                # terminal state must not require FK or delay motor cleanup.
                atomic_json(
                    self.directory / "execution.json",
                    {
                        "mode": "simulation" if self.simulate else "robot",
                        "state": self.status,
                        "reason": str(self.error) if self.error else None,
                        "keyframe": self.index,
                        "total": len(self.plan["keyframes"]),
                        "commands": self.commands_sent,
                    },
                )
                atomic_json(
                    self.directory / "motor_trace.json",
                    {
                        "total_ticks": self._motor_trace_total,
                        "samples": list(self._motor_trace),
                    },
                )

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.thread is None and not self.simulate and self.robot.is_connected:
            self.robot.disconnect()


def main():
    from lerobot.gui.api.robot import _make_robot_from_profile

    directory = Path(sys.argv[1])
    config = json.loads((directory / "config.json").read_text())
    setup = config["demo_execution"]
    folder = Path(setup["plan_directory"])
    plan = json.loads((folder / "plan.json").read_text())
    load_reference(folder / "reference.npz", [obj["id"] for obj in plan["document"]["objects"]])
    robot = _make_robot_from_profile(setup["robot"])
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    motor = ExecutionLoop(
        plan,
        robot,
        directory,
        stop,
        simulate=setup["simulate"],
        rest_position=setup["robot"].get("rest_position"),
    )

    def session_factory(settings, source, model, kernels, pool):
        objects = plan["document"]["objects"]
        source = Sam3MaskSource(
            {obj["id"]: obj["name"] for obj in objects},
            camera_id=plan["document"]["camera"],
            resolution=source.resolution,
            adapter_factory=source.adapter_factory,
            seed_points_by_role={obj["id"]: obj["point"] for obj in objects if obj["point"] is not None},
            detection_correction=source.detection_correction,
            click_sequence_start=source.next_click_sequence,
        )
        return DemoRunTracker(
            plan, folder / "reference.npz", source, model=model, kernels=kernels, search_pool=pool
        )

    try:
        motor.start()
        run(directory, config, session_factory=session_factory, on_observation=motor.observe, stop_event=stop)
        if motor.error:
            raise motor.error
    finally:
        motor.close()


if __name__ == "__main__":
    main()
