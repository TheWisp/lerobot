"""Editor execution adapter around the existing live camera/depth/tracker worker."""

import json
import signal
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread

from lerobot.gui.flash_live_scene import object_evidence

from .appearance import Sam3MaskSource
from .demo_document import atomic_json
from .demo_execution import DemoRunTracker, KeyframeExecutor, load_reference
from .live_surface_session import LIVE_METHOD
from .live_surface_worker import run


class ExecutionLoop:
    """Latest perception, fixed motor rate; simulation never connects a robot."""

    def __init__(self, plan, robot, directory, stop, *, simulate):
        self.robot, self.directory, self.stop_event = robot, directory, stop
        self.simulate = simulate
        self.model = robot.cartesian_model()
        if (
            robot.geometry.to_dict() != plan["robot_geometry"]
            or self.model.description() != plan["robot_cartesian"]
        ):
            raise ValueError("Robot geometry or joint units changed since preparation")
        self.executor = KeyframeExecutor(plan, self.model)
        self.state = dict(plan["keyframes"][0]["state"])
        self.observations = {}
        self.lock = Lock()
        self.thread = None
        self.error = None
        self.commands_sent = 0

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
        # Robot model construction above is connection-free. Motor access is
        # confined to this explicit branch; the test run uses the same solver.
        if not self.simulate:
            self.robot.connect(calibrate=False)
        self.thread = Thread(target=self._run, name="flash-editor-motors", daemon=True)
        self.thread.start()

    def tick(self, now, dt):
        if self.stop_event.is_set():
            return
        if not self.simulate:
            self.state = self.robot.get_state_observation()
        with self.lock:
            observations = self.observations
        action = self.executor.step(now, dt, self.state, observations)
        if action is not None and not self.stop_event.is_set():
            if self.simulate:
                self.state.update(action)
            else:
                self.robot.send_action(action)
            self.commands_sent += 1
        return action

    def _run(self):
        period, previous, deadline = 1 / 30, time.perf_counter(), time.perf_counter()
        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                self.tick(now, now - previous)
                previous = now
                atomic_json(
                    self.directory / "execution.json",
                    {
                        "mode": "simulation" if self.simulate else "robot",
                        "state": self.executor.status,
                        "keyframe": self.executor.index,
                        "total": len(self.executor.plan["keyframes"]),
                        "commands": self.commands_sent,
                        "tool": self.model.forward(self.state).tolist(),
                        "target": self.executor.target.tolist() if self.executor.target is not None else None,
                        "updated_s": now,
                        "evidence": self.executor.evidence,
                    },
                )
                deadline += period
                if deadline < time.perf_counter():
                    deadline = time.perf_counter()
                self.stop_event.wait(max(0, deadline - time.perf_counter()))
        except Exception as exc:
            self.error = exc
            atomic_json(self.directory / "execution.json", {"state": "failed", "reason": str(exc)})
            self.stop_event.set()
        finally:
            # Stop motors independently of camera/history shutdown, which may
            # still be flushing data. The motor thread owns its serial I/O.
            if not self.simulate and self.robot.is_connected:
                self.robot.disconnect()

    def close(self):
        self.stop_event.set()
        self.executor.stop()
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
    motor = ExecutionLoop(plan, robot, directory, stop, simulate=setup["simulate"])

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
