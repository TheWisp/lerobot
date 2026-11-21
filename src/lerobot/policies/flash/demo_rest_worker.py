"""Explicit motor-only return to saved rest, owned by the GUI's Run process."""

import json
import signal
import sys
from pathlib import Path
from threading import Event

from lerobot.robots.rest_position import move_to_rest_position

from .demo_document import atomic_json
from .rollout_recording import RecordedRobot, RunJournal


def return_to_rest(directory, profile, stop, make_robot):
    journal = RunJournal(directory)
    robot = None
    failure = None
    state = "stopped"

    def publish(value):
        atomic_json(
            directory / "execution.json",
            {
                "operation": "rest",
                "mode": "robot",
                "state": value,
                "reason": str(failure) if failure else None,
            },
        )
        journal.emit("execution.state", state=value, reason=str(failure) if failure else None)
        print(str(failure) if failure else value, flush=True)

    try:
        if not profile.get("rest_position"):
            raise ValueError("Save a rest position in the robot profile before returning to rest")
        if not stop.is_set():
            robot = make_robot(profile)  # Motor-only factory; no camera or tracker.
            publish("connecting")
            robot.connect(calibrate=False)
            if not stop.is_set():
                publish("returning to saved rest")
                move_to_rest_position(
                    RecordedRobot(robot, journal), profile["rest_position"], stop_event=stop
                )
                state = "stopped" if stop.is_set() else "complete"
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if robot is not None and robot.is_connected:
                robot.disconnect()
        except BaseException as exc:
            failure = failure or exc
        try:
            publish("failed" if failure else state)
        finally:
            journal.close()
    if failure:
        raise failure


def main():
    from lerobot.gui.api.robot import _make_robot_from_profile

    directory = Path(sys.argv[1])
    profile = json.loads((directory / "config.json").read_text())["robot"]
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    return_to_rest(directory, profile, stop, _make_robot_from_profile)


if __name__ == "__main__":
    main()
