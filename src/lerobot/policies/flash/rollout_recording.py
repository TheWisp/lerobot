"""Durable, independently timed evidence for a demo application.

Camera payloads live beside this append-only journal. Producers hand off owned
snapshots; filesystem work never runs in the motor callback. An overflow is a
recording failure, not permission to silently discard an event.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from queue import Full, Queue
from threading import Lock, Thread

from .demo_document import atomic_json


def _json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot record {type(value).__name__}")


class RunJournal:
    def __init__(self, directory, *, capacity=4096):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.queue = Queue(maxsize=capacity)
        self.lock = Lock()
        self.sequence = 0
        self.error = None
        self.closed = False
        self.thread = Thread(target=self._write, name="flash-run-journal", daemon=False)
        self.thread.start()

    def emit(self, kind, **data):
        with self.lock:
            if self.error:
                raise RuntimeError("Rollout recording failed") from self.error
            if self.closed:
                raise RuntimeError("Rollout journal is closed")
            event = {"sequence": self.sequence, "time_s": time.perf_counter(), "kind": kind, **data}
            # Serialize here to detach mutable controller/tracker state. Only
            # small metadata enters the journal; pixels use the camera writer.
            encoded = json.dumps(event, default=_json_value, allow_nan=False, separators=(",", ":"))
            try:
                self.queue.put_nowait(encoded)
            except Full as exc:
                self.error = exc
                raise RuntimeError("Rollout event writer fell behind; recording is incomplete") from exc
            self.sequence += 1

    def _write(self):
        try:
            with (self.directory / "events.jsonl").open("a") as stream:
                while (line := self.queue.get()) is not None:
                    stream.write(line + "\n")
                    stream.flush()
        except Exception as exc:
            self.error = exc

    def close(self):
        with self.lock:
            self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except Full:
                continue
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Rollout event writer did not finish")
        if self.error:
            raise RuntimeError("Rollout event recording is incomplete") from self.error


class RecordedRobot:
    """Include reset motion and unsuccessful I/O, outside MotorLoop as well."""

    def __init__(self, robot, journal):
        self.robot, self.journal = robot, journal

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def _read(self, method):
        started = time.perf_counter()
        try:
            state = getattr(self.robot, method)()
        except Exception as exc:
            self.journal.emit("robot.read_failed", started_s=started, reason=str(exc))
            raise
        self.journal.emit("robot.read", started_s=started, observed=state)
        return state

    def get_state_observation(self):
        return self._read("get_state_observation")

    def get_observation(self):
        return self._read("get_observation")

    def send_action(self, action):
        self.journal.emit("robot.command_requested", requested=action)
        started = time.perf_counter()
        try:
            accepted = self.robot.send_action(action)
        except Exception as exc:
            self.journal.emit("robot.command_failed", started_s=started, reason=str(exc))
            raise
        self.journal.emit("robot.command_accepted", started_s=started, accepted=accepted)
        return accepted


def finish_run(directory, *, state, reason=None, commands=0, keyframe=0):
    path = Path(directory) / "run.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    manifest.update(state=state, reason=reason, ended_at=time.time(), commands=commands, keyframe=keyframe)
    # Issuing the entire path does not establish physical task success.
    manifest["outcome"] = "unknown"
    atomic_json(path, manifest)
