"""Send already planned reset segments through the GUI's single hardware owner.

No serial connection, implicit approach ramp, or automatic park. A fresh scene
and the appropriate attachment/contact model must be supplied by the reset
runner. An interrupted client loses its one-second execution lease.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import replace

import numpy as np
from motion_planner import NoPathError


class ExecutionStoppedError(RuntimeError):
    pass


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))


def checked_samples(segment, scene, start):
    """Bridge from measured arrival and check EVERY resampled joint/tip step.

    A stopped jaw may be more open than its closing command. Check all openings
    between the measured and commanded grip during transit, not just the
    fictitious fully closed mesh. The command itself remains the closing goal.
    """
    robot = scene.robot
    observed, commanded = start
    observed, commanded = np.asarray(observed, float), np.asarray(commanded, float)
    rows = np.asarray(segment.joints, float)
    if not np.isfinite([observed, commanded]).all() or not np.isfinite(rows).all():
        raise NoPathError("nonfinite path/start")
    if np.max(np.abs(rows[0, :-1] - observed[:-1])) > 0.7:
        raise NoPathError("arm moved away from the previous segment; replan")
    if abs(rows[0, -1] - commanded[-1]) > 0.25:
        raise NoPathError("segment would change the grasp command at handoff")
    first = observed.copy()
    first[-1] = commanded[-1]
    samples = [first]
    contact = segment.phase == "close"
    grip_changes = bool(np.any(np.diff(rows[:, -1])))
    if grip_changes:
        # A stationary gripper operation uses the actual settled arm pose.
        # Never blend a nominal arm endpoint and the opening in the same step.
        if np.max(np.abs(rows[:, :-1] - rows[0, :-1])) > 1e-8:
            raise NoPathError("mixed arm and gripper motion")
        rows = rows.copy()
        rows[:, :-1] = first[:-1]
    waypoints = np.concatenate(([first], rows))

    def check(q):
        openings = [q[-1]]
        if segment.attached or contact:
            openings = np.linspace(
                min(observed[-1], q[-1]),
                max(observed[-1], q[-1]),
                max(2, int(np.ceil(abs(observed[-1] - q[-1]))) + 1),
            )
        for opening in openings:
            at = q.copy()
            at[-1] = opening
            if not scene.check(
                at,
                attached=segment.attached,
                contact=segment.contact,
                near=segment.near,
                support=segment.support,
            ):
                raise NoPathError(f"{segment.phase}: {scene.reason}")

    check(first)
    for a, b in zip(waypoints, waypoints[1:], strict=False):
        # Joint spacing alone does not bound Cartesian travel. Densify until
        # both limits hold, then collision-check the newly introduced samples.
        n = max(1, int(np.ceil(np.max(np.abs(b - a) / np.r_[np.full(len(a) - 1, 0.5), 1.0]))))
        for _ in range(16):
            between = np.linspace(a, b, n + 1)
            poses = [robot.tip(q) for q in between]
            if all(
                np.linalg.norm(y[:3, 3] - x[:3, 3]) <= 0.002 and _angle(x, y) <= 2
                for x, y in zip(poses, poses[1:], strict=False)
            ):
                break
            n *= 2
        else:
            raise NoPathError("cannot bound tip steps")
        for q in between[1:]:
            check(q)
            if not np.array_equal(q, samples[-1]):
                samples.append(q)
    if len(samples) == 1:
        samples.append(samples[0].copy())
    if len(samples) > 6000:
        raise NoPathError("segment needs too many samples")
    return replace(segment, joints=np.asarray(samples))


class GuiExecutor:
    """The reset owns the path token until explicit finish or a frozen stop.

    ``scene`` must remain unchanged while a segment runs. The caller checks a
    fresh camera view between phases; this class monitors encoder geometry.
    """

    def __init__(self, scene, log, call=None, stopped=lambda: False):
        self.scene, self.log, self.stopped = scene, log, stopped
        self.call = self._http if call is None else call
        self.identity = None

    @staticmethod
    def _http(path, body=None):
        request = urllib.request.Request(
            "http://127.0.0.1:9100/api/jog/path/" + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return json.loads(response.read())
        except (OSError, urllib.error.HTTPError) as e:
            raise ExecutionStoppedError(f"path API {path}: {e}") from e

    def _read(self, state, segment=None):
        self.log({"at": time.time(), "state": state})
        if self.stopped():
            raise ExecutionStoppedError("reset stop requested")
        age = state.get("read_age_s")
        if not state.get("connected") or state.get("halted") or age is None or not 0 <= age < 0.25:
            raise ExecutionStoppedError(f"arm telemetry unavailable, stale or frozen: {state.get('reason')}")
        names = self.scene.robot.names
        try:
            observed = np.array([state["q_obs"][n] for n in names], float)
            commanded = np.array([state["q_cmd"][n] for n in names], float)
        except (KeyError, TypeError, ValueError) as e:
            raise ExecutionStoppedError("incomplete encoder/command state") from e
        if not np.isfinite([observed, commanded]).all():
            raise ExecutionStoppedError("nonfinite encoder/command state")
        if segment is not None and not self.scene.check(
            observed,
            attached=segment.attached,
            contact=segment.contact,
            near=segment.near,
            support=segment.support,
        ):
            raise ExecutionStoppedError(f"observed arm/payload: {self.scene.reason}")
        return observed, commanded

    def run(self, segment):
        """Run one checked phase; no unchecked ramp if the arm misses a waypoint."""
        try:
            state = self.call("state")
            start = self._read(state, segment)
            checked = checked_samples(segment, self.scene, start)
            # Collision sampling can take time. Require a fresh start again;
            # the GUI performs another encoder read before acquiring the path.
            state = self.call("state")
            current, _ = self._read(state, segment)
            if np.max(np.abs(current[:-1] - checked.joints[0, :-1])) > 0.25:
                raise ExecutionStoppedError("arm moved during collision validation")
            state = self.call(
                "start",
                {
                    "samples": checked.joints.tolist(),
                    "tip_poses": [self.scene.robot.tip(q).tolist() for q in checked.joints],
                    "grip_contact": segment.phase == "close",
                    "previous_id": self.identity,
                },
            )
            self.identity = state["path"]["id"]
            started = time.monotonic()
            while True:
                self._read(state, segment)
                path = state.get("path") or {}
                if path.get("id") != self.identity or path.get("status") == "failed":
                    raise ExecutionStoppedError(f"path stopped or ownership changed: {path.get('reason')}")
                if path.get("status") == "done":
                    return self._read(state, segment)
                if time.monotonic() - started > 190:
                    raise ExecutionStoppedError("segment wall-clock timeout")
                # Only renew after checking fresh actual arm/payload geometry.
                state = self.call("heartbeat", {"id": self.identity})
                time.sleep(0.05)
                state = self.call("state")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        # Best effort: an unreachable server still expires its lease. Never
        # auto-park through the obstacle that stopped this checked trajectory.
        if self.identity is not None:
            try:
                self.call("stop", {"id": self.identity})
            except Exception as e:
                self.log({"stop_error": str(e), "at": time.time()})

    def finish(self):
        if self.identity is None:
            return
        self._read(self.call("state"))
        self.call("finish", {"id": self.identity})
        self.identity = None
