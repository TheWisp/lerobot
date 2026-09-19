"""Taps for the live-video tests: a synthetic one, and one of real footage.

The synthetic tap is the fast fixture: four cameras at the rig's sizes and
rate, with content that costs the encoder something. What it cannot say is
what the profile costs on a real scene, because a gradient with a moving
square is not a room with an arm in it — and bytes follow content.

The footage tap plays a recorded episode of a real SO-101 through the same
shared memory a run writes: real cameras, real motion, real joint
trajectories, at the dataset's own frame rate. It reads a public dataset
from the local cache, read-only, and skips when it is not there.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.obs_stream import ObservationStream

FPS = 30

#: The rig's camera set and sizes, for the synthetic tap.
CAMERAS = {
    "top": (720, 1280, 3),
    "front": (600, 960, 3),
    "left_wrist": (600, 960, 3),
    "right_wrist": (720, 1280, 3),
}

#: The rig's own motor names, so what reads a tap resolves a robot the way it
#: will in a run rather than a shape only tests have.
JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

#: A public recording of an SO-101 picking and placing: two cameras at
#: 640×480, 30 fps, and this motor set. Read from the local cache only —
#: these tests never reach the Hub, and never write to it.
FOOTAGE_REPO = "lerobot/svla_so101_pickplace"
FOOTAGE_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "lerobot" / "svla_so101_pickplace"


class _TapWriter:
    """Writes one cycle per period into the tap, from whatever supplies frames."""

    def __init__(self, obs_features: dict, fps: int) -> None:
        self.fps = fps
        self.stream = ObservationStream(obs_features, dict.fromkeys(JOINTS, float))
        self.cycles_written = 0
        self.written_at: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tap-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.time() + 10.0
        while self.cycles_written < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert self.cycles_written >= 2, "the tap wrote nothing"

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)
        self.stream.cleanup()

    def cycle(self, i: int) -> dict:
        raise NotImplementedError

    def _run(self) -> None:
        period = 1.0 / self.fps
        t_next = time.perf_counter()
        while not self._stop.is_set():
            i = self.cycles_written
            self.stream.mark_observation_start()
            obs = self.cycle(i)
            obs["cycle"] = float(i + 1)
            self.stream.write_obs(obs)
            self.stream.write_action({j: obs.get(j, 0.0) for j in JOINTS})
            self.cycles_written = i + 1
            self.written_at.append(time.time())
            t_next += period
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def written_between(self, t0: float, t1: float) -> int:
        return sum(1 for t in self.written_at if t0 <= t <= t1)


class FootageTap(_TapWriter):
    """A recorded episode played into the tap, as a run would write it.

    Every camera in the recording becomes a camera of the run, at the
    recording's own resolution; the joints are the recording's own, so the
    visualizer draws the arm that was there.
    """

    def __init__(self, frames: int = 300, episode: int = 0, fps: int = FPS) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(FOOTAGE_REPO, root=FOOTAGE_ROOT)
        self.cameras = [k.split(".")[-1] for k in dataset.meta.camera_keys]
        start = int(dataset.meta.episodes["dataset_from_index"][episode])
        end = int(dataset.meta.episodes["dataset_to_index"][episode])
        count = min(frames, end - start)
        assert count >= 30, f"episode {episode} has {end - start} frames"

        # Read the whole slice up front: what is measured is the stream, not
        # a video decoder racing it.
        self._frames: list[dict[str, np.ndarray]] = []
        self._joints: list[dict[str, float]] = []
        for f in range(start, start + count):
            item = dataset[f]
            self._frames.append(
                {
                    name: np.ascontiguousarray((item[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    for name, key in zip(self.cameras, dataset.meta.camera_keys, strict=True)
                }
            )
            state = item["observation.state"].tolist()
            self._joints.append(dict(zip(JOINTS, state, strict=False)))

        shapes = {name: self._frames[0][name].shape for name in self.cameras}
        obs_features = {"cycle": float, **dict.fromkeys(JOINTS, float), **shapes}
        super().__init__(obs_features, fps)

    def cycle(self, i: int) -> dict:
        at = i % len(self._frames)
        return {**self._joints[at], **self._frames[at]}


def footage_tap(frames: int = 300) -> FootageTap:
    """A footage tap, or a skip when the recording is not on this machine."""
    if not (FOOTAGE_ROOT / "meta" / "info.json").exists():
        pytest.skip(
            f"{FOOTAGE_REPO} is not in the local cache; "
            f"fetch it once with `huggingface-cli download --repo-type dataset {FOOTAGE_REPO}`"
        )
    return FootageTap(frames=frames)
