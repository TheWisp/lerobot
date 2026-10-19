"""Record FLASH attempts as ordinary LeRobot episodes plus a trace sidecar."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Any, Protocol

from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .execution import ActionReceipt
from .trajectory import PoseEstimate


class DatasetWriter(Protocol):
    root: Path
    features: dict

    def add_frame(self, frame: dict) -> None: ...

    def save_episode(self) -> None: ...


@dataclass(frozen=True)
class RecordedTick:
    receipt: ActionReceipt
    camera_observation: dict[str, Any]
    camera_source_time_s: float | None
    pose_version: int | None
    progress_events: tuple[dict, ...] = ()
    camera_frame_id: int | None = None
    pose_estimates: dict[str, PoseEstimate] = field(default_factory=dict)


class FlashEpisodeWriter:
    """Write normal observations/actions with a joinable decision trace.

    This object belongs to the recorder worker, never to the motor tick. Its
    dataset is created through ``LeRobotDataset.create`` or ``resume`` by the
    host rollout. Missing image/action features fail instead of producing a
    silently incomplete training episode.
    """

    def __init__(self, dataset: DatasetWriter, episode_index: int, task: str) -> None:
        self.dataset = dataset
        self.episode_index = episode_index
        self.task = task
        self.frame_index = 0
        self.extension_dir = Path(dataset.root) / "flash" / "v1"
        self.extension_dir.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.extension_dir / f"episode_{episode_index:06d}.jsonl.partial"
        self.path = self.extension_dir / f"episode_{episode_index:06d}.jsonl"
        self._trace = self.partial_path.open("w", encoding="utf-8")

    def write(self, record: RecordedTick) -> None:
        receipt = record.receipt
        observation = {**receipt.observed, **record.camera_observation}
        obs_frame = build_dataset_frame(self.dataset.features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.dataset.features, receipt.accepted, prefix=ACTION)
        self.dataset.add_frame({**obs_frame, **action_frame, "task": self.task})
        trace = {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "motor_tick": receipt.tick,
            "scheduled_s": receipt.scheduled_s,
            "dispatched_s": receipt.dispatched_s,
            "camera_source_time_s": record.camera_source_time_s,
            "camera_frame_id": record.camera_frame_id,
            "pose_version": record.pose_version,
            "poses": {
                role: {
                    "pose": estimate.pose,
                    "source_time_s": estimate.source_time_s,
                    "available_time_s": estimate.available_time_s,
                    "evidence": estimate.evidence.value,
                    "reference_version": estimate.reference_version,
                    "translation_supported": estimate.translation_supported,
                    "rotation_supported": estimate.rotation_supported,
                }
                for role, estimate in record.pose_estimates.items()
            },
            "interval": receipt.requested.interval,
            "interval_complete": receipt.requested.interval_complete,
            "plan_revision": receipt.revision,
            "event_id": receipt.requested.event_id,
            "requested": receipt.requested.action,
            "accepted": receipt.accepted,
            "read_duration_s": receipt.read_duration_s,
            "write_duration_s": receipt.write_duration_s,
            "progress_events": record.progress_events,
        }
        self._trace.write(json.dumps(trace, default=_json_value) + "\n")
        self.frame_index += 1

    def save(self, outcome: str, reason: str) -> None:
        if outcome not in {"success", "failure", "unknown", "interrupted"}:
            raise ValueError("outcome must be explicit")
        self.dataset.save_episode()
        self._trace.close()
        os.replace(self.partial_path, self.path)
        manifest_path = self.extension_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {"schema_version": 1, "episodes": {}}
        )
        manifest["episodes"][str(self.episode_index)] = {
            "trace": self.path.name,
            "frames": self.frame_index,
            "outcome": outcome,
            "reason": reason,
        }
        staged = manifest_path.with_suffix(".json.tmp")
        staged.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staged, manifest_path)


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


class RecorderWorker:
    """Bounded handoff: disk/video encoding cannot hold the motor tick."""

    def __init__(self, writer: FlashEpisodeWriter, capacity: int = 256) -> None:
        self.writer = writer
        self._queue: Queue[RecordedTick | None] = Queue(maxsize=capacity)
        self._error: Exception | None = None
        self._thread = Thread(target=self._run, name="flash-dataset-writer", daemon=False)

    def start(self) -> None:
        self._thread.start()

    def submit(self, record: RecordedTick) -> None:
        if self._error:
            raise RuntimeError("FLASH recording worker failed") from self._error
        try:
            self._queue.put_nowait(record)
        except Full as exc:
            raise RuntimeError(
                "FLASH recording queue overflow; episode cannot be considered complete"
            ) from exc

    def finish(self, outcome: str, reason: str, timeout_s: float = 30) -> None:
        self._queue.put(None, timeout=timeout_s)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("FLASH recording worker did not stop")
        if self._error:
            raise RuntimeError("FLASH recording worker failed") from self._error
        self.writer.save(outcome, reason)

    def _run(self) -> None:
        try:
            while (record := self._queue.get()) is not None:
                self.writer.write(record)
        except Exception as exc:
            self._error = exc
