"""Record FLASH attempts as ordinary LeRobot episodes plus a trace sidecar."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Any, Protocol

from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .execution import ActionReceipt, MotorTick
from .perception import LatestPoseStore, LatestRGBDFrameStore
from .pose_disagreement import PoseComparison, PoseComparisonVerdict, SurfaceSupport
from .trajectory import Evidence, PoseEstimate


class DatasetWriter(Protocol):
    root: Path
    features: dict

    def add_frame(self, frame: dict) -> None: ...

    def save_episode(self) -> None: ...


@dataclass(frozen=True)
class RecordedTick:
    receipt: ActionReceipt | None
    camera_observation: dict[str, Any]
    camera_source_time_s: float | None
    pose_version: int | None
    progress_events: tuple[dict, ...] = ()
    camera_frame_id: int | None = None
    pose_estimates: dict[str, PoseEstimate] = field(default_factory=dict)
    pose_comparisons: dict[str, PoseComparison] = field(default_factory=dict)
    motor_tick: MotorTick | None = None


class TickRecordAssembler:
    """Join one motor tick to already captured camera and pose snapshots.

    This runs in the motor owner's callback, so it only reads short-lock
    snapshots and submits references to immutable camera arrays. Dataset
    encoding remains in ``RecorderWorker``. A missing or stale RGB-D frame
    fails the episode instead of writing a training row without its image.
    """

    def __init__(
        self,
        frames: LatestRGBDFrameStore,
        poses: LatestPoseStore,
        *,
        rgb_feature: str,
        depth_feature: str | None,
        max_camera_age_s: float,
        progress_events: Callable[[], list[dict]],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not rgb_feature or (depth_feature is not None and not depth_feature):
            raise ValueError("camera feature names must be nonempty")
        if not math.isfinite(max_camera_age_s) or max_camera_age_s <= 0:
            raise ValueError("maximum camera age must be finite and positive")
        self.frames = frames
        self.poses = poses
        self.rgb_feature = rgb_feature
        self.depth_feature = depth_feature
        self.max_camera_age_s = max_camera_age_s
        self.progress_events = progress_events
        self.clock = clock
        self._event_cursor = 0

    def assemble(self, tick: MotorTick) -> RecordedTick:
        now_s = self.clock()
        frame = self.frames.snapshot(no_later_than_s=now_s)
        if frame is None or not 0 <= now_s - frame.source_time_s <= self.max_camera_age_s:
            raise RuntimeError("no fresh aligned RGB-D frame for motor tick recording")
        version, poses = self.poses.snapshot()
        visible_poses = {
            role: estimate for role, estimate in poses.items() if estimate.available_time_s <= now_s
        }
        events = self.progress_events()
        delta = tuple(events[self._event_cursor :])
        self._event_cursor = len(events)
        camera = {self.rgb_feature: frame.rgb}
        if self.depth_feature is not None:
            camera[self.depth_feature] = frame.depth
        return RecordedTick(
            receipt=tick.receipt,
            camera_observation=camera,
            camera_source_time_s=frame.source_time_s,
            pose_version=version,
            progress_events=delta,
            camera_frame_id=frame.frame_id,
            pose_estimates=visible_poses,
            motor_tick=tick,
        )


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
        self.timing_regular = True

    def write(self, record: RecordedTick) -> None:
        tick = record.motor_tick
        receipt = record.receipt if record.receipt is not None else (tick.receipt if tick else None)
        if tick is None and receipt is None:
            raise ValueError("record needs a motor tick or action receipt")
        observed = tick.observed if tick is not None else receipt.observed
        effective_action = tick.effective_action if tick is not None else receipt.requested.action
        observation = {**observed, **record.camera_observation}
        obs_frame = build_dataset_frame(self.dataset.features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.dataset.features, effective_action, prefix=ACTION)
        self.dataset.add_frame({**obs_frame, **action_frame, "task": self.task})
        trace = {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "motor_tick": tick.tick if tick is not None else receipt.tick,
            "skipped_before": tick.skipped_before if tick is not None else 0,
            "scheduled_s": tick.scheduled_s if tick is not None else receipt.scheduled_s,
            "dispatched_s": receipt.dispatched_s if receipt is not None else None,
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
            "pose_comparisons": {
                role: {
                    "verdict": comparison.verdict.value,
                    "object_center_difference_m": comparison.object_center_difference_m,
                    "rotation_difference_rad": comparison.rotation_difference_rad,
                    "candidate": _surface_support_json(comparison.candidate),
                    "prior": _surface_support_json(comparison.prior),
                }
                for role, comparison in record.pose_comparisons.items()
            },
            "interval": receipt.requested.interval if receipt is not None else None,
            "interval_complete": receipt.requested.interval_complete if receipt is not None else False,
            "plan_revision": receipt.revision if receipt is not None else None,
            "event_id": receipt.requested.event_id if receipt is not None else None,
            "requested": receipt.requested.action if receipt is not None else None,
            "accepted": receipt.accepted if receipt is not None else None,
            "effective_action": effective_action,
            "action_source": tick.action_source if tick is not None else "commanded",
            "idle_reason": tick.idle_reason if tick is not None else None,
            "read_duration_s": tick.read_duration_s if tick is not None else receipt.read_duration_s,
            "write_duration_s": receipt.write_duration_s if receipt is not None else 0.0,
            "progress_events": record.progress_events,
        }
        self._trace.write(json.dumps(trace, default=_json_value) + "\n")
        if tick is not None and tick.skipped_before:
            self.timing_regular = False
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
            "timing_regular": self.timing_regular,
            "outcome": outcome,
            "reason": reason,
        }
        staged = manifest_path.with_suffix(".json.tmp")
        staged.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staged, manifest_path)


def _trace_entries(
    dataset_root: str | Path, episode_index: int, *, expected_frames: int | None = None
) -> list[dict]:
    extension_dir = Path(dataset_root) / "flash" / "v1"
    manifest = json.loads((extension_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported FLASH pose trace version")
    item = manifest["episodes"][str(episode_index)]
    name = item["trace"]
    if Path(name).name != name:
        raise ValueError("FLASH trace name must be local to its extension directory")
    entries = []
    with (extension_dir / name).open(encoding="utf-8") as stream:
        for expected_index, line in enumerate(stream):
            entry = json.loads(line)
            if entry["episode_index"] != episode_index or entry["frame_index"] != expected_index:
                raise ValueError("FLASH trace is not aligned with its LeRobot episode")
            entries.append(entry)
    if len(entries) != item["frames"] or (expected_frames is not None and len(entries) != expected_frames):
        raise ValueError("FLASH trace frame count differs from its LeRobot episode")
    return entries


def load_pose_estimates(
    dataset_root: str | Path, episode_index: int, *, expected_frames: int | None = None
) -> dict[int, dict[str, PoseEstimate]]:
    """Read the pose layer joined to an ordinary LeRobot episode by frame index."""
    result = {}
    for frame_index, entry in enumerate(
        _trace_entries(dataset_root, episode_index, expected_frames=expected_frames)
    ):
        estimates = {}
        for role, pose in entry.get("poses", {}).items():
            estimates[role] = PoseEstimate(
                role=role,
                pose=pose["pose"],
                source_time_s=pose["source_time_s"],
                available_time_s=pose["available_time_s"],
                evidence=Evidence(pose["evidence"]),
                reference_version=pose["reference_version"],
                translation_supported=tuple(pose["translation_supported"]),
                rotation_supported=tuple(pose["rotation_supported"]),
            )
        result[frame_index] = estimates
    return result


def load_pose_comparisons(
    dataset_root: str | Path, episode_index: int, *, expected_frames: int | None = None
) -> dict[int, dict[str, PoseComparison]]:
    """Replay candidate/prior conflicts alongside the regular episode frames."""
    result = {}
    for frame_index, entry in enumerate(
        _trace_entries(dataset_root, episode_index, expected_frames=expected_frames)
    ):
        comparisons = {}
        for role, item in entry.get("pose_comparisons", {}).items():
            candidate = item["candidate"]
            prior = item["prior"]
            comparisons[role] = PoseComparison(
                PoseComparisonVerdict(item["verdict"]),
                item["object_center_difference_m"],
                item["rotation_difference_rad"],
                _surface_support_from_json(candidate),
                _surface_support_from_json(prior),
            )
        result[frame_index] = comparisons
    return result


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _surface_support_json(support: SurfaceSupport) -> dict[str, int | float | None]:
    return {
        "projected_points": support.projected_points,
        "masked_depth_points": support.masked_depth_points,
        "projected_point_mask_fraction": support.projected_point_mask_fraction,
        "depth_residual_p90_m": (
            support.depth_residual_p90_m if math.isfinite(support.depth_residual_p90_m) else None
        ),
    }


def _surface_support_from_json(item: dict) -> SurfaceSupport:
    return SurfaceSupport(
        item["projected_points"],
        item["masked_depth_points"],
        item["projected_point_mask_fraction"],
        item["depth_residual_p90_m"] if item["depth_residual_p90_m"] is not None else float("inf"),
    )


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
