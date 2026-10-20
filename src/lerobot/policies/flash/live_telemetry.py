"""Best-effort live tracker inspection, outside the motor and perception loops."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from threading import Condition, Thread

import cv2
import numpy as np

from .perception import EndEffectorKinematics, RGBDFrame
from .pose_disagreement import PoseComparison
from .trajectory import AttachmentState, Evidence, PoseEstimate


def _mask_boundary(mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    return max(contours, key=cv2.contourArea)[:, 0, :].tolist()


def _estimate_json(estimate: PoseEstimate) -> dict:
    return {
        "role": estimate.role,
        "pose": estimate.pose.tolist() if estimate.pose is not None else None,
        "source_time_s": estimate.source_time_s,
        "available_time_s": estimate.available_time_s,
        "evidence": estimate.evidence.value,
        "estimator": estimate.estimator,
        "translation_supported": estimate.translation_supported,
        "rotation_supported": estimate.rotation_supported,
        "last_direct_observation_s": estimate.last_direct_observation_s,
    }


class FlashLiveTelemetry:
    """Publish the latest layers to the GUI and append the same view for replay.

    Producers only replace small in-memory references and notify a writer. The
    writer serializes and performs all filesystem I/O on its own thread; a slow
    viewer or disk cannot delay a camera frame, pose update, or motor tick.
    """

    def __init__(
        self,
        live_path: Path,
        trace_path: Path,
        centers_by_role: dict[str, np.ndarray],
        *,
        points_by_role: dict[str, np.ndarray] | None = None,
        kinematics: EndEffectorKinematics | None = None,
        period_s: float = 0.1,
    ) -> None:
        self.live_path = live_path
        self.trace_path = trace_path
        self.centers = {role: center.tolist() for role, center in centers_by_role.items()}
        self.points = {
            role: np.asarray(points, dtype=float).tolist() for role, points in (points_by_role or {}).items()
        }
        self.period_s = period_s
        self.kinematics = kinematics
        self._condition = Condition()
        self._revision = 0
        self._closed = False
        self._frame: RGBDFrame | None = None
        self._proposals: dict[str, PoseEstimate] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._mask_frame_id: int | None = None
        self._layers: dict[str, dict[str, PoseEstimate]] = {}
        self._comparisons: dict[str, dict] = {}
        self._accepted: dict[str, PoseEstimate] = {}
        self._held_attachment: tuple[str, AttachmentState, np.ndarray, float, bool] | None = None
        self._thread = Thread(target=self._write_loop, name="flash-live-telemetry", daemon=True)
        self._thread.start()

    def _changed(self) -> None:
        self._revision += 1
        self._condition.notify()

    def frame(self, frame: RGBDFrame) -> None:
        with self._condition:
            self._frame = frame
            self._changed()

    def proposals(
        self,
        estimates: tuple[PoseEstimate, ...],
        masks: dict[str, np.ndarray] | None = None,
        frame_id: int | None = None,
    ) -> None:
        with self._condition:
            for estimate in estimates:
                self._proposals[estimate.role] = estimate
            if masks is not None:
                self._masks = dict(masks)
                self._mask_frame_id = frame_id
            self._changed()

    def layer(self, estimate: PoseEstimate) -> None:
        with self._condition:
            self._layers.setdefault(estimate.role, {})[estimate.estimator] = estimate
            self._changed()

    def comparison(self, role: str, result: PoseComparison) -> None:
        with self._condition:
            self._comparisons[role] = {
                **asdict(result),
                "verdict": result.verdict.value,
                "available_at_s": time.perf_counter(),
            }
            self._changed()

    def accepted(self, poses: dict[str, PoseEstimate]) -> None:
        with self._condition:
            self._accepted = dict(poses)
            self._changed()

    def held_attachment(
        self,
        role: str | None,
        attachment: AttachmentState | None,
        joints_deg: np.ndarray | None,
        observed_at_s: float,
        *,
        independently_observed: bool,
    ) -> None:
        """Record the controller's rigid-grasp prediction without doing FK on the motor loop."""
        with self._condition:
            self._held_attachment = (
                (
                    role,
                    attachment,
                    np.asarray(joints_deg, dtype=float).copy(),
                    observed_at_s,
                    independently_observed,
                )
                if role is not None and attachment is not None and joints_deg is not None
                else None
            )
            self._changed()

    def _snapshot(self) -> dict:
        with self._condition:
            frame = self._frame
            proposals = self._proposals.copy()
            masks = self._masks.copy()
            mask_frame_id = self._mask_frame_id
            layers = {role: values.copy() for role, values in self._layers.items()}
            comparisons = self._comparisons.copy()
            accepted = self._accepted.copy()
            held = self._held_attachment
        held_prediction = None
        if held is not None and self.kinematics is not None:
            role, attachment, joints, observed_at_s, independently_observed = held
            pose = self.kinematics.forward_kinematics(joints) @ attachment.gripper_from_object
            held_prediction = _estimate_json(
                PoseEstimate(
                    role,
                    pose,
                    observed_at_s,
                    observed_at_s,
                    Evidence.PREDICTED,
                    attachment.reference_version,
                    (True,) * 3,
                    (True,) * 3,
                    estimator="grasp_attachment_prediction",
                    last_direct_observation_s=attachment.source_time_s if independently_observed else None,
                )
            )
        return {
            "schema_version": 1,
            "recorded_at_s": time.perf_counter(),
            "camera": (
                {
                    "frame_id": frame.frame_id,
                    "source_time_s": frame.source_time_s,
                    "width": frame.rgb.shape[1],
                    "height": frame.rgb.shape[0],
                    "intrinsics": frame.intrinsics.tolist(),
                    "base_from_camera": frame.base_from_camera.tolist(),
                }
                if frame is not None
                else None
            ),
            "centers_object_m": self.centers,
            "points_object_m": self.points,
            "observed_masks": {
                role: {"frame_id": mask_frame_id, "boundary_xy": _mask_boundary(mask)}
                for role, mask in masks.items()
            },
            "proposals": {role: _estimate_json(value) for role, value in proposals.items()},
            "layers": {
                role: {name: _estimate_json(value) for name, value in entries.items()}
                for role, entries in layers.items()
            },
            "direct_group_comparisons": comparisons,
            "accepted": {role: _estimate_json(value) for role, value in accepted.items()},
            "held_prediction": held_prediction,
        }

    def _write_loop(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        seen = -1
        with self.trace_path.open("w", encoding="utf-8") as trace:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda seen=seen: self._revision != seen or self._closed)
                    revision = self._revision
                    closing = self._closed
                if revision != seen:
                    snapshot = self._snapshot()
                    encoded = json.dumps(snapshot, separators=(",", ":"), allow_nan=False)
                    staged = self.live_path.with_suffix(self.live_path.suffix + ".tmp")
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_text(encoded, encoding="utf-8")
                    os.replace(staged, self.live_path)
                    trace.write(encoded + "\n")
                    trace.flush()
                    seen = revision
                if closing:
                    return
                with self._condition:
                    deadline = time.monotonic() + self.period_s
                    while not self._closed and time.monotonic() < deadline:
                        self._condition.wait(timeout=deadline - time.monotonic())

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("FLASH telemetry writer did not stop")
