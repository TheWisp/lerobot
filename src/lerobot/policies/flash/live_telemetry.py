"""Best-effort live tracker inspection, outside the motor and perception loops."""

from __future__ import annotations

import gzip
import json
import os
import time
from collections import deque
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from threading import Condition, Thread

import cv2
import numpy as np

from .causal_service import AnchorDecision
from .execution import MotorTick
from .perception import EndEffectorKinematics, RGBDFrame
from .pose_disagreement import FreeSpaceComparison, PoseComparison
from .pose_evidence import PoseProposal
from .trajectory import AttachmentState, Evidence, PoseEstimate


def live_frame_path(live_path: Path, frame_id: int, kind: str) -> Path:
    """Locate a bounded, display-only RGB-D frame beside the live snapshot."""
    if frame_id < 0 or kind not in {"rgb", "depth"}:
        raise ValueError("invalid live frame request")
    extension = "jpg" if kind == "rgb" else "bin"
    return live_path.with_suffix(".frames") / f"{frame_id:08d}.{extension}"


def _mask_boundary(mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    return max(contours, key=cv2.contourArea)[:, 0, :].tolist()


def _mask_runs(mask: np.ndarray) -> list[list[int]]:
    flat = np.asarray(mask, dtype=np.uint8).ravel()
    edges = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
    return [[int(start), int(end - start)] for start, end in edges.reshape(-1, 2)]


def _group_support_json(groups: dict[str, dict]) -> dict[str, dict]:
    return {
        role: {
            "verified": bool(group["verified"]),
            "measured_this_frame": bool(group["measured_this_frame"]),
            "ids": group["ids"].tolist(),
            "pixels": group["xy"].tolist(),
            "base_xyz_m": np.round(group["base_xyz"], 6).tolist(),
            "last_group_measurement_s": group["last_group_measurement_s"],
        }
        for role, group in groups.items()
    }


def _estimate_json(estimate: PoseEstimate) -> dict:
    return {
        "role": estimate.role,
        "pose": estimate.pose.tolist() if estimate.pose is not None else None,
        "source_time_s": estimate.source_time_s,
        "source_frame_id": estimate.source_frame_id,
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
        frame_history: int = 8,
        decision_history_frames: int | None = None,
    ) -> None:
        if frame_history < 1 or (decision_history_frames is not None and decision_history_frames < 1):
            raise ValueError("live frame history must contain at least one frame")
        self.live_path = live_path
        self.trace_path = trace_path
        self.decision_path = trace_path.with_suffix(".decisions.jsonl")
        self.mask_path = trace_path.with_suffix(".masks.jsonl.gz")
        self.group_path = trace_path.with_suffix(".groups.jsonl.gz")
        self.centers = {role: center.tolist() for role, center in centers_by_role.items()}
        self.points = {
            role: np.asarray(points, dtype=float).tolist() for role, points in (points_by_role or {}).items()
        }
        self.period_s = period_s
        self.frame_history = frame_history
        # One source-frame decision per role and retained display frame is the
        # bounded writer backlog. A stalled disk must fail explicitly, never
        # grow an unbounded queue in the estimator process.
        # The writer can be behind by one display window when event-time
        # replay revises every retained source-frame decision. Keep both
        # bounded windows without silently dropping any decision.
        self.decision_backlog_limit = (frame_history + (decision_history_frames or 0)) * max(
            1, len(centers_by_role)
        )
        self.kinematics = kinematics
        self._written_frame_id: int | None = None
        self._frame_ids: deque[int] = deque()
        self._condition = Condition()
        self._revision = 0
        self._closed = False
        self._frame: RGBDFrame | None = None
        self._proposals: dict[str, PoseEstimate] = {}
        self._raw_proposals: dict[str, dict] = {}
        self._proposal_frame_id: int | None = None
        self._masks: dict[str, np.ndarray] = {}
        self._mask_frame_id: int | None = None
        self._pending_mask_events: deque[tuple[int, float, dict[str, np.ndarray]]] = deque()
        self._mask_events_dropped = 0
        self._group_support: dict[str, dict] = {}
        self._group_frame_id: int | None = None
        self._pending_group_events: deque[tuple[int, float, dict[str, dict]]] = deque()
        self._group_events_dropped = 0
        self._layers: dict[str, dict[str, PoseEstimate]] = {}
        self._comparisons: dict[str, dict] = {}
        self._point_comparisons: dict[str, dict] = {}
        self._free_space_comparisons: dict[str, dict] = {}
        self._accepted: dict[str, PoseEstimate] = {}
        self._latest_decisions: dict[str, dict] = {}
        self._pending_decisions: deque[dict] = deque()
        self._held_attachment: tuple[str, AttachmentState, np.ndarray, float, bool, bool] | None = None
        self._held_visual_evidence: dict | None = None
        self._motor_tick: dict | None = None
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
        source_proposals: tuple[PoseProposal, ...] = (),
    ) -> None:
        with self._condition:
            for estimate in estimates:
                self._proposals[estimate.role] = estimate
            self._proposal_frame_id = frame_id
            self._raw_proposals = {
                proposal.role: {
                    "frame_id": proposal.frame_id,
                    "calibration_version": proposal.calibration_version,
                    "pose": proposal.raw_pose.tolist() if proposal.raw_pose is not None else None,
                    "estimator": proposal.estimate.estimator,
                    "failure": proposal.failure,
                    "visible_support": asdict(proposal.visible_support),
                }
                for proposal in source_proposals
            }
            if masks is not None:
                copied_masks = {role: np.asarray(mask, dtype=bool).copy() for role, mask in masks.items()}
                self._masks = copied_masks
                self._mask_frame_id = frame_id
                if frame_id is not None:
                    # The camera and perception callbacks run at different rates.
                    # Retain the source frame's masks for exact replay alignment;
                    # contour-only display snapshots can miss that frame entirely.
                    if len(self._pending_mask_events) >= self.frame_history:
                        self._pending_mask_events.popleft()
                        self._mask_events_dropped += 1
                    self._pending_mask_events.append((frame_id, time.perf_counter(), copied_masks))
            self._changed()

    def layer(self, estimate: PoseEstimate) -> None:
        with self._condition:
            self._layers.setdefault(estimate.role, {})[estimate.estimator] = estimate
            self._changed()

    def scene_support(self, frame_id: int, groups: dict[str, dict]) -> None:
        """Retain measured group support at its camera frame, off the writer thread."""
        copied = {
            role: {
                **group,
                "ids": np.asarray(group["ids"]).copy(),
                "xy": np.asarray(group["xy"]).copy(),
                "base_xyz": np.asarray(group["base_xyz"]).copy(),
            }
            for role, group in groups.items()
        }
        with self._condition:
            self._group_support = copied
            self._group_frame_id = frame_id
            if len(self._pending_group_events) >= self.frame_history:
                self._pending_group_events.popleft()
                self._group_events_dropped += 1
            self._pending_group_events.append((frame_id, time.perf_counter(), copied))
            self._changed()

    def comparison(self, role: str, result: PoseComparison) -> None:
        with self._condition:
            self._comparisons[role] = {
                **asdict(result),
                "verdict": result.verdict.value,
                "available_at_s": time.perf_counter(),
            }
            self._changed()

    def point_comparison(self, role: str, result: PoseComparison) -> None:
        with self._condition:
            self._point_comparisons[role] = {
                **asdict(result),
                "verdict": result.verdict.value,
                "available_at_s": time.perf_counter(),
            }
            self._changed()

    def free_space_comparison(self, role: str, result: FreeSpaceComparison) -> None:
        with self._condition:
            self._free_space_comparisons[role] = {
                **asdict(result),
                "verdict": result.verdict.value,
                "available_at_s": time.perf_counter(),
            }
            self._changed()

    def accepted(self, poses: dict[str, PoseEstimate]) -> None:
        with self._condition:
            self._accepted = dict(poses)
            self._changed()

    def anchor_decision(self, decision: AnchorDecision) -> None:
        """Keep every source-frame verdict, even if display snapshots coalesce."""
        raw_pose = decision.proposal.raw_pose if decision.proposal is not None else decision.candidate.pose
        event = {
            "schema_version": 1,
            "frame_id": decision.frame_id,
            "calibration_version": decision.calibration_version,
            "reference_version": decision.candidate.reference_version,
            "role": decision.candidate.role,
            "outcome": decision.outcome,
            "decided_at_s": decision.decided_at_s,
            "candidate": _estimate_json(decision.candidate),
            "raw_pose": raw_pose.tolist() if raw_pose is not None else None,
            "visible_support": (
                asdict(decision.proposal.visible_support) if decision.proposal is not None else None
            ),
            "backend_failure": decision.proposal.failure if decision.proposal is not None else None,
            "accepted": _estimate_json(decision.accepted) if decision.accepted is not None else None,
        }
        with self._condition:
            if len(self._pending_decisions) >= self.decision_backlog_limit:
                raise RuntimeError("FLASH decision journal fell behind the estimator")
            self._latest_decisions[event["role"]] = event
            self._pending_decisions.append(event)
            self._changed()

    def held_attachment(
        self,
        role: str | None,
        attachment: AttachmentState | None,
        joints_deg: np.ndarray | None,
        observed_at_s: float,
        *,
        independently_observed: bool,
        visual_ambiguous: bool = False,
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
                    visual_ambiguous,
                )
                if role is not None and attachment is not None and joints_deg is not None
                else None
            )
            if role is None:
                self._held_visual_evidence = None
            self._changed()

    def held_visual_evidence(self, evidence: dict) -> None:
        """Publish the source-frame visual/gripper comparison without changing control state."""
        with self._condition:
            self._held_visual_evidence = evidence.copy()
            self._changed()

    def motor_tick(self, tick: MotorTick) -> None:
        """Expose the evidence behind the latest command without motor-loop I/O."""
        with self._condition:
            self._motor_tick = {
                "tick": tick.tick,
                "action_source": tick.action_source,
                "plan_revision": tick.receipt.revision if tick.receipt is not None else None,
                "plan_evidence": (
                    asdict(tick.effective_plan_evidence) if tick.effective_plan_evidence is not None else None
                ),
            }
            self._changed()

    def _snapshot(
        self,
    ) -> tuple[
        dict,
        RGBDFrame | None,
        tuple[dict, ...],
        tuple[tuple[int, float, dict[str, np.ndarray]], ...],
        tuple[tuple[int, float, dict[str, dict]], ...],
    ]:
        with self._condition:
            frame = self._frame
            proposals = self._proposals.copy()
            raw_proposals = self._raw_proposals.copy()
            proposal_frame_id = self._proposal_frame_id
            masks = self._masks.copy()
            mask_frame_id = self._mask_frame_id
            layers = {role: values.copy() for role, values in self._layers.items()}
            comparisons = self._comparisons.copy()
            point_comparisons = self._point_comparisons.copy()
            free_space_comparisons = self._free_space_comparisons.copy()
            accepted = self._accepted.copy()
            latest_decisions = self._latest_decisions.copy()
            pending_decisions = tuple(self._pending_decisions)
            pending_mask_events = tuple(self._pending_mask_events)
            self._pending_mask_events.clear()
            mask_events_dropped = self._mask_events_dropped
            group_support = self._group_support.copy()
            group_frame_id = self._group_frame_id
            pending_group_events = tuple(self._pending_group_events)
            self._pending_group_events.clear()
            group_events_dropped = self._group_events_dropped
            held = self._held_attachment
            held_visual_evidence = self._held_visual_evidence.copy() if self._held_visual_evidence else None
            motor_tick = self._motor_tick.copy() if self._motor_tick is not None else None
        held_prediction = None
        if held is not None and self.kinematics is not None:
            role, attachment, joints, observed_at_s, independently_observed, visual_ambiguous = held
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
            if visual_ambiguous:
                held_prediction["evidence"] = "ambiguous_visual_prediction"
        snapshot = {
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
                    "calibration_version": frame.calibration_version,
                }
                if frame is not None
                else None
            ),
            "centers_object_m": self.centers,
            "points_object_m": self.points,
            "observed_masks": {
                role: {
                    "frame_id": mask_frame_id,
                    "boundary_xy": _mask_boundary(mask),
                    **(
                        {"runs": _mask_runs(mask)}
                        if frame is not None and mask_frame_id == frame.frame_id
                        else {}
                    ),
                }
                for role, mask in masks.items()
            },
            "mask_evidence_dropped": mask_events_dropped,
            "group_evidence_dropped": group_events_dropped,
            "scene_support": (
                {"frame_id": group_frame_id, "groups": _group_support_json(group_support)}
                if frame is not None and group_frame_id == frame.frame_id
                else None
            ),
            "proposals": {role: _estimate_json(value) for role, value in proposals.items()},
            "proposal_frame_id": proposal_frame_id,
            "raw_proposals": raw_proposals,
            "layers": {
                role: {name: _estimate_json(value) for name, value in entries.items()}
                for role, entries in layers.items()
            },
            "direct_group_comparisons": comparisons,
            "direct_point_comparisons": point_comparisons,
            "direct_free_space_comparisons": free_space_comparisons,
            "accepted": {role: _estimate_json(value) for role, value in accepted.items()},
            "anchor_decisions": latest_decisions,
            "held_prediction": held_prediction,
            "held_visual_evidence": held_visual_evidence,
            "motor_tick": motor_tick,
        }
        return snapshot, frame, pending_decisions, pending_mask_events, pending_group_events

    def _write_frame(self, frame: RGBDFrame) -> None:
        if frame.frame_id == self._written_frame_id:
            return
        depth_mm = frame.metric_depth().astype(np.float64) * 1000.0
        valid = np.isfinite(depth_mm) & (depth_mm > 0) & (depth_mm <= np.iinfo(np.uint16).max)
        depth_bytes = np.rint(np.where(valid, depth_mm, 0)).astype("<u2").tobytes()
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("cannot encode live RGB frame")
        for kind, payload in (("rgb", encoded.tobytes()), ("depth", depth_bytes)):
            destination = live_frame_path(self.live_path, frame.frame_id, kind)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.with_suffix(destination.suffix + ".tmp")
            staged.write_bytes(payload)
            os.replace(staged, destination)
        self._written_frame_id = frame.frame_id
        self._frame_ids.append(frame.frame_id)
        while len(self._frame_ids) > self.frame_history:
            expired = self._frame_ids.popleft()
            for kind in ("rgb", "depth"):
                # safe-destruct: prune only this telemetry writer's bounded temporary display files.
                live_frame_path(self.live_path, expired, kind).unlink(missing_ok=True)

    def _write_loop(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        seen = -1
        with (
            self.trace_path.open("w", encoding="utf-8") as trace,
            self.decision_path.open("w", encoding="utf-8") as decisions,
            gzip.open(self.mask_path, "wt", encoding="utf-8") as mask_events,
            gzip.open(self.group_path, "wt", encoding="utf-8") as group_events,
        ):
            while True:
                with self._condition:
                    self._condition.wait_for(lambda seen=seen: self._revision != seen or self._closed)
                    revision = self._revision
                    closing = self._closed
                if revision != seen:
                    snapshot, frame, pending_decisions, pending_masks, pending_groups = self._snapshot()
                    for frame_id, available_at_s, masks in pending_masks:
                        if not masks:
                            continue
                        image_size = next(iter(masks.values())).shape
                        if any(mask.shape != image_size for mask in masks.values()):
                            raise ValueError("FLASH object masks do not share one camera frame")
                        mask_events.write(
                            json.dumps(
                                {
                                    "frame_id": frame_id,
                                    "available_at_s": available_at_s,
                                    "image_size": list(image_size),
                                    "masks_rle": {role: _mask_runs(mask) for role, mask in masks.items()},
                                    "boundaries_xy": {
                                        role: _mask_boundary(mask) for role, mask in masks.items()
                                    },
                                },
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    mask_events.flush()
                    for frame_id, available_at_s, groups in pending_groups:
                        group_events.write(
                            json.dumps(
                                {
                                    "frame_id": frame_id,
                                    "available_at_s": available_at_s,
                                    "groups": _group_support_json(groups),
                                },
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    group_events.flush()
                    for decision in pending_decisions:
                        decisions.write(json.dumps(decision, separators=(",", ":"), allow_nan=False) + "\n")
                    decisions.flush()
                    with self._condition:
                        for _ in pending_decisions:
                            self._pending_decisions.popleft()
                    if frame is not None:
                        try:
                            self._write_frame(frame)
                        except (OSError, RuntimeError, ValueError, cv2.error):
                            snapshot["camera"]["rgbd_available"] = False
                        else:
                            snapshot["camera"]["rgbd_available"] = True
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
        # Remove only the temporary display files this writer created. The
        # completed LeRobot recording remains available in Replay.
        for frame_id in self._frame_ids:
            for kind in ("rgb", "depth"):
                # safe-destruct: these frame IDs were written by this telemetry instance.
                live_frame_path(self.live_path, frame_id, kind).unlink(missing_ok=True)
        with suppress(OSError):
            # safe-destruct: remove this writer's empty temporary display directory.
            self.live_path.with_suffix(".frames").rmdir()
