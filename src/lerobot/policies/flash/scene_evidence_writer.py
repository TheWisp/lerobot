"""Stream per-frame FLASH scene evidence beside a prepared demo version."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .perception import RGBDFrame
from .perception_step import PerceptionResult
from .trajectory import PoseEstimate

SCENE_EVIDENCE_CAPTURE_VERSION = 3


def _mask_runs(mask: np.ndarray) -> list[list[int]]:
    flat = np.asarray(mask, dtype=np.uint8).ravel()
    edges = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
    return [[int(start), int(end - start)] for start, end in edges.reshape(-1, 2)]


class SceneEvidenceWriter:
    """Write one source frame at a time; a partial pass is never marked complete."""

    def __init__(
        self, directory: Path, *, total_frames: int, roles: tuple[str, ...], source_sha256: str = ""
    ):
        if total_frames < 1 or not roles:
            raise ValueError("scene evidence needs frames and object roles")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "total_frames": total_frames,
            "frames_completed": 0,
            "complete": False,
            "roles": list(roles),
        }
        self._save_manifest()

    def _save_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def record(
        self,
        frame: RGBDFrame,
        result: PerceptionResult,
        groups: dict,
        estimates: dict[str, PoseEstimate],
    ) -> None:
        index = self.manifest["frames_completed"]
        if frame.frame_id != index or result.frame_id != index or index >= self.manifest["total_frames"]:
            raise ValueError("scene evidence frames must match the prepared source order")
        data = {
            "frame_index": index,
            "image_size": list(frame.depth.shape),
            "masks_rle": {role: _mask_runs(mask) for role, mask in result.masks.items()},
            "groups": {
                role: {
                    "verified": bool(group["verified"]),
                    "measured_this_frame": bool(group["measured_this_frame"]),
                    "ids": group["ids"].tolist(),
                    "pixels": group["xy"].tolist(),
                    "base_xyz_m": np.round(group["base_xyz"], 6).tolist(),
                    "last_group_measurement_s": group["last_group_measurement_s"],
                }
                for role, group in groups.items()
            },
            "estimates": {
                role: {
                    "evidence": estimate.evidence.value,
                    "estimator": estimate.estimator,
                    "pose": estimate.pose.tolist() if estimate.pose is not None else None,
                    "source_time_s": estimate.source_time_s,
                    "last_direct_observation_s": estimate.last_direct_observation_s,
                }
                for role, estimate in estimates.items()
            },
            "proposals": {
                proposal.role: {
                    "evidence": proposal.estimate.evidence.value,
                    "estimator": proposal.estimate.estimator,
                    "raw_pose": proposal.raw_pose.tolist() if proposal.raw_pose is not None else None,
                    "motion_prior_pose": (
                        proposal.motion_prior_pose.tolist()
                        if proposal.motion_prior_pose is not None
                        else None
                    ),
                    "failure": proposal.failure,
                    "visible_support": asdict(proposal.visible_support),
                }
                for proposal in result.proposals
            },
        }
        frame_path = self.directory / f"{index:06d}.json.gz"
        temporary = frame_path.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
        temporary.replace(frame_path)
        self.manifest["frames_completed"] = index + 1
        self._save_manifest()

    def finish(self, *, source_sha256: str | None = None) -> None:
        if self.manifest["frames_completed"] != self.manifest["total_frames"]:
            raise ValueError("scene evidence stopped before the last prepared frame")
        if source_sha256 is not None:
            self.manifest["source_sha256"] = source_sha256
        self.manifest["complete"] = True
        self._save_manifest()
