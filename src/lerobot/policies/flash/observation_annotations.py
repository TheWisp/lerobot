"""Review annotations for RGB-D episodes without robot actions.

These are human intent, not inferred motion or an executable robot skill. Pose
evidence stays in the separately versioned observation sidecar.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def _path(root: Path, episode: int) -> Path:
    return root / "flash" / "v1" / "observation-annotations" / f"episode_{episode:06d}.json"


def source_identity(root: Path) -> str:
    source = root / "flash" / "v1" / "source.json"
    metadata = json.loads(source.read_text(encoding="utf-8"))
    if metadata.get("status") != "perception_only":
        raise ValueError("this dataset is not an RGB-D-only observation")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def load_annotations(root: Path, episode: int) -> dict:
    root = root.expanduser().resolve()
    identity = source_identity(root)
    path = _path(root, episode)
    if not path.is_file():
        return {"schema_version": 1, "source_metadata_sha256": identity, "keyframes": []}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("keyframes"), list):
        raise ValueError("unsupported observation annotation schema")
    if document.get("source_metadata_sha256") != identity:
        raise ValueError("observation source changed since these annotations were saved")
    return document


def validate_annotations(document: dict, replay: dict, identity: str) -> dict:
    if document.get("schema_version") != 1 or document.get("source_metadata_sha256") != identity:
        raise ValueError("annotation version or source identity does not match this episode")
    keyframes = document.get("keyframes")
    if not isinstance(keyframes, list):
        raise ValueError("keyframes must be a list")
    if not replay.get("evidence_available"):
        raise ValueError("extract object evidence before marking relationships")
    frames = replay["frames"]
    roles = set().union(*(frame["snapshot"]["observed_masks"] for frame in frames))
    normalized = []
    seen = set()
    for keyframe in keyframes:
        if not isinstance(keyframe, dict) or isinstance(keyframe.get("frame_index"), bool):
            raise ValueError("each key frame needs a frame index")
        index = keyframe["frame_index"]
        moving = keyframe.get("moving_role")
        reference = keyframe.get("reference_role")
        if not isinstance(index, int) or not 0 <= index < len(frames):
            raise ValueError("key frame is outside this episode")
        if moving not in roles or reference not in roles or moving == reference:
            raise ValueError("select two different tracked objects")
        pair = (index, moving, reference)
        if pair in seen:
            raise ValueError("this object relationship is already marked at this frame")
        seen.add(pair)
        required = keyframe.get("required")
        if not isinstance(required, bool):
            raise ValueError("required must be true or false")
        turn = keyframe.get("allowed_turn")
        if turn is not None:
            if not isinstance(turn, dict) or set(turn) != {
                "pivot_reference_m",
                "axis_reference",
                "min_angle_rad",
                "max_angle_rad",
                "source_evidence",
            }:
                raise ValueError("allowed turn needs a point, direction and angle range")
            if turn["source_evidence"] not in {"observed", "predicted"}:
                raise ValueError("allowed turn must record whether its pivot was observed")
            reference_estimate = frames[index]["snapshot"]["accepted"].get(reference)
            if not reference_estimate or turn["source_evidence"] != reference_estimate["evidence"]:
                raise ValueError("turn pivot evidence does not match the reference pose at this frame")
            pivot, axis = turn["pivot_reference_m"], turn["axis_reference"]
            if not all(
                isinstance(v, list)
                and len(v) == 3
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
                for v in (pivot, axis)
            ):
                raise ValueError("turn point and direction must be finite 3D vectors")
            if math.sqrt(sum(x * x for x in axis)) < 1e-6:
                raise ValueError("turn direction must be nonzero")
            lower, upper = turn["min_angle_rad"], turn["max_angle_rad"]
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (lower, upper)):
                raise ValueError("turn range must be finite")
            if not -math.pi <= lower < upper <= math.pi:
                raise ValueError("turn range must lie between -180° and 180°")
        normalized.append(
            {
                "frame_index": index,
                "moving_role": moving,
                "reference_role": reference,
                "required": required,
                "allowed_turn": turn,
            }
        )
    return {
        "schema_version": 1,
        "source_metadata_sha256": identity,
        "keyframes": sorted(
            normalized, key=lambda item: (item["frame_index"], item["moving_role"], item["reference_role"])
        ),
    }


def save_annotations(root: Path, episode: int, document: dict, replay: dict) -> dict:
    root = root.expanduser().resolve()
    if episode < 0 or replay.get("source_episode") != episode:
        raise ValueError("annotation episode does not match the replay")
    validated = validate_annotations(document, replay, source_identity(root))
    path = _path(root, episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(validated, indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=".annotation-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            # safe-destruct: remove only the temporary file this save created.
            temporary.unlink(missing_ok=True)
    return validated
