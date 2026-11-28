"""Small, editable FLASH annotations beside an ordinary robot episode.

Images, depth, measured joints and actions remain in LeRobot. This document
only selects objects and marks the frames that should guide an execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Demos are recorded and replayed at the motor loop's rate; other rates are refused (#277).
DEMO_FPS = 30


class DemoObject(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(pattern=r"^object_[1-3]$")
    name: str = Field(min_length=1, max_length=80)
    point: tuple[float, float] | None = None

    @model_validator(mode="after")
    def prompt(self):
        self.name = self.name.strip()
        if not self.name or "." in self.name:
            raise ValueError("Use one object name without periods")
        return self


class DemoKeyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame: int = Field(ge=0, strict=True)
    reference: str = Field(pattern=r"^(world|object_[1-3])$")
    label: str = Field(default="", max_length=80)


class DemoDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: int = Field(default=1, ge=1, le=1)
    source: str
    revision: int = Field(default=0, ge=0, strict=True)
    name: str = Field(default="Untitled demo", min_length=1, max_length=80)
    camera: str = ""
    reference_frame: int = Field(default=0, ge=0, strict=True)
    objects: list[DemoObject] = Field(default_factory=list, max_length=3)
    # Dense contact intervals retain every recorded sample; episode bounds,
    # rather than an arbitrary keyframe count, limit the annotation.
    keyframes: list[DemoKeyframe] = Field(default_factory=list)

    @model_validator(mode="after")
    def references(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Name the demo")
        ids = [obj.id for obj in self.objects]
        names = [obj.name for obj in self.objects]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("Objects need distinct names and IDs")
        if any(key.reference not in {"world", *ids} for key in self.keyframes):
            raise ValueError("A keyframe refers to an object that is not selected")
        frames = [key.frame for key in self.keyframes]
        if len(set(frames)) != len(frames):
            raise ValueError("Only one keyframe per image")
        self.keyframes.sort(key=lambda key: key.frame)
        return self


def episode_source(dataset, episode: int) -> dict:
    if isinstance(episode, bool) or not 0 <= episode < dataset.meta.total_episodes:
        raise ValueError("Episode is outside this dataset")
    first, length = dataset.episode_rows(episode)
    features = dataset.meta.features
    cameras = {
        key: value
        for key, value in features.items()
        if value.get("dtype") in {"image", "video"}
        and not value.get("info", {}).get("is_depth_map")
        and value.get("info", {}).get("modality") != "infrared"
    }
    paths = {dataset.meta.get_data_file_path(episode)}
    paths.update(dataset.meta.get_video_file_path(episode, key) for key in dataset.meta.video_keys)
    files = []
    for relative in sorted(paths):
        stat = (Path(dataset.root) / relative).stat()
        files.append((str(relative), stat.st_size, stat.st_mtime_ns))
    source = {
        "episode": episode,
        "first": first,
        "length": length,
        "fps": dataset.fps,
        "robot_type": dataset.meta.info.get("robot_type", ""),
        "robot_geometry": dataset.meta.info.get("robot_geometry"),
        "features": features,
        # repo_id is an alias chosen by the opener, not a property of a local
        # episode. The GUI and CLI can legitimately use different aliases.
        "files": files,
    }
    identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return {
        "identity": identity,
        "length": length,
        "fps": dataset.fps,
        "robot_type": source["robot_type"],
        "robot_geometry": source["robot_geometry"],
        "cameras": cameras,
        "has_robot": "observation.state" in features and "action" in features,
    }


def document_path(dataset, episode: int) -> Path:
    return Path(dataset.root) / "flash" / "v1" / "editor" / f"episode_{episode:06d}.json"


def tracking_selection(document: DemoDocument) -> dict:
    """Keyframe edits and renaming the demo do not change measured object poses."""
    return document.model_dump(
        mode="json", include={"version", "source", "camera", "reference_frame", "objects"}
    )


def load_document(dataset, episode: int) -> tuple[DemoDocument, dict]:
    source = episode_source(dataset, episode)
    path = document_path(dataset, episode)
    document = (
        DemoDocument.model_validate_json(path.read_text())
        if path.exists()
        else DemoDocument(source=source["identity"], camera=next(iter(source["cameras"]), ""))
    )
    if document.source != source["identity"]:
        raise ValueError("The episode has changed; reopen it before editing")
    return document, source


def save_document(dataset, episode: int, document: DemoDocument) -> DemoDocument:
    current, source = load_document(dataset, episode)
    if document.source != source["identity"]:
        raise ValueError("These edits belong to another episode")
    if document.revision != current.revision:
        raise ValueError("A newer edit was saved; reopen before saving")
    if document.camera not in source["cameras"]:
        raise ValueError("Select a recorded color camera")
    if document.reference_frame >= source["length"] or any(
        key.frame >= source["length"] for key in document.keyframes
    ):
        raise ValueError("A selected frame is outside the episode")
    # LeRobot metadata stores image shapes as HWC, including video features.
    height, width = source["cameras"][document.camera]["shape"][:2]
    for obj in document.objects:
        if obj.point is not None and not (0 <= obj.point[0] < width and 0 <= obj.point[1] < height):
            raise ValueError("Object click is outside the recorded image")
    saved = document.model_copy(update={"revision": document.revision + 1})
    atomic_json(document_path(dataset, episode), saved.model_dump(mode="json"))
    return saved


def tracking_stale_reason(inputs, document, *, settings=None, depth_source=None):
    """One freshness rule for display, regeneration and execution preparation."""
    from .pipeline_version import is_current

    if not is_current(inputs):
        return "Tracking code changed"
    if tracking_selection(DemoDocument.model_validate(inputs["document"])) != tracking_selection(document):
        return "Object selection changed"
    if settings is not None and inputs.get("settings") != settings:
        return "Tracking settings changed"
    if depth_source is not None and inputs.get("depth_source") != depth_source:
        return "Depth source changed"
    return None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            # safe-destruct: only this writer's unique temporary file.
            temporary.unlink(missing_ok=True)
