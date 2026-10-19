# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A recording, whichever kind: a demo, an act, or the live stream are one kind of thing. Each is camera frames with
their times, the arm's samples, and where the camera stands; a demo adds the operator's annotations (key frames, the
objects clicked). What processes or draws a recording reads it through :class:`Recording` and does not care which
kind it is.

Today a demo and an act are stored in different layouts; :func:`open_recording` reads either as it is. Times are the
wall clock, the one clock both keep.

- A demo folder: ``recording/`` (``rgb/%06d.jpg``, ``depth/%06d.png`` in mm, ``times.txt``, ``cam_K.txt``),
  ``showservo_demo.npz`` (the arm: times from ``t0``, joints, commands, fingertips), ``keypoints.json`` and
  ``objects.npz`` (the annotations; ``objects.npz`` also keeps the objects' tracks, which are derived, not read here).
- An act folder: ``camera/`` (``rgb/``, ``depth/``, ``timeline.json``: one row per recording tick, a new image only on
  a "captured" one), ``arm.npz``, ``act.json`` (intrinsics and the camera's pose). An act has no annotations of its
  own; ``demo`` names the demo it played.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

DEMO_CAMERA, DEMO_ARM, DEMO_KEYPOINTS, DEMO_OBJECTS = (
    "recording",
    "showservo_demo.npz",
    "keypoints.json",
    "objects.npz",
)
ACT_CAMERA, ACT_ARM, ACT_META = "camera", "arm.npz", "act.json"


@dataclasses.dataclass(frozen=True)
class ArmSamples:
    """The arm as it was sampled: wall-clock times (N,), joints read and commanded (N, J), and the fingertip
    (N, 4, 4, base <- fingertip) the arm's own kinematics put at each."""

    t: np.ndarray
    q_obs: np.ndarray
    q_cmd: np.ndarray
    tip: np.ndarray


@dataclasses.dataclass(frozen=True)
class Annotations:
    """What an operator added to a recording to make it a demo: the key frames and marks as saved, the place's
    landing rule, and each object clicked: the camera frame it was clicked on, the click, its declared symmetry and
    its mask there."""

    keypoints: list[dict[str, Any]]
    landing: str | None
    landing_axis: str | None
    objects: dict[str, dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class Recording:
    """One recording. Pre: ``times`` is increasing, one per camera frame; ``images`` and ``depths`` the frames' files
    in the same order."""

    root: pathlib.Path
    kind: str  # "demo" or "act": where it came from, nothing more
    times: np.ndarray  # each camera frame's capture time, wall clock (s)
    images: tuple[pathlib.Path, ...]
    depths: tuple[pathlib.Path, ...]
    K: np.ndarray  # 3x3 intrinsics
    arm: ArmSamples
    base_from_camera: np.ndarray | None  # None where the recording did not keep it
    annotations: Annotations | None
    demo: pathlib.Path | None = None  # the demo an act played
    # Each object as the recording designated it: the camera frame and its mask there. A demo's are its annotations';
    # an act's, the masks its own run started tracking from (its first tracker frame).
    designations: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.times)

    def color(self, i: int) -> np.ndarray:
        """Frame ``i``'s colour image, RGB uint8."""
        import cv2

        bgr = cv2.imread(str(self.images[i]))
        if bgr is None:
            raise FileNotFoundError(self.images[i])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth(self, i: int) -> np.ndarray:
        """Frame ``i``'s depth in metres, aligned to its colour image; 0 where there is none."""
        import cv2

        mm = cv2.imread(str(self.depths[i]), cv2.IMREAD_UNCHANGED)
        if mm is None:
            raise FileNotFoundError(self.depths[i])
        return mm.astype(np.float32) / 1000.0

    def frame_at(self, t: float) -> int:
        """The camera frame captured nearest wall-clock time ``t``."""
        return int(np.argmin(np.abs(self.times - t)))


def open_recording(root: str | pathlib.Path) -> Recording:
    """A demo folder or an act folder as a :class:`Recording`. Raises ValueError for a folder that is neither."""
    root = pathlib.Path(root)
    if (root / DEMO_CAMERA / "times.txt").exists() and (root / DEMO_ARM).exists():
        return _open_demo(root)
    if (root / ACT_CAMERA / "timeline.json").exists() and (root / ACT_ARM).exists():
        return _open_act(root)
    raise ValueError(f"{root} is neither a demo nor an act recording")


def _open_demo(root: pathlib.Path) -> Recording:
    cam = root / DEMO_CAMERA
    times = np.atleast_1d(np.loadtxt(cam / "times.txt"))
    z = np.load(root / DEMO_ARM, allow_pickle=False)
    arm = ArmSamples(
        t=float(z["t0"]) + np.asarray(z["t"], dtype=float),
        q_obs=np.asarray(z["q_obs"], dtype=float),
        q_cmd=np.asarray(z["q_cmd"], dtype=float),
        tip=np.asarray(z["tips"], dtype=float),
    )
    return Recording(
        root=root,
        kind="demo",
        times=times,
        images=tuple(cam / "rgb" / f"{k:06d}.jpg" for k in range(len(times))),
        depths=tuple(cam / "depth" / f"{k:06d}.png" for k in range(len(times))),
        K=np.loadtxt(cam / "cam_K.txt").reshape(3, 3),
        arm=arm,
        base_from_camera=None,
        annotations=(annotations := _demo_annotations(root)),
        designations={n: {"frame": o["frame"], "mask": o["mask"]} for n, o in annotations.objects.items()},
    )


def _demo_annotations(root: pathlib.Path) -> Annotations:
    keypoints, landing, landing_axis = [], None, None
    if (root / DEMO_KEYPOINTS).exists():
        saved = json.loads((root / DEMO_KEYPOINTS).read_text())
        keypoints = list(saved.get("keypoints", []))
        landing, landing_axis = saved.get("landing"), saved.get("landing_axis")
    objects: dict[str, dict[str, Any]] = {}
    if (root / DEMO_OBJECTS).exists():
        z = np.load(root / DEMO_OBJECTS, allow_pickle=False)
        for n, (name, frame, click, *rest) in enumerate(json.loads(str(z["names"]))):
            objects[name] = {
                "frame": int(frame),
                "click": [int(c) for c in click],
                "symmetry": rest[0] if rest else None,
                "mask": np.asarray(z[f"o{n}_mask"], dtype=bool),
            }
    return Annotations(keypoints, landing, landing_axis, objects)


def _open_act(root: pathlib.Path) -> Recording:
    cam = root / ACT_CAMERA
    timeline = json.loads((cam / "timeline.json").read_text())
    times, images, depths = [], [], []
    for row in timeline["rows"]:  # one per tick; a new image only on a captured one
        if row.get("status") != "captured" or row.get("image") is None:
            continue
        # the tick's wall time, moved by how long before the tick the camera took the frame
        capture = float(row["t_wall"]) + float(row.get("camera_mono", row["t_mono"])) - float(row["t_mono"])
        times.append(capture)
        images.append(cam / "rgb" / f"{int(row['image']):06d}.jpg")
        depths.append(cam / "depth" / f"{int(row['image']):06d}.png")
    meta = json.loads((root / ACT_META).read_text()) if (root / ACT_META).exists() else {}
    z = np.load(root / ACT_ARM, allow_pickle=False)
    arm = ArmSamples(
        t=np.asarray(z["t"], dtype=float),
        q_obs=np.asarray(z["q_obs"], dtype=float),
        q_cmd=np.asarray(z["q_cmd"], dtype=float),
        tip=np.asarray(z["tip_obs"], dtype=float),
    )
    i = meta.get("intr") or {}
    k = np.array(
        [
            [i.get("fx", 0.0), 0.0, i.get("cx", 0.0)],
            [0.0, i.get("fy", 0.0), i.get("cy", 0.0)],
            [0.0, 0.0, 1.0],
        ]
    )
    return Recording(
        root=root,
        kind="act",
        times=np.asarray(times, dtype=float),
        images=tuple(images),
        depths=tuple(depths),
        K=k,
        arm=arm,
        base_from_camera=np.asarray(meta["t_bc"], dtype=float) if meta.get("t_bc") else None,
        annotations=None,
        demo=pathlib.Path(meta["demo_root"]) if meta.get("demo_root") else None,
        designations=_act_designations(root, meta, np.asarray(times, dtype=float)),
    )


def _act_designations(
    root: pathlib.Path, meta: dict[str, Any], times: np.ndarray
) -> dict[str, dict[str, Any]]:
    """The masks an act's own run started tracking from: its first tracker frame's, the session object's in its mask
    image and each other's in the frame's arrays, at the camera frame captured nearest it."""
    frames = meta.get("frames") or []
    names = meta.get("objects") or []
    if not frames or not names or not len(times):
        return {}
    import cv2

    first, d = frames[0], root / "frames"
    at = int(np.argmin(np.abs(times - float(first.get("t_capture", first.get("t_frame", times[0]))))))
    out: dict[str, dict[str, Any]] = {}
    mask = cv2.imread(str(d / f"{int(first['i']):06d}_mask.png"), cv2.IMREAD_UNCHANGED)
    if mask is not None:
        out[names[0]] = {"frame": at, "mask": mask > 0}
    arrays = d / f"{int(first['i']):06d}.npz"
    if arrays.exists():
        z = np.load(arrays, allow_pickle=False)
        for k, name in enumerate(first.get("others") or []):
            if f"other_mask_{k}" in z.files:
                out[name] = {"frame": at, "mask": np.asarray(z[f"other_mask_{k}"], dtype=bool)}
    return out
