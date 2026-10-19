# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What one pipeline makes of a recording (lerobot.showservo.recordings), the same for a demo and an act, so a demo's
beliefs come from the rules the runtime uses (docs/act_execution.md, "Recordings and their perception").

The pipeline, every frame in order, no clock involved: the point groups (benchmarks/group_live.py in P2P's
environment, with the settings the GUI's live view runs with), each object designated where the recording
designated it; then frame by frame the pose estimator the engine uses on the groups' poses (SeenThrough), against
the frame's own depth. Everything is in the camera's frame: a recording need not know where the camera stood.

The result is cached beside the recording under ``perception/<version>/``, the version a hash of the pipeline's
source files: a change of rule is a new version, and the recording is processed again (:func:`process`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

import numpy as np

from .estimation import GROUPS, Candidate, Frame, SeenThrough
from .recordings import Recording

ROOT = pathlib.Path(__file__).resolve().parents[3]
GROUP_SCRIPT = ROOT / "benchmarks" / "group_live.py"
# The live view's TAPIR settings (lerobot.gui.api.manipulation.groups._start_groups runs it with these too).
GROUP_VIEW_ARGS = ("--pips", "2")
SOURCES = (
    pathlib.Path(__file__).parent / "groups.py",
    pathlib.Path(__file__).parent / "groups_scene.py",
    pathlib.Path(__file__).parent / "wireframe.py",
    pathlib.Path(__file__).parent / "pose.py",
    pathlib.Path(__file__).parent / "placement.py",
    pathlib.Path(__file__).parent / "estimation.py",
    pathlib.Path(__file__),
    GROUP_SCRIPT,
)
RECORD = "perception.json"
SURFACE_STEP = 4  # every fourth pixel of an object's mask, as the act's surfaces (planning._object_points)


def version() -> str:
    """The pipeline's version: a hash of its source files, so any change of rule is a new version."""
    h = hashlib.sha1(usedforsecurity=False)
    for path in SOURCES:
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def cache_dir(recording: Recording) -> pathlib.Path:
    return recording.root / "perception" / version()


@dataclasses.dataclass(frozen=True)
class Perception:
    """A recording's perception, frame by frame: ``t`` each frame's time (wall clock), each object's pose as the
    groups had it (camera <- object) and whether it was moving to another group, its designation mask, the
    estimator's rows (the engine's feed format: candidates, taken or not and why), and each frame's groups picture
    (its fields here, its arrays by :meth:`picture`)."""

    root: pathlib.Path
    version: str
    names: list[str]
    t: np.ndarray
    poses: dict[str, np.ndarray]
    moving: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    rows: list[dict[str, Any]]
    pictures: list[dict[str, Any]]

    def motion(self, name: str, k: int) -> np.ndarray:
        """``name``'s motion from its designation to frame ``k``, camera frame."""
        return self.poses[name][k] @ np.linalg.inv(self.poses[name][0])

    def picture(self, k: int) -> dict[str, np.ndarray]:
        """Frame ``k``'s picture arrays, under the names an act's record keeps them (groups_points, ...)."""
        z = np.load(self.root / "groups.frames" / f"{k:06d}.npz", allow_pickle=False)
        return {key: np.asarray(z[key]) for key in z.files}


def load(recording: Recording) -> Perception | None:
    """The recording's perception by the pipeline as it is now, or None when it has not been processed by it."""
    return load_dir(cache_dir(recording))


def load_dir(d: pathlib.Path) -> Perception | None:
    """The perception cached in ``d``, whichever version made it; None when there is none there."""
    if not (d / RECORD).exists():
        return None
    record = json.loads((d / RECORD).read_text())
    z = np.load(d / "groups.npz", allow_pickle=False)
    names = json.loads(str(z["names"]))
    return Perception(
        root=d,
        version=record["version"],
        names=names,
        t=np.asarray(z["t"], dtype=float),
        poses={n: np.asarray(z["poses"][:, i]) for i, n in enumerate(names)},
        moving={n: np.asarray(z["moving"][:, i]) for i, n in enumerate(names)},
        masks={n: np.asarray(z[f"mask_{i}"]) for i, n in enumerate(names) if f"mask_{i}" in z.files},
        rows=record["frames"],
        pictures=json.loads(str(z["pictures"])),
    )


def designation_clicks(recording: Recording) -> dict[str, tuple[int, int]]:
    """Where each designated object is clicked to designate it in the groups: the pixel deepest inside its mask."""
    import cv2

    out = {}
    for name, d in recording.designations.items():
        mask = np.asarray(d["mask"], dtype=np.uint8)
        if not mask.any():
            continue
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        out[name] = (int(x), int(y))
    return out


def process(
    recording: Recording,
    *,
    python: str | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Perception:
    """The recording through the pipeline, every frame in order, into its cache; the cache as it is when the pipeline
    has processed it already and not ``force``. ``python`` is P2P's interpreter (lerobot.p2p_runtime.python_path)."""
    if not force and (done := load(recording)) is not None:
        return done
    from lerobot.p2p_runtime import python_path

    d = cache_dir(recording)
    d.mkdir(parents=True, exist_ok=True)
    stage = _stage(recording, d / "stage")
    clicks = designation_clicks(recording)
    if not clicks:
        raise ValueError(f"{recording.root}: nothing designated to follow")
    log(f"the point groups over {len(recording)} frames of {recording.root}")
    with open(d / "groups.log", "w") as out:
        subprocess.run(  # nosec B603: our own script, in P2P's own interpreter
            group_command(python or python_path(), stage, d / "tracks.npz", clicks),
            check=True,
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
    done = finish(recording, d)
    log(f"perception of {recording.root}: {d}")
    return done


def group_command(
    python: str, stage: pathlib.Path, dump: pathlib.Path, clicks: dict[str, tuple[int, int]]
) -> list[str]:
    """The groups' script over the staged frames, dumping to ``dump``, with the live view's settings."""
    cmd = [python, str(GROUP_SCRIPT), "--recording", str(stage), "--dump", str(dump)]
    for name, (x, y) in clicks.items():
        cmd += ["--object", f"{name}={x},{y}"]
    return cmd + list(GROUP_VIEW_ARGS)


def finish(recording: Recording, d: pathlib.Path) -> Perception:
    """The groups' dump in ``d`` (tracks.npz and beside it) made the recording's perception: their files in place, the
    estimator's rows written, the version last, so a cache is complete only once its record is there."""
    os.replace(d / "tracks.npz.groups.npz", d / "groups.npz")
    if (d / "groups.frames").exists():
        # safe-destruct: this cache's own pictures, from an earlier run of it
        shutil.rmtree(d / "groups.frames")
    os.replace(d / "tracks.npz.frames", d / "groups.frames")
    rows = estimate(recording, d)
    (d / RECORD).write_text(
        json.dumps(
            {"version": version(), "recording": str(recording.root), "kind": recording.kind, "frames": rows}
        )
    )
    done = load(recording)
    assert done is not None
    return done


def estimate(recording: Recording, d: pathlib.Path) -> list[dict[str, Any]]:
    """Frame by frame, each designated object's candidate from the groups and what the engine's estimator makes of it
    against the frame's depth (its rows, as an act's engine.feed keeps them)."""
    z = np.load(d / "groups.npz", allow_pickle=False)
    names = json.loads(str(z["names"]))
    poses, moving, t = np.asarray(z["poses"]), np.asarray(z["moving"]), np.asarray(z["t"], dtype=float)
    fx, fy, cx, cy = recording.K[0, 0], recording.K[1, 1], recording.K[0, 2], recording.K[1, 2]
    surfaces = {}
    for i, name in enumerate(names):
        if f"mask_{i}" not in z.files:
            continue
        k0 = recording.frame_at(float(t[0]))
        surfaces[name] = surface(np.asarray(z[f"mask_{i}"]), recording.depth(k0), recording.K)
    est = SeenThrough(GROUPS, surfaces)
    rows = []
    for k in range(len(t)):
        frame = Frame(recording.depth(recording.frame_at(float(t[k]))), fx, fy, cx, cy, np.eye(4))
        candidates = {}
        for i, name in enumerate(names):
            motion = poses[k, i] @ np.linalg.inv(poses[0, i])
            ok = not bool(moving[k, i])
            candidates[name] = [Candidate(GROUPS, motion, ok, "" if ok else "moving to another group")]
        row: dict[str, Any] = {"i": k, "t": float(t[k]), "objects": {}}
        for name, e in est.estimate(float(t[k]), candidates, frame).items():
            c = candidates[name][0]
            row["objects"][name] = {
                "candidates": [
                    {"source": c.source, "ok": c.ok, "why": c.why, "motion": np.round(c.motion, 5).tolist()}
                ],
                "taken": e.motion is not None,
                "why": e.why,
                "seen_through": None if e.seen_through is None else round(e.seen_through, 4),
            }
        rows.append(row)
    return rows


def surface(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:  # noqa: N803
    """An object's surface at its designation, camera frame: the depth under its mask, every SURFACE_STEP-th pixel."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool) & (depth_m > 0))
    ys, xs = ys[::SURFACE_STEP], xs[::SURFACE_STEP]
    z = depth_m[ys, xs]
    return np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], axis=1)


def _stage(recording: Recording, stage: pathlib.Path) -> pathlib.Path:
    """The recording's frames in the layout the groups' script reads (rgb/%06d.jpg, depth/%06d.png in mm, times.txt,
    cam_K.txt), linked, not copied."""
    for sub in ("rgb", "depth"):
        (stage / sub).mkdir(parents=True, exist_ok=True)
    for k, (image, depth) in enumerate(zip(recording.images, recording.depths, strict=True)):
        for src, dst in ((image, stage / "rgb" / f"{k:06d}.jpg"), (depth, stage / "depth" / f"{k:06d}.png")):
            if dst.is_symlink() or dst.exists():
                dst.unlink()  # safe-destruct: a link this stage made, to the recording's file, never the file
            dst.symlink_to(src.resolve())
    np.savetxt(stage / "times.txt", recording.times, fmt="%.6f")
    np.savetxt(stage / "cam_K.txt", recording.K)
    return stage


def main() -> None:
    import argparse

    from .recordings import open_recording

    ap = argparse.ArgumentParser(description="process a demo or an act recording into its perception cache")
    ap.add_argument("recording")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    p = process(open_recording(args.recording), force=args.force)
    taken = sum(o["taken"] for r in p.rows for o in r["objects"].values())
    print(f"{len(p.rows)} frames, {taken} poses taken of {sum(len(r['objects']) for r in p.rows)}: {p.root}")


if __name__ == "__main__":
    main()
