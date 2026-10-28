"""Measured RGB-D geometry for inspecting a prepared FLASH demonstration.

Meshes are display evidence only. They do not fill unseen surfaces or change the
prepared skill. A reference mesh is stored in the same object coordinates as
the prepared reference snippet, so the saved pose can place it in each frame.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import transform


def reference_surfaces(dataset_root: Path, bundle_path: Path, *, step_px: int = 2) -> dict:
    root = dataset_root.expanduser().resolve()
    bundle = bundle_path.expanduser().resolve()
    if step_px < 1:
        raise ValueError("mesh sampling step must be positive")
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    intrinsics = np.loadtxt(root / "recording" / "cam_K.txt").reshape(3, 3)
    result: dict[str, dict] = {}
    for reference in manifest["references"]:
        if not reference["snippets"]:
            continue
        positions_by_view, colors_by_view, faces_by_view = [], [], []
        frame_indices = []
        vertex_count = 0
        for snippet in reference["snippets"]:
            frame_index = int(snippet["frame_index"])
            mask_path = (root / snippet["mask_artifact"]).resolve()
            mask_path.relative_to((root / "flash" / "v1" / "masks").resolve())
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            depth_mm = cv2.imread(
                str(root / "recording" / "depth" / f"{frame_index:06d}.png"), cv2.IMREAD_UNCHANGED
            )
            rgb = cv2.imread(str(root / "recording" / "rgb" / f"{frame_index:06d}.jpg"))
            if mask_image is None or depth_mm is None or rgb is None:
                raise ValueError(f"reference RGB-D or mask is missing for {reference['role']}")
            if mask_image.shape != depth_mm.shape or rgb.shape[:2] != depth_mm.shape:
                raise ValueError(f"reference RGB-D and mask are misaligned for {reference['role']}")
            depth = depth_mm[::step_px, ::step_px].astype(np.float64) / 1000
            valid = (mask_image[::step_px, ::step_px] > 0) & np.isfinite(depth) & (depth > 0)
            rows, cols = np.nonzero(valid)
            z = depth[rows, cols]
            x = (cols * step_px - intrinsics[0, 2]) * z / intrinsics[0, 0]
            y = (rows * step_px - intrinsics[1, 2]) * z / intrinsics[1, 1]
            camera_points = np.stack((x, y, z, np.ones_like(z)), axis=1)
            object_from_camera = transform(snippet["object_from_camera"])
            positions = (camera_points @ object_from_camera.T)[:, :3]
            indices = np.full(valid.shape, -1, dtype=np.int32)
            indices[rows, cols] = np.arange(len(rows), dtype=np.int32)
            a, b, c, d = indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]
            faces = np.concatenate(
                (
                    np.stack((a, c, b), axis=-1).reshape(-1, 3),
                    np.stack((b, c, d), axis=-1).reshape(-1, 3),
                )
            )
            faces = faces[np.all(faces >= 0, axis=1)]
            # A projected pixel footprint gives the local scale of a continuous
            # surface. Do not bridge a larger depth edge with a display triangle.
            depths = z[faces]
            pixel_footprint = step_px * depths.mean(axis=1) / min(intrinsics[0, 0], intrinsics[1, 1])
            faces = faces[np.ptp(depths, axis=1) <= 3 * pixel_footprint]
            positions_by_view.append(positions)
            colors_by_view.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[rows * step_px, cols * step_px])
            faces_by_view.append(faces + vertex_count)
            vertex_count += len(positions)
            frame_indices.append(frame_index)
        result[reference["role"]] = {
            "source_frame_index": frame_indices[0],
            "source_frame_indices": frame_indices,
            "positions_object_m": np.concatenate(positions_by_view).round(6).ravel().tolist(),
            "colors_rgb": np.concatenate(colors_by_view).tolist(),
            "indices": np.concatenate(faces_by_view).ravel().tolist(),
        }
    return result


def scene_evidence_status(bundle_path: Path) -> dict:
    """Report whether a same-version RGB-D replay has rebuilt scene evidence."""
    bundle = bundle_path.expanduser().resolve()
    sidecar = bundle.parent / "scene-evidence"
    manifest_path = sidecar / "manifest.json"
    if not manifest_path.is_file():
        return {"available": False, "frames_completed": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = hashlib.sha256(
        bundle.read_bytes() + (bundle.parent / "annotation.json").read_bytes()
    ).hexdigest()
    if manifest.get("schema_version") != 1 or manifest.get("source_sha256") != identity:
        return {"available": False, "frames_completed": 0, "reason": "sidecar version differs"}
    return {
        "available": True,
        "frames_completed": int(manifest["frames_completed"]),
        "total_frames": int(manifest["total_frames"]),
        "complete": bool(manifest["complete"]),
    }


def scene_evidence_frame(bundle_path: Path, frame_index: int) -> dict:
    if frame_index < 0:
        raise ValueError("frame index must be nonnegative")
    status = scene_evidence_status(bundle_path)
    if not status["available"] or frame_index >= status["frames_completed"]:
        return {"available": False, "status": status}
    path = bundle_path.expanduser().resolve().parent / "scene-evidence" / f"{frame_index:06d}.json.gz"
    if not path.is_file():
        return {"available": False, "status": status}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if data["frame_index"] != frame_index:
        raise ValueError("scene evidence frame does not match the requested index")
    return {"available": True, **data}
