"""Scan-to-3D: SAM3 mask + SAM 3D Objects → a mesh shown in the URDF viewer.

Prototype. The scan (SAM3 segment + SAM 3D Objects in the isolated ~/.cache/sam3d
venv) is slow/offline, so the result — a metric GLB + a base-frame placement — is
cached here and the URDF viewer (gui/static/urdf_viz.html) overlays it on the robot.

Pose anchoring (metric scale from RealSense depth + table plane + camera→base
extrinsic) is the do-as-i-do reconstruction; this module currently serves a cached
mesh with a placeholder placement so the render path can be built + validated first.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/run/scan3d", tags=["scan3d"])

SCAN_DIR = Path.home() / ".cache/huggingface/lerobot/gui/scan3d"
MESH_PATH = SCAN_DIR / "object.glb"
PLACEMENT_PATH = SCAN_DIR / "placement.json"

# Placement in the robot BASE frame (three.js Y-up, metres). Real anchoring replaces
# this; until then a scan writes placement.json and we fall back to this default.
_DEFAULT_PLACEMENT = {
    "position": [0.20, 0.0, 0.0],  # ~20 cm in front of the base
    "quaternion": [0.0, 0.0, 0.0, 1.0],  # identity (xyzw)
    "scale": 1.0,
}


def _read_placement() -> dict:
    if PLACEMENT_PATH.exists():
        try:
            return {**_DEFAULT_PLACEMENT, **json.loads(PLACEMENT_PATH.read_text())}
        except Exception:
            logger.warning("scan3d: bad placement.json, using default")
    return dict(_DEFAULT_PLACEMENT)


@router.get("/object")
async def scan_object() -> dict:
    """Whether a scanned object is available, and where to place it on the robot."""
    if not MESH_PATH.exists():
        return {"available": False}
    return {"available": True, "version": int(MESH_PATH.stat().st_mtime), **_read_placement()}


@router.get("/mesh")
async def scan_mesh() -> Response:
    """The scanned object's mesh (GLB, vertex-coloured)."""
    if not MESH_PATH.exists():
        return Response(status_code=404)
    return FileResponse(MESH_PATH, media_type="model/gltf-binary", headers={"Cache-Control": "no-store"})
