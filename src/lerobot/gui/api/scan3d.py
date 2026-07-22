"""Scan-to-3D: SAM3 mask + SAM 3D Objects → a mesh shown in the URDF viewer.

Prototype. The scan (SAM3 segment + SAM 3D Objects in the isolated ~/.cache/sam3d
venv) is slow/offline, so the result — a metric GLB + a base-frame placement — is
cached here and the URDF viewer (gui/static/urdf_viz.html) overlays it on the robot.

Pose anchoring (metric scale from RealSense depth + table plane + camera→base
extrinsic) is the do-as-i-do reconstruction; this module currently serves a cached
mesh with a placeholder placement so the render path can be built + validated first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

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


# ============================================================================
# 3D Scanner tab — capture a frame from a detected camera + run the scan pipeline.
# ============================================================================

# The cached cylinder_ring frame from the #37 scan3d work — the static-image test input.
STATIC_TEST_IMAGE = SCAN_DIR / "in_rgb.png"
# The latest camera capture (the scan input for the real-object path).
CAPTURE_PATH = SCAN_DIR / "capture.png"
# Multi-view capture set: one photo per object placement/angle — the future fusion input.
CAPTURES_DIR = SCAN_DIR / "captures"

# Single in-flight scan; the tab polls /scan/status.
_scan: dict = {"running": False, "done": False, "error": None, "log": "", "image": ""}


class ScanRequest(BaseModel):
    prompt: str = "object"
    image: str | None = None  # explicit path; else the latest capture; else the static test frame
    use_static: bool = False  # force the cached cylinder_ring test frame (the "test static image" button)
    threshold: float = 0.5


async def _run_scan(image_path: str, prompt: str, threshold: float) -> None:
    _scan.update(running=True, done=False, error=None, log="", image=image_path)
    args = [
        sys.executable, "-u", "-m", "lerobot.gui.api.scan3d_worker",
        "--image", image_path, "--prompt", prompt, "--threshold", str(threshold),
    ]  # fmt: skip
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            _scan["log"] += raw.decode(errors="replace")
        rc = await proc.wait()
        _scan["error"] = None if rc == 0 else f"scan worker exited {rc} (see log)"
    except Exception as e:  # noqa: BLE001
        _scan["error"] = str(e)
    finally:
        _scan.update(running=False, done=True)


@router.post("/scan")
async def start_scan(req: ScanRequest) -> dict:
    """Start the SAM3 + SAM 3D pipeline on the given image (default: the latest capture, else the
    static cylinder_ring test frame). The mesh lands in the cache and /object flips available."""
    if _scan["running"]:
        return {"running": True, "message": "a scan is already in progress"}
    if req.use_static:
        img = str(STATIC_TEST_IMAGE)
    else:
        img = req.image or (str(CAPTURE_PATH) if CAPTURE_PATH.exists() else str(STATIC_TEST_IMAGE))
    if not Path(img).exists():
        return {"running": False, "error": f"no image to scan ({img}) — capture a frame first"}
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(_run_scan(img, req.prompt, req.threshold))
    return {"running": True, "image": img}


@router.get("/scan/status")
async def scan_status() -> dict:
    return {k: _scan[k] for k in ("running", "done", "error", "log", "image")}


class CaptureRequest(BaseModel):
    camera_id: str  # the ``id`` from POST /api/robot/detect-cameras


async def _grab_rgb(camera_id: str):
    """One RGB frame from a detected+opened preview camera, or (None, error)."""
    from lerobot.gui.api import robot as robot_mod

    infos = getattr(robot_mod, "_preview_camera_info", [])
    cams = getattr(robot_mod, "_preview_cameras", [])
    idx = next((i for i, ci in enumerate(infos) if str(ci.get("id")) == str(camera_id)), None)
    if idx is None or idx >= len(cams):
        return None, "camera not open — click 'Detect cameras' first"
    import numpy as np

    frame = await asyncio.get_event_loop().run_in_executor(None, cams[idx].read)
    if isinstance(frame, tuple):  # some cameras return (frame, meta)
        frame = frame[0]
    rgb = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])  # detect opens BGR -> RGB
    return rgb, None


@router.post("/capture")
async def capture_frame(req: CaptureRequest) -> dict:
    """Grab ONE frame from a detected+opened preview camera → capture.png (the scan input)."""
    try:
        rgb, err = await _grab_rgb(req.camera_id)
        if rgb is None:
            return {"ok": False, "error": err}
        from PIL import Image

        SCAN_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(CAPTURE_PATH)
        return {"ok": True, "shape": list(rgb.shape)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@router.get("/capture/preview")
async def capture_preview() -> Response:
    """The latest captured frame (or the static test frame if nothing captured yet)."""
    path = CAPTURE_PATH if CAPTURE_PATH.exists() else STATIC_TEST_IMAGE
    if not path.exists():
        return Response(status_code=404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


# --- Multi-view capture set (fusion input; today: collect + eyeball coverage) -------


def _capture_names() -> list[str]:
    if not CAPTURES_DIR.exists():
        return []
    return sorted(p.name for p in CAPTURES_DIR.glob("cap_*.png"))


@router.post("/captures")
async def captures_add(req: CaptureRequest) -> dict:
    """Grab one frame and APPEND it to the capture set (one photo per placement/angle)."""
    try:
        rgb, err = await _grab_rgb(req.camera_id)
        if rgb is None:
            return {"ok": False, "error": err}
        from PIL import Image

        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        seq = 1 + max((int(n[4:7]) for n in _capture_names()), default=0)
        name = f"cap_{seq:03d}.png"
        Image.fromarray(rgb).save(CAPTURES_DIR / name)
        return {"ok": True, "name": name, "captures": _capture_names()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@router.get("/captures")
async def captures_list() -> dict:
    return {"captures": _capture_names()}


@router.delete("/captures")
async def captures_clear() -> dict:
    for n in _capture_names():
        # safe-destruct: user clicked Clear; only our own cap_*.png in the cache dir
        (CAPTURES_DIR / n).unlink(missing_ok=True)
    return {"ok": True, "captures": []}


@router.get("/captures/{name}")
async def captures_image(name: str) -> Response:
    if name not in _capture_names():  # whitelist: no traversal, only real set members
        return Response(status_code=404)
    return FileResponse(CAPTURES_DIR / name, media_type="image/png", headers={"Cache-Control": "no-store"})
