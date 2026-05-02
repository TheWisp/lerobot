"""Robot and teleop profile management, camera detection, and port scanning."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/robot", tags=["robot"])

# Module-level state (same pattern as datasets.py)
_app_state: "AppState" = None  # type: ignore

# Config directories
ROBOT_PROFILES_DIR = Path.home() / ".config" / "lerobot" / "robots"
TELEOP_PROFILES_DIR = Path.home() / ".config" / "lerobot" / "teleops"

# Camera preview state
_preview_cameras: list = []
_preview_camera_info: list[dict] = []

# Rest-position recording state (holds robot connection between start/finish)
_rest_recording_robot = None


def set_app_state(state: "AppState") -> None:
    global _app_state
    _app_state = state


# ============================================================================
# Config loading helpers
# ============================================================================

_configs_loaded = False


def _ensure_configs_loaded():
    """Import all config modules to trigger @register_subclass decorators.

    Some modules have optional hardware dependencies (hebi, pyrealsense2, etc.)
    that may not be installed, so each import is wrapped in try/except.
    """
    global _configs_loaded
    if _configs_loaded:
        return

    _config_modules = [
        # Robot configs
        "lerobot.robots.bi_openarm_follower.config_bi_openarm_follower",
        "lerobot.robots.bi_so107_follower.config_bi_so107_follower",
        "lerobot.robots.bi_so_follower.config_bi_so_follower",
        "lerobot.robots.earthrover_mini_plus.config_earthrover_mini_plus",
        "lerobot.robots.hope_jr.config_hope_jr",
        "lerobot.robots.koch_follower.config_koch_follower",
        "lerobot.robots.lekiwi.config_lekiwi",
        "lerobot.robots.omx_follower.config_omx_follower",
        "lerobot.robots.openarm_follower.config_openarm_follower",
        "lerobot.robots.reachy2.configuration_reachy2",
        "lerobot.robots.so_follower.config_so_follower",
        "lerobot.robots.unitree_g1.config_unitree_g1",
        # Teleoperator configs
        "lerobot.teleoperators.bi_openarm_leader.config_bi_openarm_leader",
        "lerobot.teleoperators.bi_so107_leader.config_bi_so107_leader",
        "lerobot.teleoperators.bi_so_leader.config_bi_so_leader",
        "lerobot.teleoperators.gamepad.configuration_gamepad",
        "lerobot.teleoperators.homunculus.config_homunculus",
        "lerobot.teleoperators.keyboard.configuration_keyboard",
        "lerobot.teleoperators.koch_leader.config_koch_leader",
        "lerobot.teleoperators.omx_leader.config_omx_leader",
        "lerobot.teleoperators.openarm_leader.config_openarm_leader",
        "lerobot.teleoperators.phone.config_phone",
        "lerobot.teleoperators.reachy2_teleoperator.config_reachy2_teleoperator",
        "lerobot.teleoperators.so_leader.config_so_leader",
        "lerobot.teleoperators.unitree_g1.config_unitree_g1",
    ]

    import importlib
    for module_name in _config_modules:
        try:
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as e:
            logger.debug(f"Skipping {module_name}: {e}")

    _configs_loaded = True


# ============================================================================
# Schema introspection
# ============================================================================

# Fields to skip in schema output (handled separately or irrelevant to GUI)
_SKIP_FIELDS = {"cameras", "calibration_dir"}


def _stringify_type(annotation: Any) -> str:
    """Convert a type annotation to a simple string for the frontend."""
    s = str(annotation)
    # Clean up common patterns
    for prefix in ("typing.", "<class '", "pathlib."):
        s = s.replace(prefix, "")
    s = s.rstrip("'>")
    return s


def _introspect_fields(cls: type) -> list[dict]:
    """Extract field info from a dataclass config class."""
    result = []
    for f in dataclasses.fields(cls):
        if f.name in _SKIP_FIELDS:
            continue
        required = (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[arg-type]
        )
        default = None
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[arg-type]
            # Don't call factory, just indicate it has a default
            default = None

        result.append({
            "name": f.name,
            "type_str": _stringify_type(f.type),
            "required": required,
            "default": default,
        })
    return result


@router.get("/schemas")
async def get_robot_schemas() -> list[dict]:
    """Return field schemas for all registered robot config types."""
    from lerobot.robots.config import RobotConfig
    _ensure_configs_loaded()

    schemas = []
    for type_name, config_cls in sorted(RobotConfig.get_known_choices().items()):
        schemas.append({
            "type_name": type_name,
            "fields": _introspect_fields(config_cls),
        })
    return schemas


@router.get("/teleop-schemas")
async def get_teleop_schemas() -> list[dict]:
    """Return field schemas for all registered teleoperator config types."""
    from lerobot.teleoperators.config import TeleoperatorConfig
    _ensure_configs_loaded()

    schemas = []
    for type_name, config_cls in sorted(TeleoperatorConfig.get_known_choices().items()):
        schemas.append({
            "type_name": type_name,
            "fields": _introspect_fields(config_cls),
        })
    return schemas


# ============================================================================
# Profile CRUD
# ============================================================================

class ProfileData(BaseModel):
    type: str
    name: str
    fields: dict[str, Any] = {}
    cameras: dict[str, dict[str, Any]] = {}
    rest_position: dict[str, float] = {}


class RenameRequest(BaseModel):
    new_name: str


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _list_profiles(profiles_dir: Path) -> list[dict]:
    _ensure_dir(profiles_dir)
    profiles = []
    for f in sorted(profiles_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            profiles.append({
                "name": data.get("name", f.stem),
                "type": data.get("type", "unknown"),
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read profile {f}: {e}")
    return profiles


def _read_profile(profiles_dir: Path, name: str) -> dict:
    path = profiles_dir / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Profile '{name}' not found")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Failed to parse profile: {e}")


def _write_profile(profiles_dir: Path, data: ProfileData) -> None:
    _ensure_dir(profiles_dir)
    path = profiles_dir / f"{data.name}.json"
    path.write_text(json.dumps(data.model_dump(), indent=2))
    logger.info(f"Saved profile: {path}")


def _delete_profile(profiles_dir: Path, name: str) -> None:
    path = profiles_dir / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Profile '{name}' not found")
    # safe-destruct: user-confirmed delete via GUI dialog
    path.unlink()
    logger.info(f"Deleted profile: {path}")


def _rename_profile(profiles_dir: Path, old_name: str, new_name: str) -> None:
    old_path = profiles_dir / f"{old_name}.json"
    if not old_path.exists():
        raise HTTPException(404, f"Profile '{old_name}' not found")
    new_path = profiles_dir / f"{new_name}.json"
    if new_path.exists():
        raise HTTPException(409, f"Profile '{new_name}' already exists")
    data = json.loads(old_path.read_text())
    data["name"] = new_name
    new_path.write_text(json.dumps(data, indent=2))
    # safe-destruct: rename: drop old after writing new
    old_path.unlink()
    logger.info(f"Renamed profile: {old_path} -> {new_path}")


# Robot profiles
@router.get("/profiles")
async def list_robot_profiles() -> list[dict]:
    return _list_profiles(ROBOT_PROFILES_DIR)


@router.post("/profiles")
async def create_robot_profile(profile: ProfileData) -> dict:
    path = ROBOT_PROFILES_DIR / f"{profile.name}.json"
    if path.exists():
        raise HTTPException(409, f"Profile '{profile.name}' already exists")
    _write_profile(ROBOT_PROFILES_DIR, profile)
    return {"status": "created", "name": profile.name}


@router.get("/profiles/{name}")
async def get_robot_profile(name: str) -> dict:
    return _read_profile(ROBOT_PROFILES_DIR, name)


@router.put("/profiles/{name}")
async def update_robot_profile(name: str, profile: ProfileData) -> dict:
    _write_profile(ROBOT_PROFILES_DIR, profile)
    return {"status": "updated", "name": profile.name}


@router.delete("/profiles/{name}")
async def delete_robot_profile(name: str) -> dict:
    _delete_profile(ROBOT_PROFILES_DIR, name)
    return {"status": "deleted", "name": name}


@router.post("/profiles/{name}/rename")
async def rename_robot_profile(name: str, req: RenameRequest) -> dict:
    _rename_profile(ROBOT_PROFILES_DIR, name, req.new_name)
    return {"status": "renamed", "old_name": name, "new_name": req.new_name}


# Teleop profiles
@router.get("/teleop-profiles")
async def list_teleop_profiles() -> list[dict]:
    return _list_profiles(TELEOP_PROFILES_DIR)


@router.post("/teleop-profiles")
async def create_teleop_profile(profile: ProfileData) -> dict:
    path = TELEOP_PROFILES_DIR / f"{profile.name}.json"
    if path.exists():
        raise HTTPException(409, f"Profile '{profile.name}' already exists")
    _write_profile(TELEOP_PROFILES_DIR, profile)
    return {"status": "created", "name": profile.name}


@router.get("/teleop-profiles/{name}")
async def get_teleop_profile(name: str) -> dict:
    return _read_profile(TELEOP_PROFILES_DIR, name)


@router.put("/teleop-profiles/{name}")
async def update_teleop_profile(name: str, profile: ProfileData) -> dict:
    _write_profile(TELEOP_PROFILES_DIR, profile)
    return {"status": "updated", "name": profile.name}


@router.delete("/teleop-profiles/{name}")
async def delete_teleop_profile(name: str) -> dict:
    _delete_profile(TELEOP_PROFILES_DIR, name)
    return {"status": "deleted", "name": name}


@router.post("/teleop-profiles/{name}/rename")
async def rename_teleop_profile(name: str, req: RenameRequest) -> dict:
    _rename_profile(TELEOP_PROFILES_DIR, name, req.new_name)
    return {"status": "renamed", "old_name": name, "new_name": req.new_name}


# ============================================================================
# Camera detection and preview
# ============================================================================

def _detect_and_open_cameras() -> list[dict]:
    """Detect cameras and open them for live preview. Runs in thread pool.

    RealSense cameras are detected and opened first via pyrealsense2.
    This claims their V4L2 device nodes so that the subsequent OpenCV
    scan naturally skips them (they appear as busy / unopenable).
    """
    global _preview_cameras, _preview_camera_info
    _close_preview_cameras()

    from lerobot.cameras.configs import ColorMode

    all_cameras: list[dict] = []

    # --- Phase 1: detect + open RealSense cameras first ---
    # This claims /dev/video* nodes owned by RealSense so OpenCV skips them.
    try:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        realsense_cams = RealSenseCamera.find_cameras()
        logger.info(f"Found {len(realsense_cams)} RealSense camera(s)")

        for cam_info in realsense_cams:
            cam_id = cam_info.get("id")
            try:
                config = RealSenseCameraConfig(
                    serial_number_or_name=str(cam_id), color_mode=ColorMode.BGR,
                )
                camera = RealSenseCamera(config)
                camera.connect(warmup=True)
                _preview_cameras.append(camera)
                _preview_camera_info.append(cam_info)
                all_cameras.append(cam_info)
                logger.info(f"Opened preview camera: RealSense {cam_id}")
            except Exception as e:
                logger.warning(f"Failed to open RealSense {cam_id}: {e}")
                all_cameras.append(cam_info)
    except Exception as e:
        logger.warning(f"RealSense camera detection failed: {e}")

    # --- Phase 2: detect + open OpenCV cameras ---
    # RealSense V4L2 nodes are now busy, so find_cameras() skips most.
    # We also explicitly filter any remaining RealSense nodes via sysfs.
    try:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        opencv_cams = OpenCVCamera.find_cameras()
        logger.info(f"Found {len(opencv_cams)} OpenCV camera(s)")

        for cam_info in opencv_cams:
            cam_id = cam_info.get("id")
            # On Linux, skip V4L2 nodes that belong to RealSense
            if isinstance(cam_id, str) and cam_id.startswith("/dev/video"):
                devname = cam_id.split("/")[-1]  # e.g. "video6"
                sysfs_name = Path(f"/sys/class/video4linux/{devname}/name")
                if sysfs_name.exists():
                    hw_name = sysfs_name.read_text().strip()
                    if "RealSense" in hw_name:
                        logger.debug(f"Skipping {cam_id} (RealSense V4L2 node: {hw_name})")
                        continue
            try:
                config = OpenCVCameraConfig(
                    index_or_path=cam_id, color_mode=ColorMode.BGR,
                )
                camera = OpenCVCamera(config)
                camera.connect(warmup=True)
                _preview_cameras.append(camera)
                _preview_camera_info.append(cam_info)
                all_cameras.append(cam_info)
                logger.info(f"Opened preview camera: OpenCV {cam_id}")
            except Exception as e:
                logger.warning(f"Failed to open OpenCV {cam_id}: {e}")
    except Exception as e:
        logger.warning(f"OpenCV camera detection failed: {e}")

    return all_cameras


def _close_preview_cameras() -> None:
    """Disconnect all preview cameras."""
    global _preview_cameras, _preview_camera_info
    for cam in _preview_cameras:
        try:
            cam.disconnect()
        except Exception:
            pass
    _preview_cameras.clear()
    _preview_camera_info.clear()


@router.post("/detect-cameras")
async def detect_cameras() -> list[dict]:
    """Detect available cameras and open them for preview."""
    loop = asyncio.get_event_loop()
    cameras = await loop.run_in_executor(None, _detect_and_open_cameras)
    return cameras


@router.get("/camera-frame/{index}")
async def get_camera_frame(index: int) -> Response:
    """Return a JPEG frame from a preview camera."""
    if index < 0 or index >= len(_preview_cameras):
        raise HTTPException(404, "Camera index out of range")
    camera = _preview_cameras[index]
    try:
        frame = camera.async_read()
        # Camera is opened with BGR color mode, so no conversion needed
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(500, f"Failed to read camera frame: {e}")


@router.post("/stop-cameras")
async def stop_cameras() -> dict:
    """Disconnect all preview cameras."""
    await asyncio.get_event_loop().run_in_executor(None, _close_preview_cameras)
    return {"status": "ok"}


# ============================================================================
# Port scanning and arm identification
# ============================================================================

@router.get("/ports")
async def scan_ports() -> list[dict]:
    """Scan for USB serial ports (ttyACM*, ttyUSB* on Linux).

    Only shows USB serial adapters, not legacy serial ports (ttyS*),
    virtual terminals (tty0-63), or kernel consoles (ttyprintk).
    Uses pyserial when available for richer device metadata.
    """
    ports = []
    try:
        from serial.tools import list_ports
        for port_info in sorted(list_ports.comports(), key=lambda p: p.device):
            # Filter to USB serial ports only
            dev = port_info.device
            devname = Path(dev).name
            if not (devname.startswith("ttyACM") or devname.startswith("ttyUSB")):
                if platform.system() == "Linux":
                    continue  # Skip non-USB ports on Linux
            ports.append({
                "path": dev,
                "name": port_info.description or devname,
                "manufacturer": port_info.manufacturer or "",
                "vid_pid": f"{port_info.vid:04x}:{port_info.pid:04x}" if port_info.vid else "",
            })
    except ImportError:
        # Fallback: glob for USB serial devices
        if platform.system() == "Linux":
            for pattern in ["ttyACM*", "ttyUSB*"]:
                for p in sorted(Path("/dev").glob(pattern)):
                    ports.append({"path": str(p), "name": p.name})
        else:
            logger.warning("pyserial not installed, cannot scan ports")
    return ports


def _wiggle_shoulder(port: str) -> dict:
    """Wiggle the shoulder motor on the given port. Runs in thread pool."""
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        return {"status": "error", "message": "lerobot motor modules not available"}

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    }

    bus = None
    try:
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        current_pos = bus.read("Present_Position", "shoulder_pan", normalize=False)
        ticks = 200

        bus.write("Goal_Position", "shoulder_pan", current_pos + ticks, normalize=False)
        time.sleep(0.5)
        bus.write("Goal_Position", "shoulder_pan", current_pos - ticks, normalize=False)
        time.sleep(0.5)
        bus.write("Goal_Position", "shoulder_pan", current_pos, normalize=False)
        time.sleep(0.3)

        return {"status": "ok", "port": port}
    except Exception as e:
        return {"status": "error", "port": port, "message": str(e)}
    finally:
        if bus:
            try:
                bus.disconnect()
            except Exception:
                pass


def _collect_all_port_assignments() -> list[dict]:
    """Read all saved profiles and extract port field values.

    Returns a flat list of {port, profile_name, profile_kind, field_name}.
    """
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig
    _ensure_configs_loaded()

    assignments = []

    for profiles_dir, kind, base_cls in [
        (ROBOT_PROFILES_DIR, "robot", RobotConfig),
        (TELEOP_PROFILES_DIR, "teleop", TeleoperatorConfig),
    ]:
        _ensure_dir(profiles_dir)
        for f in profiles_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            profile_name = data.get("name", f.stem)
            profile_type = data.get("type")
            fields_data = data.get("fields", {})

            # Find which fields are port fields from the schema
            choices = base_cls.get_known_choices()
            config_cls = choices.get(profile_type)
            if not config_cls:
                continue
            for field in dataclasses.fields(config_cls):
                if field.name in _SKIP_FIELDS:
                    continue
                if "port" in field.name and "str" in _stringify_type(field.type).lower():
                    port_value = fields_data.get(field.name)
                    if port_value and isinstance(port_value, str) and port_value.strip():
                        assignments.append({
                            "port": port_value.strip(),
                            "profile_name": profile_name,
                            "profile_kind": kind,
                            "field_name": field.name,
                        })

    return assignments


@router.get("/all-port-assignments")
async def get_all_port_assignments() -> list[dict]:
    """Return all port assignments across all saved profiles."""
    return _collect_all_port_assignments()


class IdentifyArmRequest(BaseModel):
    port: str


@router.post("/open-in-files")
async def open_in_file_manager(body: dict) -> dict:
    """Open a profile directory in the system file manager."""
    import subprocess as _subprocess

    kind = body.get("kind", "robot")
    profiles_dir = ROBOT_PROFILES_DIR if kind == "robot" else TELEOP_PROFILES_DIR
    _ensure_dir(profiles_dir)

    try:
        _subprocess.Popen(["xdg-open", str(profiles_dir)])
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="xdg-open not found")

    return {"status": "ok"}


@router.post("/identify-arm")
async def identify_arm(request: IdentifyArmRequest) -> dict:
    """Wiggle the shoulder motor on the given port to identify which arm it is."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _wiggle_shoulder, request.port)
    return result


# ============================================================================
# Rest position
# ============================================================================


def _make_robot_from_profile(profile: dict):
    """Instantiate a Robot from GUI profile data (no subprocess).

    Builds the correct RobotConfig via draccus type dispatch, skipping cameras
    (not needed for motor-only operations like rest position).
    """
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.utils import make_robot_from_config
    _ensure_configs_loaded()

    config_dict = {"type": profile["type"]}
    config_dict.update(profile.get("fields", {}))

    import draccus
    config = draccus.decode(RobotConfig, config_dict)
    return make_robot_from_config(config)


def _disable_torque(device) -> None:
    """Disable torque on all motors regardless of device type."""
    if hasattr(device, "bus"):
        device.bus.disable_torque()
    elif hasattr(device, "left_arm") and hasattr(device, "right_arm"):
        device.left_arm.bus.disable_torque()
        device.right_arm.bus.disable_torque()
    else:
        raise RuntimeError(f"Cannot disable torque: unsupported device type {type(device).__name__}")


def _do_start_rest_recording(profile: dict) -> dict:
    """Connect to robot and disable torque so user can move it. Runs in thread pool."""
    global _rest_recording_robot

    # Clean up any previous recording session
    _do_cancel_rest_recording()

    robot = _make_robot_from_profile(profile)
    try:
        robot.connect()
        _disable_torque(robot)
        _rest_recording_robot = robot
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to start rest position recording")
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass
        return {"status": "error", "message": str(e)}


def _do_finish_rest_recording() -> dict:
    """Read current positions from the held robot, then disconnect. Runs in thread pool."""
    global _rest_recording_robot
    from lerobot.robots.rest_position import record_rest_position

    robot = _rest_recording_robot
    if robot is None:
        return {"status": "error", "message": "No recording session active"}

    try:
        rest_pos = record_rest_position(robot)
        return {"status": "ok", "rest_position": rest_pos}
    except Exception as e:
        logger.exception("Failed to finish rest position recording")
        return {"status": "error", "message": str(e)}
    finally:
        _rest_recording_robot = None
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass


def _do_cancel_rest_recording() -> dict:
    """Disconnect the held robot without reading positions. Runs in thread pool."""
    global _rest_recording_robot

    robot = _rest_recording_robot
    _rest_recording_robot = None
    if robot is None:
        return {"status": "ok"}
    try:
        if robot.is_connected:
            robot.disconnect()
    except Exception:
        pass
    return {"status": "ok"}


def _do_move_to_rest_position(profile: dict, rest_position: dict, duration_s: float) -> dict:
    """Connect to robot, interpolate to rest, disconnect. Runs in thread pool."""
    from lerobot.robots.rest_position import move_to_rest_position

    robot = _make_robot_from_profile(profile)
    try:
        robot.connect()
        move_to_rest_position(robot, rest_position, duration_s=duration_s)
        _disable_torque(robot)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to move to rest position")
        return {"status": "error", "message": str(e)}
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass


class RestPositionRecordRequest(BaseModel):
    robot: dict[str, Any]


class RestPositionMoveRequest(BaseModel):
    robot: dict[str, Any]
    rest_position: dict[str, float]
    duration_s: float = 3.0


@router.post("/start-rest-recording")
async def start_rest_recording_endpoint(req: RestPositionRecordRequest) -> dict:
    """Connect to robot and disable torque so user can move it to rest pose."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_start_rest_recording, req.robot)
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


@router.post("/finish-rest-recording")
async def finish_rest_recording_endpoint() -> dict:
    """Read positions from the held robot connection and disconnect."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_finish_rest_recording)
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


@router.post("/cancel-rest-recording")
async def cancel_rest_recording_endpoint() -> dict:
    """Cancel recording session — disconnect without reading."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_cancel_rest_recording)
    return result


@router.post("/move-to-rest-position")
async def move_to_rest_position_endpoint(req: RestPositionMoveRequest) -> dict:
    """Connect to robot, smoothly move to rest position, disconnect."""
    if not req.rest_position:
        raise HTTPException(400, "rest_position is empty")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _do_move_to_rest_position, req.robot, req.rest_position, req.duration_s
    )
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result
