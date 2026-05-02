"""Environment (sim) profile management.

Mirror of robot.py's profile CRUD, but for `EnvConfig` subclasses (gym-hil,
aloha, libero, metaworld, isaaclab_arena, EnvHub) instead of `RobotConfig`.

Profiles live at ~/.config/lerobot/envs/<name>.json with the shape
``{name, type, fields}``. Dispatch (writing CLI args / config files for the
sim subprocess) lives in run.py — this module only deals with profile
storage and config-class introspection.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/env", tags=["env"])

ENV_PROFILES_DIR = Path.home() / ".config" / "lerobot" / "envs"


# ============================================================================
# EnvConfig registry loading
# ============================================================================

_envs_loaded = False


def _ensure_envs_loaded() -> None:
    """Import lerobot.envs.configs to register all @EnvConfig.register_subclass."""
    global _envs_loaded
    if _envs_loaded:
        return
    import lerobot.envs.configs  # noqa: F401  registers subclasses

    _envs_loaded = True


# Fields whose values come from the GUI as nested config objects (or which we
# choose not to expose in the simple form). Matches robot.py's _SKIP_FIELDS in
# spirit. ``processor`` is a nested dataclass for HILSerlRobotEnvConfig — too
# deep for the v1 form; user can edit raw JSON if needed.
_SKIP_FIELDS = {"features", "features_map", "robot", "teleop", "processor"}


def _stringify_type(annotation: Any) -> str:
    s = str(annotation)
    for prefix in ("typing.", "<class '", "pathlib."):
        s = s.replace(prefix, "")
    s = s.rstrip("'>")
    return s


def _introspect_fields(cls: type) -> list[dict]:
    result = []
    for f in dataclasses.fields(cls):
        if f.name in _SKIP_FIELDS:
            continue
        required = (
            f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING  # type: ignore[arg-type]
        )
        default = None
        if f.default is not dataclasses.MISSING:
            default = f.default
        result.append(
            {
                "name": f.name,
                "type_str": _stringify_type(f.type),
                "required": required,
                "default": default,
            }
        )
    return result


@router.get("/schemas")
async def get_env_schemas() -> list[dict]:
    """Return field schemas for all registered EnvConfig subclasses."""
    from lerobot.envs.configs import EnvConfig

    _ensure_envs_loaded()

    schemas = []
    for type_name, config_cls in sorted(EnvConfig.get_known_choices().items()):
        schemas.append(
            {
                "type_name": type_name,
                "fields": _introspect_fields(config_cls),
            }
        )
    return schemas


# ============================================================================
# Profile CRUD
# ============================================================================


class EnvProfileData(BaseModel):
    type: str
    name: str
    fields: dict[str, Any] = {}


class RenameRequest(BaseModel):
    new_name: str


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _list_profiles() -> list[dict]:
    _ensure_dir(ENV_PROFILES_DIR)
    profiles = []
    for f in sorted(ENV_PROFILES_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            profiles.append(
                {
                    "name": data.get("name", f.stem),
                    "type": data.get("type", "unknown"),
                }
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read env profile {f}: {e}")
    return profiles


def _read_profile(name: str) -> dict:
    path = ENV_PROFILES_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Env profile '{name}' not found")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Failed to parse env profile: {e}") from e


def _write_profile(data: EnvProfileData) -> None:
    _ensure_dir(ENV_PROFILES_DIR)
    path = ENV_PROFILES_DIR / f"{data.name}.json"
    path.write_text(json.dumps(data.model_dump(), indent=2))
    logger.info(f"Saved env profile: {path}")


@router.get("/profiles")
async def list_env_profiles() -> list[dict]:
    return _list_profiles()


@router.post("/profiles")
async def create_env_profile(profile: EnvProfileData) -> dict:
    path = ENV_PROFILES_DIR / f"{profile.name}.json"
    if path.exists():
        raise HTTPException(409, f"Env profile '{profile.name}' already exists")
    _write_profile(profile)
    return {"status": "created", "name": profile.name}


@router.get("/profiles/{name}")
async def get_env_profile(name: str) -> dict:
    return _read_profile(name)


@router.put("/profiles/{name}")
async def update_env_profile(name: str, profile: EnvProfileData) -> dict:
    _write_profile(profile)
    return {"status": "updated", "name": profile.name}


@router.delete("/profiles/{name}")
async def delete_env_profile(name: str) -> dict:
    path = ENV_PROFILES_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Env profile '{name}' not found")
    # safe-destruct: user-confirmed delete via GUI dialog
    path.unlink()
    logger.info(f"Deleted env profile: {path}")
    return {"status": "deleted", "name": name}


@router.post("/profiles/{name}/rename")
async def rename_env_profile(name: str, req: RenameRequest) -> dict:
    old_path = ENV_PROFILES_DIR / f"{name}.json"
    if not old_path.exists():
        raise HTTPException(404, f"Env profile '{name}' not found")
    new_path = ENV_PROFILES_DIR / f"{req.new_name}.json"
    if new_path.exists():
        raise HTTPException(409, f"Env profile '{req.new_name}' already exists")
    data = json.loads(old_path.read_text())
    data["name"] = req.new_name
    new_path.write_text(json.dumps(data, indent=2))
    # safe-destruct: rename: drop old after writing new
    old_path.unlink()
    logger.info(f"Renamed env profile: {old_path} -> {new_path}")
    return {"status": "renamed", "old_name": name, "new_name": req.new_name}
