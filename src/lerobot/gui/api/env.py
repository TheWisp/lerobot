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


# ----- Enum-alike choices for str-typed fields ------------------------------
#
# TODO(literal-types): Upstream's EnvConfig subclasses type these fields as
# plain `str` and validate allowed values inside `__post_init__`. Once they
# migrate to `typing.Literal[...]`, `_introspect_fields` should derive
# `choices` automatically via `typing.get_args(f.type)` (the standard Python
# mechanism — works for any Literal-typed dataclass field). At that point
# this hand-curated registry can be retired for fields whose types switch
# to Literal. Keep it for any that stay plain `str`.
#
# Sources for each entry are the matching __post_init__ branches in
# src/lerobot/envs/configs.py — i.e. these are the values that the config
# would actually accept at runtime; passing anything else raises
# ValueError or silently no-ops.

# Field-name -> choices, applies regardless of env type. Use for things that
# are universally enum-alike (gym standards, ML conventions).
_GLOBAL_FIELD_CHOICES: dict[str, list[str]] = {
    # Gym-standard render modes; not all envs implement all three but
    # passing one that's unsupported is a runtime error, not silent.
    "render_mode": ["rgb_array", "human", "rgb_array_list"],
    # PyTorch device specs the user is likely to type. Unguarded otherwise.
    "device": ["cuda", "cpu", "mps"],
}

# (type_name, field_name) -> choices, takes precedence over _GLOBAL above.
# Entries are mined from the __post_init__ accept-lists of each EnvConfig
# subclass, so values map 1:1 to "what won't crash".
_TYPE_FIELD_CHOICES: dict[tuple[str, str], list[str]] = {
    # AlohaEnv.__post_init__ branches on these two (configs.py:172-176).
    ("aloha", "obs_type"): ["pixels", "pixels_agent_pos"],
    # PushtEnv.__post_init__ branches on these two (configs.py:219-223).
    ("pusht", "obs_type"): ["pixels_agent_pos", "environment_state_agent_pos"],
    # LiberoEnv.__post_init__ raises on anything outside this set
    # (configs.py:356-399).
    ("libero", "obs_type"): ["pixels", "pixels_agent_pos"],
    # libero comment at configs.py:353: 'or "absolute"'.
    ("libero", "control_mode"): ["relative", "absolute"],
    # MetaworldEnv.__post_init__ raises on anything outside this set
    # (configs.py:468-476).
    ("metaworld", "obs_type"): ["pixels", "pixels_agent_pos"],
    # HILSerlRobotEnvConfig.name discriminates: "real_robot" makes
    # make_robot_env (gym_manipulator.py:333) take the hardware path,
    # which asserts cfg.robot is not None and dies for our sim profiles.
    # Any other value is treated as a gym package name (gym.make uses
    # `{name}/{task}` as the env id). The GUI's Environment tab is
    # sim-only, so "real_robot" is the wrong answer here — surface only
    # the gym packages we've wired through. Today that's just gym_hil;
    # extend as more packages get consumable env types.
    ("gym_manipulator", "name"): ["gym_hil"],
}


def _choices_for(type_name: str, field_name: str) -> list[str] | None:
    """Return enum-alike choices for a (type, field), or None if unknown.

    Type-specific entries override globals. None means free-form text.
    """
    specific = _TYPE_FIELD_CHOICES.get((type_name, field_name))
    if specific is not None:
        return specific
    return _GLOBAL_FIELD_CHOICES.get(field_name)


def _introspect_fields(cls: type, type_name: str | None = None) -> list[dict]:
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
        entry: dict[str, Any] = {
            "name": f.name,
            "type_str": _stringify_type(f.type),
            "required": required,
            "default": default,
        }
        choices = _choices_for(type_name, f.name) if type_name else None
        if choices is not None:
            entry["choices"] = list(choices)
        result.append(entry)
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
                "fields": _introspect_fields(config_cls, type_name=type_name),
            }
        )
    return schemas


# ============================================================================
# Registered-task discovery
# ============================================================================

# Hardcoded fallback list, used when the package can't be imported or
# registration somehow fails. Captured 2026-05-02 from gym_hil 0.1.13. We'd
# rather show the dropdown with known options than silently disable it; if
# upstream adds tasks the live registry path will pick them up automatically.
_FALLBACK_TASKS: dict[str, list[str]] = {
    "gym_hil": [
        "PandaPickCubeBase-v0",
        "PandaPickCubeViewer-v0",
        "PandaPickCube-v0",
        "PandaPickCubeGamepad-v0",
        "PandaPickCubeKeyboard-v0",
        "PandaArrangeBoxesBase-v0",
        "PandaArrangeBoxesViewer-v0",
        "PandaArrangeBoxes-v0",
        "PandaArrangeBoxesGamepad-v0",
        "PandaArrangeBoxesKeyboard-v0",
    ],
}


@router.get("/registered-tasks")
async def list_registered_tasks(name: str = "gym_hil") -> dict:
    """Enumerate gym tasks registered under a namespace, e.g. 'gym_hil'.

    The env profile's `task` field stores the ID *without* the namespace
    prefix (gym_manipulator prepends it as `gym.make(f"{name}/{task}")`),
    so this endpoint returns just the suffixes — exactly what the
    dropdown binds to.

    Falls back to ``_FALLBACK_TASKS`` when the package isn't installed or
    its import fails, so the dropdown stays usable. The response shape
    includes a `source` field ("registry" | "fallback") so the frontend
    can flag the degraded path if it wants to.
    """
    import importlib

    fallback = list(_FALLBACK_TASKS.get(name, []))
    prefix = f"{name}/"

    try:
        importlib.import_module(name)
    except ImportError as e:
        return {
            "name": name,
            "source": "fallback",
            "tasks": fallback,
            "warning": f"package '{name}' is not installed ({e})",
        }
    except Exception as e:  # pragma: no cover - plugin import-time bug
        logger.exception("registered-tasks: import of '%s' failed", name)
        return {"name": name, "source": "fallback", "tasks": fallback, "warning": str(e)}

    try:
        import gymnasium as gym
    except ImportError as e:
        # gymnasium is a hard dep of lerobot; this branch is defensive.
        return {
            "name": name,
            "source": "fallback",
            "tasks": fallback,
            "warning": f"gymnasium not importable ({e})",
        }

    ids = sorted(k.removeprefix(prefix) for k in gym.envs.registry if k.startswith(prefix))
    if not ids:
        return {
            "name": name,
            "source": "fallback",
            "tasks": fallback,
            "warning": f"package '{name}' imported but registered no envs",
        }
    return {"name": name, "source": "registry", "tasks": ids}


# ============================================================================
# Task instructions: single source of truth for controls + behaviour + docs.
#
# Both the env editor (env.js renders HTML) and the runtime banner
# (run.py prints text) read from `get_task_instructions(task)` so the
# two surfaces can never drift. If gym-hil renames a button or changes
# its termination rules, fix it once here.
# ============================================================================


_GYM_HIL_REPO = "https://github.com/huggingface/gym-hil"
_GYM_HIL_REPO_BLOB = f"{_GYM_HIL_REPO}/blob/main"
_LEROBOT_HILSERL_DOCS = "https://huggingface.co/docs/lerobot/hilserl_sim"


def _gym_hil_source_url(task: str) -> str:
    """Map a gym-hil task id to the GitHub URL of its env class.

    Behaviour rules — termination triggers, success thresholds, reward
    shaping — live in this Python file with no user-facing prose.
    Pointing at the source is the most useful link we can give without
    re-documenting upstream code.
    """
    if "PandaPickCube" in task:
        return f"{_GYM_HIL_REPO_BLOB}/gym_hil/envs/panda_pick_gym_env.py"
    if "PandaArrangeBoxes" in task:
        return f"{_GYM_HIL_REPO_BLOB}/gym_hil/envs/panda_arrange_boxes_gym_env.py"
    return f"{_GYM_HIL_REPO_BLOB}/gym_hil/envs/"


# Per-task-family termination/reward summaries. Mined from each env's
# step()/_is_success()/_compute_reward() implementations. Source-of-truth
# is still the URL we link, but a one-line summary is easier to skim.
_TASK_FAMILY_BEHAVIOUR: dict[str, list[str]] = {
    "PandaPickCube": [
        "Natural success: cube lifted >10cm above its starting Z (terminates with reward=1).",
        "Out-of-bounds: cube knocked off-table beyond sampling bounds (terminates).",
        "Step cap: configurable via the profile's max_episode_steps field.",
    ],
    "PandaArrangeBoxes": [
        "Termination + reward rules vary by reward_type (sparse / dense). See source for the exact",
        "success condition; out-of-bounds termination applies to the manipulated objects.",
        "Step cap: configurable via the profile's max_episode_steps field.",
    ],
}


def _task_family(task: str) -> str | None:
    if "PandaPickCube" in task:
        return "PandaPickCube"
    if "PandaArrangeBoxes" in task:
        return "PandaArrangeBoxes"
    return None


def get_task_instructions(task: str) -> dict:
    """Single source of truth for per-task controls / behaviour / docs.

    Returns a dict shaped:
      {
        "task": str,
        "input_type": "keyboard" | "gamepad" | None,  # None for autonomous variants
        "controls": {...} | None,
        "behaviour": [str, ...],   # natural termination rules
        "docs": [{"label": str, "url": str}, ...],
        "global_warnings": [str, ...],  # e.g. system-wide pynput capture
      }

    Empty/unknown task returns the dict with input_type=None and a
    generic gym-hil package URL.
    """
    task = str(task or "")
    family = _task_family(task)

    is_keyboard = "Keyboard" in task
    is_gamepad = "Gamepad" in task

    controls: dict[str, Any] | None = None
    if is_keyboard:
        controls = {
            "intervention": {
                "model": "toggle",
                "label": "Space",
                "note": (
                    "every press flips on/off; no LED indicator. The arm stays still until you toggle on."
                ),
            },
            "movement": [
                {"keys": "Arrows", "function": "X/Y plane"},
                {"keys": "Shift / RShift", "function": "Z down/up"},
                {"keys": "LCtrl / RCtrl", "function": "gripper close/open"},
            ],
            "episode_end": [
                {"key": "Enter", "result": "success"},
                {
                    "key": "Esc",
                    "result": (
                        "failure (despite the printed 'ESC: Exit' label, it actually ends the episode)"
                    ),
                },
                {"key": "r", "result": "rerecord"},
            ],
        }
    elif is_gamepad:
        controls = {
            "intervention": {
                "model": "hold",
                "label": "RB",
                "note": "hold-to-engage, not a toggle — release RB and the arm goes back to autonomous",
            },
            "movement": [
                {"keys": "Left analog stick", "function": "X/Y plane"},
                {"keys": "Right analog stick (vertical)", "function": "Z"},
                {"keys": "LT / RT", "function": "gripper close/open"},
            ],
            "episode_end": [
                {"key": "Y / Triangle", "result": "success"},
                {"key": "A / Cross", "result": "failure"},
                {"key": "X / Square", "result": "rerecord"},
                {"key": "B / Circle", "result": "exit"},
            ],
            "controller_note": (
                "Button labels are Logitech F310 defaults; gym-hil reads mappings from "
                "controller_config.json so PS/Xbox controllers map to their own labels but the "
                "same functions."
            ),
        }

    behaviour = list(_TASK_FAMILY_BEHAVIOUR.get(family or "", []))

    global_warnings: list[str] = []
    if is_keyboard:
        global_warnings.append(
            "System-wide keyboard capture (pynput): keys from any focused window reach gym-hil. "
            "Typing Enter / Esc in the terminal, browser, or IDE will end the current episode."
        )

    docs = [
        {"label": "📖 Source for this task family", "url": _gym_hil_source_url(task)},
        {"label": "gym-hil package", "url": _GYM_HIL_REPO},
        {"label": "LeRobot HIL-SERL sim guide", "url": _LEROBOT_HILSERL_DOCS},
    ]

    return {
        "task": task,
        "input_type": "keyboard" if is_keyboard else ("gamepad" if is_gamepad else None),
        "controls": controls,
        "behaviour": behaviour,
        "docs": docs,
        "global_warnings": global_warnings,
    }


@router.get("/task-instructions")
async def get_task_instructions_endpoint(task: str = "") -> dict:
    """Return the structured controls / behaviour / docs bundle for a task.

    Frontend (env editor) renders HTML from this; backend (run.py
    runtime banner) reads the same data internally. Single source of
    truth so the two surfaces can't disagree on whether intervention
    is hold-vs-toggle, whether Esc is documented as 'Exit' or
    'failure', etc.
    """
    return get_task_instructions(task)


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
