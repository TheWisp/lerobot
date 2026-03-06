"""Model tab API: browse training outputs, inspect checkpoints, launch training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

_app_state: "AppState" = None  # type: ignore

SOURCES_FILE = Path.home() / ".config" / "lerobot" / "model_sources.json"


def set_app_state(state: "AppState") -> None:
    global _app_state
    _app_state = state


# ============================================================================
# Source folder persistence
# ============================================================================

_DEFAULT_SOURCE = str(Path.cwd() / "outputs")


def _read_sources() -> list[dict]:
    default = {"path": _DEFAULT_SOURCE, "removable": False, "expanded": True}
    if not SOURCES_FILE.exists():
        return [default]
    try:
        data = json.loads(SOURCES_FILE.read_text())
        sources = data.get("sources", [])
        if not any(s["path"] == _DEFAULT_SOURCE for s in sources):
            sources.insert(0, default)
        return sources
    except Exception:
        logger.warning("Failed to read model sources, using defaults", exc_info=True)
        return [default]


def _write_sources(sources: list[dict]) -> None:
    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "sources": sources}
    SOURCES_FILE.write_text(json.dumps(data, indent=2))


# ============================================================================
# Model scanning
# ============================================================================


def _read_checkpoint_meta(ckpt_dir: Path) -> dict | None:
    """Read metadata from a single checkpoint directory."""
    pretrained = ckpt_dir / "pretrained_model"
    config_file = pretrained / "config.json"
    if not config_file.is_file():
        return None

    try:
        config = json.loads(config_file.read_text())
    except Exception:
        return None

    # Read training step
    step = None
    step_file = ckpt_dir / "training_state" / "training_step.json"
    if step_file.is_file():
        try:
            step = json.loads(step_file.read_text()).get("step")
        except Exception:
            pass

    # Model file size
    model_file = pretrained / "model.safetensors"
    model_size = model_file.stat().st_size if model_file.is_file() else 0

    has_training_state = (ckpt_dir / "training_state").is_dir()

    return {
        "step": step,
        "model_size_bytes": model_size,
        "has_training_state": has_training_state,
        "policy_type": config.get("type", ""),
    }


def _scan_training_run(run_dir: Path) -> dict | None:
    """Scan a single training run directory for checkpoints."""
    ckpts_dir = run_dir / "checkpoints"
    if not ckpts_dir.is_dir():
        return None

    # Find checkpoint subdirs (numeric names or 'last')
    checkpoints = []
    last_target = None

    # Resolve 'last' symlink
    last_link = ckpts_dir / "last"
    if last_link.exists():
        try:
            last_target = last_link.resolve().name
        except Exception:
            pass

    for child in sorted(ckpts_dir.iterdir()):
        if not child.is_dir() or child.name == "last":
            continue
        meta = _read_checkpoint_meta(child)
        if meta:
            meta["name"] = child.name
            meta["path"] = str(child)
            meta["is_last"] = child.name == last_target
            checkpoints.append(meta)

    if not checkpoints:
        return None

    # Use the latest checkpoint for run-level metadata
    latest = next((c for c in checkpoints if c["is_last"]), checkpoints[-1])

    # Read train_config.json from latest checkpoint
    train_config = _read_train_config(Path(latest["path"]))

    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "policy_type": latest["policy_type"],
        "dataset": train_config.get("dataset", {}).get("repo_id", "") if train_config else "",
        "current_step": latest.get("step"),
        "total_steps": train_config.get("steps") if train_config else None,
        "batch_size": train_config.get("batch_size") if train_config else None,
        "model_size_bytes": latest["model_size_bytes"],
        "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        "wandb_run_id": (train_config.get("wandb", {}) or {}).get("run_id") if train_config else None,
        "wandb_project": (train_config.get("wandb", {}) or {}).get("project") if train_config else None,
    }


def _read_train_config(ckpt_dir: Path) -> dict | None:
    """Read train_config.json from a checkpoint directory."""
    tc_file = ckpt_dir / "pretrained_model" / "train_config.json"
    if not tc_file.is_file():
        return None
    try:
        return json.loads(tc_file.read_text())
    except Exception:
        return None


def _scan_source(source_path: str, max_depth: int = 2) -> list[dict]:
    """Scan a directory for training runs (dirs containing checkpoints/)."""
    root = Path(source_path)
    if not root.is_dir():
        return []

    found = []
    _scan_recursive(root, root, found, max_depth, 0)
    found.sort(key=lambda d: d["name"])
    return found


def _scan_recursive(
    base: Path, current: Path, found: list[dict], max_depth: int, depth: int
) -> None:
    if depth > max_depth:
        return
    try:
        # Check if this directory is a training run
        if (current / "checkpoints").is_dir():
            run_meta = _scan_training_run(current)
            if run_meta:
                # Use relative path from base as name
                try:
                    run_meta["name"] = str(current.relative_to(base))
                except ValueError:
                    pass
                found.append(run_meta)
            return  # Don't recurse into training run subdirs

        # Recurse
        if depth < max_depth:
            for child in sorted(current.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    _scan_recursive(base, child, found, max_depth, depth + 1)
    except PermissionError:
        pass
    except Exception:
        pass


# ============================================================================
# Pydantic models
# ============================================================================


class SourceRequest(BaseModel):
    path: str


class SourceInfo(BaseModel):
    path: str
    removable: bool
    expanded: bool


class ModelSourceEntry(BaseModel):
    name: str
    path: str
    policy_type: str
    dataset: str
    current_step: int | None
    total_steps: int | None
    batch_size: int | None
    model_size_bytes: int
    num_checkpoints: int
    wandb_run_id: str | None = None
    wandb_project: str | None = None


class CheckpointInfo(BaseModel):
    name: str
    path: str
    step: int | None
    model_size_bytes: int
    has_training_state: bool
    is_last: bool
    policy_type: str


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/sources")
async def list_sources() -> list[SourceInfo]:
    return [SourceInfo(**s) for s in _read_sources()]


@router.post("/sources")
async def add_source(req: SourceRequest) -> SourceInfo:
    path = str(Path(req.path).expanduser().resolve())
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {path}")

    sources = _read_sources()
    if any(s["path"] == path for s in sources):
        raise HTTPException(status_code=409, detail="Source already exists")

    new_source = {"path": path, "removable": True, "expanded": True}
    sources.append(new_source)
    _write_sources(sources)
    logger.info(f"Added model source: {path}")
    return SourceInfo(**new_source)


@router.delete("/sources/{encoded_path:path}")
async def remove_source(encoded_path: str) -> dict[str, str]:
    path = unquote(encoded_path)
    sources = _read_sources()
    source = next((s for s in sources if s["path"] == path), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")
    if not source.get("removable", True):
        raise HTTPException(status_code=400, detail="Cannot remove default source")

    sources = [s for s in sources if s["path"] != path]
    _write_sources(sources)
    logger.info(f"Removed model source: {path}")
    return {"status": "ok"}


@router.put("/sources/{encoded_path:path}/expanded")
async def set_source_expanded(encoded_path: str, expanded: bool = True) -> dict[str, str]:
    path = unquote(encoded_path)
    sources = _read_sources()
    source = next((s for s in sources if s["path"] == path), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")

    source["expanded"] = expanded
    _write_sources(sources)
    return {"status": "ok"}


@router.get("/sources/{encoded_path:path}/models")
async def scan_source(encoded_path: str) -> list[ModelSourceEntry]:
    """Scan a source folder for training runs."""
    import asyncio

    path = unquote(encoded_path)
    sources = _read_sources()
    if not any(s["path"] == path for s in sources):
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")

    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(None, _scan_source, path)
    return [ModelSourceEntry(**{k: v for k, v in m.items() if k != "checkpoints"}) for m in models]


@router.get("/run/{encoded_path:path}/checkpoints")
async def list_checkpoints(encoded_path: str) -> list[CheckpointInfo]:
    """List checkpoints for a training run."""
    path = unquote(encoded_path)
    run_dir = Path(path)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {path}")

    run_meta = _scan_training_run(run_dir)
    if not run_meta:
        raise HTTPException(status_code=404, detail="No checkpoints found")

    return [CheckpointInfo(**c) for c in run_meta["checkpoints"]]


@router.get("/run/{encoded_path:path}/config")
async def get_train_config(encoded_path: str) -> dict:
    """Return train_config.json for a checkpoint."""
    path = unquote(encoded_path)
    ckpt_dir = Path(path)

    # Try as checkpoint dir first, then as training run (use last checkpoint)
    config = _read_train_config(ckpt_dir)
    if not config:
        last = ckpt_dir / "checkpoints" / "last"
        if last.exists():
            config = _read_train_config(last.resolve())
    if not config:
        raise HTTPException(status_code=404, detail="train_config.json not found")

    return config


@router.post("/open-in-files")
async def open_in_file_manager(body: dict) -> dict:
    """Open a directory in the system file manager."""
    import subprocess as _subprocess

    path = body.get("path", "")
    if not path or not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {path}")

    try:
        _subprocess.Popen(["xdg-open", path])
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="xdg-open not found")

    return {"status": "ok"}
