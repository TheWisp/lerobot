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
_CONVERTED_SOURCE = str(Path.home() / ".cache" / "lerobot" / "converted")


def _read_sources() -> list[dict]:
    defaults = [
        {"path": _DEFAULT_SOURCE, "removable": False, "expanded": True},
    ]
    # Add converted checkpoints source if it exists
    if Path(_CONVERTED_SOURCE).is_dir():
        defaults.append({"path": _CONVERTED_SOURCE, "removable": False, "expanded": True})

    if not SOURCES_FILE.exists():
        return defaults
    try:
        data = json.loads(SOURCES_FILE.read_text())
        sources = data.get("sources", [])
        # Ensure defaults are present
        for d in defaults:
            if not any(s["path"] == d["path"] for s in sources):
                sources.insert(0, d)
        return sources
    except Exception:
        logger.warning("Failed to read model sources, using defaults", exc_info=True)
        return defaults


def _write_sources(sources: list[dict]) -> None:
    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "sources": sources}
    SOURCES_FILE.write_text(json.dumps(data, indent=2))


# ============================================================================
# Model scanning
# ============================================================================


def _count_safetensor_params(filepath: Path) -> int | None:
    """Count total parameters from a safetensors file header (no weight loading)."""
    import struct

    try:
        with open(filepath, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        total = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            params = 1
            for s in info["shape"]:
                params *= s
            total += params
        return total
    except Exception:
        return None


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

    # Model file size and parameter count
    model_file = pretrained / "model.safetensors"
    model_size = model_file.stat().st_size if model_file.is_file() else 0
    num_params = _count_safetensor_params(model_file) if model_file.is_file() else None

    has_training_state = (ckpt_dir / "training_state").is_dir()

    return {
        "step": step,
        "model_size_bytes": model_size,
        "num_parameters": num_params,
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
        "dataset_root": train_config.get("dataset", {}).get("root") if train_config else None,
        "current_step": latest.get("step"),
        "total_steps": train_config.get("steps") if train_config else None,
        "batch_size": train_config.get("batch_size") if train_config else None,
        "model_size_bytes": latest["model_size_bytes"],
        "num_parameters": latest.get("num_parameters"),
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


def _read_flat_checkpoint(ckpt_dir: Path) -> dict | None:
    """Read metadata from a flat checkpoint directory (model.safetensors + config.json).

    Used for converted checkpoints that don't have the standard
    checkpoints/pretrained_model/ structure (e.g. HVLA S2 VLM).
    """
    config_file = ckpt_dir / "config.json"
    model_file = ckpt_dir / "model.safetensors"
    if not config_file.is_file() or not model_file.is_file():
        return None

    try:
        config = json.loads(config_file.read_text())
    except Exception:
        return None

    model_size = model_file.stat().st_size
    num_params = _count_safetensor_params(model_file)

    return {
        "name": ckpt_dir.name,
        "path": str(ckpt_dir),
        "policy_type": config.get("type", "unknown"),
        "dataset": "",
        "dataset_root": None,
        "current_step": None,
        "total_steps": None,
        "batch_size": None,
        "model_size_bytes": model_size,
        "num_parameters": num_params,
        "num_checkpoints": 1,
        "checkpoints": [{
            "name": ckpt_dir.name,
            "path": str(ckpt_dir),
            "step": None,
            "model_size_bytes": model_size,
            "num_parameters": num_params,
            "has_training_state": False,
            "is_last": True,
            "policy_type": config.get("type", "unknown"),
        }],
        "wandb_run_id": None,
        "wandb_project": None,
    }


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
        # Check if this directory is a training run (standard LeRobot format)
        if (current / "checkpoints").is_dir():
            run_meta = _scan_training_run(current)
            if run_meta:
                try:
                    run_meta["name"] = str(current.relative_to(base))
                except ValueError:
                    pass
                found.append(run_meta)
            return  # Don't recurse into training run subdirs

        # Check if this is a flat checkpoint dir (model.safetensors + config.json, no checkpoints/)
        # Used by converted checkpoints (e.g. HVLA S2 VLM)
        if (current / "model.safetensors").is_file() and (current / "config.json").is_file():
            meta = _read_flat_checkpoint(current)
            if meta:
                try:
                    meta["name"] = str(current.relative_to(base))
                except ValueError:
                    pass
                found.append(meta)
            return

        # Check if this is an RLT run dir (latest/actor.pt inside it)
        # Mirrors how training runs are detected by checkpoints/ subdir.
        if (current / "latest" / "actor.pt").is_file():
            meta = _read_rlt_checkpoint(current / "latest", base)
            if meta:
                meta["path"] = str(current)  # run dir, not checkpoint subdir
                try:
                    meta["name"] = str(current.relative_to(base))
                except ValueError:
                    pass
                found.append(meta)
            return

        # Recurse
        if depth < max_depth:
            for child in sorted(current.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    _scan_recursive(base, child, found, max_depth, depth + 1)
    except PermissionError:
        pass
    except Exception:
        pass


def _read_rlt_checkpoint(ckpt_dir: Path, base: Path) -> dict | None:
    """Read metadata from an RLT checkpoint (actor.pt + optional training_state.pt)."""
    meta = {
        "type": "rlt",
        "path": str(ckpt_dir),
        "policyType": "rlt",
    }
    try:
        meta["name"] = str(ckpt_dir.relative_to(base))
    except ValueError:
        meta["name"] = str(ckpt_dir)

    # Read training state if available
    ts_path = ckpt_dir / "training_state.pt"
    if ts_path.exists():
        try:
            import torch
            ts = torch.load(str(ts_path), weights_only=True, map_location="cpu")
            meta["episode"] = ts.get("episode", 0)
            meta["updates"] = ts.get("total_updates", 0)
            meta["successes"] = sum(ts.get("successes", []))
        except Exception as e:
            logger.warning("Failed to read RLT training state from %s: %s", ts_path, e)

    # Check for RL token encoder in parent or sibling dirs
    # (convention: rlt_token checkpoint is separate from actor checkpoint)
    meta["has_replay_buffer"] = (ckpt_dir / "replay_buffer.pt").exists()
    meta["has_critic"] = (ckpt_dir / "critic.pt").exists()

    return meta


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
    num_parameters: int | None = None
    num_checkpoints: int
    wandb_run_id: str | None = None
    wandb_project: str | None = None


class CheckpointInfo(BaseModel):
    name: str
    path: str
    step: int | None
    model_size_bytes: int
    num_parameters: int | None = None
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
    # Filter out RLT checkpoints from main browser (they show in HVLA form instead)
    models = [m for m in models if m.get("type") != "rlt"]
    return [ModelSourceEntry(**{k: v for k, v in m.items() if k != "checkpoints"}) for m in models]


@router.get("/rlt-checkpoints")
async def list_rlt_checkpoints() -> list[dict]:
    """List RLT checkpoints across all source directories."""
    import asyncio

    sources = _read_sources()
    all_rlt = []

    loop = asyncio.get_event_loop()
    for s in sources:
        models = await loop.run_in_executor(None, _scan_source, s["path"])
        all_rlt.extend(m for m in models if m.get("type") == "rlt")

    return all_rlt


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
