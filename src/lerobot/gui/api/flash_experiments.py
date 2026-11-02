"""Experiment playback and an explicit rebuild subprocess; no models in the GUI."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

router = APIRouter(prefix="/api/run/flash/experiments", tags=["flash-experiments"])
_files = ThreadPoolExecutor(max_workers=2, thread_name_prefix="flash-experiment-files")
_jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-experiment-rebuild")
_scenes = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-experiment-scene")
_scene_gate = asyncio.Semaphore(1)
_logger = logging.getLogger(__name__)


def _root():
    return (
        Path(os.environ.get("LEROBOT_FLASH_EXPERIMENT_DIR", "~/.cache/flash-tracker-restart/browser"))
        .expanduser()
        .resolve()
    )


def _asset(identity: str, name: str):
    if not re.fullmatch(r"[a-f0-9]{20}", identity):
        raise HTTPException(404, "Unknown experiment")
    root = _root()
    path = (root / identity / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Experiment file is unavailable")
    return path


def _list():
    items = []
    for directory in sorted(_root().glob("*")):
        if not re.fullmatch(r"[a-f0-9]{20}", directory.name):
            continue
        try:
            manifest = json.loads(_asset(directory.name, "manifest.json").read_text())
            _asset(directory.name, "video.mp4")
            _asset(directory.name, "overlays.json")
            items.append({**manifest, "id": directory.name})
        except (OSError, ValueError, HTTPException):
            _logger.warning("Skipping incomplete experiment %s", directory.name)
    latest = {}
    for item in items:
        slot = item.get("slot_id", item["id"])
        if slot not in latest or item.get("built_at_ns", 0) > latest[slot].get("built_at_ns", 0):
            latest[slot] = item
    result = sorted(latest.values(), key=lambda item: (item["name"], item["experiment"]))
    for item in result:
        try:
            saved = _root() / item["id"] / "summary.json"
            # Older exports predate the embedded score. Read their exact source
            # result, never the most recent result for a similarly named case.
            if not saved.is_file():
                saved = Path(item["source_run"]) / "summary.json"
            summary = json.loads(saved.read_text())
            if summary.get("input_sha256") == item["input_sha256"] and summary["frames"] == item["frames"]:
                item["metrics"] = summary
        except (OSError, ValueError, KeyError):
            pass  # Playback remains useful when an older score is unavailable.
    return result


def _rebuild_status():
    directory = _root() / ".rebuild"
    pointer = directory / "current.json"
    if not pointer.exists():
        return {"state": "idle"}
    job = Path(json.loads(pointer.read_text())["job"])
    state = json.loads((job / "status.json").read_text())
    if state["state"] in {"starting", "running"}:
        # An OS lock outlives a server restart but cannot survive a dead worker.
        with (directory / "lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if (job / "cancel").exists():
                    state.update(state="cancelling", phase="Stopping")
            else:
                if (job / "cancel").exists():
                    state.update(state="cancelled", phase="Stopped")
                else:
                    state.update(
                        state="failed", error="Rerun was interrupted. Previous results are available."
                    )
    return state


def _cancel_rebuild():
    state = _rebuild_status()
    if state["state"] not in {"starting", "running", "cancelling"}:
        return state
    job = Path(json.loads((_root() / ".rebuild/current.json").read_text())["job"])
    (job / "cancel").touch(exist_ok=True)
    try:
        import psutil
    except ImportError:
        # Current workers observe the request at frame boundaries. Signalling
        # additionally supports workers launched before cancellation existed.
        return _rebuild_status()
    identity = job / "worker.json"
    pid = json.loads(identity.read_text())["pid"] if identity.exists() else state.get("pid")
    if pid:
        # psutil retains process creation time and checks PID reuse before
        # signalling. Also verify the exact worker module and job directory.
        try:
            process = psutil.Process(pid)
            args = process.cmdline()
            module = "benchmarks.flash_rebuild_examples"
            if module not in args or "--job" not in args:
                raise HTTPException(
                    409, "Cannot verify the benchmark process; cancellation remains requested"
                )
            if Path(args[args.index("--job") + 1]).resolve() != job.resolve():
                raise HTTPException(409, "Benchmark process belongs to another run")
            process.terminate()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            raise HTTPException(409, "Cannot signal the worker; cancellation remains requested") from exc
    return _rebuild_status()


def _start_rebuild(*, force=False):
    from benchmarks.flash_rebuild_examples import write_json

    directory = _root() / ".rebuild"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HTTPException(409, "Examples are already rebuilding") from exc
        items = _list()
        if not items:
            raise HTTPException(400, "No examples to rebuild")
        name = "rebuild-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        job = directory / name
        job.mkdir()
        state = {
            "id": name,
            "state": "starting",
            "phase": "Starting",
            "total": len(items),
            "completed": 0,
            "succeeded": 0,
            "errors": [],
            "current": "",
            "frame": 0,
            "frames": 0,
        }
        write_json(job / "request.json", {"browser_root": str(_root()), "items": items, "force": force})
        write_json(job / "status.json", state)
        write_json(directory / "current.json", {"job": str(job)})
        python = os.environ.get("LEROBOT_FLASH_TRACKING_PYTHON", sys.executable)
        try:
            with (job / "rebuild.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        python,
                        "-m",
                        "benchmarks.flash_rebuild_examples",
                        "--job",
                        str(job),
                        "--lock-fd",
                        str(lock.fileno()),
                    ],
                    cwd=Path(__file__).resolve().parents[4],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                    start_new_session=True,
                )
            threading.Thread(target=process.wait, name="flash-rebuild-reaper", daemon=True).start()
            write_json(job / "worker.json", {"pid": process.pid})
        except Exception as exc:
            state.update(state="failed", error=str(exc))
            write_json(job / "status.json", state)
            raise HTTPException(500, f"Could not start rebuild: {exc}") from exc
    return state


@router.get("")
async def list_experiments():
    return await asyncio.get_running_loop().run_in_executor(_files, _list)


@router.post("/rebuild", status_code=202)
async def rebuild_examples(force: bool = False):
    return await asyncio.get_running_loop().run_in_executor(_jobs, partial(_start_rebuild, force=force))


@router.get("/rebuild")
async def rebuild_status():
    return await asyncio.get_running_loop().run_in_executor(_files, _rebuild_status)


@router.post("/rebuild/cancel", status_code=202)
async def cancel_rebuild():
    return await asyncio.get_running_loop().run_in_executor(_jobs, _cancel_rebuild)


def _report_asset(job_id: str, asset: str):
    if not re.fullmatch(r"rebuild-\d{8}-\d{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(404, "Unknown benchmark run")
    if asset not in {"report.html", "report.json"}:
        raise HTTPException(404, "Unknown benchmark report")
    root = _root() / ".rebuild"
    path = (root / job_id / asset).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Benchmark report is unavailable")
    return path


@router.get("/rebuild/{job_id}/{asset}")
async def benchmark_report(job_id: str, asset: str):
    path = await asyncio.get_running_loop().run_in_executor(_files, _report_asset, job_id, asset)
    return FileResponse(
        path,
        media_type="application/json" if asset == "report.json" else "text/html",
        filename="flash-benchmark-results.json" if asset == "report.json" else None,
    )


@router.get("/{identity}/{asset}")
async def experiment_asset(identity: str, asset: str):
    types = {"video.mp4": "video/mp4", "overlays.json": "application/json"}
    if asset not in types:
        raise HTTPException(404, "Unknown experiment asset")
    path = await asyncio.get_running_loop().run_in_executor(_files, _asset, identity, asset)
    return FileResponse(path, media_type=types[asset])


def _scene(identity: str, frame: int, method: str):
    from lerobot.gui.flash_experiment_scene import scene_packet

    manifest = json.loads(_asset(identity, "manifest.json").read_text())
    try:
        return scene_packet(manifest, frame, method)
    except (OSError, ValueError, IndexError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{identity}/scene/{frame}")
async def experiment_scene(identity: str, frame: int, method: str):
    # Do not queue obsolete scrubs behind disk decoding. The player retries its
    # latest requested frame when another client is already reading one.
    if _scene_gate.locked():
        raise HTTPException(429, "Another scene frame is loading")
    async with _scene_gate:
        data = await asyncio.get_running_loop().run_in_executor(_scenes, _scene, identity, frame, method)
    return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})
