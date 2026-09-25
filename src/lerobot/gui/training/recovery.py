"""Opt-in local training recovery, driven by the GUI backend, never the browser.

State lives beside the original run; each attempt still uses ordinary Resume.
No model, optimizer, scheduler, or checkpoint is modified here. File checks are
structural checks, not a guarantee of numerical correctness of saved weights.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import struct
import subprocess
import threading
import time
import zipfile
from pathlib import Path

from lerobot.gui.training.jobs import atomic_write_json
from lerobot.gui.training.runs import TERMINAL_STATES, RunPaths, RunState
from lerobot.gui.training.transport import SubprocessTransport

logger = logging.getLogger(__name__)
STATE_FILE = "auto_recovery.json"
FINISHED = {"completed", "blocked", "disabled"}


def _read_json(path: Path):
    return json.loads(path.read_text())


def _check_tensor_file(path: Path) -> None:
    """Read only the safetensors header and check the declared data extent."""
    with path.open("rb") as stream:
        size = path.stat().st_size
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated tensor file: {path.name}")
        header_size = struct.unpack("<Q", raw)[0]
        if not 2 <= header_size <= min(size - 8, 16 * 1024 * 1024):
            raise ValueError(f"Invalid tensor header: {path.name}")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid tensor metadata: {path.name}")
    offsets = sorted(v["data_offsets"] for k, v in header.items() if k != "__metadata__")
    end = 0
    for start, stop in offsets:
        if start != end or stop < start:
            raise ValueError(f"Invalid tensor offsets: {path.name}")
        end = stop
    if not offsets or 8 + header_size + end != size:
        raise ValueError(f"Incomplete tensor data: {path.name}")


def check_checkpoint(path: Path, step: int, hvla: bool = False) -> tuple[bool, str]:
    """Validate the formats used by standard LeRobot and HVLA flow_s1."""
    try:
        pretrained = path / "pretrained_model"
        config = _read_json(pretrained / "train_config.json")
        if not isinstance(config, dict):
            raise ValueError("Invalid training configuration")
        _read_json(pretrained / "config.json")
        model = pretrained / "model.safetensors"
        if not model.exists() and config.get("peft"):
            model = pretrained / "adapter_model.safetensors"
            _read_json(pretrained / "adapter_config.json")
        _check_tensor_file(model)
        state = path / "training_state"
        if _read_json(state / "training_step.json")["step"] != step:
            raise ValueError("Checkpoint directory and saved step disagree")
        if hvla:
            # A torch.save ZIP writes its directory last. No unpickling in GUI.
            with zipfile.ZipFile(state / "optimizer.pt") as archive:
                if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                    raise ValueError("Missing HVLA optimizer metadata")
        else:
            _check_tensor_file(state / "rng_state.safetensors")
            _check_tensor_file(state / "optimizer_state.safetensors")
            _read_json(state / "optimizer_param_groups.json")
            if config.get("scheduler") is not None:
                _read_json(state / "scheduler_state.json")
        return True, ""
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        return False, str(exc)


def crash_materials(started_at: float) -> tuple[list[str], bool]:
    """Best-effort paths; do not open or copy multi-GB core dumps."""
    paths = []
    for directory in (Path("/var/lib/apport/coredump"), Path("/var/crash")):
        try:
            for path in directory.iterdir():
                if path.is_file() and path.stat().st_mtime >= started_at:
                    paths.append(str(path))
        except OSError:
            continue
    # This workstation runs one training job. Conservatively let any active
    # native-crash collector finish rather than compete for its resources.
    busy = False
    for comm in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = comm.read_text().strip()
            if name in {"apport", "systemd-coredum"}:
                busy = True
        except OSError:
            continue
    return sorted(paths), busy


class RecoveryManager:
    """One small backend loop and persisted records for opted-in local runs."""

    def __init__(self, orch, *, clock=time.time, interval=5, quiet_seconds=10, settle_timeout=600):
        self.orch = orch
        self.root = orch._runs.runs_dir
        self.clock = clock
        self.interval = interval
        self.quiet_seconds = quiet_seconds
        self.settle_timeout = settle_timeout
        self._mutex = threading.RLock()
        self._lock_depth = threading.local()
        self._shutdown = threading.Event()
        self._thread = None

    @contextlib.contextmanager
    def locked(self):
        """Serialize UI actions and recovery, including two local GUI servers."""
        with self._mutex:
            if getattr(self._lock_depth, "value", 0):
                yield
                return
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / ".auto-recovery.lock").open("a") as stream:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    self._lock_depth.value = 1
                    yield
                finally:
                    self._lock_depth.value = 0
                    if os.name == "posix":
                        fcntl.flock(stream, fcntl.LOCK_UN)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._loop, name="training-auto-recovery", daemon=True)
        self._thread.start()

    def close(self):
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._shutdown.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("automatic training recovery tick failed")
            self._shutdown.wait(self.interval)

    def _records(self):
        for path in self.root.glob(f"*/{STATE_FILE}"):
            try:
                yield _read_json(path)
            except (OSError, ValueError):
                logger.exception("could not read recovery record %s", path)

    def get(self, run_id):
        for record in self._records():
            if run_id in record["run_ids"]:
                return record
        return None

    def _save(self, record):
        atomic_write_json(self.root / record["root_run_id"] / STATE_FILE, record)

    def validate_host(self, host_id):
        host = self.orch._hosts.get(host_id)
        if host is None or not isinstance(host.transport, SubprocessTransport):
            raise ValueError("Automatic recovery currently supports training on this GUI server only")

    def configure(self, run_id, *, enabled, max_retries=3, delay_seconds=60, _initial=False):
        with self.locked():
            run = self.orch._runs.load(run_id)
            if run is None:
                raise ValueError("Training run not found")
            self.validate_host(run.host_id)
            if not 1 <= max_retries <= 10 or not 0 <= delay_seconds <= 3600:
                raise ValueError("Retries must be 1–10 and delay 0–3600 seconds")
            record = self.get(run_id)
            if record is None:
                record = {
                    "root_run_id": run_id,
                    "active_run_id": run_id,
                    "run_ids": [run_id],
                    "attempts": 0,
                    "incidents": [],
                    "status": "monitoring",
                }
            active = self.orch._runs.load(record["active_run_id"])
            if enabled and not _initial and (active is None or active.state in TERMINAL_STATES):
                raise ValueError(
                    "Enable recovery before training exits; use Resume for an already stopped run"
                )
            record.update(
                enabled=enabled,
                max_retries=max_retries,
                delay_seconds=delay_seconds,
                status="monitoring" if enabled else "disabled",
                message="",
            )
            record.pop("waiting_since", None)
            self._save(record)
            return record

    def stop(self, run_id):
        with self.locked():
            record = self.get(run_id)
            if record:
                record.update(enabled=False, status="disabled", message="Stopped by user")
                self._save(record)  # intent is durable BEFORE signalling the process
                run_id = record["active_run_id"]
            return self.orch.stop(run_id)

    def protect_delete(self, run_id=None):
        for record in self._records():
            if (
                record["enabled"]
                and record["status"] not in FINISHED
                and (run_id is None or run_id in record["run_ids"])
            ):
                raise ValueError("Disable automatic recovery before deleting its run history")

    def tick(self):
        # Taking the lock creates the lock file, and the runs directory with
        # it. Nobody has opted in until a record exists, so until one does
        # this must leave no trace: a monitor running beside a GUI that never
        # enables recovery should be indistinguishable from one that is not
        # running. A record written between here and the next tick is that
        # tick's to pick up -- configure() takes the same lock to create it.
        if not any(self.root.glob(f"*/{STATE_FILE}")):
            return
        with self.locked():
            for record in list(self._records()):
                # Inside the try, so a record this build cannot make sense of
                # costs its own run rather than every other run's sweep.
                try:
                    if not record["enabled"] or record["status"] in FINISHED:
                        continue
                    self._tick(record)
                except Exception as exc:
                    logger.exception("automatic recovery failed for %s", record.get("root_run_id"))
                    if "root_run_id" in record:
                        self._block(record, f"Recovery needs attention: {exc}")

    def _block(self, record, message):
        record.update(status="blocked", message=message)
        self._save(record)

    def _tick(self, record):
        # A GUI shutdown between Resume and recording the child is repaired
        # using the same persisted idempotency key, never by launching twice.
        if record.get("pending_key"):
            child = self.orch._runs.find_by_idempotency_key(record["pending_key"])
            if child is not None:
                self._adopt(record, child)
        run = self.orch._runs.load(record["active_run_id"])
        if run is None:
            return self._block(record, "Active run record is missing")
        self.orch.refresh(run.run_id)
        run = self.orch._runs.load(run.run_id)
        # Preparation belongs to the launcher; only handle exited runs here.
        if run.state not in TERMINAL_STATES:
            return
        events_path = self.root / run.run_id / "events.jsonl"
        events = []
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        if any(e["type"] in {"stop_requested", "aborted_by_user"} for e in events):
            record.update(enabled=False, status="disabled", message="Stopped by user")
            self._save(record)
            return
        exit_code = next((e.get("exit_code") for e in reversed(events) if e["type"] == "process_exit"), None)
        # After a GUI restart, completion can be inferred from a checkpoint
        # directory name alone. Reuse the settling and validation below before
        # trusting that inference; an explicit successful exit stays unchanged.
        verify_completion = (
            run.state == RunState.COMPLETED and exit_code is None and run.args.get("steps") is not None
        )
        if run.state == RunState.COMPLETED and not verify_completion:
            record.update(status="completed", message="Training completed")
            self._save(record)
            return
        if run.started_at is None:
            return self._block(record, "Training could not start; inspect the startup log")

        now = self.clock()
        if "waiting_since" not in record:
            record.update(status="waiting", waiting_since=now, quiet_since=now)
            progress_path = self.root / run.run_id / "progress.json"
            try:
                progress = _read_json(progress_path) if progress_path.exists() else None
            except (OSError, ValueError):
                progress = None
            record["incidents"].append(
                {
                    "run_id": run.run_id,
                    "detected_at": now,
                    "reason": (
                        "Completion has no exit code; checking checkpoint integrity"
                        if verify_completion
                        else (run.error or "Process exited")[:500]
                    ),
                    "exit_code": exit_code,
                    "last_reported_progress": progress,
                    "log_path": str(self.root / run.run_id / "stderr.log"),
                    "events_path": str(events_path),
                    "crash_materials": [],
                }
            )
            record["message"] = "Waiting for the previous run and crash diagnostics to settle"
            self._save(record)
        incident = record["incidents"][-1]
        materials, collector_busy = crash_materials(run.started_at)
        paths = RunPaths.for_run(run.run_id, self.root)
        if run.args.get("__recipe__") != "__fake__":
            # The docker CLI can disappear while its container is still alive.
            # Every training recipe bind-mounts this run's directory. Match that
            # source so unrelated containers do not hold recovery open.
            try:
                result = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=5)
                collector_busy = collector_busy or result.returncode != 0
                container_ids = result.stdout.split()
                if result.returncode == 0 and container_ids:
                    mounts = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--type",
                            "container",
                            "--format",
                            "{{json .Mounts}}",
                            *container_ids,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if mounts.returncode != 0:
                        # A container may disappear between listing and inspection;
                        # let the existing wait loop query again before resuming.
                        collector_busy = True
                    else:
                        run_root = paths.root.resolve()
                        collector_busy = collector_busy or any(
                            mount["Type"] == "bind" and Path(mount["Source"]).resolve() == run_root
                            for line in mounts.stdout.splitlines()
                            for mount in json.loads(line)
                        )
            except subprocess.TimeoutExpired:
                collector_busy = True  # Retry within the existing settling deadline.
        incident["crash_materials"] = materials
        # Fingerprint file sizes/timestamps, never hash or copy tensor payloads.
        files = []
        for checkpoint, _ in self.orch._iter_checkpoint_dirs(
            self.orch._client_for_host(self.orch._hosts.get(run.host_id), paths, run), run, paths
        ):
            files.extend(p for p in checkpoint.rglob("*") if p.is_file())
        files.extend(Path(p) for p in materials)
        signature = []
        for path in sorted(files):
            try:
                stat = path.stat()
                signature.append([str(path), stat.st_size, stat.st_mtime_ns])
            except OSError:
                collector_busy = True
        if signature != record.get("file_signature") or collector_busy:
            record.update(file_signature=signature, quiet_since=now)
        self._save(record)
        if collector_busy or now - record["quiet_since"] < self.quiet_seconds:
            if now - record["waiting_since"] >= self.settle_timeout:
                self._block(
                    record,
                    "Previous checkpoint or crash diagnostics have not settled; inspect before resuming",
                )
            return
        if now - record["waiting_since"] < record["delay_seconds"]:
            return
        selected, skipped = self._checkpoint(record)
        incident["skipped_checkpoints"] = skipped
        if selected is None:
            return self._block(record, "No complete checkpoint is available; training was not restarted")
        source_id, step = selected
        incident.update(checkpoint_run_id=source_id, checkpoint_step=step)
        target = run.args.get("steps")
        if target is not None and step >= int(target):
            if verify_completion:
                # This was a completion check, not a confirmed crash.
                record["incidents"].pop()
                record.update(status="completed", message="Training completed; checkpoint verified")
                self._save(record)
                return
            return self._block(
                record, "The target checkpoint already exists; inspect the post-training error"
            )
        if record["attempts"] >= record["max_retries"]:
            return self._block(record, "Automatic recovery retry limit reached")
        busy = self.orch._runs.active_run_on_host(run.host_id)
        if busy is not None:
            return self._block(record, f"Another training run is active: {busy.run_id}")
        record["pending_key"] = record.get("pending_key") or (
            f"auto-recovery:{record['root_run_id']}:{record['attempts'] + 1}"
        )
        record["status"] = "resuming"
        self._save(record)
        if self._shutdown.is_set():
            return
        child = self.orch.resume(
            source_id,
            checkpoint_step=step,
            idempotency_key=record["pending_key"],
            **{key: run.args.get(key) for key in ("batch_size", "num_workers", "save_freq", "steps")},
        )
        self._adopt(record, child)

    def _adopt(self, record, child):
        if child.run_id not in record["run_ids"]:
            record["run_ids"].append(child.run_id)
            record["attempts"] += 1
        record["incidents"][-1]["resumed_run_id"] = child.run_id
        record.update(active_run_id=child.run_id, status="monitoring", message="Resumed training")
        for key in ("pending_key", "waiting_since", "quiet_since", "file_signature"):
            record.pop(key, None)
        self._save(record)

    def _checkpoint(self, record):
        candidates = []
        # Include earlier attempts: the last retry may have died before saving.
        pending = [(run_id, None) for run_id in record["run_ids"]]
        visited = set()
        for run_id, ceiling in pending:
            if run_id in visited:
                continue
            visited.add(run_id)
            run = self.orch._runs.load(run_id)
            if run is None:
                continue
            parent = run.args.get("__resumed_from_run__")
            parent_step = run.args.get("__resumed_from_step__")
            if parent and parent_step is not None:
                pending.append((parent, min(parent_step, ceiling) if ceiling is not None else parent_step))
            paths = RunPaths.for_run(run_id, self.root)
            client = self.orch._client_for_host(self.orch._hosts.get(run.host_id), paths, run)
            for path, step in self.orch._iter_checkpoint_dirs(client, run, paths):
                if ceiling is None or step <= ceiling:
                    candidates.append((step, run.created_at, run, path))
        skipped = []
        for step, _, run, path in sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True):
            ok, reason = check_checkpoint(path, step, run.args.get("__recipe__") == "hvla_flow_s1")
            if ok:
                return (run.run_id, step), skipped
            skipped.append({"path": str(path), "reason": reason})
        return None, skipped
