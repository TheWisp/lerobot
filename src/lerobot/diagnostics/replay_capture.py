"""Optional SmolVLA replay inputs, independent of videos and model checkpoints.

Capture only actual sampler calls. CPU snapshots go to a bounded background writer;
the writer never calls the policy or a robot. Completed files are atomically named.
An interrupted session stays open/incomplete; discarded takes are explicitly marked.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _json_write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        # A synchronous, owned copy: no reference to mutable CUDA/queue storage
        # escapes to the writer. Its measured cost is recorded per prediction.
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_copy(v) for v in value]
    return value


class ReplayCapture:
    def __init__(self, root: Path, metadata: dict, max_pending_bytes: int = 256 * 1024**2):
        self.root = Path(root) / "replay_capture" / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.max_pending_bytes = max_pending_bytes
        self._pending = 0
        self.peak_pending_bytes = 0
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._error = None
        self._closed = False
        self._attempt = None
        self._attempt_count = 0
        self._frame = None
        self._next_prediction = 0
        self._metadata = {"schema_version": 1, "status": "open", "created_unix_s": time.time(), **metadata}
        _json_write(self.root / "session.json", self._metadata)
        self._thread = threading.Thread(target=self._write_loop, name="smolvla-replay-writer", daemon=True)
        self._thread.start()
        logger.info(
            "Replay capture enabled: %s (CPU buffer limit %.0f MiB)", self.root, max_pending_bytes / 1024**2
        )

    def begin_episode(self, episode_index):
        if self._attempt is not None:
            raise RuntimeError("Previous capture attempt has not been finished")
        self._attempt = {
            "attempt": self._attempt_count,
            "episode_index": episode_index,
            "status": "recording",
            "submitted": 0,
            "written": 0,
            "missing": 0,
        }
        self._attempt_count += 1
        self._frame = None
        self._queue.put(("begin", self._attempt, None, 0))

    def set_frame(self, frame_index):
        self._frame = {
            "frame_index": int(frame_index),
            "input_unix_ns": time.time_ns(),
            "input_monotonic_ns": time.monotonic_ns(),
        }

    def record_model_call(self, **inputs):
        """Best effort capture: data failure must not change control flow or RNG."""
        attempt = self._attempt
        if attempt is None or self._frame is None or self._closed:
            return
        prediction = self._next_prediction
        self._next_prediction += 1
        size = _tensor_bytes(inputs)
        with self._lock:
            if self._error or self._pending + size > self.max_pending_bytes:
                attempt["missing"] += 1
                if attempt["missing"] == 1:
                    logger.error(
                        "Replay capture incomplete: writer error or full buffer; inference continues"
                    )
                return
            self._pending += size
            self.peak_pending_bytes = max(self.peak_pending_bytes, self._pending)
        try:
            start = time.perf_counter()
            payload = _cpu_copy(inputs)
            payload["capture"] = {
                **self._frame,
                "prediction_index": prediction,
                "episode_index": attempt["episode_index"],
                "attempt": attempt["attempt"],
                "copy_ms": (time.perf_counter() - start) * 1000,
                "tensor_bytes": size,
            }
            attempt["submitted"] += 1
            self._queue.put(("prediction", attempt, payload, size))
        except Exception as exc:
            with self._lock:
                self._pending -= size
            attempt["missing"] += 1
            logger.exception("Replay snapshot failed; inference continues: %s", exc)

    def finish_episode(self, disposition, recorded_frames=None):
        if self._attempt is not None:
            attempt = self._attempt
            self._queue.put(
                ("finish", attempt, {"disposition": disposition, "recorded_frames": recorded_frames}, 0)
            )
            self._attempt = None
            self._frame = None

    def _save_prediction(self, path, payload):
        temp = path.with_suffix(".pt.tmp")
        torch.save(payload, temp)
        os.replace(temp, path)

    def _write_loop(self):
        while True:
            kind, attempt, payload, size = self._queue.get()
            try:
                if kind == "close":
                    self._metadata.update(
                        status="incomplete" if self._error else "closed",
                        error=self._error,
                        peak_pending_bytes=self.peak_pending_bytes,
                    )
                    _json_write(self.root / "session.json", self._metadata)
                    return
                folder = self.root / f"attempt_{attempt['attempt']:06d}"
                folder.mkdir(exist_ok=True)
                if kind == "begin":
                    _json_write(folder / "attempt.json", dict(attempt))
                elif kind == "prediction":
                    if not self._error:
                        filename = f"prediction_{payload['capture']['prediction_index']:06d}.pt"
                        self._save_prediction(folder / filename, payload)
                        attempt["written"] += 1
                        with (folder / "index.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"file": filename, **payload["capture"]}) + "\n")
                elif kind == "finish":
                    status = {**attempt, **payload}
                    status["status"] = (
                        "complete"
                        if (
                            not self._error
                            and not attempt["missing"]
                            and attempt["written"] == attempt["submitted"]
                        )
                        else "incomplete"
                    )
                    _json_write(folder / "attempt.json", status)
                    logger.info(
                        "Replay capture episode %s: %s, %s (%s predictions, %s missing)",
                        attempt["episode_index"],
                        payload["disposition"],
                        status["status"],
                        attempt["written"],
                        attempt["missing"],
                    )
            except Exception as exc:
                self._error = str(exc)
                logger.exception("Replay writer failed; capture is incomplete, inference continues")
                if kind == "close":
                    return
            finally:
                with self._lock:
                    self._pending -= size
                self._queue.task_done()

    def close(self, timeout=5.0):
        if not self._closed:
            self.finish_episode("aborted")
            self._closed = True
            self._queue.put(("close", None, None, 0))
        self._thread.join(timeout)
        done = not self._thread.is_alive()
        if not done:
            logger.error("Replay capture still saving after %.1fs; session is not marked complete", timeout)
        elif self._error:
            logger.error("Replay capture finished with missing data: %s", self._error)
        return done and not self._error


def create_smolvla_capture(policy, dataset, record_config):
    if policy.config.type != "smolvla" or policy.config.rtc_config is not None:
        raise ValueError("Replay input capture currently supports standard SmolVLA only (no RTC)")
    if policy.config.compile_model:
        raise ValueError("Replay input capture currently requires compile_model=false")
    if record_config.interpolation_multiplier != 1:
        raise ValueError("Replay input capture currently requires interpolation_multiplier=1")
    checkpoint = Path(policy.config.pretrained_path).expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError("Replay input capture requires a local SmolVLA checkpoint")
    # One hash per session, before robot.connect(); weights are NOT copied.
    with (checkpoint / "model.safetensors").open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    source = Path(__file__).resolve()
    revision = subprocess.run(
        ["git", "-C", str(source.parent), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source.parent), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    metadata = {
        "policy_type": "smolvla",
        "checkpoint": str(checkpoint),
        "weights_sha256": fingerprint,
        "policy_config": dataclasses.asdict(policy.config),
        "record_config": dataclasses.asdict(record_config),
        "image_keys": list(policy.config.image_features),
        "code_revision": revision,
        "code_dirty": bool(dirty),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "transformers_version": importlib.metadata.version("transformers"),
        "parameter_dtypes": sorted({str(p.dtype) for p in policy.parameters()}),
        "sampler_source_sha256": hashlib.sha256(
            Path(__import__(type(policy.model).__module__, fromlist=["__file__"]).__file__).read_bytes()
        ).hexdigest(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tensor_contract": "Exact sample_actions inputs after resize/normalization; initial noise; full normalized padded action chunk. No KV cache or gradients saved.",
        "frame_contract": "frame_index is episode-local input frame at a NEW sampler call; queued actions have no new capture. Read only complete, saved attempts for normal replay.",
    }
    capture = ReplayCapture(dataset.root, metadata)
    try:
        artifacts = capture.root / "checkpoint_config"
        artifacts.mkdir()
        for path in checkpoint.iterdir():
            if path.is_file() and (
                path.suffix == ".json" or (path.suffix == ".safetensors" and path.name != "model.safetensors")
            ):
                shutil.copy2(path, artifacts / path.name)
    except Exception:
        capture.close()
        raise
    policy.model._replay_capture_sink = capture
    return capture
