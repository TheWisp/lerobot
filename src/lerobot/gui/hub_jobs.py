# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-file Hub upload/download loops with progress accounting.

Background
----------
`huggingface_hub` 1.x exposes a clean progress hook only on the download
side (``tqdm_class=`` on ``snapshot_download``/``hf_hub_download``). Uploads
go through ``_commit_api`` which fans out to three different transports
(regular HTTP, LFS multipart, Xet content-addressed) — there is no single
hook that captures all three.

Rather than monkey-patch HF internals, we own the loop on both sides:

* **Uploads** enumerate files locally, call ``upload_file`` per file. We
  know each file's size from ``os.stat`` upfront, so totals are exact.
* **Downloads** enumerate remote files via ``HfApi.dataset_info`` (siblings
  + sizes), then call ``hf_hub_download`` per file. Same shape as upload.

Both loops update a shared :class:`HubJobState` dataclass that the GUI
polls. ``cancel_event.set()`` interrupts the loop between files (we cannot
interrupt a single ``upload_file`` / ``hf_hub_download`` mid-call without
killing the executor thread).

Trade-offs
----------
We give up ``snapshot_download``'s parallel ``max_workers`` and
``upload_folder``'s atomic single-commit semantics. In return we get:

* file-level progress on both sides without internal-module assumptions
* a cancel point between files that does not require killing the thread
* identical code shape for both directions (one ``HubJobState``, one
  polling endpoint, one frontend renderer)

If parallel downloads become important later, we can swap the download loop
back to ``snapshot_download(tqdm_class=...)`` while keeping the same
``HubJobState`` schema; the frontend would not change.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# Paths we never want to push: GUI-local metadata + HF cache lock files +
# temp artifacts left by interrupted writes. Kept conservative — anything
# under the dataset root that isn't one of these is uploaded as-is.
_DEFAULT_UPLOAD_IGNORES: tuple[str, ...] = (
    ".cache/",
    ".lerobot_gui_edits.json",
    ".huggingface/",
    ".DS_Store",
)

# Direction values are kept narrow to make the GUI status renderer dispatch
# trivially (no branching on free-form strings).
HubDirection = Literal["upload", "download"]
HubStatus = Literal["pending", "running", "complete", "failed", "cancelled"]


@dataclass
class HubJobState:
    """One in-flight Hub transfer, polled by the GUI.

    All counters are monotonic within a single job; ``files_done`` only
    increments after the per-file call returns successfully, so a poll
    that catches mid-file will under-report (never over-report).

    Mutated from the executor thread, read from asyncio. CPython's GIL
    makes individual int/str assignments atomic; ``to_dict`` reads each
    field once and may briefly observe slightly stale values across
    fields. That's acceptable for a progress display.
    """

    job_id: str
    dataset_id: str
    direction: HubDirection
    repo_id: str
    status: HubStatus
    started_at: float
    finished_at: float | None = None
    files_total: int = 0
    files_done: int = 0
    bytes_total: int = 0
    bytes_done: int = 0
    current_file: str | None = None
    message: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot, suitable for the progress endpoint."""
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "direction": self.direction,
            "repo_id": self.repo_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "files_total": self.files_total,
            "files_done": self.files_done,
            "bytes_total": self.bytes_total,
            "bytes_done": self.bytes_done,
            "current_file": self.current_file,
            "message": self.message,
        }


def make_job(dataset_id: str, direction: HubDirection, repo_id: str) -> HubJobState:
    """Build a new ``HubJobState`` in ``pending`` status."""
    return HubJobState(
        job_id=uuid.uuid4().hex,
        dataset_id=dataset_id,
        direction=direction,
        repo_id=repo_id,
        status="pending",
        started_at=time.time(),
    )


def _is_ignored(rel_path: str, ignores: tuple[str, ...]) -> bool:
    """Match ``rel_path`` (posix-style, relative to root) against ignore prefixes/names.

    Patterns ending in ``/`` are treated as directory prefixes; everything
    else matches by basename or exact path. Keeps the pattern set small —
    we don't need full gitignore semantics, just a hard skip list.
    """
    base = rel_path.rsplit("/", 1)[-1]
    for pat in ignores:
        if pat.endswith("/"):
            if rel_path.startswith(pat) or f"/{pat}" in f"/{rel_path}":
                return True
        elif base == pat or rel_path == pat:
            return True
    return False


def enumerate_upload_files(
    root: Path,
    *,
    ignore_patterns: tuple[str, ...] = _DEFAULT_UPLOAD_IGNORES,
) -> list[Path]:
    """List every regular file under ``root`` not matching an ignore pattern.

    Pre: ``root`` exists and is a directory.
    Post: returned paths are absolute and ordered (sorted) so the upload
    sequence is deterministic — makes the progress display predictable and
    helps test reproducibility.
    """
    assert root.exists() and root.is_dir(), f"upload root missing or not a dir: {root}"

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_ignored(rel, ignore_patterns):
            continue
        files.append(path)
    return files


def run_upload_sync(
    *,
    root: Path,
    repo_id: str,
    job: HubJobState,
    create_repo: bool = True,
    private: bool = True,
) -> None:
    """Per-file upload loop. Synchronous — meant for an executor thread.

    Pre: ``job.status`` is ``"pending"`` or ``"running"``. Caller has
    already verified Hub auth.
    Post: ``job.files_done`` equals ``job.files_total`` on success; on
    cancel ``job.status == "cancelled"`` and ``files_done < files_total``;
    on HF error the exception propagates and the caller sets
    ``status="failed"``.

    The HF call (`upload_file`) is not interruptible mid-flight, so the
    cancel check happens between files only.
    """
    from huggingface_hub import HfApi, upload_file

    if create_repo:
        HfApi().create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)

    files = enumerate_upload_files(root)
    job.files_total = len(files)
    job.bytes_total = sum(p.stat().st_size for p in files)
    job.status = "running"

    for path in files:
        if job.cancel_event.is_set():
            job.status = "cancelled"
            return
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        job.current_file = rel
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {rel} from LeRobot GUI",
        )
        # Increment AFTER the call returns — guarantees files_done counts
        # only files that have actually been committed remotely. A poll
        # that hits mid-call observes the previous count + current_file
        # pointing at the in-flight file, which is the truthful view.
        job.files_done += 1
        job.bytes_done += size

    job.current_file = None


def run_download_sync(
    *,
    root: Path,
    repo_id: str,
    job: HubJobState,
) -> None:
    """Per-file download loop. Mirror of :func:`run_upload_sync`.

    Pre: ``job.status`` is ``"pending"`` or ``"running"``.
    Post: every sibling of the remote repo lives under ``root`` (HF's
    ``hf_hub_download`` skips files already present with matching etag).
    On cancel ``status="cancelled"`` and ``files_done < files_total``.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=True)
    siblings = info.siblings or []
    job.files_total = len(siblings)
    job.bytes_total = sum((s.size or 0) for s in siblings)
    job.status = "running"

    os.makedirs(root, exist_ok=True)

    for sib in siblings:
        if job.cancel_event.is_set():
            job.status = "cancelled"
            return
        rel = sib.rfilename
        size = sib.size or 0
        job.current_file = rel
        hf_hub_download(
            repo_id=repo_id,
            filename=rel,
            repo_type="dataset",
            local_dir=str(root),
        )
        job.files_done += 1
        job.bytes_done += size

    job.current_file = None
