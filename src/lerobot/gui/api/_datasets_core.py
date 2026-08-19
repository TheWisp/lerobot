# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Whole-dataset operations shared by the FastAPI routes and the MCP tools.

Same split as ``_edits_core``: the substantive logic lives here as sync free
functions that take ``AppState`` and raise typed exceptions, so neither caller
re-implements validation and MCP never self-calls FastAPI.

Mapping:

- ``NotADatasetError``     → FastAPI 404 / MCP error
- ``InvalidNameError``     → FastAPI 400 / MCP error
- ``DatasetExistsError``   → FastAPI 409 / MCP error
- ``DatasetBusyError``     → FastAPI 423 / MCP error

Both functions block: they copy or remove a directory tree that runs to
gigabytes. The FastAPI handlers hand them to a dedicated executor; the MCP
tools call them directly, being sync already.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.gui.api._edits_core import DatasetBusyError

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

# Marks a copy in flight. Dot-prefixed so the source scanner skips it, and
# distinctive so a leftover is recognisable as ours rather than a user folder.
STAGING_PREFIX = ".copying-"
# Marks a dataset renamed aside for removal, so a failed delete can never leave
# a half-removed dataset visible in the tree.
DELETING_PREFIX = ".deleting-"


class NotADatasetError(ValueError):
    """Path does not hold ``meta/info.json``, so it is not a dataset root."""


class InvalidNameError(ValueError):
    """Requested folder name is empty, a dot entry, or carries a separator."""


class DatasetExistsError(ValueError):
    """Something already occupies the destination path."""


class CopyFailedError(RuntimeError):
    """The copy could not be completed; the message says why, in one line."""


class DeleteFailedError(RuntimeError):
    """The files could not be removed. The message must say what state that left."""


def _describe_copy_failure(err: shutil.Error, src_root: Path) -> str:
    """Turn a per-file error list into one honest, readable sentence.

    ``shutil.copytree`` lists the source once and then copies each entry, so a
    source that is deleted or renamed part-way through fails on every remaining
    file and raises a ``shutil.Error`` carrying one ``(src, dst, reason)`` tuple
    for each. Measured on a 4000-file tree that is 14 KB of text when the source
    is deleted and 228 KB when it is renamed — which reached the user as an
    alert() of that length.

    The count and one example are the honest summary: they say what happened and
    how widespread it was without pretending a single file was at fault. Whether
    the source still exists distinguishes the two cases the user can act on —
    it moved, versus this disk is failing.
    """
    failures = err.args[0] if err.args and isinstance(err.args[0], list) else []
    n = len(failures)
    first = ""
    if failures:
        f_src, _f_dst, reason = failures[0]
        first = f" First: {Path(f_src).name} — {str(reason).strip()}"
    if not src_root.exists():
        return (
            f"The source dataset was moved or deleted while copying, so {n} file(s) could not be read.{first}"
        )
    return f"{n} file(s) could not be copied.{first}"


def sweep_remnants(source_root: Path) -> list[str]:
    """Remove staging and deletion remnants under ``source_root``.

    Pre: ``source_root`` is a configured source folder.
    Post: every ``.copying-*`` / ``.deleting-*`` directory one level under an
    owner directory is gone; returns what was removed.

    Both operations rename rather than write in place, so an interrupted one
    leaves a dot-prefixed directory. That is deliberate — invisible beats
    half-formed — but invisible also means nobody notices it occupying disk.
    Called when a source is scanned, which is the moment the cost would
    otherwise be silently carried.
    """
    removed: list[str] = []
    if not source_root.is_dir():
        return removed
    try:
        owners = list(source_root.iterdir())
    except OSError:
        return removed
    for owner in owners:
        if not owner.is_dir():
            continue
        try:
            children = list(owner.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and child.name.startswith((STAGING_PREFIX, DELETING_PREFIX)):
                logger.info(f"Sweeping remnant of an interrupted operation: {child}")
                # safe-destruct: only our own dot-prefixed remnants, never a dataset
                shutil.rmtree(child, ignore_errors=True)
                removed.append(str(child))
    return removed


def _dataset_root(path: str) -> Path:
    """Resolve ``path`` to a dataset root.

    Pre: ``path`` is the filesystem path used as the GUI's registry key.
    Post: returns it, having verified it holds ``meta/info.json``.

    This check is the safety gate for both operations below — it is what stops
    a copy or a delete being pointed at a source folder, a home directory, or
    anything else that is not a dataset. Datasets merely listed in the tree are
    accepted; neither operation requires the dataset to be open.
    """
    root = Path(path)
    if not (root / "meta" / "info.json").is_file():
        raise NotADatasetError(f"Not a dataset directory: {path}")
    return root


def _registry_keys_for(app_state: AppState, root: Path) -> list[str]:
    """Every registry key whose dataset lives at ``root``.

    The key is whatever ``open_dataset`` was handed: the filesystem path for a
    local open, but the ``repo_id`` for a Hub-style one — which is what the MCP
    bridge and the ``#/dataset/<owner>/<name>`` deep link both use. Callers here
    only ever have a path, so matching on the path alone would miss a
    repo_id-keyed entry and leave its registry row, its lock, its staged edits
    and the opened-state file all pointing at a directory that no longer exists.

    Matching is by resolved root rather than by key, so it holds however the
    dataset was opened.
    """
    keys: list[str] = []
    target = root.resolve()
    for key, ds in app_state.datasets.items():
        try:
            if Path(ds.root).resolve() == target:
                keys.append(key)
        except OSError:
            continue
    return keys


def _reject_if_busy(app_state: AppState, dataset_id: str, keys: list[str] | None = None) -> None:
    """Refuse while any key for this dataset holds a lock.

    Checking only the caller's key would let a copy or a delete run under an
    apply that took the lock under the dataset's other name.
    """
    for key in {dataset_id, *(keys or [])}:
        if app_state.is_locked(key):
            raise DatasetBusyError(f"Dataset {key} is busy (operation in progress)")


def duplicate_dataset(app_state: AppState, path: str, new_name: str) -> dict[str, Any]:
    """Copy the dataset at ``path`` to a sibling directory named ``new_name``.

    Pre: ``path`` is a dataset root, not busy; ``new_name`` is a single path
    component that does not already exist beside it.
    Post: the copy is a complete, openable dataset and the original is
    untouched. On failure the partial copy is removed rather than left behind,
    since a half-written directory still holds ``meta/info.json`` and would
    list in the tree as a real dataset.

    Identity needs no rewriting: ``repo_id`` is derived from the last two path
    components and the registry keys on the path, so a copy at a new path is
    already a distinct dataset. Nothing inside ``meta/`` names it.
    """
    src_root = _dataset_root(path)
    _reject_if_busy(app_state, path, _registry_keys_for(app_state, src_root))

    name = new_name.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise InvalidNameError("Name must be a single folder name, without slashes")

    dst_root = src_root.parent / name
    if dst_root.exists():
        raise DatasetExistsError(f"Already exists: {dst_root}")

    # Copy into a dot-prefixed sibling and rename it into place. Two reasons the
    # destination is never written directly: `_scan_recursive` skips dot-prefixed
    # directories, so an in-flight copy cannot list or open as a real dataset;
    # and rename within a filesystem is atomic, so the destination is either
    # absent or whole. Cleaning up in `except` alone would not do — a server
    # killed mid-copy never runs it, and would strand a partial directory
    # holding meta/info.json.
    staging = dst_root.parent / f"{STAGING_PREFIX}{dst_root.name}"
    if staging.exists():
        # safe-destruct: a staging dir from a copy that never finished
        shutil.rmtree(staging, ignore_errors=True)

    logger.info(f"Duplicating dataset {src_root} -> {dst_root}")
    try:
        shutil.copytree(src_root, staging)
        staging.rename(dst_root)
    except shutil.Error as e:
        if staging.exists():
            # safe-destruct: removes only the staging dir this call just made
            shutil.rmtree(staging, ignore_errors=True)
        raise CopyFailedError(_describe_copy_failure(e, src_root)) from e
    except Exception:
        if staging.exists():
            # safe-destruct: removes only the staging dir this call just made
            shutil.rmtree(staging, ignore_errors=True)
        raise

    parts = dst_root.parts[-2:]
    repo_id = "/".join(parts) if len(parts) >= 2 else dst_root.name
    logger.info(f"Duplicated dataset to {dst_root}")
    return {"status": "ok", "root": str(dst_root), "repo_id": repo_id}


def delete_dataset(app_state: AppState, path: str) -> dict[str, Any]:
    """Remove the dataset at ``path`` from disk, closing it first if open.

    Pre: ``path`` is a dataset root and is not busy.
    Post: the directory is gone and no registry entry, cache, lock or staged
    edit still refers to it.

    Irreversible — there is no trash. Confirmation belongs to the caller: the
    GUI names the dataset and its episode count in a dialog, and the MCP tool
    requires an explicit flag.
    """
    root = _dataset_root(path)
    keys = _registry_keys_for(app_state, root)
    _reject_if_busy(app_state, path, keys)

    # Order matters: drop the in-memory state first. Removing the files while
    # the registry still held the dataset would leave every later read of it
    # failing against a directory that no longer exists — and the opened-state
    # file, rewritten inside forget_dataset, would reopen it on next launch.
    from lerobot.gui.api.datasets import forget_dataset

    for key in keys:
        forget_dataset(key)

    # Rename out of the way first, then remove — the mirror of how a copy is
    # staged. `rmtree` deletes bottom-up, so a failure part-way through would
    # otherwise leave a directory that is neither gone nor whole: with
    # `meta/info.json` still present it lists and opens as a real dataset whose
    # shards are missing, and without it an orphan directory nothing in the GUI
    # can see. Renaming first makes the dataset disappear atomically, and any
    # remnant is dot-prefixed, invisible to the scanner, and swept later.
    logger.info(f"Deleting dataset from disk: {root}")
    doomed = root.parent / f"{DELETING_PREFIX}{root.name}"
    if doomed.exists():
        # safe-destruct: a remnant from an earlier delete that never finished
        shutil.rmtree(doomed, ignore_errors=True)
    try:
        root.rename(doomed)
    except OSError as e:
        # Nothing has been destroyed: the dataset is intact where it was.
        logger.error(f"Delete failed for {root} (rename): {e}")
        raise DeleteFailedError(
            f"Could not delete {root}: {e}. Nothing was removed — the dataset is "
            f"still on disk, and has been closed in the GUI; reopen it to continue."
        ) from e

    try:
        # safe-destruct: caller confirmed; path verified to hold meta/info.json
        shutil.rmtree(doomed)
    except Exception as e:
        # The dataset is already gone from the tree and cannot be opened, so
        # this is a disk-space problem rather than a data-integrity one.
        logger.error(f"Delete failed for {root} (remove): {e}")
        raise DeleteFailedError(
            f"{root} was removed from the GUI, but its files could not all be "
            f"deleted: {e}. The remainder is in {doomed.name}, which the tree "
            f"ignores and the next scan of this source reclaims."
        ) from e
    return {"status": "ok", "message": f"Deleted: {root}"}
