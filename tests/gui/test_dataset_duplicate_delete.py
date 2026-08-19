# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Duplicating and deleting a dataset from the GUI.

The tree could open, merge, upload, download and reveal a dataset but not copy
or remove one, so post-processing meant mutating the only copy or dropping to
``cp -r`` in a terminal (#91).

Both routes are destructive in different directions, and the assertions below
are mostly about what they refuse. Delete is irreversible and has no trash;
duplicate writes gigabytes and must not leave a half-copy behind, because a
partial directory still holds ``meta/info.json`` and would list in the tree as
a real dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState

REPO_ID = "test/dup_delete"


@pytest.fixture
def client(tmp_path, lerobot_dataset_factory, monkeypatch):
    """A GUI client over one hermetic dataset; nothing outside tmp_path is touched."""
    hf_home = tmp_path / "hf_home"
    root = hf_home / REPO_ID
    root.mkdir(parents=True)
    lerobot_dataset_factory(
        root=root,
        repo_id=REPO_ID,
        total_episodes=2,
        total_frames=20,
        total_tasks=1,
        use_videos=False,
        camera_features={},
    )
    monkeypatch.setattr(datasets_module, "HF_LEROBOT_HOME", hf_home)

    app = FastAPI()
    app.include_router(datasets_module.router)
    original_state = datasets_module._app_state
    datasets_module.set_app_state(AppState(frame_cache=FrameCache(max_bytes=1_000_000)))
    try:
        with TestClient(app) as c:
            yield c, root
    finally:
        datasets_module._app_state = original_state


def _dup(client, root: Path, name: str):
    return client.post("/api/datasets/duplicate", json={"path": str(root), "new_name": name})


def _delete(client, root) -> object:
    return client.delete("/api/datasets/files", params={"path": str(root)})


def test_duplicate_produces_an_independent_openable_copy(client):
    c, root = client
    r = _dup(c, root, "copy_of_it")
    assert r.status_code == 200, r.text

    dst = Path(r.json()["root"])
    assert dst == root.parent / "copy_of_it"
    assert (dst / "meta" / "info.json").is_file()

    # Same content, and the two agree on episode count — a copy that cannot be
    # opened is worse than no copy, because the tree lists it as a dataset.
    src_files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst) for p in dst.rglob("*") if p.is_file())
    assert src_files == dst_files
    assert json.loads((dst / "meta" / "info.json").read_text())["total_episodes"] == 2

    # Identity comes from the path, so the copy is already a distinct dataset
    # with no metadata rewrite.
    assert r.json()["repo_id"] == f"{root.parent.name}/copy_of_it"

    # The original survives.
    assert (root / "meta" / "info.json").is_file()


def test_duplicate_refuses_to_overwrite(client):
    c, root = client
    assert _dup(c, root, "taken").status_code == 200
    again = _dup(c, root, "taken")
    assert again.status_code == 409, again.text


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "a/b", "../escape"])
def test_duplicate_rejects_names_that_are_not_a_single_folder(client, name):
    """A name with a separator would write outside the source's parent."""
    c, root = client
    r = _dup(c, root, name)
    assert r.status_code == 400, f"{name!r} was accepted: {r.text}"
    assert not (root.parent / "escape").exists()


def test_duplicate_leaves_no_partial_copy_when_it_fails(client, monkeypatch):
    """A half-written directory still holds meta/info.json and would list as real."""
    c, root = client

    def boom(src, dst):
        Path(dst).mkdir(parents=True)
        (Path(dst) / "meta").mkdir()
        (Path(dst) / "meta" / "info.json").write_text("{}")
        raise OSError("no space left on device")

    monkeypatch.setattr(shutil, "copytree", boom)
    r = _dup(c, root, "doomed")
    assert r.status_code == 500, r.text
    assert not (root.parent / "doomed").exists(), "partial copy was left behind"


def test_delete_removes_the_directory(client):
    c, root = client
    r = _delete(c, root)
    assert r.status_code == 200, r.text
    assert not root.exists()


def test_delete_refuses_a_path_that_is_not_a_dataset(client, tmp_path):
    """The meta/info.json check is what stops this deleting arbitrary folders."""
    c, root = client
    bystander = tmp_path / "not_a_dataset"
    bystander.mkdir()
    (bystander / "precious.txt").write_text("keep me")

    r = _delete(c, bystander)
    assert r.status_code == 404, r.text
    assert (bystander / "precious.txt").is_file()

    # The source folder that *contains* datasets is not itself one.
    r = _delete(c, root.parent)
    assert r.status_code == 404, r.text
    assert root.is_dir()


def test_delete_drops_the_dataset_from_the_registry(client):
    """Files gone while the registry still pointed at them would fail every later read."""
    c, root = client
    assert c.post("/api/datasets", json={"local_path": str(root)}).status_code == 200
    assert str(root) in datasets_module._app_state.datasets

    assert _delete(c, root).status_code == 200
    assert str(root) not in datasets_module._app_state.datasets
    assert not root.exists()


def test_busy_dataset_is_refused_by_both_routes(client):
    """Copying or deleting under a running edit would race the writer."""
    c, root = client
    assert c.post("/api/datasets", json={"local_path": str(root)}).status_code == 200

    datasets_module._app_state.get_lock(str(root))
    monkeyed = datasets_module._app_state.is_locked
    datasets_module._app_state.is_locked = lambda ds_id: ds_id == str(root)
    try:
        assert _delete(c, root).status_code == 423
        assert _dup(c, root, "while_busy").status_code == 423
    finally:
        datasets_module._app_state.is_locked = monkeyed

    assert root.is_dir(), "a busy dataset must survive both"
    assert not (root.parent / "while_busy").exists()


@pytest.mark.parametrize("folder", ["files", "duplicate"])
def test_close_still_closes_a_dataset_whose_folder_shares_a_route_name(
    tmp_path, lerobot_dataset_factory, monkeypatch, folder
):
    """Close must keep meaning close, whatever the dataset folder is called.

    ``DELETE /{dataset_id:path}`` is greedy. A first cut at this feature hung
    the two new operations off path suffixes — ``/{id}/files`` and
    ``/{id}/duplicate`` — which silently stole close from any dataset whose
    directory carried one of those names, routing it into delete-from-disk with
    the id truncated to the parent. Hence the literal-segment endpoints with the
    path in the body.
    """
    hf_home = tmp_path / "hf_home"
    root = hf_home / "owner" / folder
    root.mkdir(parents=True)
    lerobot_dataset_factory(
        root=root,
        repo_id=f"owner/{folder}",
        total_episodes=1,
        total_frames=10,
        total_tasks=1,
        use_videos=False,
        camera_features={},
    )
    monkeypatch.setattr(datasets_module, "HF_LEROBOT_HOME", hf_home)

    app = FastAPI()
    app.include_router(datasets_module.router)
    original_state = datasets_module._app_state
    datasets_module.set_app_state(AppState(frame_cache=FrameCache(max_bytes=1_000_000)))
    try:
        with TestClient(app) as c:
            assert c.post("/api/datasets", json={"local_path": str(root)}).status_code == 200
            r = c.delete(f"/api/datasets/{root}")
            assert r.status_code == 200, r.text
            assert str(root) not in datasets_module._app_state.datasets, "close must close"
            assert (root / "meta" / "info.json").is_file(), "close must not touch the files"
    finally:
        datasets_module._app_state = original_state


def test_delete_forgets_a_dataset_opened_under_its_repo_id(tmp_path, lerobot_dataset_factory, monkeypatch):
    """The registry key is whatever ``open_dataset`` was handed.

    A local open keys on the filesystem path, but a Hub-style open keys on the
    ``repo_id`` — which is what the MCP bridge and the ``#/dataset/<repo_id>``
    deep link produce. Both delete callers only ever have a path, so keying on
    it alone left the registry row, the lock and the opened-state file pointing
    at a directory that had just been removed, with MCP's own edit tools still
    accepting the repo_id and writing to it.
    """
    hf_home = tmp_path / "hf_home"
    root = hf_home / "owner" / "pick_place"
    root.mkdir(parents=True)
    ds = lerobot_dataset_factory(
        root=root,
        repo_id="owner/pick_place",
        total_episodes=1,
        total_frames=10,
        total_tasks=1,
        use_videos=False,
        camera_features={},
    )
    monkeypatch.setattr(datasets_module, "HF_LEROBOT_HOME", hf_home)

    app = FastAPI()
    app.include_router(datasets_module.router)
    original_state = datasets_module._app_state
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    datasets_module.set_app_state(state)
    try:
        # Registered under the repo_id, as a bridge/deep-link open does.
        state.datasets["owner/pick_place"] = ds
        with TestClient(app) as c:
            assert _delete(c, root).status_code == 200
            assert not root.exists()
            assert "owner/pick_place" not in state.datasets, (
                "a repo_id-keyed entry outlived the directory it points at"
            )
            # The opened-state file is only rewritten inside forget_dataset, so
            # a skipped forget leaves the deleted root listed and the next
            # launch reopens a directory that is gone. This is the symptom a
            # user actually meets.
            opened = datasets_module.OPENED_FILE
            listed = opened.read_text() if opened.exists() else ""
            assert str(root) not in listed, (
                f"deleted dataset still listed for reopen on next launch: {listed!r}"
            )
    finally:
        datasets_module._app_state = original_state


def test_busy_under_the_repo_id_key_blocks_a_path_addressed_delete(
    tmp_path, lerobot_dataset_factory, monkeypatch
):
    """An apply holds the lock under the key it opened with, not under the path."""
    hf_home = tmp_path / "hf_home"
    root = hf_home / "owner" / "busy_one"
    root.mkdir(parents=True)
    ds = lerobot_dataset_factory(
        root=root,
        repo_id="owner/busy_one",
        total_episodes=1,
        total_frames=10,
        total_tasks=1,
        use_videos=False,
        camera_features={},
    )
    monkeypatch.setattr(datasets_module, "HF_LEROBOT_HOME", hf_home)

    app = FastAPI()
    app.include_router(datasets_module.router)
    original_state = datasets_module._app_state
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    datasets_module.set_app_state(state)
    try:
        state.datasets["owner/busy_one"] = ds
        real_is_locked = state.is_locked
        state.is_locked = lambda key: key == "owner/busy_one"
        with TestClient(app) as c:
            assert _delete(c, root).status_code == 423
            assert _dup(c, root, "copy_while_busy").status_code == 423
        state.is_locked = real_is_locked
        assert root.is_dir(), "a dataset busy under any of its keys must survive"
    finally:
        datasets_module._app_state = original_state


def test_a_source_that_vanishes_mid_copy_reports_one_readable_sentence(client, monkeypatch):
    """``shutil`` reports per-file, and the GUI shows the message in an alert().

    ``copytree`` lists the source once and then copies each entry, so a source
    deleted or renamed part-way fails on every remaining file and raises a
    ``shutil.Error`` carrying one tuple each. Measured on a 4000-file tree:
    14 KB of text when deleted, 228 KB when renamed. Passing that through as the
    detail put a quarter-megabyte string in front of the user.
    """
    import shutil as _shutil

    c, root = client
    failures = [
        (f"{root}/data/chunk-000/file-{i:03d}.parquet", "/dst", "[Errno 2] No such file or directory")
        for i in range(3628)
    ]

    def vanish(src, dst):
        Path(dst).mkdir(parents=True, exist_ok=True)
        _shutil.rmtree(root)  # the source goes away mid-copy
        raise _shutil.Error(failures)

    monkeypatch.setattr(_shutil, "copytree", vanish)
    r = _dup(c, root, "doomed")

    assert r.status_code == 500, r.text
    detail = r.json()["detail"]
    assert len(detail) < 300, f"message is {len(detail)} chars; it is shown in an alert()"
    assert "moved or deleted" in detail, f"must say what happened: {detail!r}"
    assert "3628" in detail, f"must say how widespread it was: {detail!r}"
    assert "file-000.parquet" in detail, f"must give one example: {detail!r}"
    assert not (root.parent / "doomed").exists(), "no partial copy may survive"


def test_a_failure_with_the_source_intact_does_not_blame_a_vanished_source(client, monkeypatch):
    """A disk or permission failure is a different problem from a moved source."""
    import shutil as _shutil

    c, root = client

    def fail(src, dst):
        Path(dst).mkdir(parents=True, exist_ok=True)
        raise _shutil.Error([(f"{root}/x.parquet", "/dst", "[Errno 28] No space left on device")])

    monkeypatch.setattr(_shutil, "copytree", fail)
    detail = _dup(c, root, "nospace").json()["detail"]
    assert "moved or deleted" not in detail, f"the source is still there: {detail!r}"
    assert "1 file(s) could not be copied" in detail
    assert "No space left" in detail
    assert root.is_dir()


# ── Error-path coverage ─────────────────────────────────────────────────────
#
# Both operations destroy something, so what they do when they cannot is the
# behaviour that matters. The enumeration below is the coverage claim: every
# failure either operation can raise, the status it maps to, and the fact the
# message must carry. A path missing from here is a path nobody checked.


def test_duplicate_cleans_a_staging_dir_left_by_an_earlier_crash(client):
    """A server killed mid-copy leaves `.copying-<name>` behind.

    It is invisible to the tree (dot-prefixed), so the only way a user meets it
    is a later copy to the same name failing for no visible reason.
    """
    c, root = client
    stale = root.parent / ".copying-retry_me"
    (stale / "junk").mkdir(parents=True)
    (stale / "junk" / "leftover.bin").write_bytes(b"partial")

    r = _dup(c, root, "retry_me")
    assert r.status_code == 200, r.text
    assert (root.parent / "retry_me" / "meta" / "info.json").is_file()
    assert not stale.exists(), "the stale staging dir must not survive"
    assert not (root.parent / "retry_me" / "junk").exists(), "nor leak into the copy"


def test_duplicate_refuses_when_a_plain_file_occupies_the_destination(client):
    """`exists()` covers files, not just directories — a copy must not clobber one."""
    c, root = client
    (root.parent / "occupied").write_text("not a dataset, but not ours to remove")
    r = _dup(c, root, "occupied")
    assert r.status_code == 409, r.text
    assert (root.parent / "occupied").read_text().startswith("not a dataset")


def test_a_delete_that_cannot_rename_destroys_nothing(client, monkeypatch):
    """If the dataset cannot even be moved aside, nothing has been removed."""
    c, root = client
    assert c.post("/api/datasets", json={"local_path": str(root)}).status_code == 200

    def denied(self, target):
        raise PermissionError(f"[Errno 13] Permission denied: '{target}'")

    monkeypatch.setattr(Path, "rename", denied)
    r = _delete(c, root)

    assert r.status_code == 500, r.text
    detail = r.json()["detail"]
    assert "Permission denied" in detail, f"must carry the cause: {detail!r}"
    assert "Nothing was removed" in detail, f"must say the data is safe: {detail!r}"
    assert "reopen" in detail.lower(), f"must say what to do next: {detail!r}"
    assert (root / "meta" / "info.json").is_file(), "the dataset is untouched"


def test_a_half_removed_dataset_is_never_visible_to_the_tree(client, monkeypatch):
    """`rmtree` deletes bottom-up, so a failure part-way leaves a directory that
    is neither gone nor whole. Renaming aside first means the tree loses the
    dataset atomically and the remnant is dot-prefixed, which `_scan_recursive`
    skips — so the half state cannot be opened, listed, or trained on.
    """
    import shutil as _shutil

    c, root = client
    real_rmtree = _shutil.rmtree  # captured before patching, or `half` recurses

    def half(p, *args, **kwargs):
        # Fail only on the removal under test; other call sites (the pre-clean,
        # the sweep) must behave normally or the test is measuring its own stub.
        if not Path(p).name.startswith(".deleting-"):
            return real_rmtree(p, *args, **kwargs)
        real_rmtree(Path(p) / "data")  # gut it, then fail
        raise OSError("[Errno 5] Input/output error")

    monkeypatch.setattr(_shutil, "rmtree", half)
    r = _delete(c, root)

    assert r.status_code == 500, r.text
    detail = r.json()["detail"]
    assert "could not all be deleted" in detail, f"must admit files remain: {detail!r}"
    assert ".deleting-" in detail, f"must name where they are: {detail!r}"
    assert "reclaims" in detail, f"must say what becomes of it: {detail!r}"

    assert not root.exists(), "the dataset path itself must be gone"
    remnant = root.parent / f".deleting-{root.name}"
    assert remnant.is_dir(), "the remainder is kept, not silently lost"
    assert remnant.name.startswith("."), "and is invisible to the source scanner"

    # The decisive property: the scanner does not list it, so it cannot be
    # opened or trained on — and the same pass reclaims the space it holds.
    # Failure injection is undone first: the sweep is a separate operation, and
    # leaving the stub in place would be measuring the stub.
    monkeypatch.undo()
    from lerobot.gui.api.datasets import _scan_source

    assert [d["root"] for d in _scan_source(str(root.parent.parent))] == [], (
        "a half-removed dataset must not list as a real one"
    )
    assert not remnant.exists(), "and the scan reclaims it, as the message promises"


def test_a_remnant_from_an_earlier_failed_delete_does_not_block_a_retry(client):
    """The user's next move after a failed delete is to try again."""
    c, root = client
    stale = root.parent / f".deleting-{root.name}"
    (stale / "meta").mkdir(parents=True)
    (stale / "meta" / "info.json").write_text("{}")

    assert _delete(c, root).status_code == 200
    assert not root.exists()
    assert not stale.exists(), "the earlier remnant is cleared too"


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("not_a_dataset", 404),
        ("bad_name", 400),
        ("destination_taken", 409),
        ("busy", 423),
        ("copy_failed", 500),
    ],
)
def test_every_duplicate_failure_maps_to_its_status(client, monkeypatch, case, expected_status):
    """One row per way duplicate can fail. Nothing may fall through to a bare 500."""
    import shutil as _shutil

    c, root = client
    if case == "not_a_dataset":
        r = _dup(c, root.parent, "x")
    elif case == "bad_name":
        r = _dup(c, root, "a/b")
    elif case == "destination_taken":
        _dup(c, root, "taken2")
        r = _dup(c, root, "taken2")
    elif case == "busy":
        real = datasets_module._app_state.is_locked
        datasets_module._app_state.is_locked = lambda k: True
        try:
            r = _dup(c, root, "while_busy2")
        finally:
            datasets_module._app_state.is_locked = real
    else:

        def fail(src, dst):
            raise _shutil.Error([(f"{root}/a", "/d", "[Errno 28] No space left on device")])

        monkeypatch.setattr(_shutil, "copytree", fail)
        r = _dup(c, root, "nospace2")

    assert r.status_code == expected_status, f"{case}: {r.status_code} {r.text}"
    assert isinstance(r.json()["detail"], str) and r.json()["detail"], f"{case}: empty detail"


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("not_a_dataset", 404),
        ("source_folder", 404),
        ("busy", 423),
        ("remove_failed", 500),
    ],
)
def test_every_delete_failure_maps_to_its_status(client, monkeypatch, tmp_path, case, expected_status):
    """One row per way delete can fail."""
    import shutil as _shutil

    c, root = client
    if case == "not_a_dataset":
        r = _delete(c, tmp_path / "nowhere")
    elif case == "source_folder":
        r = _delete(c, root.parent)
    elif case == "busy":
        real = datasets_module._app_state.is_locked
        datasets_module._app_state.is_locked = lambda k: True
        try:
            r = _delete(c, root)
        finally:
            datasets_module._app_state.is_locked = real
    else:
        monkeypatch.setattr(
            _shutil, "rmtree", lambda p: (_ for _ in ()).throw(OSError("[Errno 30] Read-only file system"))
        )
        r = _delete(c, root)

    assert r.status_code == expected_status, f"{case}: {r.status_code} {r.text}"
    assert isinstance(r.json()["detail"], str) and r.json()["detail"], f"{case}: empty detail"
    if case != "remove_failed":
        assert root.is_dir(), f"{case}: a refused delete must not touch the files"


def test_scanning_reclaims_remnants_of_interrupted_operations(client):
    """Invisible must not mean forgotten.

    A killed server leaves `.copying-*` or `.deleting-*` behind. The tree
    ignores them by design, which is also why nobody would ever notice the disk
    they hold — so a scan, which already walks this tree, clears them.
    """
    from lerobot.gui.api.datasets import _scan_source

    c, root = client
    source = root.parent.parent
    copying = root.parent / ".copying-abandoned"
    deleting = root.parent / ".deleting-abandoned"
    for d in (copying, deleting):
        (d / "data").mkdir(parents=True)
        (d / "data" / "big.bin").write_bytes(b"x" * 4096)

    listed = [d["root"] for d in _scan_source(str(source))]

    assert str(root) in listed, "the real dataset still lists"
    assert not copying.exists(), "an abandoned copy is reclaimed"
    assert not deleting.exists(), "an abandoned delete is reclaimed"
    assert (root / "meta" / "info.json").is_file(), "and nothing real is touched"
