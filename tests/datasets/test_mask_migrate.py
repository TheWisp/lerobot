# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Datasets written before the namespace moved must not read as unmasked.

Masks were first stored as `observation.masks.<camera>`. After the rename the
reader looks under `masks.`, finds nothing, and composites nothing -- so a
dataset that was deliberately masked would train on raw pixels, the run would
look entirely healthy, and only the pixels would be wrong. That is the failure
`--ignore-saved-masks` exists to make deliberate, so it must not be reachable
by accident.

Two halves: asking for saved masks on such a dataset raises, and the migration
renames it so that it does not.
"""

import json

import pandas as pd
import pytest

from lerobot.datasets.mask_compositing import (
    LEGACY_MASK_NAMESPACE,
    MASK_NAMESPACE,
)
from lerobot.datasets.mask_migrate import migrate_root, plan

OLD = f"{LEGACY_MASK_NAMESPACE}.top"
NEW = f"{MASK_NAMESPACE}.top"


def _spec() -> dict:
    return {
        "dtype": "string",
        "shape": [1],
        "names": None,
        "mask_encoding": "coco_rle",
        "mask_labels": ["tray"],
        "mask_size": [8, 12],
    }


@pytest.fixture
def legacy_root(tmp_path):
    """A dataset on the pre-rename namespace: metadata and parquet agree."""
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": {
                    "action": {"dtype": "float32", "shape": [6], "names": None},
                    OLD: _spec(),
                },
            }
        )
    )
    shard = tmp_path / "data" / "chunk-000"
    shard.mkdir(parents=True)
    pd.DataFrame({"episode_index": [0, 0], OLD: ['[[0,"a"]]', '[[0,"b"]]']}).to_parquet(
        shard / "file-000.parquet"
    )
    return tmp_path


def test_the_migration_moves_metadata_and_data_together(legacy_root):
    """A dataset whose info.json names columns its parquet lacks will not open,
    so neither half may move without the other."""
    migrate_root(legacy_root)
    info = json.loads((legacy_root / "meta" / "info.json").read_text())
    assert NEW in info["features"] and OLD not in info["features"]
    assert info["features"][NEW]["mask_labels"] == ["tray"], "the spec must survive the rename"

    df = pd.read_parquet(next(legacy_root.rglob("data/**/*.parquet")))
    assert NEW in df.columns and OLD not in df.columns
    assert df[NEW].tolist() == ['[[0,"a"]]', '[[0,"b"]]'], "rows must be untouched"


def test_migrating_twice_is_a_no_op(legacy_root):
    """Safe to run over a tree of datasets without tracking which are done."""
    assert migrate_root(legacy_root) == {OLD: NEW}
    assert migrate_root(legacy_root) == {}
    assert plan(legacy_root) == {}


def test_a_dry_run_changes_nothing(legacy_root):
    assert migrate_root(legacy_root, dry_run=True) == {OLD: NEW}
    info = json.loads((legacy_root / "meta" / "info.json").read_text())
    assert OLD in info["features"], "dry run must not write"


def _write_shard(root, name, columns):
    shard = root / "data" / "chunk-000"
    shard.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_parquet(shard / name)


def test_a_half_migrated_dataset_is_refused(legacy_root):
    """Both names present would produce two columns of one name in one file --
    which pyarrow refuses to read back ("Can't unify schema with duplicate
    field names") and whose info.json keeps only one of the two specs. Renaming
    it silently corrupted the dataset before this check."""
    info_path = legacy_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][NEW] = _spec()
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match="already has"):
        plan(legacy_root)
    with pytest.raises(ValueError, match="already has"):
        migrate_root(legacy_root)


def test_every_shard_is_migrated(legacy_root):
    """A dataset is many parquet files; renaming only the first leaves the rest
    unreadable against the new metadata."""
    _write_shard(legacy_root, "file-001.parquet", {"episode_index": [1, 1], OLD: ["c", "d"]})
    _write_shard(legacy_root, "file-002.parquet", {"episode_index": [2], OLD: ["e"]})
    migrate_root(legacy_root)
    shards = sorted(legacy_root.rglob("data/**/*.parquet"))
    assert len(shards) == 3
    for s in shards:
        cols = pd.read_parquet(s).columns
        assert NEW in cols and OLD not in cols, f"{s.name} was not migrated"


def test_a_shard_without_the_column_is_left_alone(legacy_root):
    """Not every file need carry every column; one that does not must not be
    rewritten, and must not stop the others."""
    _write_shard(legacy_root, "file-001.parquet", {"episode_index": [1], "action": [0.0]})
    migrate_root(legacy_root)
    other = legacy_root / "data" / "chunk-000" / "file-001.parquet"
    assert list(pd.read_parquet(other).columns) == ["episode_index", "action"]


def test_an_interrupted_migration_is_completed_by_re_running(legacy_root):
    """A crash between shards leaves some renamed and some not, with info.json
    still on the old name. That state must be recoverable by running again
    rather than by hand-editing parquet."""
    _write_shard(legacy_root, "file-001.parquet", {"episode_index": [1], OLD: ["c"]})
    # Simulate the interruption: rename one shard's column, leave metadata.
    first = legacy_root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(first).rename(columns={OLD: NEW})
    df.to_parquet(first)
    assert plan(legacy_root) == {OLD: NEW}, "metadata still names the old column"

    migrate_root(legacy_root)
    for s in sorted(legacy_root.rglob("data/**/*.parquet")):
        assert NEW in pd.read_parquet(s).columns, f"{s.name} left behind"
    info = json.loads((legacy_root / "meta" / "info.json").read_text())
    assert NEW in info["features"] and OLD not in info["features"]


def test_several_mask_columns_migrate_together(legacy_root):
    """A multi-camera dataset has one mask column per camera; a migration that
    handled only the first would leave the dataset half-renamed."""
    second_old = f"{LEGACY_MASK_NAMESPACE}.wrist"
    second_new = f"{MASK_NAMESPACE}.wrist"
    info_path = legacy_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][second_old] = _spec()
    info_path.write_text(json.dumps(info))
    first = legacy_root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(first)
    df[second_old] = ["x", "y"]
    df.to_parquet(first)

    assert migrate_root(legacy_root) == {OLD: NEW, second_old: second_new}
    cols = pd.read_parquet(first).columns
    assert NEW in cols and second_new in cols
    assert OLD not in cols and second_old not in cols


def test_a_dataset_with_no_mask_columns_is_untouched(tmp_path):
    """The migration runs over whole caches; a dataset that never had masks
    must not be rewritten at all."""
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps({"features": {"action": {"dtype": "float32", "shape": [6], "names": None}}})
    )
    shard = tmp_path / "data" / "chunk-000"
    shard.mkdir(parents=True)
    pd.DataFrame({"episode_index": [0], "action": [0.0]}).to_parquet(shard / "f.parquet")
    before = (shard / "f.parquet").read_bytes()

    assert plan(tmp_path) == {}
    assert migrate_root(tmp_path) == {}
    assert (shard / "f.parquet").read_bytes() == before, "an unmasked dataset was rewritten"
