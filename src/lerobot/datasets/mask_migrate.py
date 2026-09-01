# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Move a dataset's mask columns to the current namespace.

Masks were first written as ``observation.masks.<camera>``. That name puts
them inside the prefix ``dataset_to_policy_features`` reads as policy STATE,
so the column was declared a model input and then dropped by the reader --
see ``MASK_NAMESPACE`` in ``mask_compositing``. They are now ``masks.<camera>``.

This renames an existing dataset in place: the parquet column and the
``meta/info.json`` entry, which must move together or the dataset describes
columns it does not have.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from lerobot.datasets.mask_compositing import (
    LEGACY_MASK_NAMESPACE,
    MASK_NAMESPACE,
    legacy_mask_columns,
)

logger = logging.getLogger(__name__)


def _renamed(key: str) -> str:
    return f"{MASK_NAMESPACE}.{key[len(LEGACY_MASK_NAMESPACE) + 1 :]}"


def plan(root: Path | str) -> dict[str, str]:
    """The renames this dataset needs, as ``{old_key: new_key}``.

    Pre: ``root`` holds ``meta/info.json``.
    Post: empty when the dataset is already on the current namespace, so
    callers can use this to test whether migration is needed.
    """
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    features = info.get("features", {})
    renames = {k: _renamed(k) for k in legacy_mask_columns(features)}
    # A dataset holding both names for one camera cannot be renamed: the result
    # is a parquet with two columns of the same name, which pyarrow refuses to
    # read back ("Can't unify schema with duplicate field names") and whose
    # info.json silently keeps only one of the two specs. Refuse instead of
    # producing a file that cannot be opened.
    clashing = {old: new for old, new in renames.items() if new in features}
    if clashing:
        raise ValueError(
            f"{Path(root)} already has {sorted(clashing.values())} alongside "
            f"{sorted(clashing)}; renaming would put two columns of the same name in one "
            "file. Decide which is current and drop the other before migrating."
        )
    return renames


def migrate_root(root: Path | str, *, dry_run: bool = False) -> dict[str, str]:
    """Rename this dataset's mask columns in place.

    Pre: ``root`` is a v3 LeRobotDataset directory, not open for writing.
    Post: every ``observation.masks.*`` column and its ``info.json`` entry is
    renamed to ``masks.*``; returns what was renamed. A no-op, returning ``{}``,
    when there is nothing to move -- so it is safe to run twice.

    ``info.json`` is written only after every parquet file has been rewritten,
    because a dataset whose metadata names columns its files do not have will
    not open at all.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    renames = plan(root)
    if not renames:
        return {}
    if dry_run:
        return renames

    files = sorted(root.rglob("data/**/*.parquet"))
    for f in files:
        table = pq.read_table(f)
        names = [renames.get(n, n) for n in table.column_names]
        if names == table.column_names:
            continue
        tmp = f.with_suffix(f.suffix + ".tmp")
        pq.write_table(table.rename_columns(names), tmp)
        # The original is only unlinked once the replacement is fully
        # written, and the rename is within one filesystem.
        # safe-destruct: same table, renamed column, written before the swap
        shutil.move(str(tmp), str(f))

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"] = {renames.get(k, k): v for k, v in info["features"].items()}
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=4))
    # Written last: metadata that names columns the parquet lacks is the one
    # state in which the dataset will not open at all.
    # safe-destruct: info.json swapped only after every parquet carries the new column
    shutil.move(str(tmp), str(info_path))

    logger.info("migrated %d mask column(s) in %s: %s", len(renames), root, renames)
    return renames
