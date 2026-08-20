# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Contracts where the fork deliberately extends an upstream type.

These are the seams an upstream sync re-litigates every time: the fork adds a
field or a subclass, upstream reorganises the base, and the extension is
quietly lost or bypassed. Each test here states the divergence explicitly so a
merge that flattens it fails with an explanation rather than an AttributeError
somewhere downstream.
"""

from __future__ import annotations

import dataclasses

import pytest


def test_record_config_keeps_fork_only_fields():
    """``lerobot-record``'s dataset config must stay a superset of upstream's.

    The fork subclasses upstream's ``DatasetRecordConfig`` to add fields that
    ``record()`` reads unconditionally. A merge that flattens the subclass back
    to upstream's would leave ``record()`` failing partway through dataset
    creation on a missing attribute, naming a class that does have it in the
    module a reader would check first.

    Pre: ``lerobot.scripts.lerobot_record`` is importable. The subclass check
    skips before the 2026-07 upstream sync, which is what introduces
    ``lerobot.configs.dataset`` — so the contract arms itself once that merge
    lands rather than needing to be switched on by hand.
    Post: the fork-only fields are present, and the base is still upstream's so
    future upstream field additions arrive by inheritance rather than needing a
    manual copy.
    """
    from lerobot.scripts.lerobot_record import DatasetRecordConfig as ForkConfig, RecordConfig

    upstream = pytest.importorskip(
        "lerobot.configs.dataset",
        reason="pre-sync tree still defines DatasetRecordConfig in lerobot_record",
    )

    assert issubclass(ForkConfig, upstream.DatasetRecordConfig), (
        "lerobot_record.DatasetRecordConfig must subclass "
        "lerobot.configs.dataset.DatasetRecordConfig so upstream field additions "
        "are inherited, not forked."
    )

    fields = {f.name for f in dataclasses.fields(ForkConfig)}
    for fork_only in ("record_images", "rename_map"):
        assert fork_only in fields, (
            f"fork-only field '{fork_only}' is gone from DatasetRecordConfig, but "
            f"record() still reads it — this is a merge regression, not a cleanup."
        )

    annotation = RecordConfig.__dataclass_fields__["dataset"].type
    # Under postponed evaluation the annotation may still be a string; accept
    # either spelling of the same contract.
    assert annotation in (ForkConfig, "DatasetRecordConfig"), (
        f"RecordConfig.dataset must be annotated with the fork's DatasetRecordConfig, got {annotation!r}"
    )


# NOTE: a per-symbol guard for load_subtasks / load_info / write_info used to
# live here. It is not reinstated: test_import_integrity covers all three.
# load_subtasks is imported absolutely from inside three test helpers, and
# load_info / write_info relatively at module scope in dataset_metadata — all
# four import styles the walker now inspects. A hand-maintained list of "the
# symbols that broke last time" does not generalise to the next sync.
