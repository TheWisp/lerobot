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
"""What a policy is actually fed, said once per run.

A trainer logs the config it was ASKED for. The contract it ends up with is
resolved afterwards -- for ``lerobot-train``, by prefix from the dataset
schema -- so the two can differ and only the checkpoint records which won.
Answering "did this model train on what I think it did" then means finding the
run's checkpoint and reading ``train_config.json``.

The reporting lives here rather than at the three call sites so that adding it
to a trainer costs one line, and so every trainer says it the same way. The
alternative, which this replaced, was the formatting expression written out
twice in ``make_policy`` and a third time, differently, in the HVLA trainer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

DEFAULT_LOGGER = logging.getLogger(__name__)


class _Feature(Protocol):
    """The shape of a ``PolicyFeature``: enough to describe one entry."""

    type: Any
    shape: tuple[int, ...]


def describe(features: Mapping[str, _Feature]) -> str:
    """``"name [TYPE (shape)], ..."`` in name order, or ``"none"``.

    Pre: every value exposes ``type`` and ``shape``. Post: sorted by name, so
    two runs of the same contract produce byte-identical lines and a diff of
    two logs shows a real change rather than dict ordering.
    """
    parts = []
    for name, feature in sorted(features.items()):
        kind = getattr(feature.type, "name", feature.type)
        parts.append(f"{name} [{kind} {tuple(feature.shape)}]")
    return ", ".join(parts) or "none"


def log_contract(
    inputs: Mapping[str, _Feature],
    outputs: Mapping[str, _Feature],
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Log the resolved input and output contract, one line each."""
    log = logger or DEFAULT_LOGGER
    log.info("Policy input contract: %s", describe(inputs))
    log.info("Policy output contract: %s", describe(outputs))


def report_undelivered(
    declared: Iterable[str],
    batch: Mapping[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Warn about declared inputs the batch never carries; return their names.

    A declared feature that no batch delivers is the signature of a column
    classified as an input by its NAME rather than by intent: the run trains
    normally without it, and the mismatch is visible only in the checkpoint
    afterwards. Returned as well as logged so a caller can assert on it.

    Only tensor-valued keys count as delivered -- a batch carries task strings
    and index bookkeeping too, and neither reaches the model as an input.
    """
    import torch

    log = logger or DEFAULT_LOGGER
    declared = set(declared)
    delivered = {k for k, v in batch.items() if isinstance(v, torch.Tensor)}

    missing = sorted(declared - delivered)
    if missing:
        log.warning(
            "Declared policy inputs absent from the batch: %s. They were resolved from the "
            "dataset schema but nothing delivers them; the model trains without them.",
            ", ".join(missing),
        )
    log.info(
        "First batch delivers: %s",
        ", ".join(f"{k} {tuple(batch[k].shape)}" for k in sorted(declared & delivered)) or "none",
    )
    return missing
