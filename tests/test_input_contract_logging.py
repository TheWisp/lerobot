# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A run must say what goes into the model.

The input contract is resolved by prefix from the dataset schema: any
non-image ``observation.*`` feature becomes a policy input whether or not
anyone meant it to. That is not hypothetical -- an annotation column named
``observation.masks.<camera>`` is classified as STATE, declared as an input,
and then dropped by the reader before collation, so the model trains without it
while the checkpoint records it as an input.

Nothing about that is visible from a run's log today, which is what these tests
fix: the contract is logged where it is resolved, and the first batch is
compared against it. The second is the one that catches a wrongly-classified
column, because the signature is "declared but never delivered".
"""

import logging

from lerobot.policies.input_contract import describe, log_contract, report_undelivered
from lerobot.utils.feature_utils import dataset_to_policy_features


def _image(h=64, w=64):
    return {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"]}


BASE = {
    "observation.state": {"dtype": "float32", "shape": [6], "names": None},
    "observation.images.top": _image(),
    "action": {"dtype": "float32", "shape": [6], "names": None},
}


def test_an_annotation_column_named_like_an_observation_becomes_a_policy_input():
    """The hazard itself, pinned so a rename cannot silently reintroduce it.

    Nothing here asserts the naming is wrong -- it asserts what the classifier
    does with it, so that moving such a column out of ``observation.*`` shows up
    as this test changing rather than as a surprise in a checkpoint.
    """
    plain = dataset_to_policy_features(BASE)
    assert "observation.masks.top" not in plain

    with_masks = dataset_to_policy_features(
        {
            **BASE,
            "observation.masks.top": {
                "dtype": "string",
                "shape": [1],
                "names": None,
                "mask_encoding": "coco_rle",
            },
        }
    )
    assert "observation.masks.top" in with_masks, (
        "a non-image observation.* feature is classified as policy STATE by prefix; "
        "if this no longer holds, the classifier changed and the rename rationale with it"
    )
    assert with_masks["observation.masks.top"].type.name == "STATE"

    # A column outside observation.* is skipped -- the property the masks.* name relies on.
    outside = dataset_to_policy_features(
        {**BASE, "masks.top": {"dtype": "string", "shape": [1], "names": None}}
    )
    assert "masks.top" not in outside


def _batch(*keys, extras=True):
    """A batch shaped like the trainer's: tensors, plus the non-tensor keys a
    real batch also carries and which never reach the model as inputs."""
    import torch

    batch = {k: torch.zeros(2, 3) for k in keys}
    if extras:
        batch["task"] = ["pick up the cube", "pick up the cube"]
        batch["index"] = torch.arange(2)
    return batch


def test_declared_but_undelivered_inputs_are_reported(caplog):
    """The check the trainer runs on its first batch, through the function the
    trainer actually calls.

    A declared feature the batch never carries means the model is training on
    less than its config claims. The warning names the keys, because the run is
    otherwise indistinguishable from a healthy one.
    """
    declared = {"observation.state", "observation.images.top", "observation.masks.top"}

    with caplog.at_level(logging.WARNING):
        missing = report_undelivered(declared, _batch("observation.state", "observation.images.top"))

    assert missing == ["observation.masks.top"]
    assert "observation.masks.top" in caplog.text
    assert "absent from the batch" in caplog.text


def test_a_healthy_contract_reports_nothing(caplog):
    """The complement: no warning when every declared input arrives.

    Without this, the check above would pass equally well for an implementation
    that warns unconditionally.
    """
    declared = {"observation.state", "observation.images.top"}

    with caplog.at_level(logging.WARNING):
        missing = report_undelivered(declared, _batch(*declared))

    assert missing == []
    assert caplog.text == ""


def test_a_non_tensor_value_does_not_count_as_delivered(caplog):
    """`task` is a list of strings and `index` is bookkeeping. A declared input
    that arrives as neither a tensor nor at all is still undelivered -- counting
    any present key would hide exactly the case this exists to surface."""
    with caplog.at_level(logging.WARNING):
        missing = report_undelivered({"task"}, _batch("observation.state"))

    assert missing == ["task"]


def test_the_contract_lines_are_stable_and_sorted(caplog):
    """Two runs of the same contract must produce identical lines, or a diff of
    two logs shows dict ordering rather than a real change."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    feats = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    assert describe(feats) == (
        "observation.images.top [VISUAL (3, 224, 224)], observation.state [STATE (6,)]"
    )
    assert describe({}) == "none"

    with caplog.at_level(logging.INFO):
        log_contract(feats, {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))})
    assert "Policy input contract: observation.images.top [VISUAL" in caplog.text
    assert "Policy output contract: action [ACTION (6,)]" in caplog.text
