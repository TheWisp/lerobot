# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the segmenter is asked for is not what the dataset stores.

An object's stored label is fixed for the life of the column: rows reference it
by position, so it can only ever be appended to. Its prompt is the text handed
to the model, and sharpening one is the normal repair -- "the yellow ball on
the desk" when plain "yellow ball" also finds a spare on the shelf.

While the two were the same string, sharpening a prompt meant renaming a label,
which the writer normalises into an append: the operator asks for a better
prompt and gets a second object with rows only in the episodes they re-ran.
Separating them makes the repair a metadata change and leaves the vocabulary
alone.

The invariant under test: for any prompt, the stored vocabulary is unchanged
and the rows still decode to the same labels.
"""

import json

import numpy as np
import pytest

from lerobot.datasets.dataset_postprocess import generate_episode_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.overlays.adapters import _parse_objects, prompt_of
from tests.datasets.test_saved_masks_training import (  # noqa: F401
    masked_dataset_root,
)

# ── the prompt/name split, at the adapter boundary ──────────────────────────


def test_an_object_without_a_prompt_is_asked_for_by_name():
    """Every object written before prompts existed behaves exactly as it did."""
    assert prompt_of({"name": "ball"}) == "ball"
    assert prompt_of({"name": "ball", "prompt": ""}) == "ball"
    assert prompt_of({"name": "ball", "prompt": "   "}) == "ball"


def test_a_prompt_overrides_the_name_for_the_model_only():
    assert prompt_of({"name": "ball", "prompt": "the yellow ball on the desk"}) == (
        "the yellow ball on the desk"
    )


def test_the_control_carries_prompts_not_labels():
    """The adapter is given text to find, and keys its signs by the same text --
    because that is what it echoes back on the masks."""
    control = {
        "objects": [
            {"name": "ball", "prompt": "the yellow ball on the desk", "sign": "+"},
            {"name": "tray", "sign": "-"},
        ]
    }
    prompts, signs = _parse_objects(control, 8)
    assert prompts == ["the yellow ball on the desk", "tray"]
    assert signs == {"the yellow ball on the desk": "+", "tray": "-"}


def test_a_clicked_object_still_never_joins_the_prompt():
    """Clicked objects are found by position, not by text; a prompt on one must
    not put it back into the query."""
    control = {"objects": [{"name": "blob", "prompt": "some text", "clicked": True}]}
    prompts, signs = _parse_objects(control, 8)
    assert prompts == []
    assert signs == {"some text": "+"}


def test_two_objects_sharing_a_prompt_are_deduped_for_the_model():
    """The model is asked once; the writer still maps the echo back to a label."""
    control = {"objects": [{"name": "a", "prompt": "cup"}, {"name": "b", "prompt": "cup"}]}
    prompts, _ = _parse_objects(control, 8)
    assert prompts == ["cup"]


# ── the same split, through the writer that stores rows ─────────────────────


class _PromptEchoAdapter:
    """Returns a mask per prompt it was given, keyed by the prompt -- which is
    what SAM3 does, and the reason the writer has to translate."""

    def __init__(self):
        self._prompts: list[str] = []

    def set_control(self, control):
        prompts, _ = _parse_objects(control, 8)
        if prompts is not None:
            self._prompts = prompts

    def set_camera(self, cam):
        pass

    def reset(self):
        pass

    def segment(self, rgb):
        h, w = rgb.shape[:2]
        out = {}
        for i, prompt in enumerate(self._prompts):
            m = np.zeros((h, w), np.float32)
            m[:, i * (w // 8) : (i + 1) * (w // 8)] = 1.0
            out[prompt] = m
        return out


def _spec_of(root):
    info = json.loads((root / "meta" / "info.json").read_text())
    return next(v for v in info["features"].values() if v.get("mask_encoding") == "coco_rle")


def _regenerate(root, repo_id, objects):
    ds = LeRobotDataset(repo_id, root=root)
    return generate_episode_masks(
        ds,
        episode=0,
        objects=objects,
        cameras=None,
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment={"key": "none", "params": {}},
        adopt=False,
        device="cpu",
        adapter=_PromptEchoAdapter(),
    )


def test_sharpening_a_prompt_does_not_touch_the_vocabulary(masked_dataset_root):  # noqa: F811
    """The case this exists for. Before the split this added a third label."""
    root, repo_id = masked_dataset_root
    before = list(_spec_of(root)["mask_labels"])
    assert before == ["tray", "ball"], "fixture changed; update the expectations"

    result = _regenerate(
        root,
        repo_id,
        [
            {"name": "tray", "sign": "+", "treatment": {"key": "none"}},
            {
                "name": "ball",
                "prompt": "the yellow ball on the desk",
                "sign": "+",
                "treatment": {"key": "none"},
            },
        ],
    )
    assert not result.get("cancelled")

    spec = _spec_of(root)
    assert spec["mask_labels"] == before, (
        "a sharpened prompt grew the vocabulary; the point of the split is that it does not"
    )
    assert spec["mask_prompts"] == {"ball": "the yellow ball on the desk"}, (
        "the prompt that produced these rows is not recorded, so the run cannot be reproduced"
    )


def test_rows_still_decode_to_the_stored_labels(masked_dataset_root):  # noqa: F811
    """The translation must land the mask under the LABEL. If it stored the
    prompt instead, the row would carry a label id outside the vocabulary."""
    from lerobot.datasets.mask_codec import decode_frame
    from lerobot.datasets.mask_compositing import mask_feature_of

    root, repo_id = masked_dataset_root
    _regenerate(
        root,
        repo_id,
        [
            {"name": "tray", "prompt": "the white tray", "sign": "+", "treatment": {"key": "none"}},
            {"name": "ball", "prompt": "the yellow ball", "sign": "+", "treatment": {"key": "none"}},
        ],
    )
    ds = LeRobotDataset(repo_id, root=root)
    cam = ds.meta.camera_keys[0]
    key = mask_feature_of(cam)
    spec = ds.meta.features[key]
    cell = ds.hf_dataset[0][key]
    raw = cell[0] if isinstance(cell, (list, tuple)) and cell else cell
    decoded = decode_frame(str(raw), spec["mask_labels"], tuple(spec["mask_size"]))
    assert set(decoded) <= set(spec["mask_labels"]), (
        f"decoded {sorted(decoded)} is not a subset of the vocabulary {spec['mask_labels']}; "
        "the writer stored prompts where labels belong"
    )
    assert decoded, "no masks were stored at all, so this asserts nothing about the mapping"


def test_no_prompts_are_recorded_when_none_differ(masked_dataset_root):  # noqa: F811
    """The spec must not grow for datasets that never sharpen a prompt."""
    root, repo_id = masked_dataset_root
    _regenerate(
        root,
        repo_id,
        [
            {"name": "tray", "sign": "+", "treatment": {"key": "none"}},
            {"name": "ball", "sign": "+", "treatment": {"key": "none"}},
        ],
    )
    assert "mask_prompts" not in _spec_of(root)


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_an_empty_prompt_is_not_recorded(masked_dataset_root, prompt):  # noqa: F811
    """An empty prompt means "ask by name", not "the prompt is the empty string"."""
    root, repo_id = masked_dataset_root
    obj = {"name": "ball", "sign": "+", "treatment": {"key": "none"}}
    if prompt is not None:
        obj["prompt"] = prompt
    _regenerate(root, repo_id, [{"name": "tray", "sign": "+", "treatment": {"key": "none"}}, obj])
    assert "mask_prompts" not in _spec_of(root)
