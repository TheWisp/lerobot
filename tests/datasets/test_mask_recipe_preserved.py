# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What a segmentation save may and may not change about the stored recipe.

A save is about ROWS. The recipe -- which effect each label renders with, and
what happens to the background -- is dataset-wide, edited in the Inspector, and
a segmentation pass has no opinion about it. But the panel that starts a save
still sends a treatment per object and a background, defaulting to `none`,
because those controls moved to the Inspector and the payload did not follow.

So every value the caller supplies here is a DEFAULT, not an intent, and
letting it win silently resets the recipe on the next save of any episode.
Reported as "the background treatment (I set it to blur) is overwritten AGAIN
by the per-frame lock step sweep" -- it was the save afterwards, not the sweep.

Per-label treatments were already guarded. The background was not, and the
asymmetry is what these tests exist to hold: the same rule for both, plus the
complement, so "stored always wins" cannot degenerate into "never writes".
"""

import json

import pytest

from lerobot.datasets.dataset_postprocess import generate_episode_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.datasets.test_mask_prompts import _PromptEchoAdapter
from tests.datasets.test_saved_masks_training import masked_dataset_root  # noqa: F401

BLUR = {"key": "blur", "params": {}}
TINT = {"key": "tint", "params": {"color": [255, 0, 0]}}
NONE = {"key": "none", "params": {}}


def _mask_specs(root) -> dict:
    """Every mask column's spec, keyed by column name."""
    info = json.loads((root / "meta" / "info.json").read_text())
    return {k: v for k, v in info["features"].items() if v.get("mask_encoding") == "coco_rle"}


def _write_recipe(root, *, treatments=None, background=None, drop_background=False) -> None:
    """Put a recipe on disk the way the Inspector's edit path leaves one.

    ``drop_background`` removes the key entirely, which is the only state that
    means "nothing has ever set one" -- a stored ``none`` is a stored value.
    """
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    for spec in info["features"].values():
        if spec.get("mask_encoding") != "coco_rle":
            continue
        if treatments is not None:
            spec["mask_treatments"] = treatments
        if drop_background:
            spec.pop("mask_background", None)
        elif background is not None:
            spec["mask_background"] = background
    path.write_text(json.dumps(info, indent=4))


def _save(root, repo_id, names, *, background=NONE):
    """Run the real save path, as the panel calls it: effects defaulted to none."""
    ds = LeRobotDataset(repo_id, root=root)
    objects = [{"name": n, "sign": "+", "treatment": dict(NONE)} for n in names]
    return generate_episode_masks(
        ds,
        episode=0,
        objects=objects,
        cameras=None,
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment=dict(background),
        adopt=False,
        device="cpu",
        adapter=_PromptEchoAdapter(),
    )


# ── the reported defect ─────────────────────────────────────────────────────


def test_a_save_does_not_reset_the_stored_background(masked_dataset_root):  # noqa: F811
    """The bug: blur set in the Inspector, gone after segmenting again."""
    root, repo_id = masked_dataset_root
    _write_recipe(root, background=BLUR)

    _save(root, repo_id, ["tray", "ball"], background=NONE)

    for key, spec in _mask_specs(root).items():
        assert spec["mask_background"] == BLUR, f"{key} lost the stored background: {spec['mask_background']}"


def test_a_save_does_not_reset_stored_per_label_treatments(masked_dataset_root):  # noqa: F811
    """Guarded before the background was; pinned so the pair cannot drift."""
    root, repo_id = masked_dataset_root
    _write_recipe(root, treatments={"tray": TINT})

    _save(root, repo_id, ["tray", "ball"])

    for key, spec in _mask_specs(root).items():
        assert spec["mask_treatments"].get("tray") == TINT, f"{key} lost the tray's treatment"


# ── the complement: "stored wins" must not become "never writes" ────────────


def test_the_first_save_records_the_background_it_is_given(masked_dataset_root):  # noqa: F811
    """The complement. Without it the test above passes on a save that ignores
    the argument entirely, and no background could ever be set at all.

    "Nothing stored" means the key is absent -- a stored ``none`` is a value
    somebody chose, and a later save carrying a defaulted ``blur`` must not
    overwrite that either.
    """
    root, repo_id = masked_dataset_root
    _write_recipe(root, drop_background=True)

    _save(root, repo_id, ["tray", "ball"], background=BLUR)

    for key, spec in _mask_specs(root).items():
        assert spec["mask_background"] == BLUR, f"{key} ignored the supplied background"


def test_a_stored_none_is_a_choice_and_is_kept(masked_dataset_root):  # noqa: F811
    """Turning every effect off is a decision, and the next save must not undo
    it by re-stamping whatever the panel happens to carry."""
    root, repo_id = masked_dataset_root
    _write_recipe(root, background=NONE)

    _save(root, repo_id, ["tray", "ball"], background=BLUR)

    for key, spec in _mask_specs(root).items():
        assert spec["mask_background"] == NONE, f"{key}: a deliberate 'none' was overwritten"


def test_a_new_object_arrives_untreated_without_disturbing_the_others(masked_dataset_root):  # noqa: F811
    """Adding an object is the common reason to segment again."""
    root, repo_id = masked_dataset_root
    _write_recipe(root, treatments={"tray": TINT}, background=BLUR)

    _save(root, repo_id, ["tray", "ball", "cube"])

    for key, spec in _mask_specs(root).items():
        tr = spec["mask_treatments"]
        assert tr.get("tray") == TINT, f"{key}: existing treatment changed"
        assert tr.get("cube", NONE)["key"] == "none", f"{key}: new object arrived treated"
        assert spec["mask_background"] == BLUR, f"{key}: background changed"


def test_every_camera_ends_with_the_same_recipe(masked_dataset_root):  # noqa: F811
    """The recipe describes the OBJECT, not the view: blurring the arm in one
    camera and tinting it in another describes nothing a model could learn."""
    root, repo_id = masked_dataset_root
    _write_recipe(root, treatments={"tray": TINT}, background=BLUR)

    _save(root, repo_id, ["tray", "ball"])

    specs = list(_mask_specs(root).values())
    assert len(specs) >= 2, "fixture needs two mask columns to compare"
    assert len({json.dumps(s["mask_treatments"], sort_keys=True) for s in specs}) == 1, "treatments diverged"
    assert len({json.dumps(s["mask_background"], sort_keys=True) for s in specs}) == 1, "backgrounds diverged"


@pytest.mark.parametrize("background", [BLUR, TINT])
def test_the_stored_background_survives_repeated_saves(masked_dataset_root, background):  # noqa: F811
    """Once is luck; the reported symptom was that it kept happening."""
    root, repo_id = masked_dataset_root
    _write_recipe(root, background=background)

    for _ in range(3):
        _save(root, repo_id, ["tray", "ball"], background=NONE)

    for key, spec in _mask_specs(root).items():
        assert spec["mask_background"] == background, f"{key} drifted after repeated saves"
