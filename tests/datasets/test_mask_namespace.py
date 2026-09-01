# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Mask columns must not be classified as policy inputs.

`dataset_to_policy_features` types a feature by key prefix: any non-image
feature under `observation.` becomes STATE. A mask column named
`observation.masks.<camera>` was therefore declared a policy input, handed to
the normalizer, and then dropped by the reader before collation -- the model
trained without it while its checkpoint recorded it as an input.

The columns live under their own top-level namespace instead, which reaches
that function's `else: continue`, exactly as `quality.human_flags` does. The
rule is checked here against the classifier itself rather than asserted in
prose, because the hazard is a property of that function and would come back
if it changed.
"""

import pytest

from lerobot.datasets.mask_compositing import (
    MASK_NAMESPACE,
    camera_feature_of,
    mask_feature_of,
)
from lerobot.utils.feature_utils import dataset_to_policy_features

CAMERA = "observation.images.top"


def _spec() -> dict:
    return {"dtype": "string", "shape": [1], "names": None, "mask_encoding": "coco_rle"}


def test_a_mask_column_is_not_a_policy_input():
    """The defect, stated as the classifier sees it."""
    features = {
        CAMERA: {"dtype": "video", "shape": [240, 320, 3], "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": [6], "names": None},
        mask_feature_of(CAMERA): _spec(),
    }
    resolved = dataset_to_policy_features(features)
    assert mask_feature_of(CAMERA) not in resolved, (
        f"{mask_feature_of(CAMERA)} was resolved as a policy input; the reader drops it before "
        "collation, so the model would train without an input its checkpoint claims to have"
    )
    # And the complement: the features that SHOULD be inputs still are, so this
    # cannot pass by resolving nothing at all.
    assert CAMERA in resolved
    assert "observation.state" in resolved


def test_the_old_name_would_have_been_a_policy_input():
    """Why the namespace moved. If this ever stops holding, the rename is moot
    and this file should say so rather than quietly still passing."""
    features = {"observation.masks.top": _spec()}
    assert "observation.masks.top" in dataset_to_policy_features(features)


def test_the_namespace_is_outside_observation():
    assert not MASK_NAMESPACE.startswith("observation")
    assert mask_feature_of(CAMERA) == f"{MASK_NAMESPACE}.top"


@pytest.mark.parametrize(
    "camera",
    ["observation.images.top", "observation.images.left_wrist", "obs.images.cam_0"],
)
def test_the_pair_round_trips(camera):
    """`camera_feature_of` is the inverse of `mask_feature_of`, which is what
    lets a mask row find its camera's recipe."""
    assert camera_feature_of(mask_feature_of(camera), [camera]) == camera


def test_a_camera_outside_the_convention_has_no_mask_column():
    """pusht's `observation.image` has no derivable mask column; the helper
    returns it unchanged so readers see "no masks" rather than a key that
    would collide with the camera itself."""
    assert mask_feature_of("observation.image") == "observation.image"


def test_the_inverse_prefers_a_real_camera_over_the_default_prefix():
    """Without the dataset's cameras the standard prefix is assumed; with them
    the actual key wins, so a non-standard prefix does not produce a key that
    does not exist."""
    odd = "sensors.images.top"
    assert camera_feature_of(f"{MASK_NAMESPACE}.top", [odd]) == odd
    assert camera_feature_of(f"{MASK_NAMESPACE}.top") == "observation.images.top"


def test_two_cameras_cannot_share_a_mask_column():
    """The narrowing this namespace introduced, refused at the writer.

    The old name kept the whole camera prefix, so `observation.images.top` and
    `sensors.images.top` had distinct mask columns. The new one is derived from
    the part after `.images.`, so they collide -- and the second pass would
    overwrite the first's rows with masks of a different scene.
    """
    from lerobot.datasets.mask_compositing import mask_keys_for

    with pytest.raises(ValueError, match="both map to the mask column"):
        mask_keys_for(["observation.images.top", "sensors.images.top"])


def test_the_standard_convention_never_collides():
    """The complement: refusing must not refuse ordinary datasets. Every camera
    under one prefix has a unique suffix by construction."""
    from lerobot.datasets.mask_compositing import mask_keys_for

    cams = [
        "observation.images.top_l",
        "observation.images.top_r",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    keys = mask_keys_for(cams)
    assert len(set(keys.values())) == len(cams)
    assert keys["observation.images.top_l"] == f"{MASK_NAMESPACE}.top_l"


def test_a_camera_without_the_infix_does_not_collide_with_another():
    """Two keys outside the convention pass through unchanged, and remain
    distinct -- the helper must not fold them onto one another."""
    from lerobot.datasets.mask_compositing import mask_keys_for

    keys = mask_keys_for(["observation.image", "observation.depth"])
    assert keys == {"observation.image": "observation.image", "observation.depth": "observation.depth"}


def test_the_suffix_may_contain_the_infix_again():
    """`.images.` inside the suffix must not be re-split; only the first
    occurrence separates prefix from suffix."""
    cam = "observation.images.rig.images.wrist"
    assert mask_feature_of(cam) == f"{MASK_NAMESPACE}.rig.images.wrist"
    assert camera_feature_of(mask_feature_of(cam), [cam]) == cam


def test_the_inverse_leaves_unrelated_keys_alone():
    """Callers pass any feature key; one that names no mask column comes back
    untouched rather than acquiring a camera prefix."""
    for key in ("observation.state", "action", "quality.human_flags", "masks", ""):
        assert camera_feature_of(key, []) == key
