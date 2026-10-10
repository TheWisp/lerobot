# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The patch grid used to be hardcoded to DINOv2's; it is now read from the config.

These pin the equivalence (DINOv2 must still get exactly 518/14/1) and the property
that made the generalisation necessary in the first place — a backbone with register
tokens must not have them returned as patches.
"""

from types import SimpleNamespace

import pytest

from lerobot.fewshot.features import _TARGET_SIZE, _grid_geometry


def test_dinov2_geometry_is_the_previous_hardcoded_one():
    assert _grid_geometry(SimpleNamespace(patch_size=14)) == (518, 14, 1)


def test_dinov3_patch_and_registers_are_read_from_the_config():
    size, patch, n_prefix = _grid_geometry(SimpleNamespace(patch_size=16, num_register_tokens=4))
    assert (size, patch, n_prefix) == (512, 16, 5)


@pytest.mark.parametrize("patch", [8, 14, 16, 32])
def test_input_square_is_always_a_whole_number_of_patches(patch):
    size, p, _ = _grid_geometry(SimpleNamespace(patch_size=patch))
    assert p == patch
    assert size % patch == 0
    # Backbones are compared at roughly equal pixel count, not equal patch count.
    assert abs(size - _TARGET_SIZE) <= patch / 2
