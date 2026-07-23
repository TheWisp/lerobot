#!/usr/bin/env python

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

"""Regression tests for the torchcodec decode backend's loadability.

torchcodec ships binaries compiled per torch minor version but declares no
torch requirement in its package metadata, so an environment can end up with
an ABI-incompatible torch/torchcodec pair (e.g. torch 2.11 + torchcodec
0.10). In that state ``importlib.util.find_spec("torchcodec")`` still
succeeds — the failure only appears when the compiled libraries are first
loaded, i.e. when a decoder is constructed. Availability checks based on
``find_spec`` (see ``get_safe_default_video_backend``) therefore cannot
detect it; only an actual decode can.
"""

import importlib.util
from pathlib import Path

import pytest

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "encoded_videos"
SRC_CLIP = TEST_ARTIFACTS_DIR / "clip_4frames.mp4"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchcodec") is None,
    reason="torchcodec is not installed on this platform (pyav is the default backend)",
)


def test_torchcodec_backend_decodes_frames():
    """Forcing backend='torchcodec' must load libtorchcodec and return real frames.

    Fails (does not skip) when the installed torchcodec is ABI-incompatible
    with the installed torch — that is exactly the regression this guards.
    """
    from lerobot.datasets.video_utils import decode_video_frames

    frames = decode_video_frames(str(SRC_CLIP), [0.0], tolerance_s=1.0, backend="torchcodec")
    assert frames.ndim == 4  # (T, C, H, W)
    assert frames.shape[0] == 1


def test_default_backend_decodes_frames():
    """The auto-selected default backend must decode, mirroring the dataset/GUI playback path."""
    from lerobot.datasets.video_utils import decode_video_frames

    frames = decode_video_frames(str(SRC_CLIP), [0.0], tolerance_s=1.0)
    assert frames.ndim == 4
    assert frames.shape[0] == 1
