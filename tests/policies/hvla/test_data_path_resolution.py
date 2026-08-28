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
"""`--data-path` is a contract, and it was the untested part of the GPU path.

Which path a run takes decides what its numbers mean. A run that asked for the
GPU path and quietly got the CPU one is not a slower run, it is a wrong
measurement -- three benchmark runs were misread that way in a single day before
the fallback was made to announce itself. The rules are therefore asserted here
rather than left to the logs:

* `cpu` never builds a pipeline.
* `auto` falls back when the GPU path cannot be had, and says why.
* `gpu` fails loudly instead of falling back.

Every case below is unsatisfiable on a CPU device, which is the point: it makes
the branch reachable without a GPU, so the contract is checked in CI too.
"""

import logging

import pytest

from lerobot.policies.hvla.s1.flow_matching.train import _resolve_data_path


def test_cpu_never_builds_a_gpu_pipeline():
    """Requesting the CPU path must not touch the dataset or the device at all."""
    assert _resolve_data_path("cpu", None, None, None, "cuda", 8) is None


def test_auto_falls_back_and_says_why(caplog):
    """A fallback that is silent is indistinguishable from a path that worked."""
    with caplog.at_level(logging.WARNING):
        assert _resolve_data_path("auto", None, None, None, "cpu", 8) is None
    assert "Data path: CPU" in caplog.text, "the fallback must be logged"
    assert "not CUDA" in caplog.text, "the log must carry the reason, not just the outcome"


def test_an_explicit_gpu_request_that_cannot_be_met_stops_the_run():
    """Silently honouring the other path is how a benchmark comes to lie."""
    with pytest.raises(NotImplementedError, match="not CUDA"):
        _resolve_data_path("gpu", None, None, None, "cpu", 8)


def test_an_unknown_choice_is_not_quietly_treated_as_auto():
    with pytest.raises(AssertionError):
        _resolve_data_path("nvdec", None, None, None, "cpu", 8)
