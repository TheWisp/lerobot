# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned machine-readable records embedded in human training logs."""

from __future__ import annotations

import json
import math

TRAINING_LOG_JSON_MARKER = "LEROBOT_TRAINING_JSON:"
TRAINING_LOG_JSON_VERSION = 1


def format_training_log_record(
    *,
    step: int,
    total_steps: int,
    **values: int | float,
) -> str:
    """Encode one validated training progress/metrics record."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"Training log step must be a non-negative integer, got {step!r}")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError(f"Training log total_steps must be a positive integer, got {total_steps!r}")

    payload: dict[str, int | float] = {
        "version": TRAINING_LOG_JSON_VERSION,
        "step": step,
        "total_steps": total_steps,
    }
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Training log field {key!r} must be numeric, got {type(value).__name__}")
        if not math.isfinite(value):
            raise ValueError(f"Training log field {key!r} must be finite, got {value}")
        payload[key] = value
    return TRAINING_LOG_JSON_MARKER + json.dumps(payload, separators=(",", ":"), sort_keys=True)
