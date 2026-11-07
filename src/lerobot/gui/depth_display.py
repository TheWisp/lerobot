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
"""How the GUI maps a depth map's distances to colour.

The stored data is never touched. A mapping is a display choice the page
sends with every frame and chunk request, and it is part of the cache key,
so two choices never serve each other's pixels.

Modes:

- ``auto``: the palette spans this episode's 1st..99th percentile of valid
  depth (``q01``/``q99``), taken from the per-episode statistics written at
  record time, or the dataset's when the episode has none, or the frame's
  own when neither exists. Stable within an episode, adapting between them.
- ``fixed``: a linear scale between two distances in metres, the user's.

On the wire a spec is ``mode`` or ``mode:lo:hi`` with the range in metres,
optionally per camera as ``camera=spec``, several joined by ``;``. A spec
without a camera is the default for every depth camera the request carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

MODES = ("auto", "fixed")
MM_PER_M = 1000.0


@dataclass(frozen=True)
class DepthMapping:
    mode: str = "auto"
    lo_m: float | None = None
    hi_m: float | None = None

    @property
    def variant(self) -> str:
        """Cache-key suffix. Auto is the plain rendering and stays ``""``."""
        if self.mode == "auto":
            return ""
        return f"d{self.mode}_{self.lo_m:g}_{self.hi_m:g}"

    def render_kwargs(
        self, auto_range_mm: tuple[float, float] | None, invalid_below_mm: float = 0.0
    ) -> dict[str, Any]:
        """Keyword arguments for the JPEG funnel's depth branch, in millimetres.

        ``invalid_below_mm`` is where "no measurement" ends up after the
        dataset round trip: the quantizer maps a zero onto its floor, so the
        marker only survives as "at or below ``depth_min``".
        """
        out: dict[str, Any] = {"depth_invalid_below": invalid_below_mm} if invalid_below_mm > 0 else {}
        if self.mode == "auto":
            if auto_range_mm is not None:
                out["depth_lo"], out["depth_hi"] = auto_range_mm
            return out
        assert self.lo_m is not None and self.hi_m is not None
        out.update(depth_lo=self.lo_m * MM_PER_M, depth_hi=self.hi_m * MM_PER_M)
        return out


AUTO = DepthMapping()


def parse_mapping(spec: str) -> DepthMapping:
    """One spec, ``mode`` or ``mode:lo:hi``. Raises ``ValueError`` on anything else."""
    parts = spec.strip().split(":")
    mode = parts[0]
    if mode not in MODES:
        raise ValueError(f"depth mode must be one of {MODES}, got {mode!r}")
    if mode == "auto":
        if len(parts) != 1:
            raise ValueError("the auto depth mapping takes no range")
        return AUTO
    if len(parts) != 3:
        raise ValueError(f"the {mode} depth mapping needs lo:hi in metres, got {spec!r}")
    try:
        lo, hi = float(parts[1]), float(parts[2])
    except ValueError as e:
        raise ValueError(f"bad depth range in {spec!r}") from e
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < 0 or hi <= lo:
        raise ValueError(f"a depth range needs 0 <= lo < hi, got {spec!r}")
    return DepthMapping(mode, lo, hi)


def parse_depth_spec(spec: str | None) -> dict[str, DepthMapping]:
    """The request's ``depth`` value as camera -> mapping; ``*`` is the default."""
    out: dict[str, DepthMapping] = {}
    if not spec:
        return out
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        cam, _, mapping = item.rpartition("=")
        out[cam or "*"] = parse_mapping(mapping)
    return out


def mapping_for(specs: dict[str, DepthMapping], cam: str) -> DepthMapping:
    return specs.get(cam) or specs.get("*") or AUTO


def _first_number(cell: Any) -> float | None:
    """Statistics cells arrive as nested lists or object arrays shaped like the
    feature, ``(1, 1, 1)`` for a depth map; the number inside."""
    while isinstance(cell, (list, tuple, np.ndarray)):
        if len(cell) == 0:
            return None
        cell = cell[0]
    if cell is None:
        return None
    value = float(cell)
    return value if math.isfinite(value) else None


def _feature_info(feature: dict) -> dict:
    """A feature's info dict, in its current or legacy spelling, the way
    ``LeRobotDatasetMetadata.depth_keys`` reads it."""
    return feature.get("info") or feature.get("video_info") or {}


def _stored_unit_to_mm(feature: dict) -> float:
    return MM_PER_M if _feature_info(feature).get("depth_unit") == "m" else 1.0


def _output_unit_to_mm(dataset) -> float:
    """Dataset-wide statistics are rescaled at load to the unit frames are read
    in (``LeRobotDataset.__init__`` calls ``meta.rescale_depth_stats``); the
    per-episode statistics stay in the recorded unit."""
    return MM_PER_M if getattr(dataset, "_depth_output_unit", "mm") == "m" else 1.0


def episode_auto_range_mm(dataset, episode_idx: int, cam: str) -> tuple[float, float] | None:
    """This episode's 1st..99th percentile of ``cam`` in millimetres, from the
    per-episode statistics; the dataset-wide ones when the episode has none;
    ``None`` when neither exists, which leaves the frame's own to the renderer.

    The statistics count holes, which are stored as zero: an episode with more
    than 1% of them has its 1st percentile at a hole, and the range then starts
    at the quantizer's floor, where a hole reads back, rather than below it.
    """
    feature = dataset.meta.features.get(cam) or {}
    candidates: list[tuple[Any, Any, float]] = []
    try:
        episodes = dataset.meta.episodes
        ep = episodes[episode_idx] if episodes is not None else None
    except Exception:
        ep = None
    if ep is not None:
        candidates.append(
            (ep.get(f"stats/{cam}/q01"), ep.get(f"stats/{cam}/q99"), _stored_unit_to_mm(feature))
        )
    stats = (getattr(dataset.meta, "stats", None) or {}).get(cam)
    if stats:
        candidates.append((stats.get("q01"), stats.get("q99"), _output_unit_to_mm(dataset)))
    floor = invalid_below_mm(feature)
    for lo_cell, hi_cell, scale in candidates:
        lo, hi = _first_number(lo_cell), _first_number(hi_cell)
        if lo is None or hi is None:
            continue
        lo, hi = max(lo * scale, floor), hi * scale
        if hi > lo:
            return lo, hi
    return None


def invalid_below_mm(feature: dict) -> float:
    """The depth at or below which a read-back pixel is "no measurement": the
    quantizer's floor, ``depth_min``, which a stored zero dequantizes to."""
    from lerobot.configs.video import encoder_config_from_video_info

    depth_min_m = getattr(encoder_config_from_video_info(_feature_info(feature)), "depth_min", None)
    return float(depth_min_m) * MM_PER_M if depth_min_m else 0.0


def render_kwargs(dataset, episode_idx: int, cam: str, mapping: DepthMapping) -> dict[str, Any]:
    """The JPEG funnel's depth arguments for ``cam`` at this episode; ``{}``
    for a camera that is not depth, which the funnel then draws as it is."""
    if cam not in dataset.meta.depth_keys:
        return {}
    auto = episode_auto_range_mm(dataset, episode_idx, cam) if mapping.mode == "auto" else None
    return mapping.render_kwargs(auto, invalid_below_mm(dataset.meta.features.get(cam) or {}))


def depth_range_hint_m(stats: dict | None, cam: str, output_unit: str = "mm") -> list[float] | None:
    """The dataset's 1st..99th percentile of ``cam`` in metres, to seed a
    range control; ``None`` without usable statistics. ``stats`` is the loaded
    dataset's, already rescaled to ``output_unit``."""
    cell = (stats or {}).get(cam)
    if not cell:
        return None
    lo, hi = _first_number(cell.get("q01")), _first_number(cell.get("q99"))
    if lo is None or hi is None or hi <= lo:
        return None
    to_m = 1.0 if output_unit == "m" else 1.0 / MM_PER_M
    return [round(max(lo, 0.0) * to_m, 3), round(hi * to_m, 3)]  # a percentile at a hole is a (negative) zero
