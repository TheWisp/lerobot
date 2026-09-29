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
"""A camera's intrinsics: how a direction in front of it lands on a pixel.

The numbers belong to one camera unit at one stream profile, which is why a
driver reads them from the device after connecting rather than from a table.
They describe the frames the camera *publishes*: after our rotation setting,
at the recorded resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import cv2


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"  # distortion model name as the vendor reports it
    coeffs: tuple[float, ...] = ()

    def rotated(self, rotation: int | None) -> CameraIntrinsics:
        """The intrinsics of the frame after ``cv2.rotate(frame, rotation)``.

        Pixel indices run 0..W-1, so a quarter turn maps column ``u`` to row
        ``u`` and row ``v`` to column ``H-1-v``. The distortion coefficients
        ride along unchanged, which is exact for radial terms only; a rotated
        camera with tangential distortion would need refitting.
        """
        if rotation is None:
            return self
        if rotation == cv2.ROTATE_90_CLOCKWISE:
            return replace(
                self, width=self.height, height=self.width, fx=self.fy, fy=self.fx,
                cx=(self.height - 1) - self.cy, cy=self.cx,
            )  # fmt: skip
        if rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            return replace(
                self, width=self.height, height=self.width, fx=self.fy, fy=self.fx,
                cx=self.cy, cy=(self.width - 1) - self.cx,
            )  # fmt: skip
        if rotation == cv2.ROTATE_180:
            return replace(self, cx=(self.width - 1) - self.cx, cy=(self.height - 1) - self.cy)
        raise ValueError(f"unknown cv2 rotation code {rotation!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["coeffs"] = list(self.coeffs)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraIntrinsics:
        return cls(
            width=int(d["width"]), height=int(d["height"]),
            fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]), cy=float(d["cy"]),
            model=str(d.get("model", "pinhole")), coeffs=tuple(float(c) for c in d.get("coeffs", ())),
        )  # fmt: skip
