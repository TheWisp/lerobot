"""Appearance-based pixel correspondences for the metric RGB-D pose backend."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .observations import RGBDFrame


class Sam3MaskSource:
    """Use LeRobot's shared SAM3 detector/tracker once per aligned live frame.

    Distinct roles need distinct prompts here. When two objects share the same
    description, a point-prompted instance source must supply their masks.
    """

    def __init__(
        self,
        prompts_by_role: dict[str, str],
        *,
        camera_id: str,
        device: str = "cuda",
        resolution: int | None = None,
        seed_points_by_role: dict[str, tuple[float, float]] | None = None,
        detection_correction: dict | None = None,
        click_sequence_start: int = 1,
        adapter_factory: Callable[..., object] | None = None,
    ) -> None:
        if (
            not prompts_by_role
            or not camera_id
            or any(
                not role or not prompt.strip() or "." in prompt for role, prompt in prompts_by_role.items()
            )
            or len(set(prompts_by_role.values())) != len(prompts_by_role)
        ):
            raise ValueError("SAM3 needs distinct nonempty role prompts without periods")
        self.prompts = dict(prompts_by_role)
        self.camera_id = camera_id
        self.device = device
        self.resolution = resolution
        self.adapter_factory = adapter_factory
        self.seed_points_by_role = dict(seed_points_by_role or {})
        self.detection_correction = dict(detection_correction) if detection_correction is not None else None
        # reset() preserves clicked instances across scrubs. Reusing an adapter
        # for a new selection must continue its monotonically increasing events.
        if click_sequence_start < 1:
            raise ValueError("click sequence must be positive")
        self.click_sequence_start = click_sequence_start
        self._next_click_seq = click_sequence_start
        if set(self.seed_points_by_role) - set(self.prompts):
            raise ValueError("point-seeded role lacks an object prompt")
        self._adapter = None
        self._last_frame_id: int | None = None
        self._calibration_version: str | None = None
        self._cached_frame: RGBDFrame | None = None
        self._cached_masks: dict[str, np.ndarray] = {}

    @property
    def next_click_sequence(self) -> int:
        return self._next_click_seq

    @property
    def last_frame_id(self) -> int | None:
        return self._last_frame_id

    def reset_tracking(self) -> None:
        """Start an independent reference view without reloading the detector."""
        if self._adapter is not None:
            self._adapter.reset()
        self._last_frame_id = None
        self._calibration_version = None
        self._cached_frame = None
        self._cached_masks = {}

    def masks(self, frame: RGBDFrame) -> dict[str, np.ndarray]:
        if frame is self._cached_frame:
            return self._cached_masks
        if self._adapter is None:
            if self.adapter_factory is None:
                from lerobot.overlays.adapters import Sam3TrackByDetectionAdapter

                factory = Sam3TrackByDetectionAdapter
            else:
                factory = self.adapter_factory
            self._adapter = factory(device=self.device, resolution=self.resolution)
            self._adapter.set_camera(self.camera_id)
            text_prompts = [
                prompt for role, prompt in self.prompts.items() if role not in self.seed_points_by_role
            ]
            control = {"prompt": " . ".join(text_prompts), "text_detection": bool(text_prompts)}
            if self.detection_correction is not None:
                control["detection_correction"] = self.detection_correction
            self._adapter.set_control(control)
            for sequence, (role, (x, y)) in enumerate(
                self.seed_points_by_role.items(), start=self.click_sequence_start
            ):
                if not 0 <= x < frame.depth.shape[1] or not 0 <= y < frame.depth.shape[0]:
                    raise ValueError(f"{role}: point seed lies outside the aligned frame")
                self._adapter.set_control(
                    {
                        "click_seq": sequence,
                        "click_name": {self.camera_id: self.prompts[role]},
                        "clicks": {self.camera_id: [[x, y, 1]]},
                    }
                )
                # Each point has its own name. Apply it to this frame before
                # queuing the next seed, as the adapter stores one pending name.
                self._adapter.segment(np.ascontiguousarray(frame.rgb))
                self._next_click_seq = sequence + 1
        if self._last_frame_id is not None and (
            frame.frame_id <= self._last_frame_id or frame.calibration_version != self._calibration_version
        ):
            self._adapter.reset()
        self._last_frame_id = frame.frame_id
        self._calibration_version = frame.calibration_version
        segmented = self._adapter.segment(np.ascontiguousarray(frame.rgb))
        masks = {
            role: np.asarray(segmented[prompt], dtype=bool)
            for role, prompt in self.prompts.items()
            if prompt in segmented
        }
        self._cached_frame, self._cached_masks = frame, masks
        return masks


def _normalize(descriptors: np.ndarray) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("descriptors must be finite NxD")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero appearance descriptor")
    return values / norms
