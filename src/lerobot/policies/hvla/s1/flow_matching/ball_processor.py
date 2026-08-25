# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Supply the ball cue at inference, from the same segmenter training used.

Hardcoded for the feasibility experiment: one prompt, one source camera named
by the checkpoint. If the experiment says the cue helps, this becomes a
declared, registry-instantiated processor; until then a narrow implementation
is the honest amount of machinery.

The training side reads a cached mask column and this reads SAM3 live, but
both hand the mask to the SAME functions in ball_cue.py, so the cue itself
cannot drift between the two paths. What does differ is cadence: the tracker
advances per frame it SEES, and inference sees only the frames it runs on.
"""

from __future__ import annotations

import logging

import numpy as np

from lerobot.policies.hvla.s1.flow_matching.ball_cue import (
    NOT_VISIBLE,
    ball_cue,
    render_ball_view,
)

logger = logging.getLogger(__name__)

PROMPT = "yellow ball"  # hardcode-ok: the experiment's single object


class BallCueProcessor:
    """Adds the ball cue to an observation, in place.

    Implements the observation-step contract the S1 loop dispatches on
    (``step.observation(obs) -> obs``), the same one CameraProcessor and the
    HIL steps implement. It is not a plain callable: run_s1 calls the method
    by name.

    Pre: ``observation`` carries the source camera's frame as HxWx3 uint8 under
    its short name. Post: ``ball.x``, ``ball.y``, ``ball.visible`` are present
    (sentinel values when nothing was found), and ``ball_view`` too when the
    checkpoint asked for the rendered image.

    Never raises on a detection failure: a miss is a value the policy was
    trained on, not an error. It DOES raise if the source frame is absent,
    because then the cue would be a fabrication.
    """

    def __init__(self, source_key: str, want_view: bool, device: str = "cuda"):
        from lerobot.overlays.adapters import build_adapter

        self.source_name = source_key.rsplit(".", 1)[-1]
        self.want_view = want_view
        self._adapter = build_adapter("sam3_track", device=device)
        self._adapter.set_control({"objects": [{"name": PROMPT}], "multi_instance": False})
        self._adapter.set_camera(self.source_name)
        self._misses = 0
        self._frames = 0
        logger.info(
            "Ball cue: segmenting %r on %s%s",
            PROMPT,
            self.source_name,
            " (+ rendered view)" if want_view else "",
        )

    def observation(self, observation: dict) -> dict:
        frame = observation.get(self.source_name)
        if frame is None:
            raise KeyError(
                f"ball cue needs camera {self.source_name!r} in the observation; "
                f"have {sorted(k for k in observation if isinstance(observation[k], np.ndarray))}"
            )
        masks = self._adapter.segment(np.asarray(frame, dtype=np.uint8))
        mask = masks.get(PROMPT) if masks else None
        x, y, visible = ball_cue(mask)
        self._frames += 1
        if not visible:
            self._misses += 1
            if self._misses % 50 == 1:
                logger.warning(
                    "Ball not found in %d of %d frames; the policy sees the not-visible sentinel",
                    self._misses,
                    self._frames,
                )
        observation["ball.x"], observation["ball.y"], observation["ball.visible"] = x, y, visible
        if self.want_view:
            observation["ball_view"] = render_ball_view(np.asarray(frame, dtype=np.uint8), mask)
        return observation

    @property
    def miss_rate(self) -> float:
        return self._misses / self._frames if self._frames else 0.0


def cue_from_observation(observation: dict) -> tuple[float, float, float]:
    """The (x, y, visible) triple a processed observation carries."""
    if "ball.x" not in observation:
        return NOT_VISIBLE
    return (
        float(observation["ball.x"]),
        float(observation["ball.y"]),
        float(observation["ball.visible"]),
    )
