#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bimanual SO-107 follower with predictive-lookahead controllers on both arms.

Composes two ``SO107FollowerPredictive`` instances — one per arm. Each
arm owns its own ``FeetechMotorsBus`` and its own 200 Hz controller
thread; the buses are independent, so the two controllers don't
contend on anything. The composer's job is just argument routing and
joint-name prefix translation (``left_*`` / ``right_*``), identical
to plain ``BiSO107Follower``.

Why per-arm threads instead of one shared loop driving both:

  * Bus blip isolation. A transient Feetech retry on one bus (~10 ms)
    would stall the shared loop and delay the other arm's writes.
    Two threads × two buses keeps the slow side from infecting the
    fast side.
  * Independent adaptive L. The two arms have measurably different
    motor τ in practice (one bus is over-cabled in white profile,
    the other isn't). Each arm's amplitude-gated xcorr converges to
    its own τ. Sharing one L would force the wrong tuning on at
    least one arm.
  * Phase drift between arms is ≤ 1 control period = 5 ms at 200 Hz —
    below motor response time and below the human-teleop floor, so
    losing per-tick sync of the goal issue costs nothing measurable.
  * ``SO107FollowerPredictive`` stays self-contained and re-usable as
    a single-arm robot; the bimanual is just two of them stacked.

GIL note: ``bus.sync_write`` releases the GIL during the serial I/O,
so the two writer threads progress in parallel even on CPython.

Distinct ``robot_type`` (``bi_so107_follower_predictive``) so the
training-side embodiment contract is unambiguous — see the module
docstring on ``SO107FollowerPredictive`` for the long-form rationale.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.types import ActionChunk

from ..bi_so107_follower.bi_so107_follower import BiSO107Follower
from ..robot import Robot
from ..so107_follower_predictive import (
    SO107FollowerPredictive,
    SO107FollowerPredictiveRobotConfig,
)
from .config_bi_so107_follower_predictive import BiSO107FollowerPredictiveConfig

logger = logging.getLogger(__name__)


class BiSO107FollowerPredictive(BiSO107Follower):
    """Bimanual SO-107 with the predictive controller on each arm.

    Everything caller-facing (``connect`` / ``get_observation`` /
    ``send_action`` / ``disconnect``) is inherited from
    ``BiSO107Follower`` — it forwards to the per-arm objects, which
    are now ``SO107FollowerPredictive`` instances that handle the
    200 Hz writer thread + lookahead + adaptive update internally.
    """

    config_class = BiSO107FollowerPredictiveConfig
    name = "bi_so107_follower_predictive"

    def __init__(self, config: BiSO107FollowerPredictiveConfig):
        # Bypass BiSO107Follower.__init__ — its arms are plain SO107Follower.
        # We want predictive arms, so call Robot.__init__ directly and rebuild
        # the per-arm + camera setup with our own arm class.
        Robot.__init__(self, config)
        self.config = config

        # Shared controller settings projected onto each per-arm config.
        # Both arms get the same lookahead / alpha / control rate — see
        # the config docstring for why this is shared rather than per-arm.
        per_arm_kwargs = {
            "lookahead_ms": config.lookahead_ms,
            "max_lookahead_ms": config.max_lookahead_ms,
            "corrector_alpha": config.corrector_alpha,
            "velocity_window_ms": config.velocity_window_ms,
            "control_rate_hz": config.control_rate_hz,
            "adaptive": config.adaptive,
            "max_step_deg": config.max_step_deg,
            "velocity_estimator": config.velocity_estimator,
        }

        left_arm_config = SO107FollowerPredictiveRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
            **per_arm_kwargs,
        )

        right_arm_config = SO107FollowerPredictiveRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
            **per_arm_kwargs,
        )

        self.left_arm = SO107FollowerPredictive(left_arm_config)
        self.right_arm = SO107FollowerPredictive(right_arm_config)

        # Cameras + RealSense depth-edge post-grab processor installation —
        # identical to BiSO107Follower.__init__. Kept inline rather than
        # extracted into a helper because changing it would require also
        # changing BiSO107Follower, and the two are independent embodiments
        # that just happen to share hardware.
        from lerobot.cameras.realsense import RealSenseCamera
        from lerobot.cameras.utils import make_cameras_from_configs
        from lerobot.processor import DepthEdgeOverlayProcessorStep

        self.cameras = make_cameras_from_configs(config.cameras)
        for cam_key, cam in self.cameras.items():
            if isinstance(cam, RealSenseCamera) and cam.use_depth:
                cam.post_grab_processor = DepthEdgeOverlayProcessorStep(
                    camera_key=cam_key,
                    threshold_percentile=90,
                    blur_kernel=3,
                    dilation_kernel=2,
                    alpha=0.7,
                    min_depth=0.2,
                    max_depth=0.6,
                )

    def attach_teleop(self, teleop) -> None:
        """Route per-arm teleop bindings to each predictive arm.

        Expects a bimanual leader teleop with ``left_arm`` / ``right_arm``
        sub-teleop attributes (BiSO107Leader / BiSO107LeaderHighRate both
        expose this). Each predictive arm's controller polls its own
        arm's teleop directly — no cross-arm coordination, no shared
        cache. The two controllers operate independently at their own
        rates against independent bus reads.

        ``None`` detaches both arms.
        """
        if teleop is None:
            self.left_arm.attach_teleop(None)
            self.right_arm.attach_teleop(None)
            return
        if not hasattr(teleop, "left_arm") or not hasattr(teleop, "right_arm"):
            # Not a bimanual leader — likely a chunk-aware source like
            # TrajectoryReplayTeleop. The dict-pull path doesn't apply
            # (we'd silently feed the same dict to both arms). Skip the
            # attach and let the send_action chunk path handle routing.
            # Logged at INFO so it's visible without being alarming.
            logger.info(
                "%s: teleop %r is not bimanual (no left_arm/right_arm) — "
                "skipping pull-path attach; send_action chunk path will be used",
                self,
                type(teleop).__name__,
            )
            return
        self.left_arm.attach_teleop(teleop.left_arm)
        self.right_arm.attach_teleop(teleop.right_arm)

    def send_action(self, action: dict[str, Any] | ActionChunk) -> dict[str, Any]:
        """Route bimanual intent to each predictive arm.

        For an ``ActionChunk``: split each frame's prefixed keys into two
        per-arm sub-chunks (same fps) and forward each to its arm. Both
        arms therefore get the full chunk horizon — exact-lookup runs
        independently per-arm. The recorded action is the prefixed
        frames[0] (= "current intent" for both arms combined), preserving
        the dataset-writer contract.

        For a plain dict: delegate to ``BiSO107Follower.send_action``,
        which handles dry-run + per-arm prefix translation.
        """
        if not isinstance(action, ActionChunk):
            return super().send_action(action)

        if self.config.dry_run:
            # Don't even route — return the current frame so callers
            # recording the action keep working.
            if not getattr(self, "_dry_run_logged", False):
                logger.warning(
                    "%s: dry_run=True - send_action is a no-op. Motors will NOT move. "
                    "Disable dry_run in the robot config to drive the arms.",
                    self,
                )
                self._dry_run_logged = True
            return dict(action.frames[0])

        # Split: each frame is {left_*: ..., right_*: ...} -> ({*: ...}, {*: ...})
        left_frames = tuple(
            {k.removeprefix("left_"): v for k, v in f.items() if k.startswith("left_")} for f in action.frames
        )
        right_frames = tuple(
            {k.removeprefix("right_"): v for k, v in f.items() if k.startswith("right_")}
            for f in action.frames
        )
        # Fail fast on mixed frames (frame missing one arm's keys). Without
        # this guard, the empty side would pass ActionChunk's non-empty-tuple
        # check, hit left_arm first, succeed, then raise on right_arm's
        # strict-key check — leaving the arms out of sync. The fix is to
        # detect the mismatch before any side-effect.
        empty_left = [i for i, f in enumerate(left_frames) if not f]
        empty_right = [i for i, f in enumerate(right_frames) if not f]
        if empty_left or empty_right:
            raise ValueError(
                f"{self}: ActionChunk frames are missing per-arm keys. "
                f"Frames with no left_* keys: {empty_left}. "
                f"Frames with no right_* keys: {empty_right}. "
                f"Every frame must carry both left_* and right_* keys to keep "
                f"the arms in sync."
            )
        left_chunk = ActionChunk(fps=action.fps, frames=left_frames)
        right_chunk = ActionChunk(fps=action.fps, frames=right_frames)

        left_sent = self.left_arm.send_action(left_chunk)
        right_sent = self.right_arm.send_action(right_chunk)

        # Re-prefix and return — the dataset writer records this as the
        # action for the current tick.
        return {
            **{f"left_{k}": v for k, v in left_sent.items()},
            **{f"right_{k}": v for k, v in right_sent.items()},
        }
