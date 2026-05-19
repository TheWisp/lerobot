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
        # Enumerated via ``PredictiveControllerConfig.__dataclass_fields__``
        # so adding a new controller knob automatically propagates here
        # — the previous hand-maintained list silently dropped
        # ``velocity_lowpass_hz`` / ``amp_gate_lo`` / ``amp_gate_hi``.
        import dataclasses

        from ..predictive.config import PredictiveControllerConfig

        per_arm_kwargs = {
            f.name: getattr(config, f.name) for f in dataclasses.fields(PredictiveControllerConfig)
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

        # Cartesian → joint-space adapter (lazy). Created on attach_teleop
        # when a Cartesian teleop (Quest VR) is bound. The adapter owns a
        # background thread that runs THIS robot's registered IK pipeline
        # at WebXR-frame rate and exposes joint-space sub-teleops to each
        # per-arm controller's pull path — same contract as a high-rate
        # leader. Embodiment (URDF, joint_map) lives in the registry, not
        # in the teleop.
        self._cart_adapter = None
        # Last motor observation, cached for the IK adapter to read.
        # Updated by ``get_observation``; the adapter polls this from its
        # own thread.
        self._last_observation_for_ik: dict[str, float] | None = None
        # ``self._cartesian_ik_pipeline`` is set by BiSO107Follower.__init__
        # (parent) — pre-built so URDF+mesh parse doesn't race with the
        # RealSense warmup window.

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

    def get_observation(self) -> dict[str, Any]:
        """Read observation and cache motor positions for the IK adapter.

        The IK adapter (if active) reads this cache from its own thread —
        no separate bus reads, no contention with the main loop. The
        adapter sees observations at the main loop's cadence (~30 Hz),
        which is plenty fresh for engage-time FK.

        Also: if a Cartesian IK adapter was constructed in attach_teleop
        but not yet started (lazy-start to avoid starving RealSense's
        background read thread during camera warm-up), kick it off here.
        By the time the main loop is calling get_observation, all cameras
        have had a chance to start producing frames steadily.
        """
        obs = super().get_observation()
        self._last_observation_for_ik = {
            k: float(v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")
        }
        if self._cart_adapter is not None and not self._cart_adapter.is_running:
            self._cart_adapter.start()
        return obs

    @property
    def last_observation_for_ik(self) -> dict[str, float] | None:
        """For :class:`BimanualCartesianIKAdapter` to consume."""
        return self._last_observation_for_ik

    def bind_teleop(self, teleop):
        """Bind ``teleop`` and return the right :class:`MotorActionBinding`.

        Three cases:

        1. **Cartesian bimanual teleop** (Quest VR, etc.): :meth:`attach_teleop`
           builds the Cartesian→joint adapter and wires it to the
           per-arm controllers' pull paths. The binding reads joint
           output from the adapter cache and reports ``pull_path_active=True``.

        2. **Joint-space bimanual leader** (BiSO107Leader / BiSO107LeaderHighRate):
           per-arm controllers poll the leader directly. The binding
           polls the leader and returns the merged prefixed joint dict;
           ``pull_path_active=True``.

        3. **Anything else** (chunk-aware sources without ``.left_arm``/
           ``.right_arm`` sub-teleops, falls back to the legacy push
           path via ``send_action``): binding polls teleop and returns
           output as-is; ``pull_path_active=False``.
        """
        from lerobot.processor.cartesian_ik_pipeline import is_cartesian_teleop
        from lerobot.robots.motor_action_binding import (
            MotorActionBinding,
            make_adapter_binding,
            make_direct_binding,
        )

        self.attach_teleop(teleop)  # creates self._cart_adapter when Cartesian

        if self._cart_adapter is not None:
            return make_adapter_binding(self._cart_adapter)

        if is_cartesian_teleop(teleop):
            # Cartesian teleop but no adapter was created (e.g., no IK
            # pipeline registered). The loop has no good way to convert
            # without it; fall through to push-path so the existing
            # script-side machinery — if any — still runs.
            return make_direct_binding(teleop, pull_path_active=False)

        if hasattr(teleop, "left_arm") and hasattr(teleop, "right_arm"):
            # Bimanual joint-space leader. Each per-arm controller is
            # polling its own sub-teleop (set up by attach_teleop). For
            # the loop driver's recording need we return the merged
            # prefixed action dict from both sub-teleops.
            return MotorActionBinding(
                get_action=lambda _obs: {
                    **{f"left_{k}": v for k, v in teleop.left_arm.get_action().items()},
                    **{f"right_{k}": v for k, v in teleop.right_arm.get_action().items()},
                },
                pull_path_active=True,
            )

        # Non-bimanual / chunk-aware source: use the legacy push-path
        # (send_action drives motors via the loop driver).
        return make_direct_binding(teleop, pull_path_active=False)

    def attach_teleop(self, teleop) -> None:
        """Route per-arm teleop bindings to each predictive arm.

        Two-mode operation:

        1. **Joint-space bimanual leader** (BiSO107Leader,
           BiSO107LeaderHighRate): teleop exposes ``.left_arm`` /
           ``.right_arm`` sub-teleops returning unprefixed joint dicts.
           Each per-arm controller polls its own sub-teleop directly.

        2. **Cartesian bimanual teleop** (BimanualQuestVRTeleop): teleop
           returns a unified prefixed Cartesian dict. We wrap it in a
           :class:`BimanualCartesianIKAdapter` which runs THIS robot's
           registered IK pipeline at WebXR-frame rate (90 Hz) in a
           background thread and exposes ``.left_arm`` / ``.right_arm``
           sub-teleops returning unprefixed joint dicts — same contract
           as case (1). The adapter caches; both per-arm controllers
           poll the cache lock-free at 200 Hz. The teleop never learns
           anything about embodiment.

        ``None`` detaches both arms (and stops the adapter if running).
        """
        # Always tear down any prior adapter first; covers both detach
        # and the rare re-attach with a different teleop type.
        if self._cart_adapter is not None:
            self._cart_adapter.stop()
            self._cart_adapter = None

        if teleop is None:
            self.left_arm.attach_teleop(None)
            self.right_arm.attach_teleop(None)
            return

        # Cartesian teleop → wrap with IK adapter.
        from lerobot.processor.cartesian_ik_pipeline import (
            is_cartesian_teleop,
            make_cartesian_ik_pipeline,
        )

        from ..predictive.cartesian_adapter import BimanualCartesianIKAdapter

        if is_cartesian_teleop(teleop):
            # Reuse the pipeline pre-built in __init__ to avoid the
            # URDF+mesh parse racing with RealSense's warmup. Fall back
            # to a fresh build only if the pre-build failed.
            pipeline = self._cartesian_ik_pipeline or make_cartesian_ik_pipeline(self)
            if pipeline is None:
                logger.warning(
                    "%s: Cartesian teleop %r attached but no IK pipeline registered "
                    "for this robot — pull path inactive, send_action path will be used",
                    self,
                    type(teleop).__name__,
                )
                return
            self._cart_adapter = BimanualCartesianIKAdapter(
                teleop=teleop,
                pipeline=pipeline,
                observation_source=self,
            )
            # Don't start the IK thread yet — defer to the first
            # ``get_observation()`` call. Starting it now races against
            # RealSense's background read thread coming up after camera
            # connect, and we've observed that race starving the camera
            # below its 500 ms freshness window. By the time the main
            # loop is polling observations, cameras are stable.
            self.left_arm.attach_teleop(self._cart_adapter.left_arm)
            self.right_arm.attach_teleop(self._cart_adapter.right_arm)
            logger.info(
                "%s: Cartesian teleop %r bound via IK adapter "
                "(will start on first observation; 90 Hz IK -> 200 Hz controllers)",
                self,
                type(teleop).__name__,
            )
            return

        if not hasattr(teleop, "left_arm") or not hasattr(teleop, "right_arm"):
            # Not a bimanual leader and not Cartesian — likely a
            # chunk-aware source like TrajectoryReplayTeleop. The
            # dict-pull path doesn't apply (we'd silently feed the same
            # dict to both arms). Skip the attach and let the
            # send_action chunk path handle routing.
            logger.info(
                "%s: teleop %r is not bimanual (no left_arm/right_arm) and not "
                "Cartesian — skipping pull-path attach; send_action chunk path "
                "will be used",
                self,
                type(teleop).__name__,
            )
            return
        self.left_arm.attach_teleop(teleop.left_arm)
        self.right_arm.attach_teleop(teleop.right_arm)

    def send_action(
        self,
        action: dict[str, Any] | ActionChunk,
        *,
        period_s: float | None = None,
    ) -> dict[str, Any]:
        """Route bimanual intent to each predictive arm.

        For an ``ActionChunk``: split each frame's prefixed keys into two
        per-arm sub-chunks (same fps) and forward each to its arm. Both
        arms therefore get the full chunk horizon — exact-lookup runs
        independently per-arm. The recorded action is the prefixed
        frames[0] (= "current intent" for both arms combined), preserving
        the dataset-writer contract.

        For a plain dict: delegate to ``BiSO107Follower.send_action``,
        which handles dry-run + per-arm prefix translation.

        ``period_s`` is forwarded to each arm's predictive controller
        (see :meth:`SO107FollowerPredictive.send_action`). For chunks it
        defaults to ``1/action.fps`` per arm.
        """
        if not isinstance(action, ActionChunk):
            # The plain-dict path goes through BiSO107Follower.send_action,
            # which doesn't accept period_s. The current dict-mode callers
            # (intervention reset, idle hold) don't need starvation
            # detection. If a future caller needs it, they can wire it
            # explicitly through left_arm / right_arm send_action calls.
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

        left_sent = self.left_arm.send_action(left_chunk, period_s=period_s)
        right_sent = self.right_arm.send_action(right_chunk, period_s=period_s)

        # Re-prefix and return — the dataset writer records this as the
        # action for the current tick.
        return {
            **{f"left_{k}": v for k, v in left_sent.items()},
            **{f"right_{k}": v for k, v in right_sent.items()},
        }
