# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Robot-owned mailbox handler for click_target calibration + goto.

Lives on the robot, not on the click_target teleop, so calibration clicks
work regardless of which teleop is currently attached — the user can run
their normal leader (or anything else) and the calibration UI still
captures pixel-frame and base-frame XYZ pairs.

Goto requires a click_target teleop ALSO attached: the teleop registers
its ``set_world_target`` setter via
:meth:`set_goto_target_callback`, and the service calls it when a goto
mailbox request arrives. With no teleop registered, goto requests return
an error response.

Commands consumed from the mailbox (see :class:`ClickMailbox`):

* ``sample_pixel {u, v}`` — read depth at the pixel, return camera-frame XYZ.
  The two-step calibration flow uses this; the GUI shows a marker at the
  unprojected point.
* ``sample_fk {arm}`` — read current arm joint observation, FK to base-frame
  EE XYZ, return it. The user has driven the gripper to the previously
  sampled pixel via whatever teleop they have running; the GUI calls this
  when the user clicks ``Confirm``. The pair is ``(prior_cam_xyz, base_xyz)``.
* ``goto {u, v}`` — unproject pixel + apply ``T_base_camera`` + push world
  XYZ into the registered click_target teleop. Returns error if no teleop
  is registered or no extrinsics file exists.
* ``clear`` — push ``None`` into the teleop's target so the IK reverts to
  holding at the latched reference.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import load_extrinsics, unproject_pixel
from .mailbox import ClickMailbox

logger = logging.getLogger(__name__)


class ClickCalibrationService:
    """Background mailbox handler that owns the click_target capture path.

    Constructed by the robot's ``__init__``; started in ``connect()`` and
    stopped in ``disconnect()``. The robot wires it to its own cameras,
    observation, and per-arm kinematics — those don't change at runtime.
    The optional goto-teleop callback is registered/unregistered through
    :meth:`set_goto_target_callback` as click_target teleops attach/detach.

    Preconditions:
        * ``cameras[top_camera_key]`` is a RealSense camera with
          ``use_depth=True`` and exposes ``read_color_and_aligned_depth``
          and ``get_color_intrinsics``.
        * ``get_observation()`` returns a dict including ``{arm}_{motor}.pos``
          keys for every motor in :data:`MOTOR_NAMES` (SO-107).
        * ``kin_left`` / ``kin_right`` may be ``None`` if the robot's
          Cartesian-IK kinematics build failed (e.g. pin-pink missing);
          in that case ``sample_fk`` returns an error response but
          ``sample_pixel`` still works.
    """

    def __init__(
        self,
        *,
        cameras: dict[str, Any],
        get_observation: Callable[[], dict[str, Any]],
        kin_left: Any,
        kin_right: Any,
        mailbox_path: str,
        extrinsics_path: str,
        top_camera_key: str,
        default_arm: str,
        mailbox_poll_hz: float = 20.0,
    ) -> None:
        self._cameras = cameras
        self._get_observation = get_observation
        self._kin = {"left": kin_left, "right": kin_right}
        self._top_camera_key = top_camera_key
        self._default_arm = default_arm
        self._mailbox_path = str(Path(mailbox_path).expanduser())
        self._extrinsics_path = str(Path(extrinsics_path).expanduser())
        self._mailbox = ClickMailbox(self._mailbox_path)
        self._last_seen_seq: int = 0
        self._poll_period_s = 1.0 / max(mailbox_poll_hz, 1.0)

        # Set when a click_target teleop is attached. The setter takes
        # an ndarray world XYZ (or None to clear).
        self._goto_setter: Callable[[np.ndarray | None], None] | None = None
        self._goto_setter_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        # Drain whatever's in the mailbox from a previous session before
        # the poll thread starts: bump ``_last_seen_seq`` to the current
        # request_seq so we don't re-fire a stale sample_fk / goto / etc.
        # from the prior run (which previously produced a one-off
        # "FK failed" exception at startup as the inherited motor read
        # raced the main loop).
        try:
            cur = self._mailbox._read()  # noqa: SLF001 — internal sync read of the JSON state
            self._last_seen_seq = int(cur.get("request_seq", 0))
        except Exception:
            logger.exception("ClickCalibrationService: failed to drain stale mailbox")
        self._thread = threading.Thread(target=self._poll_loop, name="click_target_service", daemon=True)
        self._thread.start()
        logger.info(
            "ClickCalibrationService started (mailbox=%s, extrinsics=%s, top_cam=%s, drained_to_seq=%d)",
            self._mailbox_path,
            self._extrinsics_path,
            self._top_camera_key,
            self._last_seen_seq,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("ClickCalibrationService stopped")

    def set_goto_target_callback(self, setter: Callable[[np.ndarray | None], None] | None) -> None:
        """Register / unregister a click_target teleop's world-target setter.

        Called by the predictive follower's ``_attach_cartesian_teleop``
        when a click_target teleop is detected, and again with ``None``
        when the teleop detaches.
        """
        with self._goto_setter_lock:
            self._goto_setter = setter

    # ── Poll loop ────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                pkt = self._mailbox.poll_request(self._last_seen_seq)
                if pkt is not None:
                    req_seq, request = pkt
                    self._last_seen_seq = req_seq
                    response = self._handle_request(request)
                    self._mailbox.post_response(req_seq, response)
            except Exception:
                logger.exception("ClickCalibrationService: poll loop error")
            self._stop_event.wait(self._poll_period_s)

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = request.get("command")
        if cmd == "sample_pixel":
            return self._handle_sample_pixel(request)
        if cmd == "sample_fk":
            return self._handle_sample_fk(request)
        if cmd == "goto":
            return self._handle_goto(request)
        if cmd == "clear":
            return self._handle_clear()
        return {"status": "error", "message": f"unknown command {cmd!r}"}

    # ── Capture: pixel ↦ camera-frame XYZ ───────────────────────────────

    # See the equivalent constants in the teleop for the rationale; the
    # service inherits them so both code paths produce identical pixels.
    _DEPTH_WINDOW_HALF: int = 7
    _MIN_PLAUSIBLE_DEPTH_M: float = 0.10

    def _read_depth_at(self, u: int, v: int) -> tuple[float, dict[str, float]]:
        cam = self._cameras.get(self._top_camera_key)
        assert cam is not None, (
            f"top_camera_key={self._top_camera_key!r} not in robot cameras "
            f"(available: {sorted(self._cameras)})"
        )
        _color, depth = cam.read_color_and_aligned_depth()
        intrinsics = cam.get_color_intrinsics()
        h, w = depth.shape[:2]
        assert 0 <= u < w and 0 <= v < h, f"pixel ({u}, {v}) out of bounds ({w}, {h})"
        hf = self._DEPTH_WINDOW_HALF
        u0, v0 = max(0, u - hf), max(0, v - hf)
        u1, v1 = min(w, u + hf + 1), min(h, v + hf + 1)
        patch = depth[v0:v1, u0:u1].astype(np.uint32).flatten()
        threshold_mm = int(self._MIN_PLAUSIBLE_DEPTH_M * 1000)
        valid = patch[patch > threshold_mm]
        if valid.size == 0:
            return 0.0, intrinsics
        # Clicking a surface (desk / fixture) — the click target — gives a
        # clean depth from the surface itself, so median is the right
        # choice here (small window already excludes outliers).
        return float(np.median(valid)) / 1000.0, intrinsics

    def _handle_sample_pixel(self, request: dict[str, Any]) -> dict[str, Any]:
        u = int(request["u"])
        v = int(request["v"])
        try:
            depth_m, intrinsics = self._read_depth_at(u, v)
        except Exception as e:
            logger.exception("sample_pixel: depth read failed")
            return {"status": "error", "message": f"depth read failed: {e!r}"}
        if depth_m <= 0.0:
            return {"status": "error", "message": f"no valid depth at ({u}, {v})"}
        cam_xyz = unproject_pixel(u, v, depth_m, intrinsics)
        response = {
            "status": "ok",
            "u": u,
            "v": v,
            "depth_m": float(depth_m),
            "cam_xyz": cam_xyz.tolist(),
            "intrinsics": intrinsics,
        }
        # When extrinsics exist, also return the base-frame surface XYZ.
        # Used by the hover preview during goto mode so the operator sees
        # the planned target BEFORE clicking — clicking on a tall object
        # (the gripper itself, a box) gives a high surface_z, which the
        # user otherwise wouldn't notice until after the goto. The
        # extrinsics are reloaded each call so a fresh save during the
        # same session is picked up on the next hover.
        T_bc = load_extrinsics(self._extrinsics_path)  # noqa: N806
        if T_bc is not None:
            world_xyz_surface = T_bc[:3, :3] @ cam_xyz + T_bc[:3, 3]
            response["world_xyz_surface"] = world_xyz_surface.tolist()
        return response

    # ── Capture: live FK ↦ base-frame EE XYZ ────────────────────────────

    def _handle_sample_fk(self, request: dict[str, Any]) -> dict[str, Any]:
        arm = request.get("arm", self._default_arm)
        if arm not in ("left", "right"):
            return {"status": "error", "message": f"arm must be left|right, got {arm!r}"}
        kin = self._kin.get(arm)
        if kin is None:
            return {
                "status": "error",
                "message": f"kinematics for arm={arm!r} unavailable (is pin-pink installed?)",
            }
        try:
            from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

            obs = self._get_observation()
            q = np.array([float(obs[f"{arm}_{m}.pos"]) for m in MOTOR_NAMES], dtype=float)
            T_ee = kin.forward_kinematics(q)  # noqa: N806
            base_xyz = T_ee[:3, 3].astype(float)
        except KeyError as e:
            return {"status": "error", "message": f"missing observation key: {e!r}"}
        except Exception as e:
            logger.exception("sample_fk: FK failed")
            return {"status": "error", "message": f"FK failed: {e!r}"}
        return {
            "status": "ok",
            "arm": arm,
            "base_xyz": base_xyz.tolist(),
        }

    # ── Goto: pixel ↦ world XYZ ↦ teleop target ─────────────────────────

    # Hard limits on the user-controlled ``z_offset`` (meters in base frame).
    # Below the floor would drive the gripper into the desk; above 30 cm is
    # past the SO-107 workspace ceiling. The frontend clips to the same
    # range but the service clips again as a defence in depth.
    _Z_OFFSET_MIN_M: float = -0.02
    _Z_OFFSET_MAX_M: float = +0.30

    def _handle_goto(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._goto_setter_lock:
            setter = self._goto_setter
        if setter is None:
            return {
                "status": "error",
                "message": "click_target teleop not attached — goto unavailable",
            }
        T_bc = load_extrinsics(self._extrinsics_path)  # noqa: N806
        if T_bc is None:
            return {
                "status": "error",
                "message": (f"no extrinsics at {self._extrinsics_path}; calibrate first"),
            }
        u = int(request["u"])
        v = int(request["v"])
        # ``z_offset`` is the requested raise above the clicked surface, in
        # base-frame Z (the IK's world frame). Default +5 cm so a click
        # onto a flat surface puts the gripper hovering above it instead of
        # diving into it. Clipped to ``[Z_OFFSET_MIN, Z_OFFSET_MAX]`` here
        # as a defence in depth — the frontend's UI already clips first.
        z_offset = float(request.get("z_offset", 0.05))
        z_offset = max(self._Z_OFFSET_MIN_M, min(self._Z_OFFSET_MAX_M, z_offset))
        try:
            depth_m, intrinsics = self._read_depth_at(u, v)
        except Exception as e:
            return {"status": "error", "message": f"depth read failed: {e!r}"}
        if depth_m <= 0.0:
            return {"status": "error", "message": f"no valid depth at ({u}, {v})"}
        cam_xyz = unproject_pixel(u, v, depth_m, intrinsics)
        world_xyz_surface = T_bc[:3, :3] @ cam_xyz + T_bc[:3, 3]
        world_xyz = world_xyz_surface.copy()
        world_xyz[2] += z_offset
        setter(world_xyz.astype(float))
        # One-line per-goto audit so a "why did the arm dive?" can be
        # reconstructed from the log. surface_z=world_xyz_surface[2] before
        # the offset; target_z is what the IK gets; depth is from the median
        # window so a noisy click is visible (depth jumps between samples).
        logger.info(
            "click_target goto: pixel=(%d, %d) depth=%.3f m → surface_z=%.3f m "
            "+ z_offset=%.3f m = target_z=%.3f m (target_xyz=%.3f, %.3f, %.3f)",
            u,
            v,
            depth_m,
            world_xyz_surface[2],
            z_offset,
            world_xyz[2],
            world_xyz[0],
            world_xyz[1],
            world_xyz[2],
        )
        return {
            "status": "ok",
            "u": u,
            "v": v,
            "depth_m": float(depth_m),
            "cam_xyz": cam_xyz.tolist(),
            "world_xyz_surface": world_xyz_surface.tolist(),
            "world_xyz": world_xyz.tolist(),
            "z_offset": z_offset,
        }

    def _handle_clear(self) -> dict[str, Any]:
        with self._goto_setter_lock:
            setter = self._goto_setter
        if setter is not None:
            setter(None)
        return {"status": "ok"}
