"""Adapter-side manager for the FoundationPose sidecar (runs in the lerobot env).

Spawns the worker (SAM3D venv), owns the shared-memory IPC, reads aligned depth from
the ObservationStream (Phase-1 depth path), and runs register/track per frame —
returning the amodal RGBA overlay for the adapter to composite. See the sidecar/IPC
limitation note in gui/TODO.md for why this is a separate process.
"""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404  spawning our own worker; args are fixed paths
import time

import numpy as np

logger = logging.getLogger(__name__)

_SAM3D_PY = os.path.expanduser("~/.cache/sam3d/venv/bin/python")
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foundationpose_worker.py")
# v1: hardcoded top-RealSense intrinsics (D435 @ 1280x720). FoundationPose only runs on a
# camera that has depth (read_depth returns None otherwise), which in our rig is "top".
# Generalize by publishing per-camera K in the ObservationStream meta.
_K_TOP = np.array([[906.0, 0, 633.0], [0, 906.0, 376.0], [0, 0, 1.0]])


class FoundationPoseClient:
    """Owns the FoundationPose worker process + IPC. ``process()`` returns an amodal
    overlay (RGBA) for the given frame, or None when unavailable (no depth, worker
    busy/dead, or registration not yet succeeded)."""

    def __init__(self, timeout_register: float = 12.0, timeout_track: float = 1.0):
        from lerobot.policies.debug_vision.foundationpose_ipc import FoundationPoseIPC

        self._ipc = FoundationPoseIPC(create=True)
        self._proc = subprocess.Popen([_SAM3D_PY, _WORKER])  # nosec B603  fixed worker path
        self._reader = None
        self._registered = False
        self._t_reg, self._t_trk = timeout_register, timeout_track
        logger.info("FoundationPose sidecar spawned (pid %s)", self._proc.pid)

    def _depth(self, cam: str | None):
        """Aligned uint16 depth for ``cam`` from the ObservationStream, or None."""
        if self._reader is None:
            try:
                from lerobot.robots.obs_stream import ObservationStreamReader

                self._reader = ObservationStreamReader()
            except Exception:
                return None
        try:
            r = self._reader.read_depth(cam)
        except Exception:
            return None
        return r[0] if r else None

    def process(self, rgb: np.ndarray, mask: np.ndarray, cam: str | None):
        """rgb HxWx3 uint8 RGB, mask HxW bool (the tracked object), cam key.
        Returns the amodal RGBA overlay or None."""
        from lerobot.policies.debug_vision.foundationpose_ipc import CMD_REGISTER, CMD_TRACK, ST_OK

        if self._proc.poll() is not None:
            return None  # worker died
        if mask is None or not mask.any():
            return None
        depth = self._depth(cam)
        if depth is None or depth.shape[:2] != rgb.shape[:2]:
            return None
        cmd = CMD_TRACK if self._registered else CMD_REGISTER
        seq = self._ipc.send(cmd, rgb, depth, mask, _K_TOP)
        timeout = self._t_trk if self._registered else self._t_reg
        t0 = time.time()
        resp = None
        while time.time() - t0 < timeout:
            resp = self._ipc.read_response(seq)
            if resp is not None:
                break
            time.sleep(0.002)
        if resp is None:
            return None
        status, overlay, _pose = resp
        if status == ST_OK:
            self._registered = True
            return overlay
        return None

    def reset(self) -> None:
        """Force a re-register on the next frame (e.g. tracking lost)."""
        self._registered = False

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                logger.debug("FoundationPose worker kill failed", exc_info=True)
        try:
            self._ipc.close(unlink=True)
        except Exception:
            logger.debug("FoundationPose IPC close failed", exc_info=True)
