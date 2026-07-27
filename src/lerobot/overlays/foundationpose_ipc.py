"""Shared-memory IPC between the debug-vision pipeline (lerobot env) and the
FoundationPose worker (SAM3D venv).

FoundationPose can't run in the debug-vision process — its env (torch 2.8 +
pytorch3d/nvdiffrast/kaolin) is ABI/dependency-incompatible with the lerobot env
(torch 2.10). So it runs as a sidecar worker and we talk over shared memory.

This module is PURE stdlib + numpy — NO lerobot imports — so the SAM3D-venv worker
can ``sys.path.insert`` this directory and ``import foundationpose_ipc`` without
pulling in the lerobot package (which isn't installed there).

Protocol (single in-flight request; the producer waits for the matching response):
  producer writes rgb/depth/mask blocks, then the ctrl block LAST with seq+1 (so the
  data is complete once the worker sees the new seq). Worker computes, writes the
  overlay block then the resp block with the same seq. Producer polls resp for its seq.
"""

from __future__ import annotations

import contextlib
import struct
from multiprocessing import shared_memory

import numpy as np

PREFIX = "lerobot_fp_"
MAX_H, MAX_W = 720, 1280  # top RealSense; buffers sized for the max, actual h/w in ctrl

CMD_IDLE, CMD_REGISTER, CMD_TRACK, CMD_RESET = 0, 1, 2, 3
ST_NONE, ST_OK, ST_FAIL = 0, 1, 2

_CTRL = struct.Struct("<q4i9f")  # seq, cmd, h, w, have_mask, K[9]
_RESP = struct.Struct("<qi16f")  # seq, status, pose[16]
_DEPTH_SCALE = 0.001  # D435 raw uint16 (mm) -> metres; v1 assumes the standard scale


def _block(name: str, size: int, create: bool) -> shared_memory.SharedMemory:
    if not create:
        return shared_memory.SharedMemory(name=name, create=False)
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:  # stale segment from a prior run — reclaim it
        old = shared_memory.SharedMemory(name=name, create=False)
        old.close()
        old.unlink()  # safe-destruct: reclaiming our own stale FoundationPose shm segment
        return shared_memory.SharedMemory(name=name, create=True, size=size)


class FoundationPoseIPC:
    """Both sides instantiate this. ``create=True`` (adapter) allocates; the worker attaches."""

    def __init__(self, create: bool):
        self._create = create
        self.ctrl = _block(PREFIX + "ctrl", _CTRL.size, create)
        self.resp = _block(PREFIX + "resp", _RESP.size, create)
        self.rgb = _block(PREFIX + "rgb", MAX_H * MAX_W * 3, create)
        self.depth = _block(PREFIX + "depth", MAX_H * MAX_W * 2, create)
        self.mask = _block(PREFIX + "mask", MAX_H * MAX_W, create)
        self.overlay = _block(PREFIX + "overlay", MAX_H * MAX_W * 4, create)
        if create:  # zero the control words so a fresh worker/adapter sees seq 0
            self.ctrl.buf[: _CTRL.size] = _CTRL.pack(0, CMD_IDLE, 0, 0, 0, *([0.0] * 9))
            self.resp.buf[: _RESP.size] = _RESP.pack(0, ST_NONE, *([0.0] * 16))

    # ---------------- producer (debug-vision adapter) ----------------
    def ctrl_seq(self) -> int:
        return _CTRL.unpack_from(self.ctrl.buf, 0)[0]

    def send(self, cmd: int, rgb, depth, mask, K) -> int:  # noqa: N803  (K = camera intrinsics)
        """Write a request; returns its seq. depth/mask may be None (track / no-mask)."""
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        self.rgb.buf[: h * w * 3] = np.ascontiguousarray(rgb, np.uint8).tobytes()
        if depth is not None:
            self.depth.buf[: h * w * 2] = np.ascontiguousarray(depth, np.uint16).tobytes()
        have_mask = 1 if mask is not None else 0
        if have_mask:
            self.mask.buf[: h * w] = np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
        seq = self.ctrl_seq() + 1
        kf = np.asarray(K, np.float32).reshape(9)
        self.ctrl.buf[: _CTRL.size] = _CTRL.pack(seq, int(cmd), h, w, have_mask, *kf)
        return seq

    def read_response(self, want_seq: int):
        """Return (status, overlay_or_None, pose4x4_or_None) for ``want_seq``, or None if not ready."""
        seq, status, *pose = _RESP.unpack_from(self.resp.buf, 0)
        if seq != want_seq or status == ST_NONE:
            return None
        _, _, h, w, _, *_ = _CTRL.unpack_from(self.ctrl.buf, 0)
        if status == ST_FAIL:
            return ST_FAIL, None, None
        ov = np.frombuffer(self.overlay.buf, np.uint8, count=h * w * 4).reshape(h, w, 4).copy()
        return ST_OK, ov, np.asarray(pose, np.float32).reshape(4, 4)

    # ---------------- consumer (FoundationPose worker) ----------------
    def poll_request(self, last_seq: int):
        """Return a request dict for a new seq (depth already scaled to metres), else None."""
        seq, cmd, h, w, have_mask, *k = _CTRL.unpack_from(self.ctrl.buf, 0)
        if seq == last_seq or cmd == CMD_IDLE:
            return None
        rgb = np.frombuffer(self.rgb.buf, np.uint8, count=h * w * 3).reshape(h, w, 3).copy()
        depth = (
            np.frombuffer(self.depth.buf, np.uint16, count=h * w).reshape(h, w).astype(np.float32)
            * _DEPTH_SCALE
        )
        mask = None
        if have_mask:
            mask = (np.frombuffer(self.mask.buf, np.uint8, count=h * w).reshape(h, w) > 0).copy()
        return {
            "seq": seq,
            "cmd": cmd,
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "mask": mask,
            # float64 to match trimesh mesh.vertices (FoundationPose does K @ pts and
            # errors on a float32/float64 mismatch).
            "K": np.asarray(k, np.float64).reshape(3, 3),
        }

    def send_response(self, seq: int, status: int, overlay=None, pose=None) -> None:
        if overlay is not None:
            h, w = int(overlay.shape[0]), int(overlay.shape[1])
            self.overlay.buf[: h * w * 4] = np.ascontiguousarray(overlay, np.uint8).tobytes()
        p = np.asarray(pose if pose is not None else np.eye(4), np.float32).reshape(16)
        self.resp.buf[: _RESP.size] = _RESP.pack(int(seq), int(status), *p)

    def close(self, unlink: bool | None = None) -> None:
        unlink = self._create if unlink is None else unlink
        for b in (self.ctrl, self.resp, self.rgb, self.depth, self.mask, self.overlay):
            b.close()
            if unlink:
                with contextlib.suppress(FileNotFoundError):
                    b.unlink()  # safe-destruct: shm segment we created
