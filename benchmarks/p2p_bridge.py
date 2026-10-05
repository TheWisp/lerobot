#!/usr/bin/env python
"""Point2Pose (arXiv 2604.10415) behind a pipe, for the live tracker's "p2p" algorithm.

The tracker (benchmarks/pregrasp_worker.py) runs in the lerobot environment; Point2Pose needs
its own (Python 3.11, its torch, gtsam, SAM2, tapnet), so it runs here as a child process and
talks NPZ over stdin/stdout.  Every message is a 4-byte big-endian length followed by an
uncompressed NPZ.

Requests carry ``kind`` ("init" or "step"), ``rgb`` (HxWx3 uint8), ``depth`` (HxW float32,
metres); "init" also carries ``K`` (3x3) and ``mask`` (HxW bool, the object).  Replies carry
``meta`` (JSON) and, when the object is posed, ``delta`` (4x4: the object's motion since the
init frame, in the camera frame), ``mask`` (HxW bool, SAM2's current mask) and ``live_uv``
(Nx2 float32, the visible tracked points).  Every reply also carries ``fit_uv`` (Mx2 float32, the
tracked points this frame's pose was fitted on) and ``fit_inlier`` (M bool, which of them the fit kept),
for a recording of the run.  The first reply, before any request, is ``{"ready": true}`` once the
models are on the GPU.

Point2Pose prints freely to stdout, so the protocol takes over file descriptor 1 and sends
every print to stderr instead.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import struct
import sys
import tempfile
import time

import numpy as np

DEPTH_FACTOR = 1000.0  # Point2Pose's RealSense path is tuned for millimetre depth


def _read_exact(fd, n: int) -> bytes | None:
    """n bytes from a pipe, which hands out whatever is in flight, or None when it closed."""
    chunks = []
    while n > 0:
        chunk = fd.read(n)
        if not chunk:
            return None
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def _read(fd) -> dict | None:
    head = _read_exact(fd, 4)
    if head is None:
        return None
    body = _read_exact(fd, struct.unpack(">I", head)[0])
    if body is None:
        return None
    z = np.load(io.BytesIO(body), allow_pickle=False)
    return {k: z[k] for k in z.files}


def _write(fd, **arrays) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    fd.write(struct.pack(">I", len(data)) + data)
    fd.flush()


def _load_config(repo: pathlib.Path, config: pathlib.Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config))
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="p2p_"))

    def under_repo(p: str) -> str:
        return p if os.path.isabs(p) else str(repo / p)

    cfg.segmenter.params.checkpoint = under_repo(cfg.segmenter.params.checkpoint)
    cfg.tracker.params.checkpoint_path = under_repo(cfg.tracker.params.checkpoint_path)
    cfg.pipeline.params.debug_dir = str(scratch)
    cfg.sampler.params.debug_dir = str(scratch / "sampler")
    cfg.criterion.params.debug_dir = str(scratch / "criterion")
    cfg.visualization.params.output_image_dir = str(scratch / "images")
    return cfg


class Session:
    """One Point2Pose pipeline from one init frame; a new init starts a new pipeline."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipe = None
        self.frame_id = 0

    def close(self) -> None:
        """Let the previous pipeline go before the next is built. Every init builds a pipeline with its own models
        on the GPU, and the wrapper on its front end refers back to the front end: a cycle only the garbage
        collector frees, which kept every old pipeline's models on the GPU until CUDA ran out of memory."""
        if self.pipe is None:
            return
        import gc

        import torch

        self.pipe.frontend.step = None  # break the wrapper's cycle, so the models go with the pipeline
        self.pipe = self._fe = None
        gc.collect()
        torch.cuda.empty_cache()

    def init(self, rgb: np.ndarray, depth_m: np.ndarray, k: np.ndarray, mask: np.ndarray) -> dict:
        from point2pose.data_types.frame import Frame
        from point2pose.pipeline.modular_pipeline import ModularPipeline

        self.close()
        self.pipe = ModularPipeline(self.cfg)
        self.frame_id = 0
        self._fe = None
        frontend_step = self.pipe.frontend.step

        def keep(*args, **kwargs):
            # The pipeline does not return the front end's result: the points the pose was fitted on.
            self._fe = frontend_step(*args, **kwargs)
            return self._fe

        self.pipe.frontend.step = keep
        frame = Frame(
            id=0,
            rgb=np.ascontiguousarray(rgb),
            depth=np.ascontiguousarray(depth_m * DEPTH_FACTOR, dtype=np.float32),
            intrinsics=np.asarray(k, dtype=np.float64),
            depth_factor=DEPTH_FACTOR,
            timestamp=time.time(),
        )
        frame.mask = np.asarray(mask, dtype=np.uint8)[None, None]
        self.pipe.step(frame)
        return self._answer(frame, 0.0)

    def step(self, rgb: np.ndarray, depth_m: np.ndarray) -> dict:
        from point2pose.data_types.frame import Frame

        self.frame_id += 1
        frame = Frame(
            id=self.frame_id,
            rgb=np.ascontiguousarray(rgb),
            depth=np.ascontiguousarray(depth_m * DEPTH_FACTOR, dtype=np.float32),
            intrinsics=self.pipe.hist_frames[-1].intrinsics if self.pipe.hist_frames else None,
            depth_factor=DEPTH_FACTOR,
            timestamp=time.time(),
        )
        if frame.intrinsics is None:
            frame.intrinsics = self._k
        self._fe = None  # a frame the front end skips must not report the previous frame's fit
        t0 = time.perf_counter()
        self.pipe.step(frame)
        return self._answer(frame, (time.perf_counter() - t0) * 1000.0)

    def _answer(self, frame, ms: float) -> dict:
        obj = self.pipe.objects[0]
        self._k = frame.intrinsics
        table = self.pipe.track_table
        visible = np.asarray(table.visible, dtype=bool) if table.visible is not None else np.zeros(0, bool)
        uv = np.asarray(table.track_2d, dtype=np.float32).reshape(-1, 2)[: len(visible)]
        mask = None
        if frame.mask is not None:
            m = frame.mask[0, 0]
            mask = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)) > 0
        fit_uv, fit_inlier, guard = self._fit(table)
        meta = {
            "ok": True,
            "lost": bool(obj.lost),
            "lost_streak": int(obj.lost_streak),
            "n_tracks": int(len(visible)),
            "n_visible": int(visible.sum()),
            "mean_residual_m": float(obj.mean_residual),
            "ms": ms,
            "n_model_points": int(len(obj.key_points)),
            "fit_points": int(len(fit_uv)),
            "fit_inliers": int(fit_inlier.sum()),
            "jump_guard_rejected": bool(guard.get("rejected", False)),
        }
        out = {
            "meta": json.dumps(meta),
            "delta": np.asarray(obj.pose, dtype=np.float64),
            "live_uv": uv[visible],
            "fit_uv": fit_uv,
            "fit_inlier": fit_inlier,
            # Its model so far: every key point it has adopted, in the first (teach) frame's coordinates.
            "model": np.asarray(obj.key_points, dtype=np.float32).reshape(-1, 3),
        }
        if mask is not None:
            out["mask"] = mask
        return out

    def _fit(self, table) -> tuple[np.ndarray, np.ndarray, dict]:
        """The points this frame's pose was fitted on, which of them the fit kept, and the jump guard's verdict."""
        empty = np.zeros((0, 2), np.float32), np.zeros(0, bool), {}
        fe = self._fe
        if fe is None:
            return empty
        stats = fe.reg_stats.get(0) or {}
        guard = stats.get("pose_jump_guard_info") or {}
        idx = np.asarray(fe.valid_indices.get(0, []), dtype=np.int64).reshape(-1)
        uv = np.asarray(table.track_2d, dtype=np.float32).reshape(-1, 2)
        if len(idx) == 0 or idx.max() >= len(uv):
            return empty[0], empty[1], guard
        inliers = np.asarray(stats.get("inliers", []), dtype=bool).reshape(-1)
        # The register's flags are per fitted point, in order; a register that reports none leaves them unknown.
        fit_inlier = inliers if len(inliers) == len(idx) else np.zeros(len(idx), bool)
        return uv[idx], fit_inlier, guard


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="the Point2Pose checkout")
    ap.add_argument("--config", required=True, help="pipeline YAML; checkpoint paths relative to --repo")
    args = ap.parse_args()
    repo = pathlib.Path(args.repo).resolve()

    # The protocol owns fd 1; everything the models print goes to stderr.
    wire_in = os.fdopen(os.dup(0), "rb", buffering=0)
    wire_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    sys.path.insert(0, str(repo))
    os.chdir(repo)  # SAM2's hydra config and the authors' relative paths resolve from here
    cfg = _load_config(repo, pathlib.Path(args.config).resolve())
    from point2pose.pipeline.modular_pipeline import ModularPipeline  # noqa: F401  (loads the models' code)

    session = Session(cfg)
    _write(wire_out, meta=json.dumps({"ready": True}))
    while True:
        req = _read(wire_in)
        if req is None:
            return
        kind = str(req["kind"])
        try:
            if kind == "init":
                reply = session.init(req["rgb"], req["depth"], req["K"], req["mask"])
            elif kind == "step":
                reply = session.step(req["rgb"], req["depth"])
            else:
                reply = {"meta": json.dumps({"ok": False, "reason": f"unknown request {kind!r}"})}
        except Exception as e:  # the request fails, the bridge lives
            import traceback

            traceback.print_exc()
            reply = {"meta": json.dumps({"ok": False, "reason": f"{type(e).__name__}: {e}"})}
        _write(wire_out, **reply)


if __name__ == "__main__":
    main()
