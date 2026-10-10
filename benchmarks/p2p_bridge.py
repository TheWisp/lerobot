#!/usr/bin/env python
"""Point2Pose (arXiv 2604.10415) behind a pipe, for the live tracker's "p2p" algorithm.

The tracker (benchmarks/pregrasp_worker.py) runs in the lerobot environment; Point2Pose needs
its own (Python 3.11, its torch, gtsam, SAM2, tapnet), so it runs here as a child process and
talks NPZ over stdin/stdout.  Every message is a 4-byte big-endian length followed by an
uncompressed NPZ.

Requests carry ``kind`` ("init" or "step"), ``rgb`` (HxWx3 uint8), ``depth`` (HxW float32,
metres); "init" also carries ``K`` (3x3) and ``mask``: HxW bool for one object, or NxHxW for N
objects followed together in one session (one SAM2 video segmenter with a mask per object, one
point tracker), in the order every reply keeps.  Replies carry ``meta`` (JSON, with ``objects``,
each object's counts) and, for each object ``i``, ``delta_i`` (4x4: its motion since the init
frame, in the camera frame), ``mask_i`` (HxW bool, SAM2's current mask), ``live_uv_i`` (Kx2
float32, its visible tracked points), ``fit_uv_i`` (Mx2 float32, the tracked points its pose was
fitted on), ``fit_inlier_i`` (M bool, which of them the fit kept) and ``model_i`` (its key points so
far, init frame).  Object 0 is also under the names without the index, with its counts at the top
of ``meta``, for a session that follows one object.  The first reply, before any request, is
``{"ready": true}`` once the models are on the GPU.

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


# ---------------------------------------------------------------------------------------------------------------------
# Speedups. Point2Pose's per-frame hot spots rewritten into the same floating-point operations in the same order, so the
# replies are bit for bit those of the unpatched pipeline (checked on recorded streams, one and two objects). Each
# applies only to the upstream code it was written against, pinned by the first 12 hex digits of its source's SHA-1;
# any other version is left as it is, and the bridge says so on stderr.


def _unchanged(fn, pin: str) -> bool:
    import hashlib
    import inspect

    if hashlib.sha1(inspect.getsource(fn).encode()).hexdigest().startswith(pin):
        return True
    print(
        f"[bridge] {fn.__qualname__} is not the version its speedup was written for; left as it is",
        file=sys.stderr,
    )
    return False


def _rewrite(owner, name: str, pin: str, edits, helpers=None) -> bool:
    """Recompile ``owner.name`` from its own source, in its own module, with lines swapped: each edit (first, stop, new)
    replaces the lines from the one starting with ``first`` up to the one starting with ``stop`` (kept) by ``new``,
    indented as the first line it replaces. ``helpers`` join the module's names."""
    import inspect
    import linecache
    import textwrap

    fn = getattr(owner, name)
    if not _unchanged(fn, pin):
        return False
    lines = textwrap.dedent(inspect.getsource(fn)).splitlines(keepends=True)
    for first, stop, new in edits:
        i = next(k for k, line in enumerate(lines) if line.strip().startswith(first))
        j = next(k for k in range(i + 1, len(lines)) if lines[k].strip().startswith(stop))
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        lines[i:j] = [textwrap.indent(textwrap.dedent(new).strip("\n") + "\n", indent)]
    source = "".join(lines)
    filename = f"<bridge speedup: {fn.__qualname__}>"
    # tracebacks show the rewritten lines
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    fn.__globals__.update(helpers or {})
    scope: dict = {}
    exec(compile(source, filename, "exec"), fn.__globals__, scope)
    scope[name].__qualname__ = fn.__qualname__
    setattr(owner, name, scope[name])
    return True


def _speed_up_sdf_refine() -> None:
    """The SDF refinement built its Jacobian in a Python loop over up to 1500 points, 8 times per object per frame
    (~30 ms an object); one stacked matmul gives the same rows, numpy running the loop's BLAS kernel per row. Its 7 SDF
    lookups per iteration (the value, and 6 offsets for the gradient) become one lookup of the stacked points, the
    lookup being elementwise per point."""
    import point2pose.modules.register.svd_cluster_ransac_register as rg

    cls = rg.SVDClusterRANSACRegister
    stacked_jacobian = """
        # J[i] = g[i] @ [I | -skew(p[i])], -skew(p) = [[-0, z, -y], [-z, -0, x], [y, -x, -0]] (its signed zeros too)
        G = np.zeros((n, 3, 6), dtype=np.float64)
        G[:, 0, 0] = G[:, 1, 1] = G[:, 2, 2] = 1.0
        G[:, 0, 3] = G[:, 1, 4] = G[:, 2, 5] = -0.0
        G[:, 0, 4], G[:, 0, 5] = pts_obj_in[:, 2], -pts_obj_in[:, 1]
        G[:, 1, 3], G[:, 1, 5] = -pts_obj_in[:, 2], pts_obj_in[:, 0]
        G[:, 2, 3], G[:, 2, 4] = pts_obj_in[:, 1], -pts_obj_in[:, 0]
        J = np.matmul(g_sdf_in[:, None, :], G)[:, 0, :]
    """
    _rewrite(
        cls,
        "_refine_pose_with_sdf",
        "69c576b2f8c6",
        [("J = np.zeros((n, 6), dtype=np.float64)", "w = self._kernel_weights(", stacked_jacobian)],
    )
    one_lookup = """
        offsets = [p for k in range(3) for p in (pts_obj + E[k][None, :], pts_obj - E[k][None, :])]
        vals, oks = self._query_sdf_signed(obj, np.concatenate([pts_obj, *offsets]))
        sdf0, v0 = vals[:n], oks[:n]
        grad = np.zeros((n, 3), dtype=np.float64)
        valid = v0.copy()
        for k in range(3):
            fp, vp = vals[(1 + 2 * k) * n : (2 + 2 * k) * n], oks[(1 + 2 * k) * n : (2 + 2 * k) * n]
            fm, vm = vals[(2 + 2 * k) * n : (3 + 2 * k) * n], oks[(2 + 2 * k) * n : (3 + 2 * k) * n]
    """
    if _unchanged(cls._query_sdf_signed, "750d39d4cc18"):
        _rewrite(
            cls,
            "_query_sdf_and_grad",
            "fb25ae67481f",
            [
                ("sdf0, v0 = self._query_sdf_signed(obj, pts_obj)", 'if getattr(obj, "sdf", None)', ""),
                ("grad = np.zeros((n, 3), dtype=np.float64)", "vk = v0 & vp & vm", one_lookup),
            ],
        )


def _speed_up_sdf_costs() -> None:
    """Within one SDF refinement the cost of the pose it stands at is evaluated again every iteration, and the seed's
    and the result's once more: about half of its cost evaluations. Each (pose, points) cost is kept for the
    refinement; the cost sees the pose and the points only as float32, which key it."""
    import hashlib

    import point2pose.modules.register.svd_cluster_ransac_register as rg

    cls = rg.SVDClusterRANSACRegister
    if not (
        _unchanged(cls._maybe_refine_with_sdf, "b0e55c8a2638")
        and _unchanged(cls._eval_sdf_cost, "953696563d46")
    ):
        return
    refine, cost = cls._maybe_refine_with_sdf, cls._eval_sdf_cost
    costs: dict | None = None  # during a refinement: (object, pose and points) -> cost

    def _maybe_refine_with_sdf(self, *args, **kwargs):
        nonlocal costs
        costs = {}
        try:
            return refine(self, *args, **kwargs)
        finally:
            costs = None

    def _eval_sdf_cost(self, pts_cur, pose, obj):
        if costs is None:
            return cost(self, pts_cur, pose, obj)
        seen = np.asarray(pose, dtype=np.float32).tobytes() + np.asarray(pts_cur, dtype=np.float32).tobytes()
        key = (id(obj), hashlib.sha1(seen).digest())
        if key not in costs:
            costs[key] = cost(self, pts_cur, pose, obj)
        return costs[key]

    cls._maybe_refine_with_sdf = _maybe_refine_with_sdf
    cls._eval_sdf_cost = _eval_sdf_cost


def _speed_up_ransac() -> None:
    """Cluster RANSAC fitted and scored its 100 hypotheses one Python iteration at a time (~5 ms a cluster). It now
    takes the loop's random draws in the same order, then fits and scores every hypothesis in stacked numpy calls that
    run the same kernels per hypothesis."""
    import point2pose.modules.register.svd_cluster_ransac_register as rg

    cls = rg.SVDClusterRANSACRegister

    def rotations(u, vt):
        rot = np.matmul(np.swapaxes(vt, 1, 2), np.swapaxes(u, 1, 2))
        flip = np.linalg.det(rot) < 0
        if flip.any():
            vt[flip, -1, :] *= -1
            rot[flip] = np.matmul(np.swapaxes(vt[flip], 1, 2), np.swapaxes(u[flip], 1, 2))
        return rot

    def fits(self, pa, qa, w):
        """_weighted_svd_fit (or _svd_fit, without weights) per sample of (K, s, 3) stacks; which converged."""
        if w is None:
            mu_p, mu_q = pa.mean(axis=1), qa.mean(axis=1)
            cov = np.matmul(np.swapaxes(pa - mu_p[:, None], 1, 2), qa - mu_q[:, None])
        else:
            wc = np.clip(np.asarray(w, dtype=float), 0.0, None)
            wn = wc / (np.sum(wc, axis=1) + 1e-12)[:, None]
            mu_p, mu_q = np.sum(pa * wn[:, :, None], axis=1), np.sum(qa * wn[:, :, None], axis=1)
            cov = np.matmul(np.swapaxes((pa - mu_p[:, None]) * wn[:, :, None], 1, 2), qa - mu_q[:, None])
        poses = np.tile(np.eye(4), (len(pa), 1, 1))
        ok = np.ones(len(pa), dtype=bool)
        try:
            u, _, vt = np.linalg.svd(cov)
        except np.linalg.LinAlgError:  # a sample did not converge: one by one, dropping it, as the loop did
            for k in range(len(pa)):
                try:
                    poses[k] = (
                        self._weighted_svd_fit(pa[k], qa[k], w[k])
                        if w is not None
                        else self._svd_fit(pa[k], qa[k])
                    )
                except np.linalg.LinAlgError:
                    ok[k] = False
            return poses, ok
        rot = rotations(u, vt)
        poses[:, :3, :3], poses[:, :3, 3] = rot, mu_q - np.matmul(rot, mu_p[:, :, None])[:, :, 0]
        return poses, ok

    def degenerate(pts, eps_area=1e-6):
        """_is_degenerate_sample per sample of a (K, s, 3) stack; the norm is the 1-D call's, a BLAS dot."""
        c = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        return np.sqrt(np.array([row.dot(row) for row in c])) < eps_area

    def best_hypothesis(self, p0, tgt_pcd, w, idx, samples):
        drawn = np.stack(samples)
        poses, ok = fits(self, p0[drawn], tgt_pcd[drawn], None if w is None else np.asarray(w)[drawn])
        if self._sample_size >= 3:
            ok &= ~degenerate(p0[drawn]) & ~degenerate(tgt_pcd[drawn])
        pts = p0[idx]
        moved = np.matmul(poses, np.c_[pts, np.ones((pts.shape[0], 1))].T)  # transform_pts, every hypothesis
        r = np.linalg.norm(np.swapaxes(moved, 1, 2)[:, :, :3] - tgt_pcd[idx][None], axis=2)
        inl = r <= self._inlier_thres
        ninl = inl.sum(axis=1)
        ok &= ninl >= self._min_inliers
        if not ok.any():
            return None, None, -1e18, 1e18
        k = int(np.argmax(np.where(ok, ninl, -1)))  # the first of the best supported, as the loop's ">"
        return poses[k].copy(), inl[k], int(ninl[k]), float(r[k][inl[k]].mean())

    batched = """
        # 1) RANSAC on remaining pool: the loop's draws in the same order, every hypothesis fitted and scored at once
        samples = [np.random.choice(idx, self._sample_size, replace=False) for _k in range(self._ransac_iters)]
        if samples:
            best_T, best_inl, best_score, best_mean = _bridge_best_hypothesis(self, p0, tgt_pcd, w, idx, samples)
    """
    if all(
        _unchanged(f, h)
        for f, h in (
            (cls._weighted_svd_fit, "95b38c8db186"),
            (cls._svd_fit, "49242e83d053"),
            (cls._is_degenerate_sample, "0a2a1e374ab4"),
            (rg.transform_pts, "4e03a2a286e1"),
        )
    ):
        _rewrite(
            cls,
            "_RANSAC",
            "528a022ce22d",
            [("# 1) RANSAC on remaining pool", "if best_T is None or best_inl is None:", batched)],
            helpers={"_bridge_best_hypothesis": best_hypothesis},
        )


def _speed_up_depth() -> None:
    """Each object's dense cloud was lifted from the depth image two or three times per frame (the register asks for
    the current frame's twice, and the previous frame's again); it is now kept for the frame. The lift also converted
    the whole depth image and computed a 5x5 window of depth statistics for every mask pixel, which only fill sampled
    depths that are not finite (there are none in the bridge's depth): now the depth is converted where it is sampled,
    and the window computed only when a sampled depth is missing."""
    import point2pose.modules.register.svd_cluster_ransac_register as rg
    import point2pose.utils.camera as cam

    orig_convert, orig_extract = cam.convert_pixel_to_world, cam.extract_cropped_point_cloud
    sampled = """
        # depth in meters, converted where it is sampled
        z = np.full(N, np.nan, dtype=np.float64)
        z[in_bounds] = (depth_image[ys[in_bounds], xs[in_bounds]].astype(np.float32) / float(depth_factor)).astype(
            np.float64
        )
    """
    window_when_needed = """
        if (fill_missing_depth or compute_depth_uncertainty) and (window_size < 1 or (window_size % 2) != 1):
            raise ValueError("window_size must be an odd integer >= 1")
        # The window only fills sampled depths that are not finite, or feeds the uncertainty.
        if compute_depth_uncertainty or (fill_missing_depth and np.any(in_bounds & ~np.isfinite(z))):
            D = depth_image.astype(np.float32) / float(depth_factor)
    """
    if _rewrite(
        cam,
        "convert_pixel_to_world",
        "b5b5e14864dc",
        [
            ("# depth in meters", "# helper: build window indices", sampled),
            (
                "if fill_missing_depth or compute_depth_uncertainty:",
                "half = window_size // 2",
                window_when_needed,
            ),
        ],
    ):
        import point2pose.pipeline.components.front_end as fe
        import point2pose.pipeline.components.key_frame_manager as kfm
        import point2pose.pipeline.modular_pipeline as mp

        for mod in (fe, kfm, mp):  # the modules that imported the name keep their own reference
            if getattr(mod, "convert_pixel_to_world", None) is orig_convert:
                mod.convert_pixel_to_world = cam.convert_pixel_to_world

    if not _unchanged(orig_extract, "86fdc14d9847"):
        return
    kept: dict = {}

    def extract_cropped_point_cloud(frame, obj_id, *args, **kwargs):
        key = (id(frame), obj_id, args, tuple(sorted(kwargs.items())))
        hit = kept.get(key)
        if hit is not None and hit[0] is frame and hit[1] is frame.mask and hit[2] is frame.depth:
            return hit[3]
        out = orig_extract(frame, obj_id, *args, **kwargs)
        out.flags.writeable = False  # shared by its callers, none of which writes to it
        while len(kept) >= 8:  # this frame's and the previous frame's, for a few objects
            kept.pop(next(iter(kept)))
        kept[key] = (frame, frame.mask, frame.depth, out)
        return out

    rg.extract_cropped_point_cloud = extract_cropped_point_cloud


def _speed_up_sam2_input() -> None:
    """SAM2 normalised each 1024x1024 frame in float64 on the CPU and uploaded it as float32 (~3 ms, and four times the
    bytes). The resize stays on the CPU; the division, cast and normalisation run on the GPU, each correctly rounded
    there as on the CPU."""
    import cv2
    import sam2.sam2_camera_predictor as scp
    import torch

    cls = scp.SAM2CameraPredictor
    if not _unchanged(cls.perpare_data, "a0a0ebeca8bc"):
        return
    orig = cls.perpare_data

    def perpare_data(
        self, img, image_size=1024, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)
    ):
        if not isinstance(img, np.ndarray):
            return orig(self, img, image_size, img_mean, img_std)
        height, width = img.shape[:2]
        x = (
            torch.from_numpy(np.ascontiguousarray(cv2.resize(img, (image_size, image_size)))).cuda().double()
            / 255.0
        )
        x = x.permute(2, 0, 1).float()
        x -= torch.tensor(img_mean, dtype=torch.float32, device=x.device)[:, None, None]
        x /= torch.tensor(img_std, dtype=torch.float32, device=x.device)[:, None, None]
        return x, width, height

    cls.perpare_data = perpare_data


def _speed_up_tapir() -> None:
    """TAPIR's step needs ~30 ms of CPU to launch thousands of small kernels, and stopped ~540 times to wait for the GPU:
    each constant tapnet builds with torch.tensor(..., device=cuda), and each gather by an index it builds on the CPU,
    is a blocking host-to-device copy. tapnet's modules get a torch that keeps those constants on the device once built
    and builds those indices there: the same values for every kernel. And the step runs in a worker thread on its own
    CUDA stream while SAM2's step, which keeps the GPU busy but needs little CPU, is queued on the main thread; the
    worker takes the main thread's autocast and grad modes (both are per thread), and the streams are joined both ways
    around it."""
    import concurrent.futures
    import hashlib
    import types

    import point2pose.modules.tracker.tapir_tracker as tt
    import point2pose.pipeline.components.front_end as fe
    import torch
    from tapnet.torch import tapir_model, utils

    if all(
        hashlib.sha1(pathlib.Path(mod.__file__).read_bytes()).hexdigest().startswith(pin)
        for mod, pin in (
            (tapir_model, "5e0c43e585a7"),
            (utils, "d2a8b347f27a"),
        )
    ):

        class TorchWithoutWaits(types.ModuleType):
            def __init__(self):
                super().__init__("torch")
                self._kept: dict = {}

            def __getattr__(self, name):
                return getattr(torch, name)

            def tensor(self, data, *args, device=None, dtype=None, **kw):
                if device is None or args or kw or torch.device(device).type != "cuda":
                    return torch.tensor(data, *args, device=device, dtype=dtype, **kw)
                key = (repr(data), str(device), dtype)  # repr keeps 256 and 256.0 apart
                if key not in self._kept:
                    self._kept[key] = torch.tensor(data, device=device, dtype=dtype)
                return self._kept[key]  # tapnet only reads its constants

            def arange(self, *args, **kw):
                if "device" not in kw and "out" not in kw:
                    kw["device"] = torch.device("cuda", torch.cuda.current_device())
                return torch.arange(*args, **kw)

            # arange's partner in estimate_trajectories, so the two stay on one device
            def randperm(self, *args, **kw):
                if "device" not in kw and "out" not in kw:
                    kw["device"] = torch.device("cuda", torch.cuda.current_device())
                return torch.randperm(*args, **kw)

        tapir_model.torch = utils.torch = TorchWithoutWaits()
    else:
        print(
            "[bridge] tapnet is not the version its speedup was written for; left as it is", file=sys.stderr
        )

    orig_step = fe.FrontEnd.step
    # The front end must still start the segmenter, then call the tracker once with the frame.
    if not (_unchanged(orig_step, "6f03eee304c8") and _unchanged(tt.TapirTracker.track_once, "59564716958b")):
        return
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tapir")
    side_streams: dict = {}

    def step(self, frame, track_table, objects):
        tracker = self.tracker
        if not isinstance(tracker, tt.TapirTracker):
            return orig_step(self, frame, track_table, objects)
        dev = torch.cuda.current_device()
        side = side_streams.get(dev) or side_streams.setdefault(dev, torch.cuda.Stream(dev))
        main = torch.cuda.current_stream(dev)
        autocast_on, autocast_dtype = torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
        grad_on = torch.is_grad_enabled()
        side.wait_stream(main)  # query points the main thread added are in place first
        track = tracker.track_once

        def run():
            with (
                torch.cuda.stream(side),
                torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_on),
                torch.set_grad_enabled(grad_on),
            ):
                return track(frame)

        fut = pool.submit(run)

        def track_once(f):
            if f is not frame:
                raise RuntimeError("the front end tracked another frame than the one TAPIR started on")
            return fut.result()

        tracker.track_once = track_once  # this step only
        try:
            return orig_step(self, frame, track_table, objects)
        finally:
            del tracker.track_once
            concurrent.futures.wait([fut])
            main.wait_stream(side)

    fe.FrontEnd.step = step


def _tapir_query_chunk(cfg) -> None:
    """Optional, and NOT exact: `tracker.params.query_chunk_size` in the config makes TAPIR estimate its queries that
    many at a time instead of Point2Pose's fixed 64. Each chunk relaunches the whole refinement, so one chunk for every
    query saves a chunk's time per further 64 tracks, but the tracks, and the poses with them, move."""
    size = cfg.tracker.params.get("query_chunk_size")
    if not size:
        return
    from tapnet.torch import tapir_model

    orig = tapir_model.TAPIR.estimate_trajectories

    def estimate_trajectories(self, *a, **kw):
        kw["query_chunk_size"] = int(size)
        return orig(self, *a, **kw)

    tapir_model.TAPIR.estimate_trajectories = estimate_trajectories


def _reuse_models() -> bool:
    """One set of models per bridge process. Each init builds a new pipeline, and each pipeline loaded its models again
    from their checkpoints: the segmenter (SAM2), the point tracker (BootsTAPIR) and the keypoint detector (SuperPoint),
    seconds of a restart, and each build left part of the last one's GPU memory behind until the process exited. A
    pipeline now gets the models the first one loaded, and its own state around them: the tracker's queries and causal
    state live on the tracker object, the keypoint detector keeps none, and the segmenter's session state is replaced
    by its first frame; its frame counter, which the first frame leaves alone, is set back to the start. Applies only
    to the upstream code it was written against (pinned like the speedups); otherwise every pipeline loads its own,
    and the bridge says so. Post: True when the models are shared."""
    import lightglue
    import point2pose.modules.sampler.super_point_fps_sampler as sp
    import point2pose.modules.segmenter.sam2_real_time_segmenter as sg
    import point2pose.modules.tracker.tapir_tracker as tp
    from sam2.sam2_camera_predictor import SAM2CameraPredictor

    pinned = [
        (tp.TapirTracker.__init__, "259b763e5b4a"),
        (sg.Sam2RealTimeSegmenter.__init__, "bc7146d73a82"),
        (sg.Sam2RealTimeSegmenter.initialize, "517a79de05a5"),
        (SAM2CameraPredictor.__init__, "39d8f89c3e61"),
        (SAM2CameraPredictor.load_first_frame, "d70d22d595d4"),
        (sp.SuperPointFPSSampler.__init__, "c20be9f445b5"),
    ]
    if not all(_unchanged(fn, pin) for fn, pin in pinned):
        return False
    built: dict = {}

    def shared(key, make):
        if key not in built:
            built[key] = make()
        return built[key]

    tapir_init, tapnet, torch = tp.TapirTracker.__init__, tp.tapir_model, tp.torch

    class _LoadedTorch:
        """torch, but its load hands back the shared model's own weights: the constructor's checkpoint load then
        copies them onto themselves."""

        def __init__(self, model):
            self._model = model

        def load(self, *_a, **_kw):
            return self._model.state_dict()

        def __getattr__(self, name):
            return getattr(torch, name)

    def tracker(self, config):
        key = (
            "tapir",
            config.get("num_pips_iter", 4),
            config.get("checkpoint_path"),
            config.get("device", "cpu"),
        )
        model = built.get(key)
        if model is None:
            tapir_init(self, config)
            built[key] = self._model
            return
        build = tapnet.TAPIR
        tapnet.TAPIR, tp.torch = (lambda **_kw: model), _LoadedTorch(model)  # the constructor runs as written
        try:
            tapir_init(self, config)
        finally:
            tapnet.TAPIR, tp.torch = build, torch

    tp.TapirTracker.__init__ = tracker
    build_predictor = sg.build_sam2_camera_predictor

    def predictor(model_cfg, checkpoint, **kw):
        p = shared(
            ("sam2", model_cfg, checkpoint, tuple(sorted(kw.items()))),
            lambda: build_predictor(model_cfg, checkpoint, **kw),
        )
        p.condition_state, p.frame_idx = {}, 0
        return p

    sg.build_sam2_camera_predictor = predictor
    superpoint = lightglue.SuperPoint
    lightglue.SuperPoint = lambda **kw: shared(
        ("superpoint", tuple(sorted(kw.items()))), lambda: superpoint(**kw)
    )
    return True


def _speedups(cfg) -> None:
    _speed_up_sdf_refine()
    _speed_up_sdf_costs()
    _speed_up_ransac()
    _speed_up_depth()
    _speed_up_sam2_input()
    _speed_up_tapir()
    _tapir_query_chunk(cfg)


class Session:
    """One Point2Pose pipeline from one init frame, following one object per init mask; a new init starts a new
    pipeline."""

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
        masks = np.asarray(mask, dtype=bool)
        frame.mask = (masks[None] if masks.ndim == 2 else masks).astype(np.uint8)[:, None]  # (N, 1, H, W)
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
        self._k = frame.intrinsics
        table = self.pipe.track_table
        visible = np.asarray(table.visible, dtype=bool) if table.visible is not None else np.zeros(0, bool)
        uv = np.asarray(table.track_2d, dtype=np.float32).reshape(-1, 2)[: len(visible)]
        masks = None
        if frame.mask is not None:
            m = frame.mask
            masks = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m))[:, 0] > 0
        owned = getattr(table, "obj2track_map", None) or {}
        out: dict = {}
        objects = []
        for i, obj in enumerate(self.pipe.objects):
            # Its own tracks; a table that keeps no owners (one object) gives them all to it.
            idx = np.asarray(owned.get(i, []) if owned else np.arange(len(visible)), dtype=np.int64).reshape(
                -1
            )
            idx = idx[idx < len(visible)]
            fit_uv, fit_inlier, guard = self._fit(table, i)
            objects.append(
                {
                    "lost": bool(obj.lost),
                    "lost_streak": int(obj.lost_streak),
                    "n_tracks": int(len(idx)),
                    "n_visible": int(visible[idx].sum()),
                    "mean_residual_m": float(obj.mean_residual),
                    "n_model_points": int(len(obj.key_points)),
                    "fit_points": int(len(fit_uv)),
                    "fit_inliers": int(fit_inlier.sum()),
                    "jump_guard_rejected": bool(guard.get("rejected", False)),
                }
            )
            out[f"delta_{i}"] = np.asarray(obj.pose, dtype=np.float64)
            out[f"live_uv_{i}"] = uv[idx[visible[idx]]]
            # Every track it owns, in the table's order, which only grows: a track's index is its identity across
            # frames, for whoever groups points by their motion (showservo.groups).
            out[f"track_idx_{i}"] = idx
            out[f"track_uv_{i}"] = uv[idx]
            out[f"track_vis_{i}"] = visible[idx]
            valid = np.asarray(table.valid, dtype=bool) if table.valid is not None else np.zeros(0, bool)
            out[f"track_valid_{i}"] = valid[idx] if len(valid) >= len(visible) else np.ones(len(idx), bool)
            out[f"fit_uv_{i}"] = fit_uv
            out[f"fit_inlier_{i}"] = fit_inlier
            # Its model so far: every key point it has adopted, in the first (init) frame's coordinates.
            out[f"model_{i}"] = np.asarray(obj.key_points, dtype=np.float32).reshape(-1, 3)
            if masks is not None and i < len(masks):
                out[f"mask_{i}"] = masks[i]
        out["meta"] = json.dumps({"ok": True, "ms": ms, "objects": objects, **objects[0]})
        for key in ("delta", "live_uv", "fit_uv", "fit_inlier", "model", "mask"):
            if f"{key}_0" in out:
                out[key] = out[f"{key}_0"]
        return out

    def _fit(self, table, i: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """The points object ``i``'s pose was fitted on this frame, which of them the fit kept, and the jump guard's
        verdict."""
        empty = np.zeros((0, 2), np.float32), np.zeros(0, bool), {}
        fe = self._fe
        if fe is None:
            return empty
        stats = fe.reg_stats.get(i) or {}
        guard = stats.get("pose_jump_guard_info") or {}
        idx = np.asarray(fe.valid_indices.get(i, []), dtype=np.int64).reshape(-1)
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
    from point2pose.pipeline.modular_pipeline import ModularPipeline

    _speedups(cfg)
    shared = _reuse_models()
    if shared:  # the models load now, before the bridge says it is ready, and every init after reuses them
        ModularPipeline(cfg)

    session = Session(cfg)
    _write(wire_out, meta=json.dumps({"ready": True, "shared_models": shared}))
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
