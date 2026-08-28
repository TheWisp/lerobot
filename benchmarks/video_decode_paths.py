#!/usr/bin/env python
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

"""Random-access video decode: codec x bitrate x resolution, CPU against NVDEC.

Answers a question the training-side A/B cannot: for one frame, decoded at a
random index, how much does the codec, the bitrate and the resolution cost on
each path. The training benchmark measures whole steps and so folds decode in
with resize, collation and the model; this isolates decode.

Four things here are deliberate, because getting each of them wrong produces a
plausible number that favours one side:

* **Real footage.** Synthetic gradients compress to almost nothing, so a
  benchmark built on them measures near-empty streams. Frames come from a real
  recording, and every codec encodes the *same* frames, so content is held
  constant across the comparison.
* **Decoders are opened once.** Building one per call measures construction
  amortised over a handful of frames, which is a different quantity and flatters
  whichever side builds more cheaply. `GpuFrameSource` holds its decoders open;
  so does this.
* **Random access.** The sampler draws chunk starts scattered through a file.
  Sequential throughput would flatter both decoders equally and answer a
  question nobody is asking.
* **Processes, not threads, for the CPU path.** That is what a DataLoader runs.
  Threads share a GIL and an allocator and would understate it.

Two ordering constraints are load-bearing rather than stylistic. Every CPU
measurement runs before any CUDA call, because a `spawn` pool created after the
parent has initialised CUDA segfaults. And each GPU measurement runs in its own
subprocess, because NVDEC decoder contexts accumulate across clips within one
interpreter until the process dies.

CRF is not comparable *across* codecs -- AV1 at CRF 30 is a different quality
point from H.264 at CRF 30 -- so read each row's CPU-vs-GPU comparison, and read
the codec column as "what this codec costs at its own CRF 30", not as a quality-
matched ranking.

Usage:
    python benchmarks/video_decode_paths.py --source <repo_id or path to mp4>
    python benchmarks/video_decode_paths.py --source <...> --json out.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

CODECS = [("h264", "h264"), ("hevc", "hevc"), ("libsvtav1", "av1")]
DEFAULT_CRFS = (20, 30)
DEFAULT_SIZES = ((640, 480), (1280, 720))


def _cpu_slice(args):
    """One worker process: open a decoder, decode its indices, return elapsed."""
    path, idx = args
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(str(path), device="cpu")
    dec[int(idx[0])]  # warm
    t = time.perf_counter()
    for i in idx:
        dec[int(i)]
    return time.perf_counter() - t


def _cpu_parallel(path, idx, n, reps, ctx):
    """Wall clock across n worker processes -- the amortised cost, not a sum."""
    parts = [p for p in np.array_split(idx, n) if len(p)]
    with ctx.Pool(len(parts)) as pool:
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            pool.map(_cpu_slice, [(path, p) for p in parts])
            times.append(time.perf_counter() - t)
    return statistics.median(times)


def _gpu_subprocess(path, n, n_frames, n_idx, reps):
    """Time NVDEC decode in a fresh interpreter; None if it dies.

    Isolated because NVDEC contexts accumulate across clips and eventually take
    the process down. A crash then costs one cell instead of the matrix -- which
    is also how the HEVC last-frame segfault stayed legible rather than
    destroying a whole run.
    """
    code = (
        "import json,statistics,sys,time\n"
        "import numpy as np, torch, PyNvVideoCodec\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        f"path,n,n_frames,n_idx,reps = {str(path)!r},{n},{n_frames},{n_idx},{reps}\n"
        "rng = np.random.default_rng(0)\n"
        "idx = np.sort(rng.choice(n_frames, min(n_idx, n_frames), replace=False)).astype(np.int64)\n"
        "parts = [p for p in np.array_split(idx, n) if len(p)]\n"
        "decs = [PyNvVideoCodec.SimpleDecoder(path, use_device_memory=True) for _ in parts]\n"
        "for d,p in zip(decs,parts): torch.from_dlpack(d[int(p[0])]).clone()\n"
        "def one(a):\n"
        "    d,p = a\n"
        "    for i in p: torch.from_dlpack(d[int(i)]).clone()\n"
        "ex = ThreadPoolExecutor(len(parts)); ts=[]\n"
        "for _ in range(reps):\n"
        "    torch.cuda.synchronize(); t=time.perf_counter()\n"
        "    list(ex.map(one, zip(decs,parts)))\n"
        "    torch.cuda.synchronize(); ts.append(time.perf_counter()-t)\n"
        "ex.shutdown()\n"
        "print(json.dumps({'ms': statistics.median(ts)/len(idx)*1e3}))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])["ms"]
    except (ValueError, KeyError, IndexError):
        return None


def _source_video(source: str) -> pathlib.Path:
    """A real mp4 to take frames from: a dataset repo_id or a path."""
    p = pathlib.Path(source)
    if p.suffix == ".mp4" and p.exists():
        return p
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(source)
    clips = sorted(pathlib.Path(ds.root).rglob("*.mp4"))
    if not clips:
        raise SystemExit(f"{source} has no video files to benchmark")
    return clips[0]


def build_clips(src_mp4, tmp, n_frames, sizes, crfs):
    """Encode the same real frames at every (size, crf, codec) point."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.video_utils import encode_video_frames

    clips = []
    for w, h in sizes:
        frames = tmp / f"frames-{w}x{h}"
        frames.mkdir()
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(src_mp4),
                "-frames:v",
                str(n_frames),
                "-vf",
                f"scale={w}:{h}",
                str(frames / "frame-%06d.png"),
            ],
            check=True,
        )
        got = len(list(frames.glob("*.png")))
        for crf in crfs:
            for vcodec, label in CODECS:
                out = tmp / f"{label}-{w}x{h}-crf{crf}.mp4"
                encode_video_frames(
                    frames,
                    out,
                    fps=30,
                    video_encoder=RGBEncoderConfig(vcodec=vcodec, g=2, crf=crf),
                )
                mb = out.stat().st_size / 1e6
                clips.append(
                    {
                        "codec": label,
                        "crf": crf,
                        "w": w,
                        "h": h,
                        "frames": got,
                        "path": out,
                        "mb": mb,
                        "kbps": mb * 8000 / (got / 30),
                    }
                )
                print(f"  encoded {label} {w}x{h} crf{crf}: {mb:.2f} MB", flush=True)
    return clips


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="dataset repo_id, or a path to an .mp4")
    ap.add_argument("--frames", type=int, default=200, help="frames taken from the source clip")
    ap.add_argument("--indices", type=int, default=256, help="random indices decoded per measurement")
    ap.add_argument("--reps", type=int, default=5, help="repetitions; the median is reported")
    ap.add_argument("--cpu-workers", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--gpu-decoders", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--json", type=str, default=None, help="also write the rows here")
    args = ap.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="decode-bench-"))
    src = _source_video(args.source)
    print(f"  source: {src.name}", flush=True)
    clips = build_clips(src, tmp, args.frames, DEFAULT_SIZES, DEFAULT_CRFS)

    rng = np.random.default_rng(0)
    ctx = mp.get_context("spawn")

    # Phase 1: every CPU measurement, before CUDA is touched anywhere.
    for c in clips:
        idx = np.sort(rng.choice(c["frames"], min(args.indices, c["frames"]), replace=False))
        c["idx"] = idx.astype(np.int64)
        for n in args.cpu_workers:
            c[f"cpu{n}"] = _cpu_parallel(c["path"], c["idx"], n, args.reps, ctx) / len(idx) * 1e3
        print(f"  cpu {c['codec']} {c['w']}x{c['h']} crf{c['crf']}: done", flush=True)

    # Phase 2: CUDA, each cell in its own process.
    for c in clips:
        for n in args.gpu_decoders:
            c[f"gpu{n}"] = _gpu_subprocess(c["path"], n, c["frames"], args.indices, args.reps)
        print(f"  gpu {c['codec']} {c['w']}x{c['h']} crf{c['crf']}: done", flush=True)

    cpu_cols = " | ".join(f"CPU x{n}" for n in args.cpu_workers)
    gpu_cols = " | ".join(f"GPU x{n}" for n in args.gpu_decoders)
    print(f"\n| res | codec | CRF | MB | kb/s | {cpu_cols} | {gpu_cols} |")
    print("|" + "---|" * (5 + len(args.cpu_workers) + len(args.gpu_decoders)))
    for c in clips:
        cells = [f"{c[f'cpu{n}']:.2f}" for n in args.cpu_workers]
        cells += ["—" if c[f"gpu{n}"] is None else f"{c[f'gpu{n}']:.2f}" for n in args.gpu_decoders]
        print(
            f"| {c['w']}x{c['h']} | {c['codec']} | {c['crf']} | {c['mb']:.2f} | "
            f"{c['kbps']:.0f} | " + " | ".join(cells) + " |"
        )
    print(f"\n  per-frame ms, median of {args.reps}, {args.indices} random indices of {args.frames}, g=2")
    print("  CPU columns are worker processes; GPU columns are NVDEC decoders on one device")
    print("  a '—' is a decoder that died on that clip; see issue #166 for one known case")

    if args.json:
        for c in clips:
            c["path"] = str(c["path"])
            c["idx"] = int(len(c["idx"]))
        pathlib.Path(args.json).write_text(json.dumps(clips, indent=1) + "\n")
        print(f"  rows written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
