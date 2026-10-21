"""Run the pinned Point2Pose pipeline with current-frame external masks.

This is an offline, causal segmentation ablation. The initial mask can come
from the sealed recording or a supplied image; subsequent masks come from
Point2Pose's SAM2 or saved causal SAM3 results. The sealed bridge and
Point2Pose checkout are not edited.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np


def read_exact(stream, size: int) -> bytes:
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError("Point2Pose bridge closed its pipe")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def receive(stream) -> dict[str, np.ndarray]:
    size = struct.unpack(">I", read_exact(stream, 4))[0]
    with np.load(io.BytesIO(read_exact(stream, size)), allow_pickle=False) as response:
        return {key: response[key] for key in response.files}


def request(stream, **fields) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    payload = buffer.getvalue()
    stream.write(struct.pack(">I", len(payload)) + payload)
    stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--current-masks", "--sam3-masks", dest="current_masks", type=Path)
    initial_mask = parser.add_mutually_exclusive_group(required=True)
    initial_mask.add_argument("--sealed-masks", type=Path)
    initial_mask.add_argument("--initial-mask", type=Path)
    parser.add_argument("--role", required=True, choices=("dowel", "ring"))
    parser.add_argument("--mask-source", choices=("external", "internal"), default="external")
    parser.add_argument("--label", default="sam3")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--diagnostic-frames", type=int, nargs="*", default=())
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--sealed-bridge", required=True, type=Path)
    parser.add_argument("--worker-bridge", type=Path)
    parser.add_argument("--worker-diagnostic", type=Path)
    parser.add_argument("--p2p-repo", required=True, type=Path)
    parser.add_argument("--p2p-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-frame", type=int, default=100)
    parser.add_argument("--last-frame", type=int, default=550)
    args = parser.parse_args()
    if args.last_frame < args.first_frame:
        parser.error("last frame precedes first frame")
    if args.mask_source == "external" and args.current_masks is None:
        parser.error("external mask source needs --current-masks")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / f"p2p_{args.role}_{args.label}_worker.log"
    bridge_script = (
        (args.worker_bridge or args.sealed_bridge)
        if args.mask_source == "internal"
        else Path(__file__).with_name("p2p_external_mask_bridge.py")
    )
    saved_masks = args.output / f"p2p_{args.role}_{args.label}_masks" / args.role
    if args.save_masks:
        saved_masks.mkdir(parents=True, exist_ok=True)
    diagnostics = args.output / f"p2p_{args.role}_{args.label}_diagnostics"
    if args.diagnostic_frames:
        diagnostics.mkdir(exist_ok=True)
    process = None
    records = []
    try:
        with log_path.open("wb") as log:
            command = [str(args.python), str(bridge_script)]
            if args.mask_source == "external" or args.worker_bridge:
                command.extend(("--sealed-bridge", str(args.sealed_bridge)))
            if args.worker_diagnostic:
                command.extend(("--diagnostic", str(args.worker_diagnostic)))
            command.extend(("--repo", str(args.p2p_repo), "--config", str(args.p2p_config)))
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                env=os.environ.copy(),
            )
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("Point2Pose bridge pipes unavailable")
            ready = json.loads(str(receive(process.stdout)["meta"]))
            if not ready.get("ready"):
                raise RuntimeError(f"Point2Pose bridge not ready: {ready}")
            intrinsics = np.loadtxt(args.dataset / "cam_K.txt").reshape(3, 3)
            if args.initial_mask is not None:
                initial = cv2.imread(str(args.initial_mask), cv2.IMREAD_GRAYSCALE)
                if initial is None:
                    raise FileNotFoundError(f"missing initial mask: {args.initial_mask}")
            else:
                with np.load(args.sealed_masks, allow_pickle=False) as initial_masks:
                    mask_index = 0 if args.role == "dowel" else 1
                    initial = initial_masks[f"o{mask_index}_masks"][args.first_frame].astype(np.uint8)
            for frame in range(args.first_frame, args.last_frame + 1):
                stem = f"{frame:06d}"
                bgr = cv2.imread(str(args.dataset / "rgb" / f"{stem}.jpg"))
                depth = cv2.imread(str(args.dataset / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
                if bgr is None or depth is None:
                    raise FileNotFoundError(f"missing RGB-D frame {frame}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if frame == args.first_frame:
                    mask = (
                        cv2.resize(initial, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    )
                    request(
                        process.stdin,
                        kind="init",
                        rgb=rgb,
                        depth=depth.astype(np.float32) / 1000,
                        K=intrinsics,
                        mask=mask,
                    )
                else:
                    fields = {"kind": "step", "rgb": rgb, "depth": depth.astype(np.float32) / 1000}
                    if args.mask_source == "external":
                        mask_image = cv2.imread(
                            str(args.current_masks / args.role / f"{stem}.png"), cv2.IMREAD_GRAYSCALE
                        )
                        if mask_image is None or mask_image.shape != depth.shape:
                            raise ValueError(f"missing or misaligned external mask at frame {frame}")
                        fields["mask"] = mask_image > 0
                    request(process.stdin, **fields)
                reply = receive(process.stdout)
                metadata = json.loads(str(reply["meta"]))
                if not metadata.get("ok"):
                    raise RuntimeError(f"Point2Pose frame {frame}: {metadata}")
                if args.save_masks:
                    cv2.imwrite(str(saved_masks / f"{stem}.png"), reply["mask"].astype(np.uint8) * 255)
                if frame in args.diagnostic_frames:
                    np.savez_compressed(
                        diagnostics / f"{stem}.npz",
                        **{
                            key: reply[key]
                            for key in (
                                "delta",
                                "mask",
                                "fit_uv",
                                "fit_inlier",
                                "track_idx_0",
                                "track_uv_0",
                                "track_vis_0",
                                "track_valid_0",
                            )
                        },
                    )
                tracked = metadata["objects"][0]
                records.append(
                    (
                        frame,
                        np.asarray(reply["delta"], dtype=np.float64),
                        bool(tracked["lost"]),
                        int(tracked["mask_pixels"] or 0),
                        int(tracked["fit_points"]),
                        int(tracked["fit_inliers"]),
                        float(metadata["ms"]),
                        int(tracked.get("fit_support", {}).get("depth_valid", -1)),
                    )
                )
                if frame % 25 == 0:
                    print(frame, "lost", tracked["lost"], "inliers", tracked["fit_inliers"], flush=True)
        output = args.output / f"p2p_{args.role}_{args.label}_{args.first_frame}_{args.last_frame}.npz"
        np.savez_compressed(
            output,
            frames=np.asarray([item[0] for item in records]),
            poses=np.stack([item[1] for item in records]),
            lost=np.asarray([item[2] for item in records]),
            mask_pixels=np.asarray([item[3] for item in records]),
            fit_points=np.asarray([item[4] for item in records]),
            fit_inliers=np.asarray([item[5] for item in records]),
            ms=np.asarray([item[6] for item in records]),
            fit_depth_valid=np.asarray([item[7] for item in records]),
        )
        print(output)
    finally:
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
