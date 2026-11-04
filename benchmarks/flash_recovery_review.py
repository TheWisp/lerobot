"""Review complete recorded replay poses at their original timestamps.

The orange wire surface is the same initial measured surface in every frame.
It isolates pose mistakes from later changes to the reconstructed geometry.
Availability is reported mechanically; correctness still needs visual review
or independent labels, never inferred from the tracker returning a pose.
"""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.spatial import cKDTree

from benchmarks.flash_experiment_inputs import samples
from lerobot.policies.flash.surface_grounding import project
from lerobot.policies.flash.surface_observation import move, spatial_sample


def intervals(rows, role):
    result = []
    for row in rows:
        item = row["objects"][role]
        key = item["group_growing"] is not None
        if not result or result[-1]["available"] != key:
            result.append({"available": key, "start_s": row["time_s"], "first_frame": row["frame"]})
        result[-1].update(end_s=row["time_s"], last_frame=row["frame"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads((args.run / "case.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    rows = [json.loads(line) for line in (args.run / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == case["expected_frames"] == summary["frames"]
    assert [r["frame"] for r in rows] == list(range(len(rows)))
    args.output.mkdir(parents=True, exist_ok=False)
    reference = np.load(args.run / "reference_points.npz")[args.role]
    reference = reference[spatial_sample(reference, 500)]
    distance, neighbors = cKDTree(reference).query(reference, k=4)
    spacing = np.median(distance[:, 1:]) * 2
    edges = [
        (i, j)
        for i in range(len(reference))
        for d, j in zip(distance[i, 1:], neighbors[i, 1:], strict=True)
        if j > i and d < spacing
    ]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    times = np.array([r["time_s"] for r in rows])
    fps = 30
    playback = np.arange(int(np.ceil(times[-1] * fps)) + 1) / fps
    indices = np.searchsorted(times, playback, side="right") - 1
    counts = Counter(indices.tolist())
    still_indices = set(np.searchsorted(times, np.arange(0, times[-1], 1)).tolist()) | {len(rows) - 1}
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "960x408",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output / "gamepad-full-replay.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    thumbnails = []
    try:
        for (frame, masks), row in zip(samples(case), rows, strict=True):
            item = row["objects"][args.role]
            pose = item["group_growing"]
            rgb = frame.rgb.copy()
            if pose is not None:
                xyz = move(reference, np.asarray(pose))
                uv = project(xyz, frame.intrinsics)
                for i, j in edges:
                    if (
                        min(xyz[i, 2], xyz[j, 2]) > 0
                        and np.isfinite(uv[[i, j]]).all()
                        and np.abs(uv[[i, j]]).max() < 10000
                    ):
                        cv2.line(
                            rgb,
                            tuple(np.rint(uv[i]).astype(int)),
                            tuple(np.rint(uv[j]).astype(int)),
                            (255, 165, 40),
                            1,
                            cv2.LINE_AA,
                        )
            mask = masks[args.role]
            ys, xs = np.nonzero(mask)
            h, w = mask.shape
            if len(xs):
                side = min(max(max(np.ptp(xs), np.ptp(ys)) * 1.8 + 32, 140), min(h, w))
                cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                x, y = np.clip(cx - side / 2, 0, w - side), np.clip(cy - side / 2, 0, h - side)
                box = tuple(map(int, (x, y, x + side, y + side)))
            else:
                box = (0, 0, w, h)
            state = (
                "no pose"
                if pose is None
                else ("prediction only" if item["group_growing_status"] == "predicted" else "pose returned")
            )
            title = f"{row['time_s']:.2f}s  |  {state}"
            local = ImageOps.pad(Image.fromarray(rgb).crop(box), (360, 360), color="#171e29")
            panel = Image.new("RGB", (960, 408), "#171e29")
            panel.paste(Image.fromarray(frame.rgb).resize((480, 360)), (0, 48))
            panel.paste(local, (540, 48))
            ImageDraw.Draw(panel).text((14, 12), "Gamepad   " + title, font=font, fill="white")
            for _ in range(counts[row["frame"]]):
                encoder.stdin.write(panel.tobytes())
            if row["frame"] in still_indices:
                small = Image.new("RGB", (360, 402), "#171e29")
                small.paste(local, (0, 42))
                ImageDraw.Draw(small).text((8, 8), title, font=font, fill="white")
                small.save(args.output / f"{row['frame']:06d}.jpg", quality=92)
                thumbnails.append(small)
    finally:
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("video encoding failed")
    for start in range(0, len(thumbnails), 12):
        chunk = thumbnails[start : start + 12]
        grid = Image.new("RGB", (1080, 402 * ((len(chunk) + 2) // 3)), "#171e29")
        for i, image in enumerate(chunk):
            grid.paste(image, ((i % 3) * 360, (i // 3) * 402))
        grid.save(args.output / f"contact-{start // 12}.jpg", quality=92)
    result = {
        "run": str(args.run),
        "frames": len(rows),
        "duration_s": times[-1],
        "availability_intervals": intervals(rows, args.role),
        "status_counts": dict(Counter(r["objects"][args.role]["group_growing_status"] for r in rows)),
        "correctness": "Not classified automatically; no independent labels for this demo",
        "display": "Fixed initial measured surface; no held-last overlay when output pose is missing",
    }
    (args.output / "availability.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
