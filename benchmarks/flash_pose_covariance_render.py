"""Actual recorded RGB with identical surface overlays for every ablation."""

import argparse
import json
from fractions import Fraction
from itertools import islice
from pathlib import Path

import av
import cv2
import matplotlib
import numpy as np
from scipy.spatial import Delaunay

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.flash_experiment_inputs import samples
from lerobot.policies.flash.surface_grounding import project
from lerobot.policies.flash.surface_observation import initial_landmarks, move, spatial_sample
from tests.policies.flash.test_gamepad_stationary_integration import ANNOTATION, stationary_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {
        p.stem: json.loads(p.read_text())
        for p in args.input.glob("*-*.json")
        if not p.stem.endswith("summary")
    }
    annotation = json.loads(ANNOTATION.read_text())
    summaries = {name: r["summary"] for name, r in reports.items()}
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, backend in zip(axes, ("lk", "ufm"), strict=True):
        for variant, label, color in (
            ("baseline", "Current", "#c26718"),
            ("correlated2", "Covariance: 2 mm noise", "#287db6"),
            ("correlated5", "Covariance: 5 mm noise", "#a855a1"),
        ):
            r = reports[f"{backend}-{variant}"]
            _, angles = stationary_errors(r, annotation["stationary_last_frame"])
            ax.plot([x["time_s"] for x in r["rows"][: len(angles)]], angles, color=color, lw=1.8, label=label)
        ax.axhline(annotation["max_rotation_deg"], color="#555555", ls="--", lw=1)
        ax.set_title(f"{backend.upper()} + surrounding groups", fontsize=16)
        ax.set_xlabel("Time in recording (seconds)")
        ax.grid(alpha=0.18)
        ax.set_ylim(0, 41)
    axes[0].set_ylabel("False rotation before grasp (degrees)")
    axes[1].legend(loc="upper left", fontsize=10)
    fig.suptitle("Covariance helps only when its assumptions fit the evidence", fontsize=17)
    fig.text(
        0.5,
        0.01,
        "Human-annotated stationary interval; no independent 6D ground truth. Dashed line: unchanged 5° test limit.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(args.output / "pregrasp-comparison.png", dpi=140)
    plt.close(fig)

    variants = ("baseline", "correlated2", "correlated5")
    titles = ("Current", "Covariance / 2 mm", "Covariance / 5 mm")
    count = min(len(r["rows"]) for r in reports.values())
    selected_reports = [reports[f"{b}-{v}"] for b in ("lk", "ufm") for v in variants]
    reference = np.array(selected_reports[0]["reference"])
    for r in selected_reports:
        np.testing.assert_array_equal(np.array(r["reference"]), reference)
    ref = reference[spatial_sample(reference, 220)]
    anchors, basis = initial_landmarks(reference)
    length = np.linalg.norm(np.ptp(reference, axis=0)) / 3
    axes3d = np.vstack([anchors[0], anchors[0] + basis.T * length])
    # Same fixed camera crop for all methods and every frame, derived from the
    # initial measured object's bounds. It is visualization only.
    crop = None
    edges = None
    panel_w, panel_h, header, footer = 600, 500, 42, 44
    width, height = panel_w * 3, (panel_h + header) * 2 + footer
    origin_time = selected_reports[0]["rows"][0]["time_s"]
    times = np.array([r["time_s"] for r in selected_reports[0]["rows"]])
    rate = Fraction(1 / float(np.median(np.diff(times)))).limit_denominator(10000)
    with av.open(str(args.output / "comparison.mp4"), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = stream.time_base
        stream.options = {"crf": "20", "preset": "fast", "bf": "0"}
        for index, (frame, masks) in enumerate(islice(samples(annotation["case"]), count)):
            if crop is None:
                uv = project(ref, frame.intrinsics)
                center = (uv.min(0) + uv.max(0)) / 2
                low = np.maximum(np.floor(center - [125, 135]).astype(int), 0)
                crop = (slice(low[1], low[1] + 250), slice(low[0], low[0] + 300))
                triangle = Delaunay(uv).simplices
                edges = np.unique(
                    np.sort(
                        np.concatenate([triangle[:, [0, 1]], triangle[:, [1, 2]], triangle[:, [0, 2]]]),
                        axis=1,
                    ),
                    axis=0,
                )
                edges = edges[np.linalg.norm(ref[edges[:, 0]] - ref[edges[:, 1]], axis=1) < 0.012]
            output = np.full((height, width, 3), 21, dtype=np.uint8)
            for cell, r in enumerate(selected_reports):
                row = r["rows"][index]
                assert row["frame_id"] == frame.frame_id and row["time_s"] == frame.source_time_s
                picture = frame.rgb.copy()
                contours, _ = cv2.findContours(
                    masks["gamepad"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(picture, contours, -1, (50, 230, 220), 1, cv2.LINE_AA)
                pose = np.array(row["camera_from_reference"])
                uv = np.rint(project(move(ref, pose), frame.intrinsics)).astype(int)
                for a, b in edges:
                    cv2.line(picture, tuple(uv[a]), tuple(uv[b]), (255, 174, 54), 1, cv2.LINE_AA)
                uv = np.rint(project(move(axes3d, pose), frame.intrinsics)).astype(int)
                for end, color in zip(uv[1:], ((255, 65, 65), (90, 235, 95), (80, 140, 255)), strict=True):
                    cv2.line(picture, tuple(uv[0]), tuple(end), color, 2, cv2.LINE_AA)
                y, x = (cell // 3) * (panel_h + header), (cell % 3) * panel_w
                panel = cv2.resize(picture[crop], (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
                output[y + header : y + header + panel_h, x : x + panel_w] = panel
                label = f"{'LK' if cell < 3 else 'UFM'}: {titles[cell % 3]}"
                cv2.putText(
                    output,
                    label,
                    (x + 15, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                output,
                f"{frame.source_time_s - origin_time:.2f}s   |   Orange: same initial surface   Cyan: visible SAM mask   |   1x speed",
                (18, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            if index in (171, 215, 280):
                cv2.imwrite(str(args.output / f"frame-{index}.jpg"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            video_frame = av.VideoFrame.from_ndarray(output, format="rgb24")
            video_frame.pts = round((frame.source_time_s - origin_time) * 1_000_000)
            video_frame.time_base = stream.time_base
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
