"""Reproduce the endpoint-filter and point-density measurements in the recovery report."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_motion_recovery import read_recorded_frame
from lerobot.policies.flash.shared_image_tracking import (
    SharedFrameTracker,
    SharedMatchingConfig,
    depth_support_mask,
    pixels_inside,
)
from lerobot.policies.flash.surface_observation import observation


def plot_filters(frames, report, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    fig.patch.set_facecolor("#101922")
    for ax, n, key in zip(axes, [5372, 5381], ["source_uv", "target_uv"], strict=True):
        frame, m = frames[n]
        mask = m["object"]
        obs = observation(frame.depth, mask, frame.intrinsics, report["observation_settings"])
        margin = report["observation_settings"]["mask_margin_px"]
        interior = cv2.erode(mask.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
        ax.imshow(frame.rgb)
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask & ~interior] = [1, 0.7, 0.2, 0.5]
        rgba[obs["valid_image"]] = [0.2, 0.9, 0.6, 0.3]
        ax.imshow(rgba)
        ax.contour(mask, levels=[0.5], colors=["#ffdf55"], linewidths=1.3)
        for st, style in [
            (2, {"marker": "x", "color": "#ff629e", "linewidths": 2.5}),
            (3, {"marker": "o", "edgecolors": "#3de1f0", "facecolors": "none", "linewidths": 2.5}),
        ]:
            pts = np.array([r[key] for r in report["pairs"] if r["stage"] == st])
            ax.scatter(pts[:, 0], pts[:, 1], s=90, **style)
        x, y, w, h = report["endpoints"][str(n)]["bbox_xywh"]
        pad = 8
        ax.set_xlim(x - pad, x + w + pad)
        ax.set_ylim(y + h + pad, y - pad)
        ax.axis("off")
        ax.set_title(f"Frame {n} · 12 LK survivors", color="#eaf0f6", fontsize=14, pad=10)
    fig.suptitle("Why nine matches were excluded", color="#eaf0f6", fontsize=19, x=0.05, ha="left", y=0.98)
    fig.subplots_adjust(top=0.85, bottom=0.23, left=0.04, right=0.96, wspace=0.12)
    handles = [
        Patch(facecolor="#d99933", label="Excluded 4-pixel boundary band"),
        Patch(facecolor="#40ac79", label="Usable surface"),
        Line2D([], [], marker="x", color="#ff629e", linestyle="None", label="9 excluded pairs"),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            color="#3de1f0",
            linestyle="None",
            label="3 usable pairs",
        ),
    ]
    legend = fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=2, frameon=False, fontsize=11
    )
    for t in legend.get_texts():
        t.set_color("#eaf0f6")
    fig.text(
        0.05,
        0.025,
        "All 12 pairs have positive depth in both 3×3 neighborhoods. Yellow outline: saved SAM mask.",
        fontsize=11,
        color="#eaf0f6",
    )
    fig.savefig(destination, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(
        description="Audit filters and query density for recorded frames 5372 and 5381."
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    root = args.recording
    settings = json.loads((root / "config.json").read_text())["pipeline_settings"]
    config = settings["config"]["observation"]
    pair = json.loads((root / "0000005381.overlay.json").read_text())["image_matches"]["objects"]["object"][
        "pair"
    ]
    frames = {n: read_recorded_frame(root, n) for n in (5372, 5381)}
    report = {
        "recording": root.name,
        "scope": "Endpoint filters from saved model depth and SAM masks; shared tracker replay only. Timing excludes groups, surface fitting, SAM and depth inference; twelve repeats of frame 5381, OpenCV one thread. No tracking defaults changed.",
        "settings": settings["shared_matching"],
        "observation_settings": config,
        "endpoints": {},
        "pairs": [],
    }
    checks = {}
    observations = {}
    for n, uvkey in ((5372, "source_uv"), (5381, "target_uv")):
        f, masks = frames[n]
        mask = masks["object"]
        depth = f.depth
        obs = observation(depth, mask, f.intrinsics, config)
        connected = np.zeros(mask.shape, np.uint8)
        uv = obs["pixels"]
        connected[uv[:, 1], uv[:, 0]] = 1
        connected = (
            cv2.dilate(
                connected, np.ones((settings["config"]["motion"]["connected_dilation_px"],) * 2, np.uint8)
            )
            > 0
        )
        pix = np.rint(pair[uvkey]).astype(int)
        x, y = pix.T
        radius = config["mask_margin_px"]
        interior = cv2.erode(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)) > 0
        neighborhood = np.ones((config["depth_neighborhood_px"],) * 2, np.uint8)
        span = cv2.dilate(depth, neighborhood) - cv2.erode(depth, neighborhood)
        tests = {
            "mask": mask,
            "positive_3x3": depth_support_mask(depth, radius_px=1),
            "mask_interior": interior,
            "depth_span": obs["depth_reliable"],
            "depth_range": ~obs["depth_range_rejected"],
            "valid_surface": obs["valid_image"],
            "connected": connected,
            "usable": obs["valid_image"] & connected,
        }
        checks[n] = {key: value[y, x] for key, value in tests.items()}
        checks[n]["span_mm"] = span[y, x] * 1000
        checks[n]["pixel_depth_m"] = depth[y, x]
        observations[n] = {
            "frame_id": n,
            "depth": depth,
            "mask": mask,
            "usable": tests["usable"],
            "intrinsics": f.intrinsics,
        }
        report["endpoints"][str(n)] = {
            "mask_pixels": int(mask.sum()),
            "interior_pixels": int(interior.sum()),
            "valid_surface_pixels": int(obs["valid_image"].sum()),
            "sampled_surface_points": len(obs["points"]),
            "bbox_xywh": list(map(int, cv2.boundingRect(mask.astype(np.uint8)))),
        }
    for i, (tid, st) in enumerate(zip(pair["track_ids"], pair["stage"], strict=True)):
        row = {
            "track_id": tid,
            "stage": st,
            "source_uv": pair["source_uv"][i],
            "target_uv": pair["target_uv"][i],
        }
        for n in frames:
            row[str(n)] = {
                key: (float(value[i]) if key.endswith(("_mm", "_m")) else bool(value[i]))
                for key, value in checks[n].items()
            }
        report["pairs"].append(row)
    print("ENDPOINTS", json.dumps(report["endpoints"]), flush=True)
    for row in report["pairs"]:
        if row["stage"] >= 1:
            print(
                row["track_id"],
                row["stage"],
                [
                    (
                        n,
                        [k for k, v in row[str(n)].items() if isinstance(v, bool) and not v],
                        round(row[str(n)]["span_mm"], 2),
                    )
                    for n in frames
                ],
                flush=True,
            )
    # Reproduce point seeding and attrition through the real shared tracker, independent of object fitting.
    ids = []
    for p in sorted(root.glob("*.overlay.json")):
        s = json.loads(p.read_text())
        if s["objects"]["object"]["diagnostics"].get("reference_frame") == 5118 and s["frame"] <= 5381:
            ids.append(s["frame"])
    loaded = {n: frames.get(n) or read_recorded_frame(root, n) for n in ids}
    report["sampling_trials"] = []
    for spacing, budget in ((5, 256), (5, 512), (3, 256), (2, 256), (2, 512)):
        tracker = SharedFrameTracker(
            SharedMatchingConfig.from_dict(
                {**settings["shared_matching"], "spacing_px": spacing, "object_points": budget}
            )
        )
        history = []
        for n in ids:
            f, m = loaded[n]
            if n == 5381:
                before_tracker = deepcopy(tracker)
            before = tracker.next_id
            tracker.update(f, m)
            px, inside = pixels_inside(tracker.xy, f.depth.shape)
            own = inside & m["object"][px[:, 1], px[:, 0]]
            if n >= 5340:
                history.append(
                    {
                        "frame": n,
                        "object_tracks": int(own.sum()),
                        "new_object_tracks": int((own & (tracker.ids >= before)).sum()),
                        "total_tracks": len(tracker.ids),
                        "matching_ms": tracker.diagnostics["matching_ms"],
                        "total_ms": tracker.diagnostics["total_ms"],
                    }
                )
            if n == 5372:
                source_ids = tracker.ids[own].copy()
        if spacing == 5 and budget == 256:
            report["source_track_ids"] = source_ids.tolist()
            report["point_history"] = history
        evidence = {}
        pairs = tracker.pairs(
            observations[5372], observations[5381], frames[5372][0].intrinsics, evidence=evidence
        )
        timing = []
        for _ in range(12):
            trial = deepcopy(before_tracker)
            trial.update(*loaded[5381])
            timing.append([trial.diagnostics["matching_ms"], trial.diagnostics["total_ms"]])
        report["sampling_trials"].append(
            {
                "spacing_px": spacing,
                "object_points": budget,
                "last_frames": history[-2:],
                "usable_pairs": len(pairs[0]),
                "lk_survivors": int((evidence["stage"] >= 1).sum()),
                "median_matching_ms": float(np.median(timing, axis=0)[0]),
                "median_shared_update_ms": float(np.median(timing, axis=0)[1]),
                "timing_repeats": len(timing),
            }
        )
    print("SAMPLING", json.dumps(report["sampling_trials"]), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.figure:
        plot_filters(frames, report, args.figure)


if __name__ == "__main__":
    main()
