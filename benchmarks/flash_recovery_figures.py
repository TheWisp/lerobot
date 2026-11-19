"""Plot actual camera evidence and measured replay geometry; no generated imagery."""

import argparse
import colorsys
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.flash_motion_recovery import read_recorded_frame
from lerobot.policies.flash.surface_observation import move

YELLOW, AMBER, CYAN, PINK = "#ffdf55", "#ff9f43", "#3de1f0", "#ff629e"


def main(recording, evidence, output):
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((evidence / "figure-evidence.json").read_text())
    data = np.load(evidence / "figure-evidence.npz")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "#101922",
            "text.color": "#eaf0f6",
            "axes.titlecolor": "#eaf0f6",
            "savefig.facecolor": "#101922",
        }
    )

    def panel(ax, n, title, crop=None):
        if f"rgb_{n}" in data:
            rgb, mask = data[f"rgb_{n}"], data[f"mask_{n}"]
        else:
            frame, masks = read_recorded_frame(recording, n)
            rgb, mask = frame.rgb, masks["object"]
        ax.imshow(rgb)
        if mask.any():
            ax.contour(mask, levels=[0.5], colors=[YELLOW], linewidths=1.3)
        ax.set_title(title, fontsize=12, pad=12)
        if crop:
            x0, y0, x1, y1 = crop
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
        ax.axis("off")

    def projected(n, pose, points=None):
        points = data[f"reference_{n}"] if points is None else points
        xyz = move(points, pose)
        k = data[f"k_{n}"]
        uv = xyz @ k.T
        return uv[:, :2] / uv[:, 2:]

    def dots(ax, n, key, color, label):
        uv = projected(n, data[f"{key}_{n}"])
        ax.scatter(*uv.T, s=8, color=color, alpha=0.55, label=label)

    def axes(ax, n, key):
        points = np.array(records[str(n)]["saved"]["objects"]["object"]["axis_points"])
        uv = projected(n, data[f"{key}_{n}"], points)
        for p, color in zip(uv[1:], ["#ff6969", "#6fe08a", "#75a8ff"], strict=True):
            ax.plot(*np.stack((uv[0], p)).T, color=color, lw=2.5)

    def figure(title, captions, name, draw, count=2):
        fig, axs = plt.subplots(1, count, figsize=(14, 6.8), squeeze=False)
        fig.suptitle(title, x=0.035, ha="left", fontsize=18, fontweight="bold", y=0.97)
        draw(axs[0])
        fig.subplots_adjust(top=0.86, bottom=0.23, left=0.03, right=0.97, wspace=0.045)
        for i, text in enumerate(captions):
            fig.text(0.035, 0.17 - i * 0.047, text, fontsize=11, va="top")
        fig.savefig(output / name, dpi=130)
        plt.close(fig)

    def one(ax):
        panel(ax[0], 5372, "Last accepted object image · frame 5372", (185, 240, 410, 380))
        panel(ax[1], 5512, "Current image · frame 5512", (135, 240, 360, 380))
        pair = records["5512"]["matches"]["objects"]["object"]["pair"]
        uv = np.array(pair["target_uv"])
        s = np.array(pair["stage"])
        uv = uv[s >= 1]
        ax[1].scatter(*uv.T, s=22, facecolors="none", edgecolors=CYAN, lw=1.1)

    figure(
        "1 · Visible again, but the old point tracks are gone",
        [
            "Yellow: saved SAM mask. Cyan circles: surviving short tracks from the immediately preceding processed image.",
            "The object estimator still refers to frame 5372. Zero point track IDs connect that accepted image to frame 5512.",
            "Confirmed in the trace/code: no fresh reference-to-current matching is attempted. Pictures alone cannot prove that omission.",
        ],
        "01-lost-identities.png",
        one,
    )

    def two(ax):
        for a in ax:
            panel(a, 2741, "", (145, 175, 315, 315))
        ax[0].set_title("Original starting pose · surface fitting fails", fontsize=12)
        ax[1].set_title("Another starting pose · geometry-compatible fit", fontsize=12)
        dots(ax[0], 2741, "seed", AMBER, "Original starting pose")
        dots(ax[1], 2741, "alternative_center", CYAN, "Alternative fitted surface")

    figure(
        "2 · SAM found the cube; surface fitting started too far away",
        [
            "Both panels show the SAME frame 2741 and the SAME mask/depth. Yellow outlines the selected cube.",
            "Orange dots: stored surface at the original starting pose. Cyan dots: fitted surface after changing the starting pose.",
            "Changing the starting pose helps this fit. Passing geometry checks does not prove the pose or the selected object is correct.",
        ],
        "02-local-alignment.png",
        two,
    )

    def three(ax):
        panel(ax[0], 5381, "The scene group is still tracked")
        s = records["5381"]["saved"]
        group = s["objects"]["object"]["group_tracks"]["group_id"]
        for g in s["scene_group_tracks"]:
            if g["id"] != group:
                continue
            uv = np.array(g.get("pixels", []) + g.get("predicted_pixels", []))
            color = colorsys.hls_to_rgb(((group * 137.508 + 278) % 360) / 360, 0.68, 0.92)
            for key, alpha in [("links", 0.75), ("predicted_links", 0.35)]:
                for a, b in g.get(key, []):
                    ax[0].plot(*uv[[a, b]].T, color=color, alpha=alpha, lw=0.7)
        panel(ax[1], 5381, "But its carried object location is stale", (240, 205, 390, 365))
        dots(ax[1], 5381, "seed", AMBER, "Group-supplied object pose")
        dots(ax[1], 5381, "alternative_prior", CYAN, "Fit from prior pose")

    figure(
        "3 · Background tracking does not establish the moved object’s location",
        [
            "Frame 5381: associated group has 425 measured environment points and 0 measured points in the usable object mask.",
            "Its 15 stored object points have no current measurements. Their displayed locations are predicted from the group.",
            "Zoom: orange = group-supplied object pose; cyan = compatible fit from the prior pose; yellow = current SAM mask.",
        ],
        "03-stale-group.png",
        three,
    )

    def four(ax):
        for a in ax:
            panel(a, 4759, "", (275, 280, 375, 380))
        ax[0].set_title("New surface fit passes geometry checks", fontsize=12)
        ax[1].set_title("After Kalman fusion: rejected", fontsize=12)
        dots(ax[0], 4759, "fit", CYAN, "Fitted surface")
        axes(ax[0], 4759, "fit")
        dots(ax[1], 4759, "fused", PINK, "Fused surface")
        axes(ax[1], 4759, "fused")

    figure(
        "4 · Combining a usable fit with the old prediction spoils the update",
        [
            "Same frame 4759. Cyan: accepted-by-geometry candidate. Pink: fused candidate. RGB lines are the same object axes.",
            "Fusion changes orientation by 8.42° relative to the fit. The fused pose fails the check against retained frame 4750.",
            "That depth inconsistency is not reliably visible in RGB alone. The tracker rolls back and publishes no measured pose.",
        ],
        "04-filter-rejection.png",
        four,
    )

    def five(ax):
        pair = records["5381"]["matches"]["objects"]["object"]["pair"]
        st = np.array(pair["stage"])
        for a, n, uvkey, title in zip(
            ax,
            [5372, 5381],
            ["source_uv", "target_uv"],
            ["Source · frame 5372", "Target · frame 5381"],
            strict=True,
        ):
            panel(a, n, title, (260, 230, 390, 355))
            uv = np.array(pair[uvkey])
            a.scatter(*uv[(st >= 1) & (st < 3)].T, s=60, marker="x", c=PINK, lw=2)
            a.scatter(*uv[st >= 3].T, s=65, facecolors="none", edgecolors=CYAN, lw=2)

    figure(
        "5 · A surviving 2D match is not necessarily a usable 3D pair",
        [
            "Frame pair 5372 → 5381: 17 object queries; 12 survive LK; only 3 pass the endpoint depth/support filters.",
            "Cyan circles: usable 3D pairs. Pink crosses: survived image tracking but failed a depth/support requirement.",
            "The rigid-motion fit needs at least 6 pairs. There is visible object depth, but too few acceptable matched endpoints.",
        ],
        "05-depth-pairs.png",
        five,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("recording", type=Path)
    p.add_argument("evidence", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.recording, a.evidence, a.output)
