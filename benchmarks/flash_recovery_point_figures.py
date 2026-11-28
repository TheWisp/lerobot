"""Plot actual saved camera images with audited LK tracks and group predictions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from benchmarks.flash_motion_recovery import read_recorded_frame


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot recorded LK measurements separately from group predictions."
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    recording, audit, output = args.recording, args.audit, args.output
    output.mkdir(parents=True, exist_ok=True)
    trace = json.loads(audit.read_text())
    points = trace["points"]
    known = set(map(int, points))
    current = [6, 591, 592]
    background = "#101922"
    foreground = "#edf2f7"
    amber = "#ff9f43"
    yellow = "#ffdf55"
    green = "#53f5bd"
    colors = {6: "#53f5bd", 591: "#69d8ff", 592: "#fca4ed"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "figure.facecolor": background,
            "savefig.facecolor": background,
            "text.color": foreground,
            "axes.titlecolor": foreground,
        }
    )
    stroke = [pe.withStroke(linewidth=3, foreground="#101922")]

    def frame(ax, n, limits):
        f, m = read_recorded_frame(recording, n)
        ax.imshow(f.rgb, interpolation="nearest")
        ax.contour(m["object"], levels=[0.5], colors=[yellow], linewidths=1.4)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[3], limits[2])
        ax.axis("off")
        return f, m

    def actual(i, n):
        return next(r for r in points[str(i)] if r["frame"] == n)

    def dot(ax, uv, color, label, offset=(7, 7), size=140):
        ax.scatter(*uv, s=size, facecolors="none", edgecolors=color, linewidths=2.7, zorder=5)
        ax.annotate(
            label,
            uv,
            xytext=offset,
            textcoords="offset points",
            color=color,
            fontsize=13,
            fontweight="bold",
            path_effects=stroke,
            zorder=6,
        )

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.6))
    fig.suptitle(
        "These 3 old points DO match between the frames",
        x=0.04,
        ha="left",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    for ax, n, title in zip(
        axs, [5372, 5381], ["Previous image · frame 5372", "Current image · frame 5381"], strict=True
    ):
        frame(ax, n, (265, 370, 235, 340))
        ax.set_title(title, fontsize=15, pad=12)
        for i in current:
            row = actual(i, n)
            assert row["active_2d_track"] and row["has_3d_measurement"] and row["lk_accepted"]
            offset = {6: (8, 7), 591: (-40, 3), 592: (8, -10)}[i]
            dot(ax, row["uv"], colors[i], str(i), offset)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.84, bottom=0.21, wspace=0.06)
    fig.text(0.04, 0.13, "Same number + color = the same LK track. All 3 also have depth.", fontsize=14)
    fig.text(
        0.04, 0.07, "The other 12 old tracks had already failed LK before these two images.", fontsize=13
    )
    fig.text(
        0.04,
        0.025,
        "Yellow: saved cube outline. Track 592 is just outside that outline.",
        fontsize=11,
        color="#b7c6d5",
    )
    fig.savefig(output / "03-lk-survivors.png", dpi=150)
    plt.close(fig)

    saved = json.loads((recording / "0000005381.overlay.json").read_text())
    g = next(g for g in saved["scene_group_tracks"] if g["id"] == 0)
    pred = {i: p for i, p in zip(g["predicted_ids"], g["predicted_pixels"], strict=True) if i in known}
    assert len(pred) == 15
    fig, ax = plt.subplots(figsize=(8, 8.1))
    fig.suptitle(
        "But the group predicts all 15 points up here",
        x=0.045,
        ha="left",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    frame(ax, 5381, (252, 373, 210, 337))
    for p in pred.values():
        ax.scatter(*p, s=85, marker="x", color=amber, linewidths=2.1, zorder=4)
    for i in current:
        p = pred[i]
        q = actual(i, 5381)["uv"]
        ax.annotate(
            "",
            xy=q,
            xytext=p,
            arrowprops={
                "arrowstyle": "->",
                "color": colors[i],
                "linewidth": 1.5,
                "linestyle": "--",
                "alpha": 0.8,
            },
            zorder=3,
        )
        ax.annotate(
            str(i),
            p,
            xytext={6: (4, 5), 591: (-33, 4), 592: (-33, -13)}[i],
            textcoords="offset points",
            color=amber,
            fontsize=12,
            fontweight="bold",
            path_effects=stroke,
        )
        dot(ax, q, colors[i], str(i), {6: (8, 7), 591: (-37, 3), 592: (8, -9)}[i])
    ax.text(
        262, 220, "15 predicted positions", color=amber, fontsize=15, fontweight="bold", path_effects=stroke
    )
    ax.text(
        289, 332, "3 measured positions", color=green, fontsize=15, fontweight="bold", path_effects=stroke
    )
    fig.subplots_adjust(left=0.04, right=0.96, top=0.91, bottom=0.16)
    fig.text(
        0.045,
        0.105,
        "Orange crosses are predictions — not LK matches.",
        fontsize=15,
        fontweight="bold",
        color=amber,
    )
    fig.text(0.045, 0.062, "Dashed arrows join the prediction and measurement of the same ID.", fontsize=12)
    fig.text(
        0.045,
        0.022,
        "Actual camera image, frame 5381. Original replay before the #2 fix.",
        fontsize=11,
        color="#b7c6d5",
    )
    fig.savefig(output / "03-predicted-versus-measured.png", dpi=150)
    plt.close(fig)
    print(output / "03-lk-survivors.png")
    print(output / "03-predicted-versus-measured.png")


if __name__ == "__main__":
    main()
