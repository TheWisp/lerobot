"""Render measured prior-in-fit results, with availability separate from accuracy."""

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def distribution(values):
    return dict(
        zip(("median", "p95", "maximum"), map(float, np.percentile(values, [50, 95, 100])), strict=True)
    )


def main(root):
    data = json.loads((root / "compact-and-controls.json").read_text())
    full = json.loads((root / "full-history.json").read_text())["recorded"]
    repo = Path(__file__).resolve().parents[1]
    summary = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": {
            **{name: version(name) for name in ("numpy", "scipy", "small_gicp")},
            "opencv": cv2.__version__,
        },
        "implementation_sha256": {
            name: hashlib.sha256((repo / "benchmarks" / name).read_bytes()).hexdigest()
            for name in ("flash_prior_registration.cpp", "flash_prior_registration.py")
        },
        "compact": {},
        "full_history": {},
        "controls": {},
        "latency": {},
    }
    for name in sorted({r["method"] for r in data["recorded"]}):
        rows = [r for r in data["recorded"] if r["method"] == name]
        summary["compact"][name] = {
            "counts": dict(Counter(r["status"] for r in rows)),
            "elapsed_ms": distribution([r["elapsed_ms"] for r in rows]),
            "without_measured_gc_ms": distribution([r["elapsed_ms"] - r.get("gc_ms", 0) for r in rows]),
            "maximum_evaluations": max(r["evaluations"] for r in rows),
            "slow_frames": [r for r in rows if r["elapsed_ms"] > 150],
        }
    for name in sorted({r["method"] for r in full}):
        rows = [r for r in full if r["method"] == name]
        summary["full_history"][name] = {
            case: dict(Counter(r["status"] for r in rows if r["case"] == case))
            for case in sorted({r["case"] for r in rows})
        }
    strengths = [0.0, 0.1, 1.0, 10.0]
    for scenario in sorted({r["scenario"] for r in data["controlled"]}):
        summary["controls"][scenario] = {}
        for strength in strengths:
            rows = [
                r
                for r in data["controlled"]
                if r["scenario"] == scenario and r["factor"] == "gicp" and r["strength"] == strength
            ]
            summary["controls"][scenario][str(strength)] = {
                "rotation_mean_deg": float(np.mean([r["rotation_error_deg"] for r in rows])),
                "rotation_error_deg": distribution([r["rotation_error_deg"] for r in rows]),
                "position_mean_mm": float(np.mean([r["position_error_mm"] for r in rows])),
                "position_error_mm": distribution([r["position_error_mm"] for r in rows]),
            }
    pairs = {}
    for row in data["fixed_input_latency"]:
        pairs.setdefault((row["repeat"], row["input"]), {})[row["prior"]] = row["elapsed_ms"]
        if row["iterations"] != row["budget"]:
            raise ValueError("Latency comparison did not use equal fixed iteration budgets")
    delta = [row[True] - row[False] for row in pairs.values()]
    summary["latency"] = {
        "paired_count": len(delta),
        "paired_extra_ms": distribution(delta),
        **{str(enabled): distribution([r[enabled] for r in pairs.values()]) for enabled in (False, True)},
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    labels = ["None", "Weaker\n(0.1)", "Unit weight\n(1)", "Stronger\n(10)"]
    colors = ["#8c98a5", "#008f95", "#d07a18", "#b04464"]
    for ax, scenario, title in zip(
        axes[0],
        ("strip", "moved"),
        ("Narrow patch: preserve known rotation", "Actual 20° turn: follow measured rotation"),
        strict=True,
    ):
        values = [summary["controls"][scenario][str(s)]["rotation_mean_deg"] for s in strengths]
        ax.bar(labels, values, color=colors)
        for index, value in enumerate(values):
            ax.text(index, value + max(values) * 0.035, f"{value:.2f}°", ha="center")
        ax.set_ylim(0, max(values) * 1.23)
        ax.set_ylabel("Mean rotation error (degrees)")
        ax.set_title(title)
        ax.set_xlabel("Prediction penalty strength")
    names = ["baseline", "native_post", "prior_1", "prior_0.1"]
    values = [
        sum(case.get("observed_candidate", 0) for case in summary["full_history"][name].values())
        for name in names
    ]
    ax = axes[1, 0]
    ax.bar(
        ["Current", "Native +\npost-filter", "Prior in fit\n(1)", "Prior in fit\n(0.1)"],
        values,
        color=["#8c98a5", "#607d9b", "#d07a18", "#008f95"],
    )
    ax.set_ylim(0, 430)
    for index, value in enumerate(values):
        ax.text(index, value + 7, f"{value}/400", ha="center")
    ax.set_title("Full recording: availability, not pose accuracy")
    ax.set_ylabel("Accepted observed updates")
    ax = axes[1, 1]
    ax.hist(np.asarray(delta) * 1000, bins=18, color="#008f95", alpha=0.8)
    ax.axvline(0, color="#555", linestyle="--")
    ax.set_xlabel("Extra native solve time with prior (microseconds)")
    ax.set_ylabel("Paired comparisons")
    ax.set_title(f"Identical inputs and iteration budgets: {len(delta)} pairs")
    fig.suptitle("Prediction inside the existing fit: benefit depends on its strength", fontsize=17)
    fig.text(
        0.5,
        0.018,
        "Top: 30 noisy synthetic trials per setting, known true pose. Bottom: recorded replay and warmed CPU timings.\n"
        "Geometry rejection remains enabled. No production defaults changed. See report for partial visibility and timing outliers.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(root / "results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
