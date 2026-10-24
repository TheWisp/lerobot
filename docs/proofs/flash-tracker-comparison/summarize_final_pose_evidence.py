"""Summarize the replay-only final-pose check over two reference samplings.

The inputs are per-frame outputs from analyze_pose_hypothesis_bank.py. A pose
dominates another only when it improves every recorded visible-surface field;
this is diagnostic evidence, not a calibrated runtime acceptance rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FIELDS = (
    ("observed_surface_p90_mm", -1),
    ("projected_mask_fraction", 1),
    ("ray_support_fraction", 1),
    ("ray_contradiction_fraction", -1),
)
PHASES = {"insertion": range(400, 501), "after_release": range(514, 551)}
ALTERNATIVES = ("depth", "two_way", "confirmed")


def load_bank(path: Path) -> dict[int, dict[str, dict[str, str]]]:
    frames: dict[int, dict[str, dict[str, str]]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            frame = int(row["frame"])
            candidates = frames.setdefault(frame, {})
            if row["candidate"] in candidates:
                raise ValueError(f"duplicate candidate at frame {frame}: {row['candidate']}")
            candidates[row["candidate"]] = row
    return frames


def verdict(original: dict[str, str], alternative: dict[str, str]) -> str:
    if any(row["status"] in ("lost", "insufficient_depth") for row in (original, alternative)):
        return "unavailable"
    advantages = [
        (float(alternative[field]) - float(original[field])) * direction for field, direction in FIELDS
    ]
    if not np.isfinite(advantages).all():
        return "unavailable"
    if all(value >= 0 for value in advantages) and any(value > 0 for value in advantages):
        return "alternative_dominates"
    if all(value <= 0 for value in advantages) and any(value < 0 for value in advantages):
        return "original_dominates"
    return "tradeoff"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("dowel", "ring"):
        for sampling in ("eroded", "full"):
            parser.add_argument(f"--{role}-{sampling}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    banks = {
        role: {sampling: load_bank(getattr(args, f"{role}_{sampling}")) for sampling in ("eroded", "full")}
        for role in ("dowel", "ring")
    }
    result: dict[str, object] = {}
    for role, samplings in banks.items():
        role_result = result[role] = {}
        for phase, frames in PHASES.items():
            phase_result = role_result[phase] = {}
            for alternative in ALTERNATIVES:
                by_sampling = {}
                for sampling, bank in samplings.items():
                    outcomes = [
                        verdict(bank[frame]["original"], bank[frame][alternative]) for frame in frames
                    ]
                    by_sampling[sampling] = {
                        outcome: outcomes.count(outcome)
                        for outcome in (
                            "alternative_dominates",
                            "original_dominates",
                            "tradeoff",
                            "unavailable",
                        )
                    }
                both = sum(
                    all(
                        verdict(bank[frame]["original"], bank[frame][alternative]) == "alternative_dominates"
                        for bank in samplings.values()
                    )
                    for frame in frames
                )
                phase_result[alternative] = {"by_reference": by_sampling, "dominates_in_both": both}
        role_result["frame_540"] = {
            sampling: {
                candidate: {field: float(bank[540][candidate][field]) for field, _ in FIELDS}
                for candidate in ("original", *ALTERNATIVES)
            }
            for sampling, bank in samplings.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
