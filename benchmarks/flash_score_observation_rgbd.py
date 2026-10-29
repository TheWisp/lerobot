"""Score a FLASH RGB-D observation against independent YCBInEOAT pose labels.

Labels are used here only for evaluation, never by observation extraction.
An unavailable pose counts as abstention rather than a zero-error estimate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from flash_full_pose_score import score_transforms, source_points, summarize

from lerobot.policies.flash.observation_replay import load_observation_replay


def score(dataset_root: Path, case: Path, role: str) -> dict:
    replay = load_observation_replay(dataset_root)
    manifest = json.loads((case / "manifest.json").read_text())
    with np.load(case / "queries.npz", allow_pickle=False) as loaded:
        reference = {name: loaded[name] for name in loaded.files}
    frame_ids = np.asarray(reference["frames"], dtype=int)
    if len(frame_ids) != len(replay["frames"]) or not np.array_equal(frame_ids, np.arange(len(frame_ids))):
        raise ValueError("observation frames differ from labeled benchmark frames")
    labels = sorted((Path(manifest["source"]) / "annotated_poses").glob("*.txt"))
    truth = np.stack(
        [np.loadtxt(labels[int(manifest["source_first_index"]) + int(i)]).reshape(4, 4) for i in frame_ids]
    )
    truth = truth @ np.linalg.inv(truth[0])
    source = source_points(reference)
    result = {
        "dataset_root": str(dataset_root),
        "case": case.name,
        "role": role,
        "frames": len(frame_ids),
        "label_visible_fraction": float(np.mean(reference["truth_visible"])),
        "accepted_estimators": dict(
            Counter(
                frame["snapshot"].get("pose_status", {}).get(role, {}).get("estimator", "unknown")
                for frame in replay["frames"]
            )
        ),
    }
    for name, key in (("accepted", "accepted"), ("raw_direct", "raw_proposals")):
        rows = [frame["snapshot"][key].get(role, {}) for frame in replay["frames"]]
        poses = [row.get("pose") for row in rows]
        if name == "raw_direct":
            # Successful direct proposals are stored as accepted poses, while
            # raw_proposals retains the rejected fits for review.
            for index, frame in enumerate(replay["frames"]):
                if poses[index] is None:
                    accepted = frame["snapshot"]["accepted"].get(role, {})
                    if accepted.get("evidence") == "observed":
                        poses[index] = accepted.get("pose")
        available = np.asarray([pose is not None for pose in poses], dtype=bool)
        transforms = np.stack(
            [np.asarray(pose if pose is not None else np.eye(4), dtype=float) for pose in poses]
        )
        if not available[0]:
            raise ValueError(f"{name} has no first-frame pose for relative-motion scoring")
        relative = transforms @ np.linalg.inv(transforms[0])
        cloud, angle, center = score_transforms(relative, available, truth, source)
        result[name] = summarize(available, cloud, angle, center)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("case", type=Path)
    parser.add_argument("role")
    args = parser.parse_args()
    print(json.dumps(score(args.dataset_root, args.case, args.role), indent=2))


if __name__ == "__main__":
    main()
