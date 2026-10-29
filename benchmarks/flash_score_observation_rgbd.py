"""Score a FLASH RGB-D observation against independent YCBInEOAT pose labels.

Labels are used here only for evaluation, never by observation extraction.
An unavailable pose counts as abstention rather than a zero-error estimate.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from flash_full_pose_score import score_transforms, source_points, summarize


def score(dataset_root: Path, case: Path, role: str, *, evidence_sidecar: Path | None = None) -> dict:
    manifest = json.loads((case / "manifest.json").read_text())
    with np.load(case / "queries.npz", allow_pickle=False) as loaded:
        reference = {name: loaded[name] for name in loaded.files}
    frame_ids = np.asarray(reference["frames"], dtype=int)
    if not np.array_equal(frame_ids, np.arange(len(frame_ids))):
        raise ValueError("observation frames differ from labeled benchmark frames")
    sidecar = evidence_sidecar or dataset_root / "flash/v1/observation-evidence/episode_000000"
    sidecar_manifest = json.loads((sidecar / "manifest.json").read_text())
    if not sidecar_manifest.get("complete") or sidecar_manifest.get("total_frames") != len(frame_ids):
        raise ValueError("observation evidence is incomplete or differs from the labeled benchmark")
    evidence = []
    for index in frame_ids:
        with gzip.open(sidecar / f"{index:06d}.json.gz", "rt", encoding="utf-8") as stream:
            row = json.load(stream)
        if row["frame_index"] != int(index):
            raise ValueError("observation evidence is not in source-frame order")
        evidence.append(row)
    labels = sorted((Path(manifest["source"]) / "annotated_poses").glob("*.txt"))
    truth = np.stack(
        [np.loadtxt(labels[int(manifest["source_first_index"]) + int(i)]).reshape(4, 4) for i in frame_ids]
    )
    truth = truth @ np.linalg.inv(truth[0])
    source = source_points(reference)
    result = {
        "dataset_root": str(dataset_root),
        "evidence_sidecar": str(sidecar),
        "case": case.name,
        "role": role,
        "frames": len(frame_ids),
        "label_visible_fraction": float(np.mean(reference["truth_visible"])),
        "accepted_estimators": dict(
            Counter(frame["estimates"].get(role, {}).get("estimator", "unknown") for frame in evidence)
        ),
    }
    for name in ("accepted", "raw_direct"):
        if name == "accepted":
            poses = [frame["estimates"].get(role, {}).get("pose") for frame in evidence]
        else:
            poses = []
            for frame in evidence:
                raw = frame["proposals"].get(role, {}).get("raw_pose")
                accepted = frame["estimates"].get(role, {})
                if raw is None and accepted.get("evidence") == "observed":
                    raw = accepted.get("pose")
                poses.append(raw)
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
    parser.add_argument("--evidence-sidecar", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            score(args.dataset_root, args.case, args.role, evidence_sidecar=args.evidence_sidecar), indent=2
        )
    )


if __name__ == "__main__":
    main()
