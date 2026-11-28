"""Score FLASH evidence against Laval's held-out 6D object-pose labels.

The labeled mesh and poses are read only by this evaluator, never by FLASH.
Laval poses use an OpenGL camera convention; FLASH uses positive-Z camera
coordinates. Both are converted to relative motion from the first frame.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from flash_full_pose_score import score_transforms, summarize


def score(sequence: Path, model: Path, evidence: Path, role: str) -> dict:
    labels_path = sequence / "poses.npy"
    labels_gl = np.load(labels_path).reshape(-1, 4, 4).astype(float)
    camera_axes = np.diag([1.0, -1.0, -1.0, 1.0])
    labels = camera_axes @ labels_gl
    truth = labels @ np.linalg.inv(labels[0])

    vertices = np.asarray(trimesh.load(model, process=False).vertices, dtype=float)
    if len(vertices) < 256:
        raise ValueError("the evaluation model needs at least 256 vertices")
    indices = np.random.default_rng(42).choice(len(vertices), 256, replace=False)
    source = vertices[indices] @ labels[0, :3, :3].T + labels[0, :3, 3]

    rows = []
    for index in range(len(labels)):
        with gzip.open(evidence / f"{index:06d}.json.gz", "rt", encoding="utf-8") as stream:
            row = json.load(stream)
        if row["frame_index"] != index:
            raise ValueError("FLASH evidence is not in labeled frame order")
        rows.append(row)

    result = {
        "source": "Laval 6DOF object tracking",
        "frames": len(rows),
        "role": role,
        "score": "median displacement of 256 labeled-model surface points under relative motion",
        "published_counts": dict(Counter(row["estimates"][role]["evidence"] for row in rows)),
        "proposal_failures": dict(
            Counter(row["proposals"][role].get("failure") or "accepted" for row in rows)
        ),
    }
    for name, field, pose_key in (
        ("published", "estimates", "pose"),
        ("raw_candidate", "proposals", "raw_pose"),
    ):
        poses = [row[field][role].get(pose_key) for row in rows]
        available = np.asarray([pose is not None for pose in poses], dtype=bool)
        if not available[0]:
            raise ValueError(f"{name} lacks a first-frame pose")
        transforms = np.stack(
            [np.asarray(pose if pose is not None else np.eye(4), dtype=float) for pose in poses]
        )
        relative = transforms @ np.linalg.inv(transforms[0])
        cloud, angle, center = score_transforms(relative, available, truth, source)
        result[name] = summarize(available, cloud, angle, center)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path, help="Laval sample sequence directory")
    parser.add_argument("model", type=Path, help="Laval geometry.ply for evaluation only")
    parser.add_argument("evidence", type=Path, help="complete FLASH observation evidence directory")
    parser.add_argument("role", help="object role in FLASH evidence")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.sequence, args.model, args.evidence, args.role)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
