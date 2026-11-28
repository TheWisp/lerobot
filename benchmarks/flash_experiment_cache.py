"""Conservative, whole-clip reuse for completed offline experiments.

Hash decoded inputs, labels, settings and the local execution environment.
There is no timestamp-only shortcut and no reuse of incomplete results. Cached
timings belong to the original run; force a rerun to measure performance again.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np

from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_experiment_labels import load_pose_labels


def file_hash(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def execution_identity(source_hashes):
    """Resolve locally, then pin model loading to these exact checkpoint files."""
    import torch
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download("infinity1096/UFM-Base-DINOv2L-init", local_files_only=True))
    external = {}
    ufm = Path(importlib.util.find_spec("uniflowmatch").origin).parent
    dino = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    for name, root in (("ufm", ufm), ("dino", dino)):
        files = sorted(root.rglob("*.py"))
        if not files:
            raise FileNotFoundError(f"Cannot fingerprint {name} implementation: {root}")
        external[name] = {str(p.relative_to(root)): file_hash(p) for p in files}
    return {
        "schema": 1,
        "source": source_hashes,
        "external_source": external,
        "model_revision": snapshot.name,
        "model_files": {
            str(p.relative_to(snapshot)): file_hash(p) for p in snapshot.rglob("*") if p.is_file()
        },
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
    }


def case_identity(case, settings, execution, *, check_cancel=lambda: None):
    digest = hashlib.sha256()
    rows = []
    for frame, masks in samples(case):
        check_cancel()
        for value in (
            frame.rgb,
            frame.depth,
            frame.intrinsics,
            frame.base_from_camera,
            np.asarray([frame.depth_m_per_unit, frame.source_time_s]),
            *(masks[role] for role in case["roles"]),
        ):
            a = np.asarray(value)
            digest.update(str((a.shape, a.dtype)).encode() + a.tobytes())
        rows.append({"frame": frame.frame_id, "time_s": frame.source_time_s})
    labels, provenance = load_pose_labels(case, rows)
    for role in sorted(labels):
        digest.update(role.encode() + labels[role].tobytes())
    # A changed evaluation model must not reuse scores for the old shape.
    surfaces = {
        role: file_hash(path) if path.is_file() else None
        for role, spec in case.get("evaluation_surfaces", {}).items()
        for path in (Path(spec["points_m"]).expanduser().resolve(),)
    }
    return canonical_hash(
        {
            "execution": execution,
            "case": case,
            "settings": settings,
            "inputs": digest.hexdigest(),
            "labels": provenance,
            "surfaces": surfaces,
        }
    )


def result_files(run):
    return {
        str(p.relative_to(run)): file_hash(p)
        for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "cache.json"
    }


def seal_result(run, identity):
    # Only call after run_case has written a complete, scored result.
    json.loads((run / "summary.json").read_text())
    value = {"identity": identity, "artifacts": result_files(run)}
    temporary = run / "cache.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(run / "cache.json")


def reusable(run, identity):
    try:
        saved = json.loads((run / "cache.json").read_text())
        if saved["identity"] != identity:
            return False
        expected = saved["artifacts"]
        required = {
            "summary.json",
            "frames.jsonl",
            "reference_points.npz",
            "case.json",
            "correspondences.jsonl.gz",
        }
        return required <= expected.keys() and expected == result_files(run)
    except (OSError, ValueError, KeyError):
        return False
