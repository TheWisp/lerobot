"""Freeze nested visible-surface query budgets from one labelled master clip."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.flash_surface_benchmark import _digest

POINT_AXES = {
    "queries": 0,
    "initial_depth_m": 0,
    "truth_xy": 1,
    "truth_xyz": 1,
    "truth_visible": 1,
    "reference_depth_gap_m": 1,
}


def farthest_order(xy: np.ndarray) -> np.ndarray:
    """Deterministic coverage order; every prefix is a subset of the next."""
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("query pixels must be finite Nx2")
    if not len(xy):
        raise ValueError("need at least one query")
    work = xy.astype(np.float64)
    selected = np.empty(len(work), dtype=np.int32)
    selected[0] = int(np.argmin(np.sum((work - work.mean(axis=0)) ** 2, axis=1)))
    nearest_sq = np.sum((work - work[selected[0]]) ** 2, axis=1)
    nearest_sq[selected[0]] = -1
    for index in range(1, len(work)):
        next_id = int(np.argmax(nearest_sq))
        selected[index] = next_id
        nearest_sq = np.minimum(nearest_sq, np.sum((work - work[next_id]) ** 2, axis=1))
        nearest_sq[selected[: index + 1]] = -1
    return selected


def build(master: Path, output: Path, budgets: tuple[int, ...]) -> dict:
    manifest = json.loads((master / "manifest.json").read_text())
    with np.load(master / "queries.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    count = len(arrays["queries"])
    if not budgets or any(budget < 3 or budget > count for budget in budgets):
        raise ValueError("budgets must fit within the labelled master query set")
    if not all(key in arrays for key in POINT_AXES):
        raise ValueError("nested budget requires material-point labels")
    order = farthest_order(arrays["queries"][:, 1:3])
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    report = {"master": str(master.resolve()), "master_query_count": count, "order_sha256": order_hash}
    for budget in sorted(set(budgets)):
        folder = output / f"budget-{budget}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, source in (
            ("recording", Path(manifest["recording"]).resolve()),
            ("masks", Path(manifest["mask_dir"]).resolve()),
        ):
            link = folder / name
            if link.is_symlink():
                if link.resolve() != source:
                    raise ValueError(f"{link}: points at a different source")
            elif link.exists():
                raise ValueError(f"{link}: source link is occupied")
            else:
                link.symlink_to(source, target_is_directory=True)
        chosen = order[:budget]
        reduced = {
            key: np.take(array, chosen, axis=POINT_AXES[key]) if key in POINT_AXES else array
            for key, array in arrays.items()
        }
        np.savez_compressed(folder / "queries.npz", **reduced)
        child = dict(manifest)
        child.update(
            {
                "recording": str((folder / "recording").resolve()),
                "mask_dir": str((folder / "masks").resolve()),
                "query_count": budget,
                "queries_sha256": _digest(folder / "queries.npz"),
                "nested_master": str(master.resolve()),
                "nested_order_sha256": order_hash,
                "nested_master_indices": chosen.tolist(),
                "labelled_visible_fraction": float(reduced["truth_visible"].mean()),
            }
        )
        (folder / "manifest.json").write_text(json.dumps(child, indent=2) + "\n")
        report[str(budget)] = {
            "query_indices": chosen.tolist(),
            "eligible_point_frames": int(reduced["truth_visible"].sum()),
        }
    (output / "nested_spec.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 256, 1024])
    args = parser.parse_args()
    print(json.dumps(build(args.master, args.output, tuple(args.budgets)), indent=2))


if __name__ == "__main__":
    main()
