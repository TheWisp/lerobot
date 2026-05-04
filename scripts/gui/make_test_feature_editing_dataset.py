"""Build a test dataset with editable features for manually exercising the
feature-editing GUI.

Copies an existing local dataset and bolts on three features:

* ``reward`` — float32[1], initialized to a -1.0 step penalty everywhere
  (RECAP-style baseline). Edit me with the slider+number widget.
* ``success`` — bool[1], initialized so each episode has uniform value
  (per-episode broadcast — should auto-detect; the Inspector card shows
  the broadcast note and edits coerce to whole-episode).
* ``subtask_index`` — int64[1], initialized to a simple 3-segment
  progression. Plus a ``meta/subtasks.parquet`` with starter strings —
  exercises the dropdown widget. Frontend may stage strings; backend
  resolves to indices at Save.

Usage:
    python /tmp/make_test_feature_editing_dataset.py \\
        --src ~/.cache/huggingface/lerobot/thewisp/intervention_cylinder_ring_assembly \\
        --dst /tmp/feature-editing-test-dataset

After it runs, open the dst path in the GUI and try:

1. Drag-select on the ``reward`` row → slider+number widget. Type a
   value or drag the slider. Confirm the chip appears in the edits bar.
2. Drag-select on the ``success`` row → checkbox widget. The card
   shows "per-episode" — the band should expand to fill the episode
   when you stage. Toggle and confirm a chip appears whose range is
   the full episode regardless of what you dragged.
3. Stage two overlapping edits on ``reward`` for the same episode →
   the second one should pop the overlap-confirmation modal.
4. Click Save → reload → values persist.

For subtask editing once the frontend dropdown ships, you'd type a new
string ("inspect knot") and confirm it appears in
``meta/subtasks.parquet`` after Save with a fresh int index.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def add_editable_features(dst: Path) -> None:
    """Mutate the dataset at ``dst`` in place: add the three features."""
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    # Schema additions ──────────────────────────────────────────────────
    new_features = {
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        "success": {"dtype": "bool", "shape": [1], "names": None},
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for name, spec in new_features.items():
        if name not in info["features"]:
            info["features"][name] = spec
            print(f"  + declared {name} in info.json")
        else:
            print(f"  ~ {name} already exists in info.json — leaving as-is")

    info_path.write_text(json.dumps(info, indent=2))

    # Data shards ───────────────────────────────────────────────────────
    data_dir = dst / "data"
    shards = sorted(data_dir.rglob("*.parquet"))
    print(f"  data shards: {len(shards)}")

    # Pre-decide per-episode success values: alternate true/false so the
    # detection picks `success` as per-episode.
    episodes_seen: dict[int, bool] = {}

    for shard in shards:
        df = pd.read_parquet(shard)
        n = len(df)
        ep_indices = df["episode_index"].astype(int).values
        frame_indices = df["frame_index"].astype(int).values

        # reward: per-frame varying so the detector doesn't false-flag it as
        # per-episode. Use a small sinusoid superimposed on the -1.0 step
        # penalty — gives a visually interesting line plot on the row too.
        df["reward"] = (-1.0 + 0.2 * np.sin(0.1 * frame_indices.astype(np.float32))).astype(np.float32)

        # success: alternate true/false per episode.
        success_vals = np.zeros(n, dtype=bool)
        for i, ep in enumerate(ep_indices):
            if ep not in episodes_seen:
                episodes_seen[ep] = bool(ep % 2)
            success_vals[i] = episodes_seen[ep]
        df["success"] = success_vals

        # subtask_index: simple frame-position bucket (3 segments per episode).
        # Approximate: 0 if first 33% of episode, 1 if middle 34%, 2 otherwise.
        # We don't know per-episode length from one row, so use frame_index ranges.
        # frame_index is already 0-based per episode in standard LeRobot data.
        # Use crude buckets: <30 → 0, <70 → 1, else 2 (works for 10-frame to
        # 1000-frame episodes, just as a placeholder).
        si = np.where(frame_indices < 30, 0, np.where(frame_indices < 70, 1, 2)).astype(np.int64)
        df["subtask_index"] = si

        df.to_parquet(shard, compression="snappy", index=False)
        print(f"  ✓ {shard.relative_to(dst)} ({n} rows)")

    # Subtasks lookup table ─────────────────────────────────────────────
    subtasks_path = dst / "meta" / "subtasks.parquet"
    subtasks = pd.DataFrame(
        {"subtask_index": [0, 1, 2]},
        index=pd.Index(["approach", "grasp", "release"], name="subtask"),
    )
    subtasks.to_parquet(subtasks_path)
    print(f"  + meta/subtasks.parquet  → {list(subtasks.index)}")

    # Episodes parquet stats columns ────────────────────────────────────
    # Re-add stats columns for the new features so the GUI doesn't error.
    # We keep this best-effort: if the parquet doesn't already track stats
    # for any feature, we skip.
    eps_dir = dst / "meta" / "episodes"
    eps_shards = sorted(eps_dir.rglob("*.parquet"))
    for shard in eps_shards:
        eps_df = pd.read_parquet(shard)
        # Sample any existing stats column to gauge the schema (e.g.
        # `stats/timestamp/min`). If none exist, skip.
        sample_stat = next(
            (c for c in eps_df.columns if c.startswith("stats/") and c.endswith("/min")),
            None,
        )
        if sample_stat is None:
            print(f"  · {shard.name}: no existing stats columns — skipping stats fill")
            continue

        for feature in ("reward", "success", "subtask_index"):
            if f"stats/{feature}/min" in eps_df.columns:
                continue  # already populated
            # Build placeholder stat lists matching the existing per-row format.
            placeholder = [[0.0]] * len(eps_df)
            for stat in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
                eps_df[f"stats/{feature}/{stat}"] = placeholder
            eps_df[f"stats/{feature}/count"] = [[1]] * len(eps_df)

        eps_df.to_parquet(shard)
        print(f"  ✓ {shard.relative_to(dst)} (stats columns added)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source dataset root (must be a local v3 LeRobot dataset).",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination directory. Will be created; refuses to overwrite an existing non-empty dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, wipe the destination first.",
    )
    args = parser.parse_args()

    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()

    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"src {src} doesn't look like a LeRobot dataset (no meta/info.json)")
    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            raise SystemExit(f"dst {dst} exists and is non-empty (pass --force to wipe)")
        # safe-destruct: user passed --force; this is a test fixture path under their control.
        shutil.rmtree(dst)

    print(f"Copying {src} → {dst}")
    shutil.copytree(src, dst)
    print(f"  ✓ copied {sum(1 for _ in dst.rglob('*'))} files")

    print("Adding editable features…")
    add_editable_features(dst)

    print()
    print(f"✓ Test dataset ready at {dst}")
    print("Open it in the GUI's Data tab and try:")
    print("  1. Drag-select on the `reward` row → slider+number widget")
    print("  2. Drag-select on the `success` row → checkbox; band should expand to whole episode")
    print("  3. Stage two overlapping reward edits → triggers the overlap-confirmation modal")
    print("  4. Save → reload → values persist")


if __name__ == "__main__":
    main()
