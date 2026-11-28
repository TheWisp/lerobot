"""Score saved predictions, optionally attaching evaluation labels and updating playback.

No images are decoded and no tracker is run. Existing scores and scorer source
are archived before replacement. Evaluation settings carry into future reruns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from benchmarks.flash_rebuild_examples import write_json
from benchmarks.flash_tracker_experiment import score_case


def rescore(run: Path, *, evaluation: dict | None = None, browser_export: Path | None = None):
    run = run.resolve()
    case = json.loads((run / "case.json").read_text())
    previous = json.loads((run / "summary.json").read_text())
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    if len(rows) != previous["frames"]:
        raise ValueError("saved predictions and summary have different frame counts")
    if browser_export is not None:
        manifest = json.loads((browser_export / "manifest.json").read_text())
        if (
            Path(manifest["source_run"]).resolve() != run
            or manifest["input_sha256"] != previous["input_sha256"]
            or manifest["frames"] != len(rows)
        ):
            raise ValueError("playback export belongs to different predictions")
    if evaluation is not None:
        case["evaluation"] = evaluation
    with np.load(run / "reference_points.npz", allow_pickle=False) as data:
        references = {role: data[role].copy() for role in data.files}
    first = {role: value["reference_frame"] for role, value in previous["roles"].items()}
    result = {**previous, **score_case(case, rows, references, first, run=run)}
    history = run / "score-history" / uuid.uuid4().hex
    history.mkdir(parents=True)
    for name in ("case.json", "summary.json"):
        shutil.copy2(run / name, history / name)
    code = {}
    for name in (
        "flash_tracker_experiment.py",
        "flash_experiment_labels.py",
        "flash_full_pose_score.py",
        "flash_surface_score.py",
        "flash_rescore_experiment.py",
    ):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, history / name)
        code[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["scoring_provenance"] = {
        "previous": str(history),
        "code_sha256": code,
        "predictions_sha256": hashlib.sha256((run / "frames.jsonl").read_bytes()).hexdigest(),
        "reference_points_sha256": hashlib.sha256((run / "reference_points.npz").read_bytes()).hexdigest(),
    }
    write_json(run / "case.json", case)
    write_json(run / "summary.json", result)
    if browser_export is not None:
        write_json(browser_export / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--evaluation", type=Path, help="JSON mapping of object roles to evaluation label files"
    )
    parser.add_argument("--browser-export", type=Path)
    args = parser.parse_args()
    result = rescore(
        args.run,
        evaluation=json.loads(args.evaluation.read_text()) if args.evaluation else None,
        browser_export=args.browser_export,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
