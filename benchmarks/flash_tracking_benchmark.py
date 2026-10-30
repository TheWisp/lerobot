"""Score complete FLASH scene-evidence journals against a frozen case manifest.

Tracking never reads evaluation labels. This program reads them only after a
completed trace exists, and keeps availability separate from conditional error.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_full_pose_score import score_transforms, summarize


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _lerobot_input_sha256(root: Path) -> str:
    """Hash RGB-D, imported masks, and LeRobot metadata; exclude derived evidence."""
    digest = hashlib.sha256()
    paths = [root / "flash/v1/source.json"]
    for directory in ("videos", "images", "data", "meta", "flash/v1/imported-masks"):
        paths.extend(sorted(path for path in (root / directory).rglob("*") if path.is_file()))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _read_evidence(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported scene evidence schema: {directory}")
    count = manifest["total_frames"]
    if not manifest.get("complete") or manifest["frames_completed"] != count:
        raise ValueError(f"incomplete evidence journal: {directory}")
    rows = []
    for index in range(count):
        path = directory / f"{index:06d}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            row = json.load(stream)
        if row["frame_index"] != index:
            raise ValueError(f"evidence frame order changed: {path}")
        rows.append(row)
    if len(list(directory.glob("*.json.gz"))) != count:
        raise ValueError(f"unexpected evidence frames: {directory}")
    return manifest, rows


def _mask(runs: list[list[int]], shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    for start, length in runs:
        if start < 0 or length < 0 or start + length > len(flat):
            raise ValueError("mask RLE exceeds frame bounds")
        flat[start : start + length] = True
    return flat.reshape(shape)


def _reference_mask(case: dict, role: str, index: int, provenance: dict | None) -> np.ndarray:
    if case["kind"] == "ycb_annotated_pose":
        assert provenance is not None
        path = _path(provenance["source"]) / "gt_mask" / provenance["names"][index]
    else:
        path = _path(case["artifact_root"]) / "flash/v1/imported-masks" / role / f"{index:06d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def _ycb_truth(provenance: dict, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    source = _path(provenance["source"])
    labels = sorted((source / "annotated_poses").glob("*.txt"))
    images = sorted((source / "rgb").glob("*.png"))
    if len(labels) != len(images):
        raise ValueError("YCB pose and image counts differ")
    if int(labels[0].stem) > 10**12 and [p.stem for p in labels] != [p.stem for p in images]:
        raise ValueError("YCB pose and image timestamps differ")
    start = provenance["start"]
    transforms = np.stack([np.loadtxt(path).reshape(4, 4) for path in labels[start : start + len(rows)]])
    if len(transforms) != len(rows) or not np.allclose(transforms[:, 3], [0, 0, 0, 1]):
        raise ValueError("YCB labels are missing or invalid")
    truth = transforms @ np.linalg.inv(transforms[0])

    # Fixed measured points make the rotation/translation metric independent of
    # the arbitrary object-coordinate origin in the supplied annotations.
    k = np.asarray(provenance["intrinsics"])
    name = provenance["names"][0]
    depth = cv2.imread(str(source / "depth" / name), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise ValueError("missing first-frame YCB depth")
    depth = depth.astype(float) * 0.001
    mask = _reference_mask({"kind": "ycb_annotated_pose"}, "object", 0, provenance)
    interior = cv2.erode(mask.astype("uint8"), np.ones((5, 5), "uint8")) > 0
    y, x = np.where(interior & (depth > 0))
    if not len(x):
        raise ValueError("no valid first-frame measured object points")
    chosen = np.random.default_rng(42).choice(len(x), min(256, len(x)), replace=False)
    x, y = x[chosen], y[chosen]
    z = depth[y, x]
    points = np.column_stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z))
    return truth, points


def _pose_accuracy(rows: list[dict], role: str, provenance: dict) -> dict:
    truth, source = _ycb_truth(provenance, rows)
    estimates = [row["estimates"][role] for row in rows]
    initial = estimates[0]["pose"]
    if initial is None:
        return {"status": "unscorable_no_initial_pose"}
    initial_inverse = np.linalg.inv(np.asarray(initial, dtype=float))
    available = np.array([item["pose"] is not None for item in estimates], dtype=bool)
    published = np.stack(
        [
            np.asarray(item["pose"] if item["pose"] is not None else np.eye(4), dtype=float)
            for item in estimates
        ]
    )
    relative = published @ initial_inverse
    cloud, angle, center = score_transforms(relative, available, truth, source)
    summary = summarize(available, cloud, angle, center)
    # The old scorer exposes fixed 10/20 mm-and-degree pass bins. This suite
    # reports distributions and availability instead of imposing task thresholds.
    summary.pop("within_10mm_10deg", None)
    summary.pop("within_20mm_20deg", None)
    summary["status"] = "scored"
    summary["metric"] = "first-frame-relative measured-point displacement; full 3D rotation"
    summary["label_source"] = "YCBInEOAT annotated poses (not independent sensor ground truth)"
    summary["scored_fraction_after_initialization"] = (
        summary["available"] / summary["total"] if summary["total"] else None
    )
    summary["per_frame_cloud_mm"] = [float(v) if np.isfinite(v) else None for v in cloud]
    summary["per_frame_angle_deg"] = [float(v) if np.isfinite(v) else None for v in angle]
    return summary


def score_case(case: dict) -> dict:
    if case["kind"] not in {"ycb_annotated_pose", "lerobot_rgbd", "flash_prepared_demo"}:
        raise ValueError(f"unsupported case kind: {case['kind']}")
    root = _path(case["artifact_root"])
    observation = None
    if case["kind"] == "flash_prepared_demo":
        evidence_dir = _path(case["evidence_dir"])
        provenance = None
        observation = None
        manifest, rows = _read_evidence(evidence_dir)
        report_path = evidence_dir.parent / "preparation-report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        capture = {
            "input_sha256": manifest["source_sha256"],
            "bundle_sha256": hashlib.sha256((evidence_dir.parent / "bundle.json").read_bytes()).hexdigest(),
            "mask_source": "prepared_demo_trace",
            "prepared_pose_trace_selection_version": report.get("pose_trace_selection_version"),
        }
    elif case["kind"] == "ycb_annotated_pose":
        evidence_dir = _path(case["evidence_dir"]) if "evidence_dir" in case else root / "evidence"
        provenance = json.loads((root / "provenance.json").read_text())
        capture = {
            "input_sha256": provenance["input_sha256"],
            "config_sha256": provenance["config_sha256"],
            "checkout": provenance["checkout"],
            "mask_source": provenance["mask_source"],
        }
    else:
        evidence_dir = (
            _path(case["evidence_dir"])
            if "evidence_dir" in case
            else root / "flash/v1" / case.get("evidence_name", "observation-evidence") / "episode_000000"
        )
        provenance = None
        observation = json.loads((evidence_dir / "observation.json").read_text())
        source_hash = hashlib.sha256((root / "flash/v1/source.json").read_bytes()).hexdigest()
        if observation.get("source_metadata_sha256") not in {None, source_hash}:
            raise ValueError(f"source metadata changed since extraction: {case['id']}")
        config_path = Path(observation["config_bundle"])
        if not config_path.is_absolute():
            config_path = root / config_path
        capture = {
            "input_sha256": _lerobot_input_sha256(root),
            "evidence_identity": observation["evidence_identity"],
            "config_bundle": observation["config_bundle"],
            "referenced_config_sha256_at_scoring": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_metadata_sha256": observation.get("source_metadata_sha256"),
            "source_trace_sha256": observation.get("source_trace_sha256"),
            "mask_source": observation["mask_source"],
        }
    timing_path = (root if provenance is not None else evidence_dir.parent) / "run.json"
    capture["offline_wall_s_including_initialization"] = (
        json.loads(timing_path.read_text())["offline_wall_s_including_initialization"]
        if timing_path.exists()
        else None
    )
    if case["kind"] != "flash_prepared_demo":
        manifest, rows = _read_evidence(evidence_dir)
    if provenance is not None and len(rows) != provenance["count"]:
        raise ValueError("YCB source and evidence frame counts differ")
    if provenance is not None and len(provenance["names"]) != len(rows):
        raise ValueError("YCB source image list differs from evidence frame count")
    if observation is not None and observation["frames"] != len(rows):
        raise ValueError("LeRobot observation and evidence frame counts differ")
    if case.get("expected_frames") is not None and len(rows) != case["expected_frames"]:
        raise ValueError("case frame count differs from frozen suite")
    roles = case.get("roles", manifest["roles"])
    if not set(roles) <= set(manifest["roles"]):
        raise ValueError("case role absent from evidence")
    results = {}
    for role in roles:
        states, failures, estimators = Counter(), Counter(), Counter()
        ious, depth_residuals = [], []
        longest_loss = loss_run = 0
        frame_states = []
        mask_present_frames = 0
        for index, row in enumerate(rows):
            size = tuple(row["image_size"])
            estimated_mask = _mask(row["masks_rle"].get(role, []), size)
            mask_present_frames += bool(estimated_mask.any())
            if case["kind"] != "flash_prepared_demo":
                reference_mask = _reference_mask(case, role, index, provenance)
                if reference_mask.shape != size:
                    raise ValueError(f"reference mask size differs: {case['id']} frame {index}")
                if not reference_mask.any():
                    frame_states.append("not_visible_in_reference")
                    loss_run = 0
                    continue
                ious.append(
                    float(
                        np.count_nonzero(estimated_mask & reference_mask)
                        / np.count_nonzero(estimated_mask | reference_mask)
                    )
                )
            estimate = row["estimates"][role]
            state = estimate["evidence"]
            if state not in {"observed", "predicted", "lost"}:
                raise ValueError(f"unknown pose state: {state}")
            if (state == "lost") != (estimate["pose"] is None):
                raise ValueError(f"pose/state contradiction: {case['id']} frame {index}")
            frame_states.append(state)
            states[state] += 1
            estimators[estimate["estimator"]] += 1
            proposal = row["proposals"].get(role, {})
            if proposal.get("failure"):
                failures[proposal["failure"]] += 1
            support = proposal.get("visible_support") or {}
            if state == "observed" and support.get("depth_residual_p90_m") is not None:
                depth_residuals.append(1000 * float(support["depth_residual_p90_m"]))
            loss_run = loss_run + 1 if state == "lost" else 0
            longest_loss = max(longest_loss, loss_run)
        visible = sum(states.values())
        result = {
            "reference_visible_frames": visible if case["kind"] != "flash_prepared_demo" else None,
            "frames_scored": visible,
            "visibility_basis": "all_frames_no_visibility_labels"
            if case["kind"] == "flash_prepared_demo"
            else "supplied_reference_mask",
            "mask_present_frames": mask_present_frames,
            "states": dict(states),
            "direct_observation_fraction": states["observed"] / visible if visible else None,
            "pose_available_fraction": (states["observed"] + states["predicted"]) / visible
            if visible
            else None,
            "longest_lost_streak_visible_frames": longest_loss,
            "mask_iou_median_vs_supplied": float(np.median(ious)) if ious else None,
            "observed_depth_residual_p90_mm_median": float(np.median(depth_residuals))
            if depth_residuals
            else None,
            "estimators": dict(estimators),
            "failure_reasons": dict(failures),
            "frame_states": frame_states,
        }
        result["pose_accuracy"] = (
            _pose_accuracy(rows, role, provenance)
            if provenance is not None
            else {
                "status": "unavailable_no_independent_pose_labels"
                if case["kind"] == "flash_prepared_demo"
                else "unavailable_no_pose_labels"
            }
        )
        results[role] = result
    return {
        "id": case["id"],
        "kind": case["kind"],
        "domain": case["domain"],
        "tags": case.get("tags", []),
        "frames": len(rows),
        "capture": capture,
        "roles": results,
    }


def score_suite(suite: dict, *, require_all: bool = False) -> dict:
    if suite.get("schema") != "flash-tracking-suite-v1":
        raise ValueError("unknown benchmark suite schema")
    cases, ids = [], set()
    for case in suite["cases"]:
        if case["id"] in ids:
            raise ValueError(f"duplicate case id: {case['id']}")
        ids.add(case["id"])
        if not _path(case["artifact_root"]).exists():
            if require_all:
                raise FileNotFoundError(case["artifact_root"])
            cases.append({"id": case["id"], "status": "missing_local_artifact", "domain": case["domain"]})
            continue
        cases.append({"status": "scored", **score_case(case)})
    return {"schema": "flash-tracking-results-v1", "suite": suite["name"], "cases": cases}


def compare_results(baseline: dict, candidate: dict) -> list[dict]:
    """Compare matching frames; avoid rewarding an estimator for dropping hard frames."""
    previous = {case["id"]: case for case in baseline["cases"] if case["status"] == "scored"}
    comparisons = []
    for case in candidate["cases"]:
        if case["status"] != "scored":
            comparisons.append({"id": case["id"], "status": case["status"]})
            continue
        old = previous.get(case["id"])
        if old is None:
            comparisons.append({"id": case["id"], "status": "no_baseline_case"})
            continue
        if case["frames"] != old["frames"] or case["kind"] != old["kind"]:
            raise ValueError(f"benchmark frame/domain mismatch: {case['id']}")
        if case["capture"].get("input_sha256") != old["capture"].get("input_sha256"):
            raise ValueError(f"benchmark source changed: {case['id']}")
        for role, new_role in case["roles"].items():
            old_role = old["roles"].get(role)
            if old_role is None:
                comparisons.append({"id": case["id"], "role": role, "status": "no_baseline_role"})
                continue
            before, after = old_role["frame_states"], new_role["frame_states"]
            if len(before) != len(after) or [s == "not_visible_in_reference" for s in before] != [
                s == "not_visible_in_reference" for s in after
            ]:
                raise ValueError(f"reference visibility changed: {case['id']} / {role}")
            result = {
                "id": case["id"],
                "role": role,
                "status": "compared",
                "visible_frames": new_role["reference_visible_frames"],
                "frames_scored": new_role["frames_scored"],
                "observed_frames_before": before.count("observed"),
                "observed_frames_after": after.count("observed"),
                "available_frames_before": sum(s in {"observed", "predicted"} for s in before),
                "available_frames_after": sum(s in {"observed", "predicted"} for s in after),
                "mask_iou_median_before": old_role["mask_iou_median_vs_supplied"],
                "mask_iou_median_after": new_role["mask_iou_median_vs_supplied"],
            }
            old_accuracy = old_role["pose_accuracy"]
            new_accuracy = new_role["pose_accuracy"]
            if old_accuracy["status"] == new_accuracy["status"] == "scored":
                old_error = old_accuracy["per_frame_cloud_mm"]
                new_error = new_accuracy["per_frame_cloud_mm"]
                paired = [
                    (a, b)
                    for a, b in zip(old_error[1:], new_error[1:], strict=True)
                    if a is not None and b is not None
                ]
                result["paired_labeled_pose_error"] = {
                    "frames": len(paired),
                    "cloud_mm_median_before": float(np.median([p[0] for p in paired])) if paired else None,
                    "cloud_mm_median_after": float(np.median([p[1] for p in paired])) if paired else None,
                }
            comparisons.append(result)
    return comparisons


def markdown(results: dict) -> str:
    lines = [
        f"# {results['suite']}",
        "",
        "Observed = accepted image/depth pose; predicted = fallback pose; lost = no pose.",
        "External mask IoU is agreement with supplied masks, not segmentation truth. Pose error is conditional on a published pose.",
        "FLASH demo rows have no independent visibility, mask, or pose labels: their counts cover every frame and measure availability only.",
        "",
    ]
    scored = [case for case in results["cases"] if case["status"] == "scored"]
    external = [case for case in scored if case["kind"] != "flash_prepared_demo"]
    if external and all(
        case["capture"]["mask_source"] == "recorded" or case["capture"]["mask_source"].startswith("supplied")
        for case in external
    ):
        lines.insert(
            5, "External traces feed supplied masks into perception, so mask IoU 1.00 does not test SAM3."
        )
    domains = sorted({case["domain"] for case in results["cases"]})
    for domain in domains:
        lines += [
            f"## {domain}",
            "",
            "| Case / role | Frames scored | Mask present | Observed / predicted / lost | Mask IoU median | Labeled pose error |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for case in results["cases"]:
            if case["domain"] != domain:
                continue
            if case["status"] != "scored":
                lines.append(f"| {case['id']} | missing local artifact | — | — | — | — |")
                continue
            for role, metrics in case["roles"].items():
                state = metrics["states"]
                accuracy = metrics["pose_accuracy"]
                error = (
                    f"{accuracy['cloud_mm_median']:.1f} mm / {accuracy['angle_deg_median']:.1f}° "
                    f"on {accuracy['available']}/{accuracy['total']} frames"
                    if accuracy["status"] == "scored" and accuracy["available"]
                    else "no labeled pose score"
                )
                iou = metrics["mask_iou_median_vs_supplied"]
                frame_label = (
                    str(metrics["reference_visible_frames"])
                    if metrics["reference_visible_frames"] is not None
                    else f"{metrics['frames_scored']} (visibility unlabeled)"
                )
                lines.append(
                    f"| {case['id']} / {role} | {frame_label} | {metrics['mask_present_frames']} | "
                    f"{state.get('observed', 0)} / {state.get('predicted', 0)} / {state.get('lost', 0)} | "
                    f"{iou:.2f} | {error} |"
                    if iou is not None
                    else f"| {case['id']} / {role} | {frame_label} | {metrics['mask_present_frames']} | "
                    f"{state.get('observed', 0)} / {state.get('predicted', 0)} / {state.get('lost', 0)} | — | {error} |"
                )
        lines.append("")
    lines += [
        "The suite intentionally has no cross-domain success percentage. See JSON for per-frame states, failures, and provenance.",
        "",
    ]
    if "comparison" in results:
        lines += [
            "## Paired comparison",
            "",
            "| Case / role | Observed before → after | Pose available before → after | Common-frame point error before → after |",
            "| --- | ---: | ---: | ---: |",
        ]
        for item in results["comparison"]:
            if item["status"] != "compared":
                lines.append(f"| {item['id']} | {item['status']} | — | — |")
                continue
            paired = item.get("paired_labeled_pose_error", {})
            error = (
                f"{paired['cloud_mm_median_before']:.1f} → {paired['cloud_mm_median_after']:.1f} mm "
                f"({paired['frames']} frames)"
                if paired.get("frames")
                else "unavailable"
            )
            lines.append(
                f"| {item['id']} / {item['role']} | "
                f"{item['observed_frames_before']} → {item['observed_frames_after']} | "
                f"{item['available_frames_before']} → {item['available_frames_after']} | {error} |"
            )
        lines += [
            "",
            "Only frames with a pose in both runs contribute to paired error. Availability is shown separately.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path(__file__).with_name("flash_tracking_suite.json"))
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON result path; Markdown is written beside it"
    )
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--baseline", type=Path, help="Compare candidate against a saved result on matching frames"
    )
    args = parser.parse_args()
    result = score_suite(json.loads(args.suite.read_text()), require_all=args.require_all)
    result["suite_sha256"] = hashlib.sha256(args.suite.read_bytes()).hexdigest()
    if args.baseline is not None:
        result["comparison"] = compare_results(json.loads(args.baseline.read_text()), result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown(result))
    print(args.output)


if __name__ == "__main__":
    main()
