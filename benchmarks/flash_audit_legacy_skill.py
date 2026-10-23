"""Audit a prepared FLASH skill against the sealed demonstration it migrated.

This checks recorded timing, calibrated EE poses, gripper commands, interval
coverage and semantic events. It cannot certify the task intent of a waypoint
that the FLASH skill schema does not represent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.policies.flash.geometry import inverse
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.sealed_demo_import import _endpoint_requirement, _nearest_index
from lerobot.policies.flash.skill import MotionMode, Relation


def audit(source_root: Path, bundle_path: Path, annotation_path: Path) -> dict:
    annotation = json.loads(annotation_path.read_text())
    keypoints = json.loads((source_root / "keypoints.json").read_text())["keypoints"]
    skill = PreparedFlashSkill.load(bundle_path).skill
    with np.load(source_root / "showservo_demo.npz", allow_pickle=False) as demo:
        times = np.asarray(demo["t"], dtype=float)
        tips = np.asarray(demo["tips"], dtype=float)
        actions = np.asarray(demo["q_cmd"], dtype=float)
    if not len(times) == len(tips) == len(actions):
        raise ValueError("demo arm samples, tips and actions differ in length")

    expected_bounds = {
        0,
        _nearest_index(times, float(annotation["grasp_time_s"])),
        _nearest_index(times, float(annotation["release_time_s"])),
        _nearest_index(times, float(annotation["start_anchored_until_s"])),
        *(_nearest_index(times, float(item["t"])) for item in keypoints),
    }
    actual_bounds = {skill.intervals[0].samples[0].source_frame_index}
    max_time_error_s = max_tip_error_m = max_tip_rotation_rad = max_gripper_error_deg = 0.0
    carried_attachment: np.ndarray | None = None
    max_attachment_translation_m = 0.0
    max_attachment_rotation_rad = 0.0
    interval_rows = []
    events = {}
    previous_end = None
    for interval in skill.intervals:
        indices = [sample.source_frame_index for sample in interval.samples]
        if any(index is None or not 0 <= index < len(times) for index in indices):
            raise ValueError(f"{interval.name}: missing or invalid source frame")
        if previous_end is not None and indices[0] != previous_end:
            raise ValueError(f"{interval.name}: gap or overlap in source frame coverage")
        previous_end = indices[-1]
        actual_bounds.add(indices[-1])
        expected_indices = (
            list(range(indices[0], indices[-1] + 1))
            if interval.mode == MotionMode.DEMONSTRATED
            else [indices[0], indices[-1]]
        )
        if indices != expected_indices:
            raise ValueError(f"{interval.name}: dense or endpoint sample coverage differs from demo")
        for sample, index in zip(interval.samples, indices, strict=True):
            max_time_error_s = max(max_time_error_s, abs(sample.time_s - times[index]))
            max_tip_error_m = max(
                max_tip_error_m, float(np.linalg.norm(sample.demo_ee[:3, 3] - tips[index, :3, 3]))
            )
            tip_delta = sample.demo_ee[:3, :3].T @ tips[index, :3, :3]
            max_tip_rotation_rad = max(
                max_tip_rotation_rad,
                float(np.arccos(np.clip((np.trace(tip_delta) - 1) / 2, -1, 1))),
            )
            max_gripper_error_deg = max(max_gripper_error_deg, abs(sample.gripper - actions[index, -1]))
            if sample.event_id:
                events[sample.event_id] = index
            if interval.relation == Relation.HELD_TO_TARGET:
                attachment = inverse(sample.demo_ee) @ sample.demo_held
                if carried_attachment is None:
                    carried_attachment = attachment
                delta = inverse(carried_attachment) @ attachment
                max_attachment_translation_m = max(
                    max_attachment_translation_m, float(np.linalg.norm(delta[:3, 3]))
                )
                max_attachment_rotation_rad = max(
                    max_attachment_rotation_rad,
                    float(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))),
                )
        interval_rows.append(
            {
                "name": interval.name,
                "frames": [indices[0], indices[-1]],
                "mode": interval.mode.value,
                "endpoint_requirement": interval.endpoint_requirement.value,
                "relation": interval.relation.value,
                "reference_role": interval.reference_role,
                "samples": len(indices),
            }
        )
    if actual_bounds != expected_bounds:
        raise ValueError(
            f"skill boundaries differ from annotated demo: {sorted(actual_bounds ^ expected_bounds)}"
        )
    expected_events = {
        "attach-held": _nearest_index(times, float(annotation["grasp_time_s"])),
        "detach-held": _nearest_index(times, float(annotation["release_time_s"])),
    }
    if events != expected_events:
        raise ValueError(f"skill event frames differ from annotated demo: {events!r}")
    keypoint_constraints = []
    for item in keypoints:
        endpoint = _nearest_index(times, float(item["t"]))
        interval = next(
            (row for row in skill.intervals if row.samples[-1].source_frame_index == endpoint), None
        )
        if item.get("frame") == "world" and (interval is None or interval.relation != Relation.EE_TO_START):
            raise ValueError(f"world-relative keypoint at arm sample {endpoint} follows another object")
        if "constraint" in item:
            expected = _endpoint_requirement(item)
            if interval is None or interval.endpoint_requirement != expected:
                raise ValueError(f"keypoint constraint at arm sample {endpoint} was not preserved")
            keypoint_constraints.append(
                {"frame": endpoint, "constraint": interval.endpoint_requirement.value}
            )
    if (
        max_time_error_s > 1e-9
        or max_tip_error_m > 1e-5
        or max_tip_rotation_rad > 1e-5
        or max_gripper_error_deg > 1e-5
    ):
        raise ValueError("prepared samples differ from recorded times, calibrated tips or gripper actions")
    if max_attachment_translation_m > 1e-8 or max_attachment_rotation_rad > 1e-6:
        raise ValueError("prepared held-object samples change their demonstrated grasp attachment")
    return {
        "source_frames": len(times),
        "intervals": interval_rows,
        "events": events,
        "max_time_error_s": max_time_error_s,
        "max_tip_position_error_m": max_tip_error_m,
        "max_tip_rotation_error_rad": max_tip_rotation_rad,
        "max_gripper_error_deg": max_gripper_error_deg,
        "max_held_attachment_translation_m": max_attachment_translation_m,
        "max_held_attachment_rotation_rad": max_attachment_rotation_rad,
        "keypoint_constraints": keypoint_constraints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("annotation", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source_root, args.bundle, args.annotation), indent=2))


if __name__ == "__main__":
    main()
