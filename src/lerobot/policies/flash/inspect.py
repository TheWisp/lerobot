"""Bounded subprocess entry point for inspecting a FLASH dataset sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .skill import FlashSkill


def inspect_skill(path: str) -> dict:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("skill file must exist and be at most 16 MiB")
    skill = FlashSkill.load(source)
    missing = []
    missing_frames = []
    for interval in skill.intervals:
        if not interval.reference_version:
            missing_frames.append(f"{interval.name}: reference object frame version")
        if interval.relation.value == "held_to_target" and not interval.held_reference_version:
            missing_frames.append(f"{interval.name}: held object frame version")
        if interval.relation.value == "held_to_target" and not interval.held_role:
            missing.append(f"{interval.name}: held object role")
        if interval.relation.value == "held_to_target" and not any(
            transition.action.value == "attach"
            and transition.role == interval.held_role
            and transition.reference_version == interval.held_reference_version
            for transition in skill.attachment_transitions
        ):
            missing.append(f"{interval.name}: matching demonstrated attachment event")
    missing = missing_frames + missing
    return {
        "schema_version": skill.schema_version,
        "source_repo_id": skill.source_repo_id,
        "source_episode": skill.source_episode,
        "milestones": [
            {
                "name": m.name,
                "depends_on_roles": m.depends_on_roles,
                "pose_relation": (
                    {
                        "moving_role": m.pose_relation.moving_role,
                        "reference_role": m.pose_relation.reference_role,
                        "translation_tolerance_m": m.pose_relation.translation_tolerance_m,
                        "rotation_tolerance_rad": m.pose_relation.rotation_tolerance_rad,
                    }
                    if m.pose_relation is not None
                    else None
                ),
            }
            for m in skill.milestones
        ],
        "attachment_transitions": [
            {
                "event_id": transition.event_id,
                "action": transition.action.value,
                "role": transition.role,
                "reference_version": transition.reference_version,
            }
            for transition in skill.attachment_transitions
        ],
        "reference_frames_declared": not missing_frames,
        "runtime_bindings_complete": not missing,
        "missing": missing,
        "intervals": [
            {
                "name": interval.name,
                "mode": interval.mode.value,
                "relation": interval.relation.value,
                "reference_role": interval.reference_role,
                "held_role": interval.held_role,
                "samples": len(interval.samples),
                "demo_duration_s": interval.demo_duration_s,
                "entry_milestone": interval.entry_milestone,
                "completion_milestone": interval.completion_milestone,
            }
            for interval in skill.intervals
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("skill_path")
    args = parser.parse_args()
    print(json.dumps(inspect_skill(args.skill_path)))


if __name__ == "__main__":
    main()
