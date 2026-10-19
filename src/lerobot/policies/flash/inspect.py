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
    for interval in skill.intervals:
        if not interval.reference_version:
            missing.append(f"{interval.name}: reference object frame version")
        if interval.relation.value == "held_to_target" and not interval.held_reference_version:
            missing.append(f"{interval.name}: held object frame version")
    return {
        "schema_version": skill.schema_version,
        "source_repo_id": skill.source_repo_id,
        "source_episode": skill.source_episode,
        "milestones": [{"name": m.name, "depends_on_roles": m.depends_on_roles} for m in skill.milestones],
        "reference_frames_declared": not missing,
        "missing": missing,
        "intervals": [
            {
                "name": interval.name,
                "mode": interval.mode.value,
                "relation": interval.relation.value,
                "reference_role": interval.reference_role,
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
