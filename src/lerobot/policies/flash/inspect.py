"""Bounded subprocess entry point for inspecting a FLASH dataset sidecar."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .prepared import PreparedFlashSkill
from .skill import FlashSkill


def inspect_skill(path: str) -> dict:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("skill file must exist and be at most 16 MiB")
    metadata = json.loads(source.read_text())
    prepared = PreparedFlashSkill.load(source) if "models" in metadata and "references" in metadata else None
    skill = prepared.skill if prepared is not None else FlashSkill.load(source)
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
    if prepared is None:
        missing.append("prepared RGB-D object references")
    warnings = []
    for interval in skill.intervals:
        if interval.relation.value == "held_to_target":
            if any(sample.held_evidence == "conflicted_prediction" for sample in interval.samples):
                warnings.append(f"{interval.name}: held-object demo pose has unresolved visual conflict")
            elif not any(
                sample.held_evidence in ("observed", "vision_corrected_prediction")
                for sample in interval.samples
            ):
                warnings.append(f"{interval.name}: no independently observed held-object demo correction")
    return {
        "schema_version": skill.schema_version,
        "vision_pipeline": prepared.vision_pipeline.to_dict()
        if prepared and prepared.vision_pipeline
        else None,
        "source_repo_id": skill.source_repo_id,
        "source_episode": skill.source_episode,
        "milestones": [
            {
                "name": m.name,
                "depends_on_roles": m.depends_on_roles,
                "require_post_motion_evidence": m.require_post_motion_evidence,
                "pose_relation": (
                    {
                        "moving_role": m.pose_relation.moving_role,
                        "reference_role": m.pose_relation.reference_role,
                        "translation_tolerance_m": m.pose_relation.translation_tolerance_m,
                        "rotation_tolerance_rad": m.pose_relation.rotation_tolerance_rad,
                        "moving_point_m": m.pose_relation.moving_point_m,
                        "moving_direction": m.pose_relation.moving_direction,
                        "direction_tolerance_rad": m.pose_relation.direction_tolerance_rad,
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
        "prepared_references": (
            [
                {
                    "role": reference.role,
                    "version": reference.version,
                    "views": len(reference.snippets),
                    "model_points": len(model.points.points_object_m),
                }
                for reference, model in zip(prepared.references, prepared.appearance_models, strict=True)
            ]
            if prepared is not None
            else []
        ),
        "descriptor_model_id": prepared.descriptor_model_id if prepared is not None else None,
        "missing": missing,
        "warnings": warnings,
        "intervals": [
            {
                "name": interval.name,
                "mode": interval.mode.value,
                "endpoint_requirement": interval.endpoint_requirement.value,
                "relation": interval.relation.value,
                "reference_role": interval.reference_role,
                "held_role": interval.held_role,
                "reference_evidence": dict(Counter(sample.reference_evidence for sample in interval.samples)),
                "held_evidence": dict(
                    Counter(
                        sample.held_evidence for sample in interval.samples if sample.demo_held is not None
                    )
                ),
                "samples": len(interval.samples),
                "demo_duration_s": interval.demo_duration_s,
                "entry_milestone": interval.entry_milestone,
                "completion_milestone": interval.completion_milestone,
            }
            for interval in skill.intervals
        ],
    }


def inspect_dataset(root: str, episode: int) -> dict:
    """Find FLASH sidecars inside one ordinary local LeRobot dataset.

    A source episode may have several skills, each with different interval or
    reference annotations. Discovery does not invent a skill from raw video.
    """
    dataset_root = Path(root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError("Select a local LeRobot dataset containing meta/info.json")
    info = json.loads(info_path.read_text())
    episode_count = info.get("total_episodes")
    if not isinstance(episode_count, int) or episode_count < 0:
        raise ValueError("LeRobot dataset metadata has no valid episode count")
    if not 0 <= episode < episode_count:
        raise ValueError(f"episode must be between 0 and {episode_count - 1}")

    extension = dataset_root / "flash" / "v1"
    candidates = sorted((extension / "prepared").glob("*/bundle.json"))
    candidates += sorted((extension / "skills").glob("*.json"))
    skills = []
    warnings = []
    for candidate in candidates:
        try:
            inspection = inspect_skill(str(candidate))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Cannot read {candidate.relative_to(dataset_root)}: {exc}")
            continue
        if inspection["source_episode"] == episode:
            skills.append(
                {
                    "path": str(candidate),
                    "kind": "prepared" if candidate.name == "bundle.json" else "annotation",
                    "inspection": inspection,
                }
            )

    missing = []
    if not skills:
        missing.append("No FLASH spatial skill is stored for this episode")
    elif not any(skill["kind"] == "prepared" for skill in skills):
        missing.append("No prepared RGB-D object references are stored for this episode")
    launchable = any(
        skill["kind"] == "prepared" and skill["inspection"]["runtime_bindings_complete"] for skill in skills
    )
    if skills and not launchable and not missing:
        missing.append("Prepared FLASH skill has unresolved runtime bindings")
    return {
        "dataset_root": str(dataset_root),
        "episode": episode,
        "has_flash_extension": extension.is_dir(),
        "skills": skills,
        "missing": missing,
        "warnings": warnings,
        "can_launch": launchable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("skill_path", nargs="?")
    parser.add_argument("--dataset-root")
    parser.add_argument("--episode", type=int)
    args = parser.parse_args()
    if args.dataset_root is not None:
        if args.skill_path is not None or args.episode is None:
            parser.error("dataset inspection needs --dataset-root and --episode only")
        result = inspect_dataset(args.dataset_root, args.episode)
    elif args.skill_path is not None and args.episode is None:
        result = inspect_skill(args.skill_path)
    else:
        parser.error("provide a skill path or a dataset root and episode")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
