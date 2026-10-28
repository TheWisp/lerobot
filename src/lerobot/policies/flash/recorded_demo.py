"""Prepare a recorded FLASH episode using the same compiler as a demonstration."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .compiler import (
    DatasetEpisode,
    ForwardKinematics,
    IntervalAnnotation,
    PoseRelationAnnotation,
    compile_skill,
)
from .prepared import PreparedFlashSkill
from .recording import load_pose_estimates
from .skill import AttachmentTransition, MilestoneSpec, Relation
from .trajectory import Evidence


def prepare_recorded_episode(
    *,
    dataset: DatasetEpisode,
    dataset_root: str | Path,
    source_repo_id: str,
    source_episode: int,
    reference_bundle_path: str | Path,
    output_dir: str | Path,
    kinematics: ForwardKinematics,
    arm_joint_names: tuple[str, ...],
    gripper_joint_name: str,
    annotations: tuple[IntervalAnnotation, ...],
    milestones: tuple[MilestoneSpec, ...] = (),
    pose_relation_annotations: tuple[PoseRelationAnnotation, ...] = (),
    attachment_transitions: tuple[AttachmentTransition, ...] = (),
    symmetry_baseline_frames: dict[str, int] | None = None,
) -> Path:
    """Compile an annotated regular episode into a portable candidate bundle.

    The caller must select a reference bundle already copied into this dataset.
    Recorded pose versions must match it. No object identity or pose is inferred
    from a label, and predicted trace poses require explicit annotation consent.
    This same path can later be called by an editor or a batch data workflow.
    """
    root = Path(dataset_root).expanduser().resolve()
    reference_path = Path(reference_bundle_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source_repo_id or source_episode < 0 or not annotations:
        raise ValueError("a recorded skill needs a source episode and at least one annotated interval")
    if not reference_path.is_relative_to(root / "flash" / "v1"):
        raise ValueError("reference bundle must be stored with the recorded dataset")
    if (
        not output.is_relative_to(root / "flash" / "v1")
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", output.name) is None
    ):
        raise ValueError("output must be a named FLASH extension folder in the dataset")
    reference = PreparedFlashSkill.load(reference_path)
    for item in reference.references:
        for snippet in item.snippets:
            path = (root / snippet.mask_artifact).resolve()
            if not path.is_relative_to(root / "flash" / "v1" / "masks") or not path.is_file():
                raise ValueError(f"missing or unsafe reference mask for {item.role}")
    _, length = dataset.episode_rows(source_episode)
    poses = load_pose_estimates(root, source_episode, expected_frames=length)
    skill = compile_skill(
        dataset=dataset,
        source_repo_id=source_repo_id,
        source_episode=source_episode,
        kinematics=kinematics,
        arm_joint_names=arm_joint_names,
        gripper_joint_name=gripper_joint_name,
        poses_by_frame=poses,
        annotations=annotations,
        milestones=milestones,
        pose_relation_annotations=pose_relation_annotations,
        attachment_transitions=attachment_transitions,
    )
    used_roles = {
        interval.reference_role for interval in skill.intervals if interval.relation != Relation.EE_TO_START
    }
    used_roles.update(interval.held_role for interval in skill.intervals if interval.held_role)
    used_roles.update(role for milestone in skill.milestones for role in milestone.depends_on_roles)
    relevant_symmetries = tuple(item for item in reference.axial_symmetries if item.role in used_roles)
    baselines = symmetry_baseline_frames or {}
    if set(baselines) != {item.role for item in relevant_symmetries}:
        raise ValueError("each used symmetry needs an observed baseline frame in the recorded episode")
    symmetries = []
    for symmetry in relevant_symmetries:
        frame_index = baselines[symmetry.role]
        estimate = poses.get(frame_index, {}).get(symmetry.role)
        if (
            estimate is None
            or estimate.pose is None
            or estimate.evidence != Evidence.OBSERVED
            or estimate.reference_version
            != next(item.version for item in reference.references if item.role == symmetry.role)
            or not estimate.supports(rotation=(True, True, True))
        ):
            raise ValueError(f"{symmetry.role} has no observed symmetry baseline at frame {frame_index}")
        symmetries.append(replace(symmetry, baseline_base_from_object=estimate.pose))
    return PreparedFlashSkill(
        skill,
        reference.references,
        reference.appearance_models,
        reference.prompts_by_role,
        reference.descriptor_model_id,
        tuple(symmetries),
        reference.vision_pipeline,
        reference.causal_tracking,
        reference.surface_models,
    ).save(output)
