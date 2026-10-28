"""Inspect and prepare an ordinary FLASH trial episode without connecting a robot."""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

from .compiler import ActionEventAnnotation, IntervalAnnotation
from .kinematics import SelectedJointKinematics
from .prepared import PreparedFlashSkill
from .recorded_demo import prepare_recorded_episode
from .recording import load_pose_estimates
from .skill import AttachmentAction, AttachmentTransition, EndpointRequirement, MotionMode, Relation
from .trajectory import Evidence


def _source(root: Path, episode: int) -> tuple[dict, Path, PreparedFlashSkill]:
    if episode < 0:
        raise ValueError("episode must be nonnegative")
    extension = root / "flash" / "v1"
    settings = json.loads((extension / "trial_config.json").read_text(encoding="utf-8"))
    relative = settings.get("source_reference_bundle")
    if not isinstance(relative, str) or not relative:
        raise ValueError(
            "this recording has no saved object references; its motion can be viewed but not prepared"
        )
    bundle_path = (root / relative).resolve()
    if not bundle_path.is_relative_to(extension / "reference_bundles"):
        raise ValueError("reference snapshot must be inside this trial dataset")
    if not bundle_path.is_file():
        raise ValueError("this trial's reference snapshot is missing")
    if not settings.get("output_root") or Path(settings["output_root"]).expanduser().resolve() != root:
        raise ValueError("trial configuration belongs to another dataset")
    if not settings.get("output_repo_id"):
        raise ValueError("trial configuration has no dataset identity")
    return settings, bundle_path, PreparedFlashSkill.load(bundle_path)


def inspect_recorded_trial(dataset_root: str | Path, episode: int) -> dict:
    """Return the exact pose evidence available to a timeline editor."""
    root = Path(dataset_root).expanduser().resolve()
    poses = load_pose_estimates(root, episode)
    result = {
        "episode": episode,
        "frame_count": len(poses),
        "ready": False,
        "reason": None,
        "roles": sorted({role for estimates in poses.values() for role in estimates}),
        "pose_counts": {},
        "observed_frames": {},
        "symmetry_roles": [],
    }
    result["pose_counts"] = {
        role: {
            evidence.value: sum(
                estimate.evidence == evidence
                for estimates in poses.values()
                if (estimate := estimates.get(role)) is not None
            )
            for evidence in Evidence
        }
        for role in result["roles"]
    }
    try:
        _, _, prepared = _source(root, episode)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result["reason"] = str(exc)
        return result
    result["ready"] = True
    result["roles"] = [item.role for item in prepared.references]
    result["symmetry_roles"] = [item.role for item in prepared.axial_symmetries]
    result["observed_frames"] = {
        item.role: [
            frame
            for frame, estimates in poses.items()
            if (estimate := estimates.get(item.role)) is not None
            and estimate.evidence == Evidence.OBSERVED
            and estimate.pose is not None
            and estimate.reference_version == item.version
            and estimate.supports(rotation=(True, True, True))
        ]
        for item in prepared.references
    }
    return result


def prepare_recorded_trial(
    dataset_root: str | Path,
    episode: int,
    name: str,
    intervals: list[dict],
    symmetry_baseline_frames: dict[str, int] | None = None,
) -> Path:
    """Compile explicitly marked intervals into a candidate FLASH skill.

    SO-107 is an initial rig adapter; the compiler and sidecar format remain
    robot independent. This adapter reads calibration and robot identity from
    the trial's own saved configuration, never from the running GUI.
    """
    root = Path(dataset_root).expanduser().resolve()
    settings, bundle_path, prepared = _source(root, episode)
    profile = json.loads(Path(settings["robot_profile"]).expanduser().read_text(encoding="utf-8"))
    if profile.get("type") != "so107_follower_gravity":
        raise ValueError("this editor currently needs an SO-107 gravity trial")
    arm = profile["fields"]["arm"]
    if arm != settings["robot_arm"]:
        raise ValueError("trial robot arm and saved profile differ")
    calibration = json.loads(Path(settings["calibration_path"]).expanduser().read_text(encoding="utf-8"))
    arm_names = tuple(joint for joint in MOTOR_NAMES if joint != "gripper")
    kinematics = SelectedJointKinematics(
        make_calibrated_so107_kinematics(calibration, arm=arm),
        tuple(MOTOR_NAMES),
        arm_names,
        {"gripper": 0.0},  # downstream of the calibrated SO-107 tool point
    )
    roles = {item.role: item.version for item in prepared.references}
    annotations = []
    transitions = []
    for index, item in enumerate(intervals):
        relation = Relation(item["relation"])
        reference = item.get("reference_role") if relation != Relation.EE_TO_START else "__start__"
        held = item.get("held_role") if relation == Relation.HELD_TO_TARGET else None
        if relation != Relation.EE_TO_START and reference not in roles:
            raise ValueError(f"section {index + 1} has no tracked reference object")
        if held is not None and (held not in roles or held == reference):
            raise ValueError(f"section {index + 1} needs a distinct tracked moving object")
        action = item.get("attachment_action")
        attachment_role = item.get("attachment_role")
        events = ()
        if action:
            if attachment_role not in roles:
                raise ValueError(f"section {index + 1} has no tracked attachment object")
            event_id = f"section-{index + 1}-{action}"
            events = (ActionEventAnnotation(item["last_frame"], event_id),)
            transitions.append(
                AttachmentTransition(
                    event_id, AttachmentAction(action), attachment_role, roles[attachment_role]
                )
            )
        annotations.append(
            IntervalAnnotation(
                name=item.get("name") or f"section-{index + 1}",
                first_frame=item["first_frame"],
                last_frame=item["last_frame"],
                mode=MotionMode(item["mode"]),
                relation=relation,
                reference_role=reference,
                held_role=held,
                allow_predicted_demo_pose=bool(item.get("allow_predicted_demo_pose", False)),
                events=events,
                endpoint_requirement=EndpointRequirement(item.get("endpoint_requirement", "required")),
            )
        )
    if not annotations or annotations[0].first_frame != 0:
        raise ValueError("the timeline must start at frame 0")
    dataset = LeRobotDataset(
        repo_id=settings["output_repo_id"], root=root, episodes=[episode], download_videos=False
    )
    _, length = dataset.episode_rows(episode)
    if annotations[-1].last_frame != length - 1:
        raise ValueError(f"the timeline must end at frame {length - 1}")
    output = prepare_recorded_episode(
        dataset=dataset,
        dataset_root=root,
        source_repo_id=settings["output_repo_id"],
        source_episode=episode,
        reference_bundle_path=bundle_path,
        output_dir=root / "flash" / "v1" / "prepared" / name,
        kinematics=kinematics,
        arm_joint_names=arm_names,
        gripper_joint_name="gripper",
        annotations=tuple(annotations),
        attachment_transitions=tuple(transitions),
        symmetry_baseline_frames=symmetry_baseline_frames,
    )
    (output.parent / "preparation-report.json").write_text(
        json.dumps(
            {
                "review_status": "candidate",
                "source": "recorded_episode",
                "source_episode": episode,
                "intervals": len(annotations),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
