"""Portable FLASH skill and metric appearance references beside a dataset."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .appearance import AppearanceModel
from .causal_service import CausalTrackingConfig
from .matched_rgbd import ObjectPointModel
from .perception import ObjectReference, ReferenceSnippet
from .skill import FlashSkill
from .symmetry import AxialSymmetry
from .vision_pipeline import VisionPipelineConfig


@dataclass(frozen=True)
class PreparedFlashSkill:
    """One direct-imitation skill with its registered multi-view object models.

    The bundle is an extension artifact. RGB-D source frames, robot actions and
    outcomes remain in the ordinary LeRobot dataset; the model file contains
    only derived points and descriptors that can be rebuilt from its snippets.
    """

    skill: FlashSkill
    references: tuple[ObjectReference, ...]
    appearance_models: tuple[AppearanceModel, ...]
    prompts_by_role: dict[str, str]
    descriptor_model_id: str
    axial_symmetries: tuple[AxialSymmetry, ...] = ()
    vision_pipeline: VisionPipelineConfig | None = None
    causal_tracking: CausalTrackingConfig | None = None

    def __post_init__(self) -> None:
        refs = {reference.role: reference for reference in self.references}
        models = {model.points.role: model for model in self.appearance_models}
        if not refs or len(refs) != len(self.references) or len(models) != len(self.appearance_models):
            raise ValueError("prepared skill needs unique object references and appearance models")
        if set(refs) != set(models) or set(refs) != set(self.prompts_by_role):
            raise ValueError("each object role needs one reference, model and prompt")
        if any(model.points.reference_version != refs[role].version for role, model in models.items()):
            raise ValueError("appearance model frame differs from its object reference")
        if any(not prompt.strip() for prompt in self.prompts_by_role.values()) or len(
            set(self.prompts_by_role.values())
        ) != len(self.prompts_by_role):
            raise ValueError("object prompts must be nonempty and distinct")
        if not self.descriptor_model_id.strip():
            raise ValueError("prepared appearance descriptors need their model identity")
        if len({item.role for item in self.axial_symmetries}) != len(self.axial_symmetries) or not {
            item.role for item in self.axial_symmetries
        } <= set(refs):
            raise ValueError("axial symmetries must name distinct prepared object roles")
        symmetries = {item.role: item for item in self.axial_symmetries}
        for interval in self.skill.intervals:
            rule = interval.turn_at_point
            if rule is None or rule.source_evidence != "symmetry_invariant":
                continue
            symmetry = symmetries.get(interval.reference_role)
            if (
                symmetry is None
                or not symmetry.lock_to_baseline
                or not np.allclose(rule.pivot_reference_m, symmetry.center_object_m, rtol=0, atol=1e-9)
                or not np.allclose(rule.axis_reference, symmetry.axis_object, rtol=0, atol=1e-9)
            ):
                raise ValueError(
                    "a symmetry-invariant turn must use the prepared reference center and direction"
                )
        if any(
            not snippet.mask_artifact
            or Path(snippet.mask_artifact).is_absolute()
            or ".." in Path(snippet.mask_artifact).parts
            for reference in self.references
            for snippet in reference.snippets
        ):
            raise ValueError("reference masks must use dataset-relative artifact paths")
        required = {
            (interval.reference_role, interval.reference_version)
            for interval in self.skill.intervals
            if interval.relation.value != "ee_to_start"
        }
        required.update(
            (interval.held_role, interval.held_reference_version)
            for interval in self.skill.intervals
            if interval.held_role is not None
        )
        if not required <= {(reference.role, reference.version) for reference in self.references}:
            raise ValueError("skill roles or frame versions lack prepared references")

    def save(self, bundle_dir: str | Path) -> Path:
        """Write a versioned extension folder with no pickle-dependent arrays."""
        root = Path(bundle_dir)
        root.mkdir(parents=True, exist_ok=False)
        self.skill.save(root / "skill.json")
        arrays = {}
        models = {model.points.role: model for model in self.appearance_models}
        references = []
        for index, reference in enumerate(self.references):
            model = models[reference.role]
            arrays[f"points_{index}"] = model.points.points_object_m
            arrays[f"descriptors_{index}"] = model.descriptors
            references.append(
                {
                    "role": reference.role,
                    "version": reference.version,
                    "model_index": index,
                    "snippets": [
                        {
                            "repo_id": snippet.repo_id,
                            "episode": snippet.episode,
                            "frame_index": snippet.frame_index,
                            "object_from_camera": snippet.object_from_camera.tolist(),
                            "mask_artifact": snippet.mask_artifact,
                            "calibration_version": snippet.calibration_version,
                        }
                        for snippet in reference.snippets
                    ],
                }
            )
        np.savez_compressed(root / "models.npz", **arrays)
        manifest = {
            "schema_version": 1,
            "vision_pipeline": self.vision_pipeline.to_dict() if self.vision_pipeline else None,
            "causal_tracking": asdict(self.causal_tracking) if self.causal_tracking else None,
            "skill": "skill.json",
            "models": "models.npz",
            "references": references,
            "prompts_by_role": self.prompts_by_role,
            "descriptor_model_id": self.descriptor_model_id,
            "axial_symmetries": [
                {
                    "role": item.role,
                    "axis_object": item.axis_object.tolist(),
                    "center_object_m": item.center_object_m.tolist(),
                    "baseline_base_from_object": item.baseline_base_from_object.tolist(),
                    "lock_to_baseline": item.lock_to_baseline,
                    "max_observed_tilt_rad": item.max_observed_tilt_rad,
                    "ends_interchangeable": item.ends_interchangeable,
                }
                for item in self.axial_symmetries
            ],
        }
        (root / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return root / "bundle.json"

    @classmethod
    def load(cls, manifest_path: str | Path) -> PreparedFlashSkill:
        source = Path(manifest_path)
        manifest = json.loads(source.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported FLASH prepared-skill version")
        for key in ("skill", "models"):
            if Path(manifest[key]).name != manifest[key]:
                raise ValueError("prepared-skill files must stay inside the bundle directory")
        skill = FlashSkill.load(source.parent / manifest["skill"])
        references, appearance_models = [], []
        with np.load(source.parent / manifest["models"], allow_pickle=False) as arrays:
            for item in manifest["references"]:
                snippets = tuple(
                    ReferenceSnippet(
                        entry["repo_id"],
                        entry["episode"],
                        entry["frame_index"],
                        item["role"],
                        entry["object_from_camera"],
                        entry["mask_artifact"],
                        entry["calibration_version"],
                    )
                    for entry in item["snippets"]
                )
                reference = ObjectReference(item["role"], item["version"], snippets)
                index = item["model_index"]
                model = AppearanceModel(
                    ObjectPointModel(reference.role, reference.version, arrays[f"points_{index}"]),
                    arrays[f"descriptors_{index}"],
                )
                references.append(reference)
                appearance_models.append(model)
        return cls(
            skill,
            tuple(references),
            tuple(appearance_models),
            manifest["prompts_by_role"],
            manifest["descriptor_model_id"],
            tuple(AxialSymmetry(**item) for item in manifest.get("axial_symmetries", [])),
            VisionPipelineConfig.from_dict(manifest["vision_pipeline"])
            if manifest.get("vision_pipeline") is not None
            else None,
            CausalTrackingConfig.from_config(manifest["causal_tracking"])
            if manifest.get("causal_tracking") is not None
            else None,
        )


def snapshot_prepared_skill(
    prepared: PreparedFlashSkill,
    *,
    source_dataset_root: str | Path,
    recording_dataset_root: str | Path,
    episode_index: int,
) -> Path:
    """Keep a trial's reference model and masks beside its ordinary dataset.

    Snippets retain their original source-episode identities and object frames.
    Only their mask artifact paths change to point at copies in the recording.
    A future demo compiler can therefore use the trial's pose trace without
    depending on a mutable reference directory outside that dataset.
    """
    if episode_index < 0:
        raise ValueError("recorded episode index must be nonnegative")
    source_root = Path(source_dataset_root).expanduser().resolve()
    recording_root = Path(recording_dataset_root).expanduser().resolve()
    source_masks = (source_root / "flash" / "v1" / "masks").resolve()
    episode_name = f"episode_{episode_index:06d}"
    relative_masks = Path("flash") / "v1" / "masks" / "reference_snapshots" / episode_name
    bundle_root = recording_root / "flash" / "v1" / "reference_bundles" / episode_name
    if bundle_root.exists() or (recording_root / relative_masks).exists():
        raise FileExistsError(f"FLASH reference snapshot already exists for {episode_name}")

    copies = []
    references = []
    for role_index, reference in enumerate(prepared.references):
        snippets = []
        for view_index, snippet in enumerate(reference.snippets):
            source = (source_root / snippet.mask_artifact).resolve()
            if not source.is_relative_to(source_masks) or not source.is_file():
                raise ValueError(f"missing or unsafe reference mask for {reference.role}")
            relative = relative_masks / f"role_{role_index:03d}_view_{view_index:03d}.png"
            copies.append((source, recording_root / relative))
            snippets.append(
                ReferenceSnippet(
                    snippet.repo_id,
                    snippet.episode,
                    snippet.frame_index,
                    snippet.object_role,
                    snippet.object_from_camera,
                    relative.as_posix(),
                    snippet.calibration_version,
                )
            )
        references.append(ObjectReference(reference.role, reference.version, tuple(snippets)))

    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    snapshot = PreparedFlashSkill(
        prepared.skill,
        tuple(references),
        prepared.appearance_models,
        prepared.prompts_by_role,
        prepared.descriptor_model_id,
        prepared.axial_symmetries,
        prepared.vision_pipeline,
        prepared.causal_tracking,
    )
    return snapshot.save(bundle_root)
