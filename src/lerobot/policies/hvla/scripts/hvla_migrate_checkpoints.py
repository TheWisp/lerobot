#!/usr/bin/env python
"""Migrate legacy HVLA checkpoints to standard LeRobot format.

Legacy format:
    outputs/flow_s1_hvla_v7/
        checkpoint-10000/
            model.safetensors
            norm_stats.pt
            optimizer.pt

Standard format (after migration):
    outputs/flow_s1_hvla_v7/
        checkpoints/
            checkpoint-10000/
                pretrained_model/
                    model.safetensors
                    norm_stats.pt
                    config.json
                    train_config.json
                training_state/
                    optimizer.pt
                    training_step.json
            last -> checkpoint-50000

Usage:
    python src/lerobot/policies/hvla/scripts/hvla_migrate_checkpoints.py outputs/flow_s1_hvla_v7
    python src/lerobot/policies/hvla/scripts/hvla_migrate_checkpoints.py outputs/flow_s1_hvla_v7 --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path

CONTRACT_FIELDS = (
    "action_feature_names",
    "robot_state_feature",
    "state_feature_names",
    "image_resize_shape",
)


def _pretrained_dirs(run_dir: Path) -> list[Path]:
    """Every ``pretrained_model/`` under a run, in either run layout."""
    found = []
    for base in (run_dir, run_dir / "checkpoints"):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and child.name.startswith("checkpoint-"):
                pretrained = child / "pretrained_model"
                if (pretrained / "config.json").is_file():
                    found.append(pretrained)
    return found


def backfill_contract(pretrained_dir: Path, dry_run: bool = False) -> str:
    """Add the ordered feature contract to a checkpoint that predates it.

    Preconditions: ``pretrained_dir`` holds ``config.json`` and the
    ``train_config.json`` written beside it at training time, and the training
    dataset is still resolvable. Postcondition: on ``"backfilled"`` the config
    satisfies :meth:`FlowMatchingS1Config.from_checkpoint_dict`.

    Names come from the training dataset's own metadata — the same source
    training read — and every one is checked against the dimensions already in
    the checkpoint. A checkpoint whose dataset disagrees is refused, never
    guessed: a wrong order is worse than a failed load, because it mis-drives a
    robot silently.
    """
    config_path = pretrained_dir / "config.json"
    config = json.loads(config_path.read_text())
    missing = [name for name in CONTRACT_FIELDS if name not in config]
    if not missing:
        return "complete"

    train_config_path = pretrained_dir / "train_config.json"
    if not train_config_path.is_file():
        return "no train_config.json — cannot verify feature order; retrain or write the contract by hand"
    train_config = json.loads(train_config_path.read_text())
    repo_id = (train_config.get("dataset") or {}).get("repo_id")
    if not repo_id:
        return "train_config.json records no dataset repo_id — cannot verify feature order"

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    try:
        features = LeRobotDatasetMetadata(repo_id).features
    except Exception as exc:  # noqa: BLE001 — any resolution failure is the same answer to the operator
        return (
            f"training dataset {repo_id!r} is unavailable ({type(exc).__name__}); cannot verify feature order"
        )

    action_names = list((features.get("action") or {}).get("names") or [])
    if len(action_names) != config.get("action_dim"):
        return (
            f"dataset {repo_id!r} has {len(action_names)} action names but the checkpoint "
            f"declares action_dim={config.get('action_dim')} — refusing to guess the order"
        )

    state_dim = config.get("state_dim")
    state_names = list((features.get("observation.state") or {}).get("names") or [])
    if state_dim:
        if len(state_names) != state_dim:
            return (
                f"dataset {repo_id!r} has {len(state_names)} state names but the checkpoint "
                f"declares state_dim={state_dim} — refusing to guess the order"
            )
    else:
        state_names = []

    dataset_cameras = {key for key in features if key.startswith("observation.images.")}
    checkpoint_cameras = set(config.get("image_features") or {})
    if not checkpoint_cameras <= dataset_cameras:
        return (
            f"checkpoint cameras {sorted(checkpoint_cameras - dataset_cameras)} are absent from "
            f"dataset {repo_id!r} — the recorded dataset is not the one this model was trained on"
        )

    resize = train_config.get("resize_images") or "224x224"
    height, _, width = resize.partition("x")
    if not (height.isdigit() and width.isdigit()):
        return f"train_config.json resize_images={resize!r} is not HxW — cannot recover the input resolution"

    config.setdefault("feature_contract_version", 1)
    config["action_feature_names"] = action_names
    config["robot_state_feature"] = bool(state_dim)
    config["state_feature_names"] = state_names
    config["image_resize_shape"] = [int(height), int(width)]

    print(f"    Backfill contract from {repo_id} ({len(action_names)} action, {len(state_names)} state)")
    if not dry_run:
        config_path.write_text(json.dumps(config, indent=2))
    return "backfilled"


def backfill_run(run_dir: Path, dry_run: bool = False) -> None:
    """Backfill the feature contract for every already-standard checkpoint in a run."""
    pretrained_dirs = _pretrained_dirs(run_dir)
    if not pretrained_dirs:
        print(f"No standard-layout checkpoints found in {run_dir}")
        return

    print(f"Found {len(pretrained_dirs)} standard-layout checkpoint(s) in {run_dir}")
    for pretrained in pretrained_dirs:
        print(f"\n  {pretrained.parent.name}:")
        status = backfill_contract(pretrained, dry_run=dry_run)
        if status == "complete":
            print("    Contract already present — nothing to do")
        elif status != "backfilled":
            print(f"    SKIPPED: {status}")


def migrate_run(run_dir: Path, dry_run: bool = False):
    """Migrate a single HVLA training run directory."""
    # Find legacy checkpoint dirs (checkpoint-N with model.safetensors, no pretrained_model/)
    legacy_ckpts = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        if (child / "model.safetensors").exists() and not (child / "pretrained_model").exists():
            legacy_ckpts.append(child)

    if not legacy_ckpts:
        # Also check under checkpoints/ subdirectory
        ckpts_dir = run_dir / "checkpoints"
        if ckpts_dir.is_dir():
            for child in sorted(ckpts_dir.iterdir()):
                if not child.is_dir() or not child.name.startswith("checkpoint-"):
                    continue
                if (child / "model.safetensors").exists() and not (child / "pretrained_model").exists():
                    legacy_ckpts.append(child)

    if not legacy_ckpts:
        # Already standard-layout: the remaining gap is the feature contract,
        # which checkpoints trained before it exists do not carry.
        backfill_run(run_dir, dry_run=dry_run)
        return

    print(f"Found {len(legacy_ckpts)} legacy checkpoint(s) in {run_dir}")

    # Determine if we need to create checkpoints/ wrapper
    needs_wrapper = not (run_dir / "checkpoints").is_dir()
    if needs_wrapper:
        ckpts_dir = run_dir / "checkpoints"
        print(f"  Will create {ckpts_dir}/")
        if not dry_run:
            ckpts_dir.mkdir(exist_ok=True)

    for ckpt in legacy_ckpts:
        print(f"\n  Migrating {ckpt.name}:")

        # Target location (may need to move into checkpoints/)
        target_dir = run_dir / "checkpoints" / ckpt.name if needs_wrapper else ckpt

        pretrained_dir = target_dir / "pretrained_model"
        training_state_dir = target_dir / "training_state"

        # Extract step from dir name
        try:
            step = int(ckpt.name.split("-")[-1])
        except ValueError:
            step = None

        if needs_wrapper and target_dir != ckpt:
            print(f"    Move {ckpt} → {target_dir}")
            if not dry_run:
                # safe-destruct: explicit migration script
                shutil.move(str(ckpt), str(target_dir))

        print(f"    Create {pretrained_dir}/")
        if not dry_run:
            pretrained_dir.mkdir(exist_ok=True)

        # Move model.safetensors and norm_stats.pt into pretrained_model/
        for fname in ["model.safetensors", "norm_stats.pt"]:
            src = target_dir / fname
            dst = pretrained_dir / fname
            if src.exists():
                print(f"    Move {fname} → pretrained_model/{fname}")
                if not dry_run:
                    # safe-destruct: explicit migration script
                    shutil.move(str(src), str(dst))

        # This migrator is intentionally for the known bimanual SO-107
        # prototype layout. Recording the verified order here is safer than
        # teaching the normal checkpoint loader to guess every 14-D model.
        feature_names = [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_forearm_roll.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_forearm_roll.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ]

        # Create config.json
        config = {
            "type": "hvla_flow_s1",
            "feature_contract_version": 1,
            "action_dim": 14,
            "action_feature_names": feature_names,
            "robot_state_feature": True,
            "state_dim": 14,
            "state_feature_names": feature_names,
            "chunk_size": 50,
            "hidden_dim": 768,
            "num_heads": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 6,
            "s2_latent_dim": 2048,
            "num_inference_steps": 10,
            "rtc_max_delay": 6,
            "rtc_drop_prob": 0.2,
            "image_features": {
                "observation.images.front": 224,
                "observation.images.left_wrist": 224,
                "observation.images.right_wrist": 224,
                "observation.images.top": 224,
            },
            "image_resize_shape": [224, 224],
            "dino_model": "dinov2_vits14",
        }
        config_path = pretrained_dir / "config.json"
        print("    Write config.json")
        if not dry_run:
            config_path.write_text(json.dumps(config, indent=2))

        # Create training_state/
        print(f"    Create {training_state_dir}/")
        if not dry_run:
            training_state_dir.mkdir(exist_ok=True)

        # Move optimizer.pt into training_state/
        opt_src = target_dir / "optimizer.pt"
        opt_dst = training_state_dir / "optimizer.pt"
        if opt_src.exists():
            print("    Move optimizer.pt → training_state/optimizer.pt")
            if not dry_run:
                # safe-destruct: explicit migration script
                shutil.move(str(opt_src), str(opt_dst))

        # Create training_step.json
        if step is not None:
            step_path = training_state_dir / "training_step.json"
            print(f"    Write training_step.json (step={step})")
            if not dry_run:
                step_path.write_text(json.dumps({"step": step}))

    # Create 'last' symlink pointing to highest checkpoint
    if not dry_run:
        ckpts_dir = run_dir / "checkpoints" if needs_wrapper else run_dir
        ckpt_dirs = sorted(
            [d for d in ckpts_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
        )
        if ckpt_dirs:
            last_link = ckpts_dir / "last"
            if last_link.exists() or last_link.is_symlink():
                # safe-destruct: explicit migration script: symlink update
                last_link.unlink()
            last_link.symlink_to(ckpt_dirs[-1].name)
            print(f"\n  Created symlink: last → {ckpt_dirs[-1].name}")

    print(f"\nMigration {'would be ' if dry_run else ''}complete for {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy HVLA checkpoints to standard format")
    parser.add_argument("run_dir", help="Path to HVLA training run directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        return

    migrate_run(run_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
