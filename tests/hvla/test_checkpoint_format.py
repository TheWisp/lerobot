"""Tests for HVLA checkpoint save/load in standard LeRobot format.

Verifies:
  - Training saves config.json, model.safetensors, norm_stats.pt, training_step.json
  - from_pretrained loads from standard format (directory path)
  - from_pretrained loads from legacy flat format (backward compat)
  - config.json values are read correctly
  - GUI model scanner can discover HVLA checkpoints
"""

import json
from pathlib import Path

import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy


def _checkpoint_config(**overrides):
    config = {
        "type": "hvla_flow_s1",
        "feature_contract_version": 1,
        "action_dim": 4,
        "action_feature_names": ["a0", "a1", "a2", "a3"],
        "robot_state_feature": True,
        "state_dim": 4,
        "state_feature_names": ["s0", "s1", "s2", "s3"],
        "image_features": {},
        "image_resize_shape": None,
        "chunk_size": 10,
        "hidden_dim": 64,
        "num_heads": 4,
        "dim_feedforward": 128,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "s2_latent_dim": 32,
        "s2_proj_hidden": 16,
        "use_dino_backbone": False,
    }
    config.update(overrides)
    return config


@pytest.fixture
def small_config():
    """Minimal config for fast tests (no DINOv2)."""
    return FlowMatchingS1Config(
        use_dino_backbone=False,
        image_features={},
        hidden_dim=64,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        action_dim=4,
        action_feature_names=["a0", "a1", "a2", "a3"],
        robot_state_feature=True,
        state_dim=4,
        state_feature_names=["s0", "s1", "s2", "s3"],
        chunk_size=10,
        s2_latent_dim=32,
        s2_proj_hidden=16,
    )


class TestStandardCheckpointFormat:
    """Test saving and loading in standard LeRobot format."""

    def test_save_creates_standard_structure(self, small_config, tmp_path):
        """save_checkpoint should create pretrained_model/ and training_state/."""
        import safetensors.torch as sft

        policy = FlowMatchingS1Policy(small_config)

        # Simulate what train.py's save_checkpoint does
        ckpt_dir = tmp_path / "checkpoint-100"
        pretrained_dir = ckpt_dir / "pretrained_model"
        pretrained_dir.mkdir(parents=True)
        training_state_dir = ckpt_dir / "training_state"
        training_state_dir.mkdir(parents=True)

        sft.save_file(dict(policy.state_dict()), str(pretrained_dir / "model.safetensors"))
        torch.save(
            {"action_mean": torch.zeros(4), "action_std": torch.ones(4)},
            str(pretrained_dir / "norm_stats.pt"),
        )
        (pretrained_dir / "config.json").write_text(json.dumps(_checkpoint_config()))
        (training_state_dir / "training_step.json").write_text(json.dumps({"step": 100}))

        # Verify structure
        assert (pretrained_dir / "model.safetensors").exists()
        assert (pretrained_dir / "config.json").exists()
        assert (pretrained_dir / "norm_stats.pt").exists()
        assert (training_state_dir / "training_step.json").exists()

    def test_load_from_standard_dir(self, small_config, tmp_path):
        """from_pretrained should load from checkpoint dir with pretrained_model/."""
        import safetensors.torch as sft

        policy = FlowMatchingS1Policy(small_config)
        pretrained_dir = tmp_path / "pretrained_model"
        pretrained_dir.mkdir()
        sft.save_file(dict(policy.state_dict()), str(pretrained_dir / "model.safetensors"))
        torch.save(
            {"action_mean": torch.zeros(4), "action_std": torch.ones(4)},
            str(pretrained_dir / "norm_stats.pt"),
        )
        (pretrained_dir / "config.json").write_text(json.dumps(_checkpoint_config()))

        # Load by passing the parent directory
        loaded = FlowMatchingS1Policy.from_pretrained(str(tmp_path))
        assert loaded.config.action_dim == 4
        assert loaded.config.hidden_dim == 64
        assert loaded.config.s2_latent_dim == 32
        assert loaded._action_mean is not None

    def test_load_from_legacy_flat(self, small_config, tmp_path):
        """from_pretrained should load from legacy flat checkpoint (no pretrained_model/)."""
        import safetensors.torch as sft

        policy = FlowMatchingS1Policy(small_config)
        sft.save_file(dict(policy.state_dict()), str(tmp_path / "model.safetensors"))
        torch.save(
            {"action_mean": torch.zeros(4), "action_std": torch.ones(4)}, str(tmp_path / "norm_stats.pt")
        )

        # Load by passing directory (no pretrained_model/ subdir)
        loaded = FlowMatchingS1Policy.from_pretrained(str(tmp_path), config=small_config)
        assert loaded._action_mean is not None

    def test_load_from_file_path(self, small_config, tmp_path):
        """from_pretrained should load from direct file path."""
        import safetensors.torch as sft

        policy = FlowMatchingS1Policy(small_config)
        model_path = tmp_path / "model.safetensors"
        sft.save_file(dict(policy.state_dict()), str(model_path))

        loaded = FlowMatchingS1Policy.from_pretrained(str(model_path), config=small_config)
        assert loaded is not None

    def test_config_json_overrides_defaults(self, small_config, tmp_path):
        """Config values from config.json should override defaults."""
        import safetensors.torch as sft

        policy = FlowMatchingS1Policy(small_config)
        pretrained_dir = tmp_path / "pretrained_model"
        pretrained_dir.mkdir()
        sft.save_file(dict(policy.state_dict()), str(pretrained_dir / "model.safetensors"))
        (pretrained_dir / "config.json").write_text(
            json.dumps(
                _checkpoint_config(
                    num_inference_steps=5,
                    rtc_max_delay=3,
                    robot_state_feature=False,
                    state_dim=0,
                    state_feature_names=[],
                    image_resize_shape=[192, 256],
                )
            )
        )

        # Load without providing config — should read from config.json
        loaded = FlowMatchingS1Policy.from_pretrained(str(tmp_path))
        assert loaded.config.num_inference_steps == 5
        assert loaded.config.rtc_max_delay == 3
        assert loaded.config.action_feature_names == ["a0", "a1", "a2", "a3"]
        assert loaded.config.robot_state_feature is False
        assert loaded.config.state_feature_names == []
        assert loaded.config.image_resize_shape == (192, 256)

    def test_visual_checkpoint_without_camera_metadata_fails(self, tmp_path):
        pretrained_dir = tmp_path / "pretrained_model"
        pretrained_dir.mkdir()
        (pretrained_dir / "config.json").write_text(json.dumps(_checkpoint_config(use_dino_backbone=True)))

        with pytest.raises(ValueError, match="does not record any image features"):
            FlowMatchingS1Policy.from_pretrained(str(tmp_path))

    def test_stateless_checkpoint_rejects_state_feature_names(self, tmp_path):
        pretrained_dir = tmp_path / "pretrained_model"
        pretrained_dir.mkdir()
        (pretrained_dir / "config.json").write_text(
            json.dumps(
                _checkpoint_config(
                    robot_state_feature=False,
                    state_dim=0,
                    state_feature_names=["joint.pos"],
                )
            )
        )

        with pytest.raises(ValueError, match="non-empty state contract"):
            FlowMatchingS1Policy.from_pretrained(str(tmp_path))

    def test_complete_unversioned_checkpoint_remains_loadable(self, tmp_path):
        pretrained_dir = tmp_path / "pretrained_model"
        pretrained_dir.mkdir()
        config = _checkpoint_config()
        config.pop("feature_contract_version")
        config.pop("robot_state_feature")
        (pretrained_dir / "config.json").write_text(json.dumps(config))

        loaded = FlowMatchingS1Config.from_checkpoint_dict(config)

        assert loaded.robot_state_feature is True
        assert loaded.state_dim == 4

    def test_ambiguous_unversioned_checkpoint_requires_verified_migration(self, tmp_path):
        config = _checkpoint_config()
        config.pop("feature_contract_version")
        config.pop("robot_state_feature")
        config["state_feature_names"] = []

        with pytest.raises(ValueError, match="ambiguous or missing"):
            FlowMatchingS1Config.from_checkpoint_dict(config)

    def test_early_stateless_checkpoint_drops_its_unused_state_dimension(self):
        config = _checkpoint_config(
            feature_contract_version=None,
            robot_state_feature=False,
            state_dim=14,
            state_feature_names=[],
        )

        loaded = FlowMatchingS1Config.from_checkpoint_dict(config)

        assert loaded.robot_state_feature is False
        assert loaded.state_dim == 0

    def test_unsupported_contract_version_requires_migration(self):
        config = _checkpoint_config(feature_contract_version=999)

        with pytest.raises(ValueError, match="unsupported feature_contract_version"):
            FlowMatchingS1Config.from_checkpoint_dict(config)

    def test_checkpoint_without_action_names_does_not_guess_runtime_order(self, tmp_path):
        pretrained_dir = tmp_path / "pretrained_model"
        pretrained_dir.mkdir()
        (pretrained_dir / "config.json").write_text(json.dumps(_checkpoint_config(action_feature_names=[])))

        with pytest.raises(ValueError, match="ordered action feature names"):
            FlowMatchingS1Policy.from_pretrained(str(tmp_path))

    def test_checkpoint_without_config_requires_explicit_contract(self, tmp_path):
        import safetensors.torch as sft

        policy = FlowMatchingS1Policy(
            FlowMatchingS1Config(
                action_dim=4,
                robot_state_feature=True,
                state_dim=4,
                hidden_dim=64,
                num_heads=4,
                dim_feedforward=128,
                num_encoder_layers=1,
                num_decoder_layers=1,
                use_dino_backbone=False,
            )
        )
        sft.save_file(dict(policy.state_dict()), str(tmp_path / "model.safetensors"))

        with pytest.raises(ValueError, match="does not contain config.json"):
            FlowMatchingS1Policy.from_pretrained(str(tmp_path))

    def test_roundtrip_weights(self, small_config, tmp_path):
        """Weights should be identical after save → load roundtrip."""
        import safetensors.torch as sft

        policy = FlowMatchingS1Policy(small_config)
        # Set known values
        with torch.no_grad():
            for p in policy.parameters():
                p.fill_(0.42)

        pretrained_dir = tmp_path / "pretrained_model"
        pretrained_dir.mkdir()
        sft.save_file(dict(policy.state_dict()), str(pretrained_dir / "model.safetensors"))
        (pretrained_dir / "config.json").write_text(json.dumps(_checkpoint_config()))

        loaded = FlowMatchingS1Policy.from_pretrained(str(tmp_path))
        for (name, p_orig), (_, p_loaded) in zip(
            policy.named_parameters(), loaded.named_parameters(), strict=False
        ):
            assert torch.equal(p_orig, p_loaded), f"Mismatch in {name}"


class TestGUIScanner:
    """Test that the GUI model scanner can discover migrated HVLA checkpoints."""

    def test_scanner_finds_hvla(self, small_config, tmp_path):
        """_scan_training_run should find HVLA checkpoints in standard format."""
        from lerobot.gui.api.models import _scan_training_run

        # Create standard structure
        ckpt_dir = tmp_path / "checkpoints" / "checkpoint-100"
        pretrained = ckpt_dir / "pretrained_model"
        pretrained.mkdir(parents=True)
        training_state = ckpt_dir / "training_state"
        training_state.mkdir()

        import safetensors.torch as sft

        policy = FlowMatchingS1Policy(small_config)
        sft.save_file(dict(policy.state_dict()), str(pretrained / "model.safetensors"))
        (pretrained / "config.json").write_text(json.dumps({"type": "hvla_flow_s1"}))
        (training_state / "training_step.json").write_text(json.dumps({"step": 100}))

        # Create 'last' symlink
        (tmp_path / "checkpoints" / "last").symlink_to("checkpoint-100")

        result = _scan_training_run(tmp_path)
        assert result is not None
        assert result["policy_type"] == "hvla_flow_s1"
        assert result["num_checkpoints"] == 1
        assert result["current_step"] == 100

    def test_scanner_emits_policy_path_standard(self, small_config, tmp_path):
        """Per-checkpoint policy_path points at <ckpt>/pretrained_model, and
        the run's default_policy_path points at the resolved last
        checkpoint's policy_path (NOT the literal `last` symlink — symlinks
        rot when run dirs move). Regression: stops the GUI from
        string-concatenating `/checkpoints/last/pretrained_model` and
        silently breaking on the docker recipe's `<run>/output/...` layout.

        Uses the numeric directory naming the GUI scanner accepts
        (``_dir_has_step_subdirs`` filters non-numeric names — that's why
        the legacy ``checkpoint-<N>`` HVLA convention is invisible to the
        scanner today; tracked separately).
        """
        import safetensors.torch as sft

        from lerobot.gui.api.models import _scan_training_run

        ckpt_dir = tmp_path / "checkpoints" / "000100"
        pretrained = ckpt_dir / "pretrained_model"
        pretrained.mkdir(parents=True)
        (ckpt_dir / "training_state").mkdir()
        policy = FlowMatchingS1Policy(small_config)
        sft.save_file(dict(policy.state_dict()), str(pretrained / "model.safetensors"))
        (pretrained / "config.json").write_text(json.dumps({"type": "hvla_flow_s1"}))
        (ckpt_dir / "training_state" / "training_step.json").write_text(json.dumps({"step": 100}))
        (tmp_path / "checkpoints" / "last").symlink_to("000100")

        result = _scan_training_run(tmp_path)
        assert result is not None
        # Per-checkpoint policy_path points at the pretrained_model subdir.
        assert result["checkpoints"][0]["policy_path"] == str(pretrained)
        # Run-level default = the is_last checkpoint's policy_path (resolved
        # to 000100/pretrained_model, not <root>/checkpoints/last/...).
        assert result["default_policy_path"] == str(pretrained)
        assert result["default_policy_path"] == result["checkpoints"][0]["policy_path"]

    def test_scanner_emits_policy_path_docker_layout(self, small_config, tmp_path):
        """GUI-managed docker recipe writes checkpoints under
        `<run>/output/checkpoints/` (extra `output/` segment because the
        bind-mount target `/runs` always pre-exists and lerobot-train
        refuses to overwrite an existing output_dir). policy_path /
        default_policy_path must reflect that nested layout — this is
        the exact form that broke the "Test on robot" button before the
        server-emits-the-path fix.
        """
        import safetensors.torch as sft

        from lerobot.gui.api.models import _scan_training_run

        ckpt_dir = tmp_path / "output" / "checkpoints" / "000050000"
        pretrained = ckpt_dir / "pretrained_model"
        pretrained.mkdir(parents=True)
        (ckpt_dir / "training_state").mkdir()
        policy = FlowMatchingS1Policy(small_config)
        sft.save_file(dict(policy.state_dict()), str(pretrained / "model.safetensors"))
        (pretrained / "config.json").write_text(json.dumps({"type": "smolvla"}))
        (ckpt_dir / "training_state" / "training_step.json").write_text(json.dumps({"step": 50000}))
        (tmp_path / "output" / "checkpoints" / "last").symlink_to("000050000")

        result = _scan_training_run(tmp_path)
        assert result is not None
        # The policy path must include the `output/` segment — not the
        # legacy `<run>/checkpoints/...` form the JS used to reconstruct.
        assert "output/checkpoints" in result["default_policy_path"]
        assert result["default_policy_path"] == str(pretrained)

    def test_scanner_emits_policy_path_flat(self, tmp_path):
        """Flat layout (converted HVLA S2 VLM etc.): no nested
        pretrained_model/ subdir; weights live at the top level of the
        checkpoint dir. policy_path == path so the frontend can treat
        flat + standard layouts uniformly (no client-side heuristic)."""
        import safetensors.torch as sft
        import torch

        from lerobot.gui.api.models import _read_flat_checkpoint

        sft.save_file({"w": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
        (tmp_path / "config.json").write_text(json.dumps({"type": "hvla_s2_vlm"}))

        result = _read_flat_checkpoint(tmp_path)
        assert result is not None
        assert result["default_policy_path"] == str(tmp_path)
        assert result["checkpoints"][0]["policy_path"] == str(tmp_path)

    def test_scanner_skips_corrupt_checkpoint(self, tmp_path):
        """A run dir whose only checkpoint is missing pretrained_model/config.json
        must not surface as a phantom run. Pins the fail-fast invariant so a
        future refactor that synthesizes policy_path outside
        `_read_checkpoint_meta` can't accidentally emit a non-existent path."""
        from lerobot.gui.api.models import _scan_training_run

        # Numeric subdir exists (so _dir_has_step_subdirs returns True) but
        # has no pretrained_model/ inside.
        (tmp_path / "checkpoints" / "000010").mkdir(parents=True)
        assert _scan_training_run(tmp_path) is None


def _v7_is_migrated() -> bool:
    """Is a *fully* migrated v7 on disk — layout and feature contract both?

    The layout migration and the contract backfill are separate steps, and a
    machine can sit between them. Asserting on a checkpoint that has had only
    the first would report the operator's pending migration as a code failure.
    """
    config = Path("outputs/flow_s1_hvla_v7/checkpoints/checkpoint-50000/pretrained_model/config.json")
    if not config.with_name("model.safetensors").exists() or not config.exists():
        return False
    return "action_feature_names" in json.loads(config.read_text())


class TestMigratedCheckpointLoads:
    """Test that the actually migrated v7 checkpoint loads correctly (integration test)."""

    @pytest.mark.skipif(
        not _v7_is_migrated(),
        reason="No fully migrated v7 checkpoint on disk (run hvla_migrate_checkpoints)",
    )
    def test_load_migrated_v7(self):
        """Load the migrated v7 checkpoint by directory path."""
        ckpt_dir = "outputs/flow_s1_hvla_v7/checkpoints/checkpoint-50000"
        policy = FlowMatchingS1Policy.from_pretrained(ckpt_dir)
        assert policy.config.action_dim == 14
        assert policy.config.hidden_dim == 768
        assert policy._action_mean is not None
        assert policy._action_mean.shape[0] == 14
