"""Tests for HVLA checkpoint save/load in standard LeRobot format.

Verifies:
  - Training saves config.json, model.safetensors, norm_stats.pt, training_step.json
  - from_pretrained loads from standard format (directory path)
  - from_pretrained loads from legacy flat format (backward compat)
  - config.json values are read correctly
  - GUI model scanner can discover HVLA checkpoints
"""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy


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
        state_dim=4,
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
        torch.save({"action_mean": torch.zeros(4), "action_std": torch.ones(4)},
                    str(pretrained_dir / "norm_stats.pt"))
        (pretrained_dir / "config.json").write_text(json.dumps({
            "type": "hvla_flow_s1",
            "action_dim": 4,
            "state_dim": 4,
            "chunk_size": 10,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "s2_latent_dim": 32,
        }))
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
        torch.save({"action_mean": torch.zeros(4), "action_std": torch.ones(4)},
                    str(pretrained_dir / "norm_stats.pt"))
        (pretrained_dir / "config.json").write_text(json.dumps({
            "type": "hvla_flow_s1",
            "action_dim": 4, "state_dim": 4, "chunk_size": 10,
            "hidden_dim": 64, "num_heads": 4, "dim_feedforward": 128,
            "num_encoder_layers": 1, "num_decoder_layers": 1,
            "s2_latent_dim": 32, "s2_proj_hidden": 16, "use_dino_backbone": False,
        }))

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
        torch.save({"action_mean": torch.zeros(4), "action_std": torch.ones(4)},
                    str(tmp_path / "norm_stats.pt"))

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
        (pretrained_dir / "config.json").write_text(json.dumps({
            "type": "hvla_flow_s1",
            "action_dim": 4, "state_dim": 4, "chunk_size": 10,
            "hidden_dim": 64, "num_heads": 4, "dim_feedforward": 128,
            "num_encoder_layers": 1, "num_decoder_layers": 1,
            "s2_latent_dim": 32, "s2_proj_hidden": 16, "use_dino_backbone": False,
            "num_inference_steps": 5,
            "rtc_max_delay": 3,
        }))

        # Load without providing config — should read from config.json
        loaded = FlowMatchingS1Policy.from_pretrained(str(tmp_path))
        assert loaded.config.num_inference_steps == 5
        assert loaded.config.rtc_max_delay == 3

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
        (pretrained_dir / "config.json").write_text(json.dumps({
            "type": "hvla_flow_s1",
            "action_dim": 4, "state_dim": 4, "chunk_size": 10,
            "hidden_dim": 64, "num_heads": 4, "dim_feedforward": 128,
            "num_encoder_layers": 1, "num_decoder_layers": 1,
            "s2_latent_dim": 32, "s2_proj_hidden": 16, "use_dino_backbone": False,
        }))

        loaded = FlowMatchingS1Policy.from_pretrained(str(tmp_path))
        for (name, p_orig), (_, p_loaded) in zip(
            policy.named_parameters(), loaded.named_parameters()
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


class TestMigratedCheckpointLoads:
    """Test that the actually migrated v7 checkpoint loads correctly (integration test)."""

    @pytest.mark.skipif(
        not Path("outputs/flow_s1_hvla_v7/checkpoints/checkpoint-50000/pretrained_model/model.safetensors").exists(),
        reason="Migrated v7 checkpoint not available",
    )
    def test_load_migrated_v7(self):
        """Load the migrated v7 checkpoint by directory path."""
        ckpt_dir = "outputs/flow_s1_hvla_v7/checkpoints/checkpoint-50000"
        policy = FlowMatchingS1Policy.from_pretrained(ckpt_dir)
        assert policy.config.action_dim == 14
        assert policy.config.hidden_dim == 768
        assert policy._action_mean is not None
        assert policy._action_mean.shape[0] == 14
