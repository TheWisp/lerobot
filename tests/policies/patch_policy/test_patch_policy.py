# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patch Policy through the LeRobot policy contract, with the offline test encoder."""

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.patch_policy.configuration_patch_policy import PatchConfig
from lerobot.policies.patch_policy.gpt import block_causal_mask
from lerobot.policies.patch_policy.modeling_patch_policy import PatchPolicy
from lerobot.policies.patch_policy.processor_patch_policy import make_patch_policy_pre_post_processors

CAMERAS = ("observation.images.front", "observation.images.wrist")
ACT_DIM = 6
IMG = (3, 40, 40)


def make_config(**overrides) -> PatchConfig:
    kwargs = {
        "n_obs_steps": 3,
        "chunk_size": 2,
        "encoder": "tiny_test",
        "encoder_pretrained": False,
        "image_size": 28,
        "gpt_n_layer": 2,
        "gpt_n_head": 2,
        "gpt_n_embd": 24,
        "gpt_output_dim": 32,
        "vqvae_latent_dim": 32,
        "vqvae_n_embed": 4,
        "vqvae_fit_steps": 2,
        "vqvae_iters": 3,
        "vqvae_batch_size": 64,
        "input_features": {
            **{cam: PolicyFeature(FeatureType.VISUAL, IMG) for cam in CAMERAS},
            "observation.state": PolicyFeature(FeatureType.STATE, (ACT_DIM,)),
        },
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
        "device": "cpu",
    }
    kwargs.update(overrides)
    return PatchConfig(**kwargs)


def make_batch(
    cfg: PatchConfig, batch_size: int = 4, with_time_axis: bool = True, shapes: dict[str, tuple] | None = None
) -> dict:
    t, w = cfg.n_obs_steps, cfg.chunk_size
    shapes = shapes or {}
    batch = {}
    for cam in CAMERAS:
        img = shapes.get(cam, IMG)
        batch[cam] = torch.rand((batch_size, t, *img) if with_time_axis else (batch_size, *img))
    if with_time_axis:
        batch["observation.state"] = torch.randn(batch_size, t, ACT_DIM)
        batch["action"] = torch.randn(batch_size, t + w - 1, ACT_DIM)
    else:
        batch["observation.state"] = torch.randn(batch_size, ACT_DIM)
        batch["action"] = torch.randn(batch_size, ACT_DIM)
    return batch


def fit_vqvae(policy: PatchPolicy, cfg: PatchConfig) -> None:
    policy.train()
    for _ in range(cfg.vqvae_fit_steps):
        policy.forward(make_batch(cfg))
    assert bool(policy.model.vqvae_is_fit)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_registered_with_the_factory():
    assert "patch_policy" in PreTrainedConfig.get_known_choices()
    assert isinstance(make_policy_config("patch_policy"), PatchConfig)
    cls = get_policy_class("patch_policy")
    assert cls is PatchPolicy
    assert cls.name == "patch_policy"
    assert cls.config_class is PatchConfig


def test_delta_indices_cover_the_window_and_one_chunk_per_frame():
    cfg = make_config(n_obs_steps=3, chunk_size=2)
    assert cfg.observation_delta_indices == [-2, -1, 0]
    assert cfg.action_delta_indices == [-2, -1, 0, 1]
    assert cfg.reward_delta_indices is None


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"vqvae_groups": 3}, "vqvae_groups"),
        ({"gpt_n_embd": 25}, "gpt_n_head"),
        ({"encoder": "dinov2_vits14", "image_size": 200}, "patch size"),
        ({"encoder": "resnet18"}, "encoder"),
    ],
)
def test_config_rejects_values_the_model_could_not_build(override, match):
    with pytest.raises(ValueError, match=match):
        make_config(**override)


def test_block_causal_mask_is_full_within_a_step_and_causal_across_steps():
    mask = block_causal_mask(n_tokens_per_step=2, n_steps=3)
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_loss_is_zero_until_the_vqvae_is_fit_then_trains_the_transformer():
    cfg = make_config()
    policy = PatchPolicy(cfg)
    policy.train()

    loss, info = policy.forward(make_batch(cfg))
    assert not bool(policy.model.vqvae_is_fit)
    assert loss.item() == 0.0
    assert info["vqvae_is_fit"] == 0.0

    loss, info = policy.forward(make_batch(cfg))  # second batch reaches vqvae_fit_steps
    assert bool(policy.model.vqvae_is_fit)
    assert info["vqvae_is_fit"] == 1.0
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert all(p.grad is not None for p in policy.model._gpt_model.parameters())
    assert all(p.grad is None for p in policy.encoder.parameters())
    assert all(not p.requires_grad for p in policy.model._vqvae_model.parameters())


def test_reduction_none_gives_one_loss_per_sample_with_the_same_mean():
    cfg = make_config()
    policy = PatchPolicy(cfg)
    fit_vqvae(policy, cfg)
    batch = make_batch(cfg, batch_size=5)
    torch.manual_seed(3)
    per_sample, _ = policy.forward(batch, reduction="none")
    torch.manual_seed(3)
    mean, _ = policy.forward(batch)
    assert per_sample.shape == (5,)
    assert torch.allclose(per_sample.mean(), mean)
    assert per_sample.std() > 0  # samples differ, so this is not a broadcast scalar
    with pytest.raises(ValueError, match="reduction"):
        policy.forward(batch, reduction="sum")


def test_train_mode_keeps_the_frozen_parts_in_eval():
    cfg = make_config()
    policy = PatchPolicy(cfg)
    fit_vqvae(policy, cfg)
    policy.train()
    assert policy.training
    assert not policy.encoder.training
    assert not policy.model._vqvae_model.training


def test_optimizer_groups_follow_the_reference_decay_rules():
    cfg = make_config()
    policy = PatchPolicy(cfg)
    fit_vqvae(policy, cfg)  # until the fit, the VQ-VAE's parameters belong to its own optimizer
    groups = policy.get_optim_params()
    grouped = {id(p) for group in groups for p in group["params"]}
    trainable = {id(p) for p in policy.parameters() if p.requires_grad}
    assert grouped == trainable
    assert not grouped & {id(p) for p in policy.encoder.parameters()}
    assert not grouped & {id(p) for p in policy.model._vqvae_model.parameters()}

    by_param = {id(p): group.get("weight_decay", "config") for group in groups for p in group["params"]}
    gpt = policy.model._gpt_model.transformer
    assert by_param[id(gpt["wte"].weight)] == "config"  # Linear weights decay at the configured value
    assert by_param[id(gpt["wte"].bias)] == 0.0
    assert by_param[id(gpt["wpe"].weight)] == 0.0
    assert by_param[id(gpt["ln_f"].weight)] == 0.0
    assert by_param[id(policy.model._map_to_cbet_preds_offset[0].weight)] == "config"
    assert by_param[id(policy.model._map_to_cbet_preds_bin[0].weight)] == 1e-2  # torch's AdamW default


def test_views_are_folded_view_major_into_the_token_axis():
    cfg = make_config()
    policy = PatchPolicy(cfg).eval()
    batch = make_batch(cfg, batch_size=2)
    front, wrist = batch[CAMERAS[0]], batch[CAMERAS[1]]
    both = policy._encode_images([front, wrist])
    per_view = torch.cat([policy._encode_images([front]), policy._encode_images([wrist])], dim=2)
    assert both.shape == (
        2,
        cfg.n_obs_steps,
        2 * 4,
        policy.encoder.embed_dim,
    )  # 28 px / 14 -> 4 tokens per view
    assert torch.equal(both, per_view)


def test_cameras_may_differ_in_resolution():
    cfg = make_config()
    shapes = {CAMERAS[0]: (3, 48, 64), CAMERAS[1]: (3, 30, 40)}
    policy = PatchPolicy(cfg)
    policy.train()
    loss, _ = policy.forward(make_batch(cfg, shapes=shapes))
    assert torch.isfinite(loss)
    policy.eval()
    action = policy.select_action(make_batch(cfg, batch_size=2, with_time_axis=False, shapes=shapes))
    assert action.shape == (2, ACT_DIM)


def test_select_action_returns_one_action_per_tick():
    cfg = make_config()
    policy = PatchPolicy(cfg)
    fit_vqvae(policy, cfg)
    policy.eval()
    policy.reset()
    for _ in range(cfg.n_obs_steps + 2):
        action = policy.select_action(make_batch(cfg, batch_size=2, with_time_axis=False))
        assert action.shape == (2, ACT_DIM)
        assert torch.isfinite(action).all()
    assert len(policy._obs_queue) == cfg.n_obs_steps


def test_predict_action_chunk_is_the_chunk_for_the_newest_frame():
    cfg = make_config()
    policy = PatchPolicy(cfg)
    fit_vqvae(policy, cfg)
    policy.eval()
    policy.reset()
    for _ in range(cfg.n_obs_steps):  # fill the window with distinct frames
        policy.predict_action_chunk(make_batch(cfg, batch_size=2, with_time_axis=False))
    batch = make_batch(cfg, batch_size=2, with_time_axis=False)

    torch.manual_seed(5)
    chunk = policy.predict_action_chunk(batch)
    obs_seq = torch.stack(list(policy._obs_queue), dim=1)  # the window now ends with `batch`
    torch.manual_seed(5)
    per_step, _, _ = policy.model(obs_seq, None)
    assert chunk.shape == (2, cfg.chunk_size, ACT_DIM)
    assert torch.equal(chunk, per_step[:, -1])
    assert not torch.equal(chunk, per_step[:, 0])


def test_select_action_averages_the_aligned_predictions_of_the_last_chunks(monkeypatch):
    cfg = make_config(chunk_size=3)
    policy = PatchPolicy(cfg).eval()
    counter = iter(range(10))

    def fake_chunk(batch):
        k = next(counter)
        return torch.zeros(2, 3, ACT_DIM) + (10 * k + torch.arange(3)).view(1, 3, 1).float()

    monkeypatch.setattr(policy, "predict_action_chunk", fake_chunk)
    batch = make_batch(cfg, batch_size=2, with_time_axis=False)
    # Chunk k predicted at tick k holds 10k + j at offset j. At tick n the action for now is offset
    # (n - k) of chunk k, averaged over the last chunk_size chunks.
    assert policy.select_action(batch)[0, 0].item() == 0.0
    assert policy.select_action(batch)[0, 0].item() == pytest.approx((1 + 10) / 2)
    assert policy.select_action(batch)[0, 0].item() == pytest.approx((2 + 11 + 20) / 3)
    assert policy.select_action(batch)[0, 0].item() == pytest.approx((12 + 21 + 30) / 3)


def test_single_frame_inputs_may_omit_the_time_axis():
    cfg = make_config(n_obs_steps=1, chunk_size=1, vqvae_fit_steps=1)
    policy = PatchPolicy(cfg)
    policy.train()
    loss, _ = policy.forward(make_batch(cfg, with_time_axis=False))
    assert bool(policy.model.vqvae_is_fit)
    assert torch.isfinite(loss)
    policy.eval()
    assert policy.select_action(make_batch(cfg, batch_size=3, with_time_axis=False)).shape == (3, ACT_DIM)


def test_save_and_load_roundtrip_keeps_the_fit_flag_and_weights(tmp_path):
    cfg = make_config()
    policy = PatchPolicy(cfg)
    fit_vqvae(policy, cfg)
    policy.eval()
    policy.save_pretrained(tmp_path)

    loaded = PatchPolicy.from_pretrained(tmp_path)
    assert bool(loaded.model.vqvae_is_fit)
    for (name, a), (_, b) in zip(policy.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert torch.equal(a, b), name

    batch = make_batch(cfg, batch_size=2, with_time_axis=False)
    torch.manual_seed(1)
    expected = policy.select_action(batch)
    torch.manual_seed(1)
    assert torch.equal(loaded.select_action(batch), expected)


def test_processor_pipelines_normalize_actions_min_max_and_leave_images_alone():
    cfg = make_config()
    stats = {
        "action": {"min": torch.zeros(ACT_DIM), "max": 2 * torch.ones(ACT_DIM)},
        "observation.state": {"min": torch.zeros(ACT_DIM), "max": 2 * torch.ones(ACT_DIM)},
    }
    pre, post = make_patch_policy_pre_post_processors(cfg, stats)
    batch = make_batch(cfg, batch_size=1)
    batch["action"] = torch.full((1, cfg.n_obs_steps + cfg.chunk_size - 1, ACT_DIM), 2.0)
    out = pre(dict(batch))
    assert torch.allclose(out["action"], torch.ones_like(out["action"]))  # 2 in [0, 2] -> 1 in [-1, 1]
    for cam in CAMERAS:
        assert torch.equal(out[cam], batch[cam])
    assert torch.allclose(post(torch.ones(1, ACT_DIM)), 2 * torch.ones(1, ACT_DIM))
