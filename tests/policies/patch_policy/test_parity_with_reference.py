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
"""Numerical parity of the port against the reference implementation.

Opt-in: set PATCH_POLICY_REFERENCE_REPO to a clone of github.com/gaoyuezhou/patch_policy. The
reference's `BehaviorTransformer` is loaded straight from that clone, its weights are copied into
`PatchPolicyModel`, and both are driven with the same inputs and the same RNG state.
"""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.patch_policy.configuration_patch_policy import PatchConfig
from lerobot.policies.patch_policy.modeling_patch_policy import PatchPolicyModel

REPO = os.environ.get("PATCH_POLICY_REFERENCE_REPO")
pytestmark = pytest.mark.skipif(not REPO, reason="PATCH_POLICY_REFERENCE_REPO not set")

T, W, V, P, E, A = 3, 2, 2, 4, 8, 6
GPT = {"n_layer": 2, "n_head": 2, "n_embd": 24}
VQ = {"latent_dim": 32, "n_embed": 4, "groups": 2, "iters": 3, "batch_size": 64}


def load_reference_bet():
    """Import `models/vq_behavior_transformer` from the clone as a standalone package.

    The clone's `models/__init__.py` pulls in the diffusion head and every encoder, and `bet.py`
    builds an `accelerate.Accelerator` at import. Neither is needed for parity, so the package is
    mounted directly and `accelerate` is replaced by a single-process stand-in.
    """
    if "accelerate" not in sys.modules:
        stub = types.ModuleType("accelerate")

        class Accelerator:
            is_local_main_process = True

            def gather(self, x):
                return x

            def wait_for_everyone(self):
                pass

            def prepare(self, module):
                return module

            def unwrap_model(self, module):
                return module

        stub.Accelerator = Accelerator
        sys.modules["accelerate"] = stub

    package = types.ModuleType("reference_vqbt")
    package.__path__ = [str(Path(REPO) / "models" / "vq_behavior_transformer")]
    sys.modules["reference_vqbt"] = package
    return importlib.import_module("reference_vqbt.bet")


@pytest.fixture(scope="module")
def reference_bet():
    return load_reference_bet()


def build_pair(reference_bet):
    torch.manual_seed(0)
    ref = reference_bet.BehaviorTransformer(
        obs_dim=E,
        act_dim=A,
        goal_dim=0,
        views=V,
        vqvae_latent_dim=VQ["latent_dim"],
        vqvae_n_embed=VQ["n_embed"],
        vqvae_groups=VQ["groups"],
        vqvae_fit_steps=None,
        vqvae_iters=VQ["iters"],
        n_patches=P,
        n_layer=GPT["n_layer"],
        n_head=GPT["n_head"],
        n_embd=GPT["n_embd"],
        dropout=0.0,
        vqvae_batch_size=VQ["batch_size"],
        act_scale=1.0,
        offset_loss_multiplier=100.0,
        secondary_code_multiplier=0.5,
        gamma=2.0,
        obs_window_size=T,
        act_window_size=W,
    )
    cfg = PatchConfig(
        n_obs_steps=T,
        chunk_size=W,
        encoder="tiny_test",
        encoder_pretrained=False,
        gpt_n_layer=GPT["n_layer"],
        gpt_n_head=GPT["n_head"],
        gpt_n_embd=GPT["n_embd"],
        gpt_output_dim=ref._gpt_model.config.output_dim,
        vqvae_latent_dim=VQ["latent_dim"],
        vqvae_n_embed=VQ["n_embed"],
        vqvae_groups=VQ["groups"],
        vqvae_iters=VQ["iters"],
        vqvae_batch_size=VQ["batch_size"],
        vqvae_fit_steps=1,
        input_features={"observation.images.cam": PolicyFeature(FeatureType.VISUAL, (3, 28, 28))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (A,))},
        device="cpu",
    )
    port = PatchPolicyModel(cfg, obs_dim=E, act_dim=A, n_patches=P * V)
    copy_weights(ref, port)
    return ref, port


def copy_weights(ref, port):
    state = {k: v for k, v in ref.state_dict().items() if not k.endswith(".attn.bias")}  # the mask buffer
    result = port.load_state_dict(state, strict=False)
    assert set(result.missing_keys) <= {"vqvae_is_fit", "_vqvae_model.vq_layer.freeze_codebook"}, (
        result.missing_keys
    )
    assert not result.unexpected_keys, result.unexpected_keys


def actions(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, T + W - 1, A, generator=g)


def observations(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, T, V * P, E, generator=g)


def fit_reference(ref, act: torch.Tensor) -> None:
    ref.train()
    ref._collected_actions.append(ref._unpack_actions(act))
    ref._maybe_fit_vq(force=True)
    ref.eval()


def test_vqvae_fit_matches_the_reference(reference_bet):
    ref, port = build_pair(reference_bet)
    act = actions(64, seed=1)

    torch.manual_seed(7)
    fit_reference(ref, act)

    torch.manual_seed(7)
    port.train()
    port._collected_actions.append(port._unpack_actions(act))
    port._maybe_fit_vq()
    port.eval()

    port_state = port._vqvae_model.state_dict()
    for name, a in ref._vqvae_model.state_dict().items():
        assert torch.allclose(a.float(), port_state[name].float(), atol=1e-6), name


def test_transformer_heads_codes_and_loss_match_the_reference(reference_bet):
    ref, port = build_pair(reference_bet)
    act = actions(64, seed=1)
    torch.manual_seed(7)
    fit_reference(ref, act)
    copy_weights(ref, port)
    port.vqvae_is_fit.fill_(True)
    port.eval()

    obs = observations(5, seed=2)
    act = actions(5, seed=3)

    assert torch.allclose(ref._gpt_model(obs), port._gpt_model(obs), atol=1e-5)

    ref_logits, ref_offsets = ref._forward_heads(ref._gpt_model(obs))
    port_logits, port_offsets = port._forward_heads(port._gpt_model(obs))
    assert torch.allclose(ref_logits, port_logits, atol=1e-5)
    assert torch.allclose(ref_offsets, port_offsets, atol=1e-5)

    ref_latent, ref_codes = ref._vqvae_model.get_code(ref._unpack_actions(act))
    port_latent, port_codes = port._vqvae_model.get_code(port._unpack_actions(act))
    assert torch.equal(ref_codes, port_codes)
    assert torch.allclose(ref_latent, port_latent, atol=1e-6)

    torch.manual_seed(11)
    ref_pred, ref_loss, ref_info = ref(obs, None, act)
    torch.manual_seed(11)
    port_pred, port_loss, port_info = port(obs, act)
    assert torch.allclose(ref_pred, port_pred, atol=1e-5)
    assert torch.allclose(ref_loss, port_loss, atol=1e-5)
    for key in ("classification_loss", "offset_loss", "action_diff", "action_diff_max"):
        assert ref_info[key] == pytest.approx(port_info[key], abs=1e-5), key


def test_short_windows_are_padded_like_the_reference(reference_bet):
    ref, port = build_pair(reference_bet)
    ref.eval()
    port.eval()
    obs = observations(3, seed=4)[:, :1]  # one frame; both pad it to T by repeating
    torch.manual_seed(5)
    ref_pred, _, _ = ref(obs, None, None)
    torch.manual_seed(5)
    port_pred, _, _ = port(obs, None)
    assert torch.allclose(ref_pred, port_pred, atol=1e-5)
