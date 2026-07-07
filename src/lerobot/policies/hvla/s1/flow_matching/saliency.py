"""Attention-map computation for FlowMatchingS1Policy — the S1-specific half of the
policy-internal overlay (see ``gui/docs/policy_saliency.md``).

Lives OUTSIDE ``model.py`` so the overlay feature doesn't grow the policy class: the policy
keeps two thin delegator methods (the documented per-policy contract), and everything the
overlay needs — the grad pass, the rollout hooks — is here. Both functions take the policy
as their first argument and leave its freeze state and inference path unchanged.
"""

from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.hvla.s1.protocol import ACTION_PREFIX_KEY

logger = logging.getLogger(__name__)

OBS_IMAGES = "observation.images"


def compute_input_saliency(policy, batch: dict[str, Tensor], num_steps: int = 4, grid: int = 64) -> dict:
    """Per-camera input-gradient saliency: where the upcoming action depends on each camera's
    pixels. Returns ``{image_feature_key: (grid, grid) float32 ndarray}`` (area-pooled), or
    ``{}`` for a policy with no image features.

    Precondition: ``batch`` is the same raw batch ``predict_action_chunk`` consumes (camera
    images + state + optional ACTION_PREFIX_KEY); norm stats loaded.
    Postcondition: the policy's freeze state and the inference path are unchanged.

    A SEPARATE grad-enabled pass — the ``no_grad`` inference path is untouched. It strips the
    sampler's ``@torch.no_grad`` (via ``__wrapped__``), unfreezes the DINOv2 backbone for the
    pass, and backprops ``|d action / d pixels|`` at the FIRST future (non-prefix) action
    position. ~30 ms on a 5090, so callers run it at a debug rate, not every inference. Unlike
    the attention overlay (captured inline, free) this costs a backward pass — but it localizes
    on the object the policy is acting on, which raw cross-attention does not.
    """
    if not policy.config.image_features:
        return {}
    t0 = time.perf_counter()  # whole-pass wall time; the first call would pay any one-time compile
    policy.eval()
    prepared = policy.prepare_batch_for_encode_observations(batch)
    imgs: dict[str, Tensor] = {}
    for k in policy.config.image_features:
        t = prepared[k].detach().clone().requires_grad_(True)
        prepared[k] = t
        imgs[k] = t
    prepared[OBS_IMAGES] = [prepared[k] for k in policy.config.image_features]  # point at the grad copies

    action_prefix = prepared.pop(ACTION_PREFIX_KEY, None)
    if action_prefix is not None and policy._action_mean is not None:
        dev = action_prefix.device
        action_prefix = (action_prefix - policy._action_mean.to(dev)) / policy._action_std.to(dev)
    prefix_len = action_prefix.shape[1] if action_prefix is not None else 0
    tok = min(prefix_len, policy.config.chunk_size - 1)  # first future (non-prefix) action position

    sample = type(policy.model).sample_actions.__wrapped__  # undecorated: run WITH grad
    was_frozen = policy.config.freeze_backbone
    policy.config.freeze_backbone = False
    try:
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            actions = sample(
                policy.model,
                prepared,
                num_steps=num_steps,
                action_prefix=action_prefix,
                prefix_len=prefix_len,
            )
            target = actions[0, tok].pow(2).sum()
            grads = torch.autograd.grad(target, list(imgs.values()), allow_unused=True)
    finally:
        policy.config.freeze_backbone = was_frozen

    out: dict = {}
    health: dict = {}
    for k, gr in zip(imgs, grads, strict=True):
        if gr is None:
            health[k] = "grad=None"  # input detached from the graph (e.g. a compiled forward)
            continue
        sal = gr[0].abs().sum(0)[None, None]  # [1,1,H,W]
        sal = torch.nn.functional.interpolate(sal, size=(grid, grid), mode="area")[0, 0]
        out[k] = sal.detach().float().cpu().numpy()
        health[k] = f"max={float(out[k].max()):.2e}"
    logger.info(
        "[saliency] input-grad %.0fms tok=%d prefix=%d | per-cam=%s | published=%s",
        (time.perf_counter() - t0) * 1000.0,
        tok,
        prefix_len,
        health,
        list(out),
    )
    return out


def compute_attention_rollout(policy, batch: dict[str, Tensor], num_steps: int = 4, grid: int = 64) -> dict:
    """Per-camera ATTENTION-ROLLOUT saliency (forward-only, GRADIENT-FREE). Composes the obs_encoder
    self-attention (residual-aware, Â=½A+½I across its layers) with the decoder cross-attention, so
    the action's attention is traced back through the obs_encoder's patch-mixing onto the original
    patches. Returns ``{image_feature_key: (grid,grid) float32}`` (area-pooled), or ``{}`` with no
    image features.

    Differs from ``compute_input_saliency``: this is a *routing* view (where attention flows from),
    needs NO backward pass, and undoes the obs_encoder MIXING but not the DINOv2 sinks. Cost: one
    extra forward that re-encodes the cameras (no backward) — cheaper than the gradient.

    Precondition: same ``batch`` contract as ``compute_input_saliency`` (the raw action-path batch),
    and NO concurrent forward on another thread — the pass temporarily swaps the process-global
    ``F.scaled_dot_product_attention`` (restored in a ``finally``); a concurrent forward would both
    pay the capture cost and pollute the captured attention. Single inference thread today; the
    async follow-up in ``gui/TODO.md`` must lock around this.
    Postcondition: hooks removed, SDPA restored, freeze state and the inference path unchanged.
    """
    if not policy.config.image_features:
        return {}
    policy.eval()
    enc, dec = [], []

    def _pre(m, args, kwargs):  # force the eager path so MHA returns per-head weights
        kwargs["need_weights"], kwargs["average_attn_weights"] = True, False
        return args, kwargs

    def _post(m, args, kwargs, output):
        enc.append(output[1].detach())  # [B, nhead, N_ctx, N_ctx]

    orig_sdpa = F.scaled_dot_product_attention

    def _msdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if q.shape[-2] != k.shape[-2]:  # action-query -> context cross-attn (T != N_ctx)
            s = scale if scale is not None else q.shape[-1] ** -0.5
            dec.append(torch.softmax((q @ k.transpose(-2, -1)) * s, dim=-1).detach())
        return orig_sdpa(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw
        )

    handles = []
    for layer in policy.model.obs_encoder.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(_pre, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(_post, with_kwargs=True))
    F.scaled_dot_product_attention = _msdpa
    try:
        with torch.no_grad():
            policy.predict_action_chunk(batch, num_steps=num_steps)
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        for h in handles:
            h.remove()
    if not enc or not dec or policy.model._ctx_layout is None:
        return {}

    A = dec[-1][0]  # [nhead, T, N_ctx] — last decoder layer, last denoise step
    pre_chunk = batch.get(ACTION_PREFIX_KEY)
    tok = min(pre_chunk.shape[1] if pre_chunk is not None else 0, A.shape[1] - 1)
    a_dec = A[:, tok, :].mean(0).double()  # head-mean attention over context [N_ctx]
    n = a_dec.shape[-1]
    eye = torch.eye(n, dtype=torch.float64, device=a_dec.device)
    R = eye
    for E in enc:  # residual-aware rollout over the obs_encoder layers
        ah = 0.5 * E[0].mean(0).double() + 0.5 * eye
        R = (ah / (ah.sum(-1, keepdim=True) + 1e-9)) @ R
    unmixed = a_dec @ R  # attention traced onto the obs_encoder input patches [N_ctx]

    p = int(policy.model._ctx_layout["patches_per_cam"])
    gg = int(round(p**0.5))
    out: dict = {}
    for i, key in enumerate(policy.config.image_features):
        blk = unmixed[i * p : (i + 1) * p]
        if blk.numel() != p or gg * gg != p:
            continue
        m = blk.reshape(gg, gg).float()[None, None]
        out[key] = torch.nn.functional.interpolate(m, size=(grid, grid), mode="area")[0, 0].cpu().numpy()
    return out
