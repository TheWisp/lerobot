#!/usr/bin/env python

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

"""Block-causal GPT over patch tokens.

Port of `models/vq_behavior_transformer/gpt.py` from the Patch Policy reference implementation
(itself an adaptation of Karpathy's nanoGPT, MIT licensed). Parameter names match the reference so a
reference checkpoint's state dict loads directly; only the attention mask buffer is not persisted.

Input is `[B, T, P, D]`: T observation steps of P patch tokens each. Within a step every token attends
to every other token; across steps attention is causal. The readout for step t is the output at its
last token.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def block_causal_mask(n_tokens_per_step: int, n_steps: int, device: torch.device | None = None) -> Tensor:
    """Boolean `[S, S]` mask with `S = n_steps * n_tokens_per_step`, True where a query may attend.

    A query in step i attends every token of steps 0..i. Same matrix as the reference's
    `generate_mask_matrix(npatch, nwindow)`.
    """
    step = torch.arange(n_steps, device=device).repeat_interleave(n_tokens_per_step)
    return step[:, None] >= step[None, :]


@dataclass
class GPTConfig:
    block_size: int  # maximum number of observation steps
    n_patches: int  # tokens per observation step (all cameras folded in)
    input_dim: int
    output_dim: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f"n_embd={config.n_embd} is not divisible by n_head={config.n_head}")
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_patches = config.n_patches
        self._mask_cache: dict[tuple[int, str], Tensor] = {}

    def _mask(self, seq_len: int, device: torch.device) -> Tensor:
        key = (seq_len, str(device))
        mask = self._mask_cache.get(key)
        if mask is None:
            if seq_len % self.n_patches != 0:
                raise ValueError(f"sequence length {seq_len} is not a multiple of n_patches={self.n_patches}")
            mask = block_causal_mask(self.n_patches, seq_len // self.n_patches, device)
            self._mask_cache[key] = mask
        return mask

    def forward(self, x: Tensor) -> Tensor:
        b, s, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        q = q.view(b, s, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, s, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, s, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self._mask(s, x.device), dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(b, s, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        # The reference's `new_gelu` is the tanh approximation.
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Linear(config.input_dim, config.n_embd),
                "wpe": nn.Embedding(config.block_size * config.n_patches, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: Tensor) -> Tensor:
        """`[B, T, P, D]` -> `[B, T, output_dim]`, the output at the last token of each step."""
        b, t, p, _ = x.shape
        if t > self.config.block_size:
            raise ValueError(f"cannot forward {t} steps, block size is {self.config.block_size}")
        if p != self.config.n_patches:
            raise ValueError(f"expected {self.config.n_patches} tokens per step, got {p}")
        pos = torch.arange(t * p, device=x.device)
        tok_emb = self.transformer["wte"](x)
        pos_emb = self.transformer["wpe"](pos).view(1, t, p, -1)
        h = self.transformer["drop"](tok_emb + pos_emb).view(b, t * p, -1)
        for block in self.transformer["h"]:
            h = block(h)
        h = self.transformer["ln_f"](h)
        logits = self.lm_head(h).view(b, t, p, -1)
        return logits[:, :, -1]

    def decay_and_no_decay_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """nanoGPT's grouping: Linear weights decay; biases, LayerNorm and Embedding weights do not."""
        decay, no_decay = [], []
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.append(param)
                elif param_name.endswith("weight") and isinstance(module, nn.Linear):
                    decay.append(param)
                elif param_name.endswith("weight") and isinstance(module, (nn.LayerNorm, nn.Embedding)):
                    no_decay.append(param)
                else:
                    raise RuntimeError(f"parameter {full_name} not assigned to a weight-decay group")
        return decay, no_decay
