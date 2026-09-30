# Patch Policy

LeRobot port of **Patch Policy: Efficient Embodied Control via Dense Visual Representations**
(Zhou, Cui, Langford, Tan, LeCun, Pinto, 2026). A small block-causal GPT reads every patch token of a
frozen pretrained ViT, for every camera and each of the last `n_obs_steps` frames, and a VQ-BeT head
turns its output into action chunks. No proprioception, no language.

- Paper: https://arxiv.org/abs/2607.18236
- Reference implementation: https://github.com/gaoyuezhou/patch_policy, commit `ebf94cf` (2026-09-29).
  Anchors below are `path:line` in that commit.
- Policy type: `patch_policy`. Code: `src/lerobot/policies/patch_policy/`.

## Train

```bash
lerobot-train \
  --policy.type=patch_policy \
  --policy.push_to_hub=false \
  --dataset.repo_id=<user>/<dataset> \
  --output_dir=outputs/patch_policy \
  --batch_size=32 --steps=50000
```

Defaults take the hyperparameters of the reference's LIBERO Goal single-GPU recipe: DINOv2 ViT-S/14
at 224 px, a 2-frame window, a chunk of 1, a 6-layer 120-wide GPT, a 2-group 16-code residual VQ.
(That recipe is goal-conditioned; this port is unconditional, see below.) The first
`vqvae_fit_steps` batches (default 1000) only collect actions; the VQ-VAE is then fit on them and the
policy loss is zero until that point, so the loss curve starts late by design.

The trainer's `--sample_weighting` is supported: `forward(batch, reduction="none")` returns one
loss per sample.

Inference goes through the usual path (`lerobot-record --policy.path=...`, the GUI Run tab). The model
is queried every control tick; with `chunk_size > 1` the executed action is the mean of the aligned
predictions from the last `chunk_size` chunks, as in the reference evaluation loop.

## Names

| Reference (`configs/train_libero_goal.yaml`) | Here                        |
| -------------------------------------------- | --------------------------- |
| `window_size`                                | `n_obs_steps`               |
| `action_window_size`                         | `chunk_size`                |
| `encoder` (`dino_patch`)                     | `encoder`, `image_size`     |
| `model.n_layer/n_head/n_embd`                | `gpt_n_layer/n_head/n_embd` |
| `model.vqvae_*`                              | `vqvae_*`                   |
| `optim.lr/weight_decay/betas`                | `optimizer_*`               |

## What is ported

| Piece                                          | Reference anchor                                    | Status                                                                                                                            |
| ---------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Frozen DINOv2 patch tokens, ImageNet norm      | `models/encoder/dino.py:8-59`                       | ported (`encoder.py`)                                                                                                             |
| Views folded into the token axis               | `train_policy.py:552`                               | ported (`_encode_images`)                                                                                                         |
| Flat learned position embedding over T×P       | `models/vq_behavior_transformer/gpt.py:181,208-211` | ported                                                                                                                            |
| Block-causal mask                              | `gpt.py:61-69`                                      | ported (`block_causal_mask`)                                                                                                      |
| Readout at the last token of each step         | `gpt.py:218`                                        | ported                                                                                                                            |
| Residual VQ-VAE over action chunks             | `vqvae.py`                                          | ported; quantizer is the fork's vendored `ResidualVQ`                                                                             |
| VQ fit on collected unique actions             | `bet.py:158-198`                                    | ported; triggered after `vqvae_fit_steps` batches                                                                                 |
| Focal code loss ×5 / ×0.5, L1 offset loss      | `bet.py:255-297`                                    | ported                                                                                                                            |
| Multinomial code sampling                      | `bet.py:299-321`                                    | ported                                                                                                                            |
| Window padding by repeating the first frame    | `bet.py:17-28`                                      | ported                                                                                                                            |
| Chunk averaging at rollout                     | `train_policy.py:399-414`                           | ported (`select_action`)                                                                                                          |
| nanoGPT weight-decay grouping                  | `gpt.py:241-292`, `bet.py:342`                      | ported (`get_optim_params`); the bin head decays at 1e-2, torch's AdamW default, which the reference's `add_param_group` inherits |
| Goal conditioning (`goal_conditional: future`) | `bet.py:227-232`, `datasets/core.py`                | **NOT IMPLEMENTED.** Unconditional only.                                                                                          |
| Diffusion Policy head                          | `models/diffusion_policy/`                          | **NOT IMPLEMENTED.** VQ-BeT head only.                                                                                            |
| Precomputed frozen embeddings                  | `train_policy.py:284-291`                           | **NOT IMPLEMENTED.** The encoder runs every step.                                                                                 |
| WebSSL, DINOv3, V-JEPA 2, SigLIP 2 encoders    | `models/encoder/`                                   | **NOT IMPLEMENTED.** `dinov2_*` hub names only.                                                                                   |
| Multi-process action gather before the fit     | `bet.py:221-222`                                    | **NOT IMPLEMENTED.** Single-process; raises under DDP.                                                                            |

## Deviations

- **Actions are MIN_MAX normalized** by the LeRobot processor. The reference feeds raw actions
  (`act_scale=1`) because its simulators already emit actions in [-1, 1].
- **The VQ-VAE fit is triggered by a batch count**, not at the end of the first epoch: LeRobot's
  trainer has no epoch boundary.
- **Gradient clipping at 10** comes from LeRobot's `AdamWConfig` default. The reference clips nothing.
- **Attention uses `scaled_dot_product_attention`** with the block-causal mask instead of the
  reference's explicit softmax. Same function, different kernel.
- **`threshold_ema_dead_code=2`** is passed explicitly: the reference's vendored quantizer defaults to 2,
  the fork's to 0.
- **The frozen encoder's weights are saved with the checkpoint** (about 88 MB for ViT-S). Loading
  still needs torch.hub for the DINOv2 code, from `facebookresearch/dinov2` at its current main; the
  reference pins commit `b48308a`.
- **Frames are resized to an `image_size` square** without keeping the aspect ratio, each camera on
  its own, so cameras may differ in resolution. The reference only ever sees 224×224 renders.

## Limitations

- **Float32 only.** The vendored quantizer's EMA update fails under autocast (shared with VQ-BeT).
- **A checkpoint without `model.vqvae_is_fit`** in its state dict, such as one converted from the
  reference, loads as unfit: the next training run collects actions again and re-fits the VQ-VAE.
- **The VQ-VAE fit is single-process**; multi-GPU training raises at the fit.

## Verification

- `tests/policies/patch_policy/test_patch_policy.py`: the LeRobot contract with an offline test encoder.
- `tests/policies/patch_policy/test_parity_with_reference.py`: with `PATCH_POLICY_REFERENCE_REPO` set
  to a clone of the reference, loads its `BehaviorTransformer`, copies the weights into the port and
  checks the transformer output, head logits and offsets, VQ codes, predicted actions and loss agree
  to 1e-5, and that the VQ-VAE fit produces the same codebook from the same seed.

## Citation

```bibtex
@misc{zhou2026patchpolicyefficientembodied,
  title={Patch Policy: Efficient Embodied Control via Dense Visual Representations},
  author={Gaoyue Zhou and Zichen Jeff Cui and Ada Langford and Bowen Tan and Yann LeCun and Lerrel Pinto},
  year={2026},
  eprint={2607.18236},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2607.18236},
}
```
