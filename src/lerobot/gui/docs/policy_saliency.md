# Attention-map overlay (policy saliency) — how it works & integrating a policy

The **Attention map** overlay (the GUI label — colloquial; technically it is gradient _saliency_ or attention _rollout_, see §1) draws, per camera, **where the _running_ policy's action depends on the pixels** — a _model-internal_ signal the policy computes about itself, not a generic vision model's guess. It rides the existing debug-vision overlay path: the policy publishes a small per-camera grid each inference, a worker colorizes it onto the camera tiles, and the GUI exposes Method / Style / Smoothing controls.

---

## 1. The two methods

The `Method` dropdown picks the **source** of the per-camera grid. Both are computed _by the policy on its own forward/backward_, so there is no re-inference and no desync with the action actually taken.

### Gradient — _causal_ (default)

Input-gradient saliency: `∂‖action‖/∂pixels`, one **backward** pass through the policy.

- **Question it answers:** _what does the action depend on_ (causal attribution).
- `FlowMatchingS1Policy.compute_input_saliency(batch, num_steps=4, grid=64)` — strips the sampler's `@torch.no_grad` (via `.__wrapped__`), unfreezes the DINOv2 backbone for the pass, backprops `actions[0, tok].pow(2).sum()` at the **first future (post-prefix) action token**, and area-pools `|grad|` to a per-camera `grid×grid` map.
- **Cost (measured, 5090, 4 cams @224):** ~30–36 ms per pass (29 ms compute; 34–36 ms through the live badge, which includes the aux write) — comparable to one 15-step inference (~36 ms). Run at a _debug rate_ (every Nth inference), not every step; the badge shows the live number (`sal N ms`).

### Rollout — _routing_

Attention rollout (Abnar–Zuidema): residual-aware composition `R = Πᵢ(½Aᵢ + ½I)` of the obs_encoder self-attention, then `a_decoder @ R` to trace the action's attention back through the patch-mixing onto the original patches.

- **Question it answers:** _where does the action read from_ (routing).
- `compute_attention_rollout(batch, num_steps=4, grid=64)` — **forward-only**, ~22 ms.
- Complementary, but a **different, less-causal** signal (see below).

### Why gradient is the default: routing ≠ importance

Attention/rollout shows _where information is read from_; the gradient shows _what the output actually depends on_. A patch can be heavily attended yet barely move the action (small value vector, washed out downstream), and a lightly-attended patch can matter via residual paths. Rollout also still propagates DINOv2 attention **sinks**. So the gradient is the faithful "source of truth" and stays the default; rollout is a useful **routing lens** to flip to live.

Prior art:

- Jain & Wallace, [_Attention is not Explanation_](https://arxiv.org/abs/1902.10186) (NAACL 2019) — attention weights ≠ feature importance (the core reason gradient is the default).
- Simonyan, Vedaldi & Zisserman, [_Deep Inside Convolutional Networks_](https://arxiv.org/abs/1312.6034) (2014) — input-gradient saliency.
- Abnar & Zuidema, [_Quantifying Attention Flow in Transformers_](https://arxiv.org/abs/2005.00928) (ACL 2020) — attention rollout (the `½A + ½I` composition used here).
- Smilkov et al., [_SmoothGrad: removing noise by adding noise_](https://arxiv.org/abs/1706.03825) (2017) — the planned denoising upgrade.
- Sundararajan, Taly & Yan, [_Axiomatic Attribution (Integrated Gradients)_](https://arxiv.org/abs/1703.01365) (ICML 2017) — the principled-but-costlier alternative considered.
- Chefer, Gur & Wolf, [_Transformer Interpretability Beyond Attention Visualization_](https://arxiv.org/abs/2012.09838) (CVPR 2021) — gradient-weighted attention (evaluated offline; inherits sink artifacts here).
- Darcet et al., [_Vision Transformers Need Registers_](https://arxiv.org/abs/2309.16588) (ICLR 2024) — why DINOv2 attention maps carry high-norm "sink" tokens.

> **Honest caveat:** vanilla input-gradient is causal but **noisy**. On a policy that isn't confidently grounding on the target (e.g. failing to grasp), the map is genuinely **diffuse** — that's the honest signal, not a bug (verified: the eager and `torch.compile`'d gradients are identical on the same inputs). The clean-and-faithful upgrade is **SmoothGrad** (the same gradient, denoised over N noisy inputs) — cleaner but 4–8× cost — tracked in `gui/TODO.md`.

---

## 2. Architecture / data flow

```
  POLICY process                     WORKER process                       GUI + browser
  (s1_inference)                     (overlays/standalone.py)
 ┌──────────────────────────┐       ┌──────────────────────────┐       ┌──────────────────────────┐
 │ every Nth inference:     │       │ PolicySaliencyAdapter:   │       │ /live/frame/<cam> -> PNG │
 │  _publish_aux(batch)     │  aux  │  read_saliency(cam)      │  ovl  │  drawn over camera tiles │
 │   -> compute_input_      │ ────► │  -> colorize (gated LUT, │ ────► │                          │
 │      saliency / rollout  │       │     smooth, percentile)  │       │ badge: fps · gpu · vram  │
 │   -> write_saliency(cam) │       │  -> write_overlay(rgba)  │       │        · sal N ms        │
 │   -> write_pass_ms(ms) ──┼───────┼───────── aux ────────────┼─────► │   (via /live/status)     │
 └──────────────────────────┘       └──────────────────────────┘       └──────────────────────────┘
        ▲ method                           ▲ style / smooth                     │
        └──────────────────────────────────┴────────────────────────────────────┘
              lerobot_overlay_control (GUI writes; policy reads method, worker reads style/smooth)

  aux = lerobot_aux_*      SharedAuxBuffer: per-camera float grids + pass_ms (policy = writer)
  ovl = lerobot_overlay_*  SharedOverlayBuffer: per-camera RGBA (worker = writer)
```

Key points:

- The **policy is the writer** of the saliency grid (`SharedAuxBuffer`, model `"policy_saliency"`) — the overlay reflects the action just taken, with zero re-inference.
- The worker + GUI are **policy-agnostic**: they only colorize and serve whatever per-camera grids the aux carries.
- The reverse channel `lerobot_overlay_control` carries the GUI controls: the **policy** reads `method`, the **worker** reads `style` / `smooth`. All switchable mid-run (verified: `gradient ↔ rollout` both directions take effect on the next publish).
- The badge's `sal N ms` is the **policy-side** cost (the pass + grid writes, published as `pass_ms` through the aux) — the worker's own fps/gpu numbers cannot see policy-process work.

---

## 3. Integrating a (new) policy

The worker and GUI need **no changes**. A policy becomes saliency-publishable by providing two things:

**(a) A saliency method** that returns per-camera grids keyed by the policy's image-feature keys:

```python
def compute_input_saliency(self, batch, num_steps=4, grid=64) -> dict[str, np.ndarray]:
    """{"observation.images.<cam>": (grid, grid) float32}, or {} if the policy has no image features.
    Precondition: `batch` is the raw batch the action path consumes; the action path is left untouched
    (a SEPARATE grad-enabled pass). Postcondition: freeze state + inference path unchanged."""
```

Optionally also `compute_attention_rollout(self, batch, ...) -> dict` for the routing view.

**(b) A `SaliencyPublisher`** (`overlays/saliency_publisher.py` — generic, shared by all policies) owned by the inference loop:

```python
from lerobot.overlays.saliency_publisher import SaliencyPublisher

pub = SaliencyPublisher(policy, image_keys, mode="saliency", every=3)  # at loop setup
...
pub.publish(batch)   # once per inference — demand-gated, cadenced, method-dispatched, timed
...
pub.cleanup()        # on stop
```

The publisher handles everything that used to be hand-rolled: the demand gate (nothing computes unless a worker is up), the cadence, the gradient↔rollout method dispatch from the GUI control block, the feature-key → obs-stream-camera mapping (prefix strip), the `pass_ms` timing the badge shows, and the optional `.npz` dump.

Requirements / gotchas:

- **`image_keys` order** must line up with the obs-stream cameras the worker reads (grid i ↔ camera i).
- **Rate:** the gradient is a backward pass, so `every=N`, not every step. RTC absorbs the latency; the first rollout call pays a one-time ~7 s hook-install.

That's the whole contract. The overlay's render (gated blue→yellow, smoothing, percentile) and the GUI controls come for free.

---

## 4. Files

- `policies/hvla/s1/flow_matching/saliency.py` — the S1 compute implementations (`model.py` keeps two thin delegator methods)
- `overlays/saliency_publisher.py` — `SaliencyPublisher` (generic: demand gate, cadence, method dispatch, key mapping, pass timing; `s1_inference` just owns one and calls `publish(batch)`)
- `overlays/aux_ipc.py` — `SharedAuxBuffer` (the aux seam)
- `overlays/adapters.py` — `PolicySaliencyAdapter` (the colorizer + render styles)
- `overlays/standalone.py` — the worker (reads aux + obs-stream, writes the overlay buffer)
- `gui/api/overlays.py` — the Method/Style/Smoothing controls + PNG serving
- Follow-ups (async off the inference thread, SmoothGrad, raw-capture-for-offline-analysis, a "warming up" badge): `gui/TODO.md`
