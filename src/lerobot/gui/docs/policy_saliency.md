# Policy saliency overlay — how it works & integrating a policy

The **policy saliency** overlay draws, per camera, **where the _running_ policy's action depends on the pixels** — a _model-internal_ signal the policy computes about itself, not a generic vision model's guess. It rides the existing debug-vision overlay path: the policy publishes a small per-camera grid each inference, a worker colorizes it onto the camera tiles, and the GUI exposes Method / Style / Smoothing controls.

---

## 1. The two methods

The `Method` dropdown picks the **source** of the per-camera grid. Both are computed _by the policy on its own forward/backward_, so there is no re-inference and no desync with the action actually taken.

### Gradient — _causal_ (default)

Input-gradient saliency: `∂‖action‖/∂pixels`, one **backward** pass through the policy.

- **Question it answers:** _what does the action depend on_ (causal attribution).
- `FlowMatchingS1Policy.compute_input_saliency(batch, num_steps=4, grid=64)` — strips the sampler's `@torch.no_grad` (via `.__wrapped__`), unfreezes the DINOv2 backbone for the pass, backprops `actions[0, tok].pow(2).sum()` at the **first future (post-prefix) action token**, and area-pools `|grad|` to a per-camera `grid×grid` map.
- **Cost:** ~40–55 ms (a backward pass) → run at a _debug rate_ (every Nth inference), not every step.

### Rollout — _routing_

Attention rollout (Abnar–Zuidema): residual-aware composition `R = Πᵢ(½Aᵢ + ½I)` of the obs_encoder self-attention, then `a_decoder @ R` to trace the action's attention back through the patch-mixing onto the original patches.

- **Question it answers:** _where does the action read from_ (routing).
- `compute_attention_rollout(batch, num_steps=4, grid=64)` — **forward-only**, ~22 ms.
- Complementary, but a **different, less-causal** signal (see below).

### Why gradient is the default: routing ≠ importance

Attention/rollout shows _where information is read from_; the gradient shows _what the output actually depends on_. A patch can be heavily attended yet barely move the action (small value vector, washed out downstream), and a lightly-attended patch can matter via residual paths. Rollout also still propagates DINOv2 attention **sinks**. So the gradient is the faithful "source of truth" and stays the default; rollout is a useful **routing lens** to flip to live. (Full prior-art writeup is in the agent memory `reference_attention_vs_gradient_saliency`.)

> **Honest caveat:** vanilla input-gradient is causal but **noisy**. On a policy that isn't confidently grounding on the target (e.g. failing to grasp), the map is genuinely **diffuse** — that's the honest signal, not a bug (verified: the eager and `torch.compile`'d gradients are identical on the same inputs). The clean-and-faithful upgrade is **SmoothGrad** (the same gradient, denoised over N noisy inputs) — cleaner but 4–8× cost — tracked in `gui/TODO.md`.

---

## 2. Architecture / data flow

```
 POLICY process (s1_inference)                WORKER process (overlays/standalone.py)        GUI + browser
 ┌───────────────────────────┐                ┌──────────────────────────────┐                 ┌──────────────┐
 │ every Nth inference:       │  lerobot_aux_* │ PolicySaliencyAdapter:        │ lerobot_overlay_*│ /live/frame  │
 │  _publish_aux(batch)       │  (per-cam grid)│  read_saliency(cam)          │  (RGBA per cam)  │  -> PNG      │
 │   -> compute_input_saliency│ ─────────────► │  -> colorize (gated LUT,     │ ───────────────► │  drawn over  │
 │      / compute_attention_  │  SharedAuxBuffer│     smooth, percentile)      │ SharedOverlayBuf │  camera tiles│
 │      rollout               │   (policy=WRITER│  -> write_overlay(cam, rgba) │                  │              │
 │   -> write_saliency(cam,g) │    worker=reader│                              │                  │              │
 └───────────────────────────┘                └──────────────────────────────┘                 └──────────────┘
        ▲ method                                       ▲ style / smooth                                 │ Method/Style/Smoothing
        └───────────────────────  lerobot_overlay_control (GUI writes, policy+worker read)  ◄───────────┘
```

Key points:

- The **policy is the writer** of the saliency grid (`SharedAuxBuffer`, model `"policy_saliency"`) — the overlay reflects the action just taken, with zero re-inference.
- The worker + GUI are **policy-agnostic**: they only colorize and serve whatever per-camera grids the aux carries.
- The reverse channel `lerobot_overlay_control` carries the GUI controls: the **policy** reads `method`, the **worker** reads `style` / `smooth`. All switchable mid-run (verified: `gradient ↔ rollout` both directions take effect on the next publish).

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

**(b) A publish hook** in the inference loop that calls it at a debug rate and writes to the aux — the reference is `s1_inference._publish_aux` / `_publish_saliency`:

```python
sal = getattr(policy, "compute_input_saliency")(batch)          # or compute_attention_rollout
obs_keys = [k.removeprefix("observation.images.") for k in image_keys]   # MAP to obs-stream camera keys
key_map  = {feat_key: obs_key for feat_key, obs_key in zip(image_keys, obs_keys)}
aux = SharedAuxBuffer(cameras={key_map[k]: g.shape for k, g in sal.items()},
                      model="policy_saliency", create=True)      # lazily, once
for k, g in sal.items():
    aux.write_saliency(key_map[k], np.asarray(g, np.float32))
```

Requirements / gotchas:

- **Key mapping is mandatory.** The worker reads by the _bare_ obs-stream camera name (`front`), the policy keys by the image feature (`observation.images.front`). Strip the prefix.
- **Grid ↔ camera order** must line up with the obs-stream cameras the worker reads.
- **Method dispatch** (optional): read the GUI selection from the control block (`OverlayControlReader().config().get("method")`) to pick gradient vs rollout per publish; default to `"gradient"`.
- **Rate:** publish every Nth inference (gradient is a backward pass). RTC absorbs the latency; the first rollout call pays a one-time ~7 s hook-install.

That's the whole contract. The overlay's render (gated blue→yellow, smoothing, percentile) and the GUI controls come for free.

---

## 4. Files

- `policies/hvla/s1/flow_matching/model.py` — `compute_input_saliency`, `compute_attention_rollout`
- `policies/hvla/s1_inference.py` — `_publish_aux` / `_publish_saliency` / `_overlay_method` (the publish + method dispatch)
- `overlays/aux_ipc.py` — `SharedAuxBuffer` (the aux seam)
- `overlays/adapters.py` — `PolicySaliencyAdapter` (the colorizer + render styles)
- `overlays/standalone.py` — the worker (reads aux + obs-stream, writes the overlay buffer)
- `gui/api/overlays.py` — the Method/Style/Smoothing controls + PNG serving
- Follow-ups (demand-gate, async off the inference thread, SmoothGrad, raw-capture-for-offline-analysis, a "warming up" badge): `gui/TODO.md`
