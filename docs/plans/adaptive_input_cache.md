# Plan: Adaptive Input Cache for Training

## Problem Statement

HVLA S1 training is input-bound: the GPU spends most of each step waiting for
frames. On the reference rig the loader delivers **124 frames/s** through the
repository's own decode path, against a full S1 training step that consumes 575
at the current batch size and **867 at the best configuration measured**. The
accelerator therefore runs at roughly **14% of its capacity**. Training time is
set by the dataloader, not by the model.

The gap widens as the GPU is used better, which is the awkward part: every knob
that raises utilisation — a larger batch, disabling gradient checkpointing —
raises the frames/s the loader must supply, and it is already 7× short. Input
throughput is the precondition for any of the rest.

The cause is not a slow disk. It is that every training step decodes 720p AV1
video and then throws away 91% of the pixels — the model consumes 224×224. We
pay full-resolution decode cost for every sample, every epoch, forever.

This document designs a cache that removes that cost, adapts its own
configuration to the machine and dataset it finds itself on, and specifies the
instrumentation needed to know whether any of it worked. The training view
already charts data-wait against update time, so the starting point is better
than nothing — but it reports residual stall with no ceiling to compare against,
which is precisely the shape of signal a caching policy cannot be built on.

## Evidence

Measured 2026-08-15 on the reference rig (RTX 5090 32 GB, 60 GB RAM, NVMe) using
`thewisp/intervention_cylinder_ring_assembly`: 15,752 frames × 4 cameras, 720p
AV1, GOP=2, 0.56 GiB on disk.

| Path                                               |                  frames/s | ms/frame |
| -------------------------------------------------- | ------------------------: | -------: |
| Repo decode path, 4 cameras (26 files) — **today** |                   **124** |     8.06 |
| Repo decode path, 1 camera (5 files)               |                       197 |     5.06 |
| Pre-opened decoders, integer index, 720p           |                       458 |     2.18 |
| Repo path, source stored at 256×256                |                     1,067 |     0.94 |
| Repo path, source stored at 224×224                |                     2,108 |     0.47 |
| Raw `uint8` memmap at 256×256 (page-resident)      |                      ~10⁶ |    0.001 |
| NVDEC hardware decode                              | unavailable in this build |        — |

**GPU demand: 575 frames/s — measured, not inferred.** A full S1 training step
(DINOv2 ViT-S/14 unfrozen with gradient checkpointing, observation encoder, flow
decoder, training-time RTC, gradient clipping, AdamW, bf16 autocast) on synthetic
pre-loaded tensors runs at **17.98 steps/s** with batch 8 × 4 cameras × 224,
peak 2.9 GiB. Config resolved from the dataset's real metadata: 4 cameras,
`action_dim=14`, `state_dim=14`, chunk 50.

This is DS-Analyzer phase 1, the ingestion ceiling, and it is the denominator for
every loader figure above. A backbone-only estimate had put it at 720 frames/s;
the full step is 20% slower, so the deficit is **4.6×** rather than 5.8×. The
cache's target of 2,108 frames/s still clears the ceiling by 3.7×.

**Cache build cost: 309 frames/s** by sequential decode with no seeking →
**3.4 minutes** for the whole four-camera dataset.

**Both branches of the policy are reachable.** Against the same four-camera,
26-file access pattern, with only the stored resolution differing:

| Dataset          | frames/s | vs ceiling | Correct policy action       |
| ---------------- | -------: | ---------: | --------------------------- |
| 720p AV1 (today) |  124–138 |      0.24× | **cache** — 4.2× short      |
| 224-native AV1   |    1,529 |      2.66× | **decline** — already ahead |

An 11× spread across the decision boundary, from resolution alone. This is the
evidence for gate 5: a policy that always caches would be wrong on the second
row, and nothing else measured here would have caught that.

### The ceiling is a curve, and the objective needs stating precisely

575 frames/s is the demand at batch 8. Batch size is a free variable, and the
goal is minimum wall clock, so the ceiling is properly a curve:

| Batch | steps/s | samples/s | frames/s demanded | VRAM GiB | % of peak |
| ----: | ------: | --------: | ----------------: | -------: | --------: |
|     4 |   25.96 |     103.9 |               415 |     2.27 |       49% |
|     8 |   17.97 |     143.8 |               575 |     2.95 |       68% |
|    16 |   11.31 |     180.9 |               724 |     4.42 |       86% |
|    32 |    6.23 |     199.3 |               797 |     7.43 |       95% |
|    64 |    3.19 |     204.1 |               816 |    13.40 |       97% |
|   128 |    1.64 |     210.3 |               841 |    25.37 |      100% |
|   256 |     OOM |         — |                 — |        — |         — |

**"The largest batch that fits VRAM" is close to right, and is not the optimum.**
Batch 32 reaches 95% of peak throughput on 29% of the memory; the last 5% costs
3.4× the VRAM. More pointedly, VRAM has competing claimants and batch is not the
best one here — gradient checkpointing is on by default, and turning it off buys
10–13%:

| Batch | Checkpointing | samples/s | VRAM GiB |
| ----: | ------------- | --------: | -------: |
|     8 | on            |     143.0 |     2.94 |
|     8 | off           |     161.6 |     4.29 |
|    32 | on            |     197.1 |     7.43 |
|    32 | **off**       | **216.8** |    12.15 |
|   128 | on            |     210.3 |    25.37 |

Batch 32 without checkpointing beats batch 128 with it — faster on less than half
the memory. Checkpointing is a default for memory-constrained hardware being
applied on a card with 20 GiB spare, and it is costing wall clock to save memory
nothing else wants.

So the objective is **maximise sustained samples/s**, not maximise batch. Batch,
checkpointing, and any other VRAM claimant are means; the memory budget should go
wherever throughput responds most steeply, which is an empirical question per
model and per GPU. A third constraint sits outside throughput entirely: batch
size changes optimisation dynamics, so wall-clock-per-step is not
wall-clock-to-quality, and the tuner must not be free to raise batch without
bound.

### Batch tuning and caching are coupled, and the order matters

Raising batch raises input demand in lockstep: 415 frames/s at batch 4, 867 at
batch 32 with checkpointing off. The loader supplies 124.

This makes the coupling directional. **Auto-scaling batch against a starved
loader buys nothing at all** — throughput is pinned at what the input pipeline
delivers, so a larger batch simply makes the GPU wait longer per step while
consuming VRAM and perturbing optimisation. Worse, it would look successful to a
tuner watching VRAM occupancy rather than samples/s.

The order is therefore fixed: **fix input first, then scale batch.** A tuner that
runs before the cache exists will converge on a large batch and no speedup. This
is also why the two cannot be separate features — the cache is what makes batch
tuning meaningful, and batch tuning is what makes the cache worth its disk.

At the best measured configuration the deficit is worse than this document's
headline: 867 frames/s demanded against 124 supplied is **7× short**, with the
GPU at roughly **14%** of capacity.

Four secondary findings shaped the design:

- **Keyframe interval dominates seek cost, not resolution.** A naive downsize to
  224 at the encoder's default GOP of 161 measured _slower_ than the 720p source
  (321 vs 561 random frames/s). The recording path writes `g=2`, which is what
  makes random access cheap at all.
- **The timestamp lookup path costs ~2.3×.** Pre-opened decoders addressed by
  integer index reach 458 frames/s where `decode_video_frames_torchcodec` with a
  float timestamp and tolerance reaches 197. The decoder LRU is not the cause —
  its default capacity is 100 against 26 files.
- **Four cameras are worse than one** (124 vs 197 frames/s) even with adequate
  decoder cache capacity, so per-sample fan-out across files carries its own
  cost.
- **GOP=2 flattens the codec axis.** Re-encoded from the same clips at the
  recorder's own `g=2`: AV1 197, H.264 219, HEVC 127 frames/s — a 1.11× spread
  between the first two and HEVC actually slower. At GOP=2 every second frame is
  intra-coded, so the inter-frame prediction where AV1 and HEVC spend their
  complexity barely runs. Compression still differs sharply (143 / 77 / 45 MB),
  so the codec choice remains a storage decision and is nearly a non-decision for
  decode speed.

## What We Can and Cannot Currently See

**More exists than this document first claimed.** `TrainingHealthTracker` already
wraps the batch fetch and host-to-device copy, and emits `data_s` alongside
`updt_s` every logging window; the training view charts both as "Data" and
"Update". An operator can see that a run is blocked on input today, without any
new tooling. An earlier draft of this plan asserted the opposite and was wrong.

What the existing signal cannot answer is everything the caching policy needs:

- **No ceiling.** `data_s` reports time spent blocked, never what the GPU could
  have consumed. A small `data_s` is equally consistent with "the input pipeline
  is comfortable" and "the model is slow too". Without the ingestion rate `G`
  there is no denominator, so there is no way to say how much of the accelerator
  is being wasted — the number this design exists to move.
- **No attribution inside `data_s`.** Fetch stalls (storage) and prep stalls
  (CPU decode and resize) are indistinguishable, and they have opposite remedies:
  one wants a cache, the other wants cheaper pixels or more workers.
- **No per-feature breakdown.** With four cameras and possibly a depth stream on
  a different backend, one expensive feature is invisible inside a single scalar.
- **It measures residual stall, not margin.** With prefetching, `data_s` is what
  leaks through _after_ the workers have hidden what they can. A pipeline running
  at 99% of capacity reports `data_s ≈ 0` and is one camera away from a cliff.
  This is the dangerous property: the metric looks healthiest immediately before
  it collapses, so it cannot be used to decide whether caching is worthwhile.
- **Window means only**, so a periodic stall — an epoch boundary, a file
  rollover — is averaged into invisibility.

**And a stopwatch cannot fix the first two.** In a pipelined loader a stall in
one stage surfaces as compute time in another; per-stage timers produce numbers
that sum correctly and attribute wrongly. That is the failure the DS-Analyzer
methodology in
[Analyzing and Mitigating Data Stalls in DNN Training](https://vldb.org/pvldb/vol14/p771-mohan.pdf)
was built to avoid, which is why §6 adds a differential probe rather than more
instrumentation points. The existing `data_s` chart stays as the cheap online
signal; the probe supplies the ceiling and the attribution it structurally
cannot.

## Scope: One Model and One GPU Are the Measurement, Not the Design

Everything measured here is HVLA S1 on an RTX 5090. That is the calibration
sample, not the target. The design has to hold for pi05 on an A100 or H100, for
smaller policies on consumer cards, and for hardware nobody has bought yet.

What generalises is the shape of the problem: a loader supply rate, a GPU
ingestion ceiling, and a decision that turns on their ratio. What does not
generalise is any number in this document. The parameters that move that ratio
by an order of magnitude are:

| Parameter          | Range in practice                   | Effect on the ratio                                                                            |
| ------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------- |
| Model size         | S1 (~30M + ViT-S) → pi05 (billions) | **Inverse.** A larger model consumes fewer samples/s, so demand falls and caching matters less |
| GPU                | 5090 32 GB → A100/H100 80 GB        | Raises both the ceiling and the batch that fits                                                |
| Batch size         | 4 → 128 measured here               | Demand scales with it, 415 → 841 frames/s                                                      |
| Cameras per sample | 1 → 4+                              | Multiplies frames per sample directly                                                          |
| Source resolution  | 224 → 720p and beyond               | The dominant term in supply — 11× measured                                                     |
| CPU cores / disk   | Workstation → cloud instance        | Sets supply; a cloud VM with few vCPUs starves sooner                                          |

**Bigger models make the decline branch ordinary, not exotic.** A pi05-scale
model on an A100 may consume tens of samples/s rather than hundreds, at which
point a 124 frames/s loader is comfortably ahead and a cache is pure cost. The
same code path that caches aggressively for S1 on a 5090 must decline there, and
that is the case most likely to be got wrong, because it is the one nobody
developing on the reference rig will see.

The design consequence is a prohibition: **no constant in this document may be
compiled in.** Not 575, not 224, not the JPEG rung, not the worker count. Each is
the output of a probe on the machine and workload at hand. The document records
them so the reasoning is checkable and so a regression is recognisable, not so
they can be defaults.

Two clarifications this scope forces on the plan:

- The batch sweep above is **part of the probe**, not a one-off. Phase 1 must
  sweep batch to find the throughput knee for the actual model, since the knee's
  location is a property of model and GPU jointly and cannot be assumed from S1.
- The cost model in §4 already takes `t_source` and GPU demand as measured
  inputs, so it generalises without change. That was the point of expressing the
  decision as a breakeven rather than a threshold.

## Design

### 1. Where the cache cuts the pipeline

The pipeline divides at the boundary between deterministic and random work:

```
  decode ──► resize ──►│──► augment ──► normalise ──► model
  └── deterministic ───┘   └────────── random ───────────┘
                       ▲
                  cache cut
```

Everything left of the cut produces the same bytes for a given frame on every
epoch, so it can be computed once. Everything right of it must vary per epoch or
it stops being augmentation. This is the autocaching rule from
[Cachew](https://www.usenix.org/conference/atc22/presentation/graur): cache at
the last deterministic operation, never after a random one.

This rule settles the resolution question that motivated the work. **The cache
resolution is the width of the deterministic prefix, not the model input.**
Today S1 applies `--resize-images 224x224` as a plain full-frame resize and
performs no image augmentation at all — verified: the only thing named
"augmentation" in the trainer is S2 latent delay. So today the cut sits at 224
and caching at 224 is exactly right.

If crop augmentation is added — the indicated fix for the visual-overfitting
result — the cut moves left, to whatever resolution crops are sampled from, and
the cache resolution must move with it. The policy must therefore treat
resolution as **derived from the configured transform chain**, never as a
constant. A cache keyed on a transform signature rebuilds itself when that chain
changes; a cache keyed on a hardcoded 224 silently serves wrong-sized frames.

### 2. The source is not one format, and the policy must not assume it is

Every number in the Evidence table describes one storage configuration: 720p
AV1 at GOP=2, RGB, decoded by torchcodec. The dataset format admits many others,
and their decode costs differ by more than the effect this design is trying to
produce.

| Axis                   | Values the format allows                                          | Why it changes the answer                                                                                                       |
| ---------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Codec                  | `h264`, `hevc`, `libsvtav1`, `libaom-av1`, plus hardware encoders | Less than expected at GOP=2 — see below. Measured 197 (AV1) / 219 (H.264) / 127 (HEVC) frames/s, all far under the ceiling      |
| `vcodec="auto"`        | Resolves to a _hardware_ encoder when one exists                  | The same recording script produces different codecs on different machines. Codec is a property of the file, not of the pipeline |
| Resolution             | Per-feature, set at record time                                   | 720p is not a constant; a 224-native dataset needs no cache at all                                                              |
| GOP / `crf` / `preset` | Per-recording encoder settings                                    | GOP dominates seek cost — the finding that inverted our first conclusion                                                        |
| `fast_decode`          | Codec-specific tuning, **default 0 (off)**                        | An encoder-side decode-speed lever we are not currently using                                                                   |
| Feature type           | RGB video, **depth video**                                        | Depth is quantised and forced onto the pyav backend, so it has a different cost curve _within the same dataset_                 |
| Codebase version       | v2.1, v3.0, converted in place by `convert_dataset_v21_to_v30`    | Layout changes under a stable `repo_id`                                                                                         |

**Codec matters far less than expected, and the reason matters more.** An earlier
draft asserted H.264 decodes several times faster than AV1, so that an H.264
dataset might already outrun the GPU and make caching pure cost. Measured on the
same clips re-encoded at the recorder's own GOP=2, that is false: H.264 reaches
219 frames/s against AV1's 197 — 1.11× — and HEVC is _slower_ at 127.

The explanation is the same finding that inverted our first conclusion. At GOP=2
every second frame is intra-coded, so inter-frame prediction — where AV1 and HEVC
spend their complexity and win their compression — barely executes. The recording
path's own choice of `g=2` flattens the codec axis into near-irrelevance for
decode, while preserving it for file size (143 MB AV1 / 77 MB H.264 / 45 MB HEVC
for the same clips).

So the axis that can make caching unnecessary is **resolution, not codec**. A
224-native dataset needs no cache; a 720p one needs it regardless of how it was
encoded. This narrows the policy's real input from a matrix to something closer
to a single dominant term, and it means gate 5 must be exercised with a
low-resolution dataset rather than a differently-encoded one.

Four consequences bind the rest of the design:

**Probe per feature, never per dataset.** A four-camera RGB dataset with a depth
stream has at least two decode paths with different throughputs. `t_source` is a
vector over features, not a scalar, and the caching decision is made per feature
— it is entirely reasonable to cache the AV1 RGB streams and leave a cheap H.264
one alone.

**The cache key must cover storage, not just content.** Codec, pixel format,
resolution, GOP, codebase version and transform signature all belong in it.
Keying on `repo_id` and resolution alone is unsound: re-encoding a dataset or
running the v2.1→v3.0 converter changes the bytes and the cost profile while the
identifier stays put, and `auto` codec selection means the same logical dataset
can differ between the workstation and the rig.

**Hardware decode availability is codec-dependent.** This box exposes
`h264_cuvid`, `hevc_cuvid` and `av1_cuvid`, so the NVDEC path is not AV1-specific
— but a dataset in a codec the local GPU cannot decode falls back to CPU
silently. Any future GPU-decode work must treat this as a capability probe, not
a build-time assumption.

### 3. Format ladder

There is no single right storage format, because the choice trades space against
decode cost and the correct point depends on how much RAM the dataset can claim.
The policy picks a rung:

| Rung                          | Bytes/frame @224 |                            Read rate | When                                            |
| ----------------------------- | ---------------: | -----------------------------------: | ----------------------------------------------- |
| **Raw `uint8` memmap**        |          147 KiB | ~10⁶/s resident, ~15–35k/s from NVMe | Working set fits comfortably in page cache      |
| **JPEG per frame**            |          ~15 KiB |                             ~5,000/s | Dataset ≫ RAM; still 7× above demand            |
| **Re-encoded video, GOP=2**   |           ~7 KiB |                             ~2,100/s | Space-constrained, or the cache must be shipped |
| **None (decode from source)** |                0 |                                124/s | Cache would not amortise                        |

Raw is 10× the space of JPEG for throughput neither the GPU nor any near-future
GPU can absorb, so **JPEG is the default rung and raw is the opt-in**, inverting
the intuition that fastest is best. The binding constraint is page-cache
residency, not peak read rate: an 8.8 GiB raw cache that fits is excellent, and
one that does not is worse than JPEG.

**Any video rung must pin `-g 2`.** The default GOP produces a cache that is
slower than no cache at all — a regression that presents as an optimisation.
This is the single most important invariant in the design and belongs in a test,
not a comment.

### 4. The caching decision

Caching is worthwhile when building it plus reading it beats not building it:

```
    build_time  +  steps × frames_per_step × t_cached
                <  steps × frames_per_step × t_source
```

Rearranged, the cache pays for itself after

```
    steps_breakeven  =  build_time / (frames_per_step × (t_source − t_cached))
```

With today's numbers — 204 s build, 32 frames/step, 8.06 ms vs 0.47 ms — that is
**≈840 steps**, well under a minute of training. Any real run clears it by
orders of magnitude, which is the quantitative reason to cache eagerly rather
than agonise over it.

The formula matters more than the answer, because it is what makes the decision
_adaptive_: on a dataset of 200 episodes, or a machine with a faster disk, or a
model whose step is 10× slower, it returns a different verdict without anyone
re-deriving it. The policy evaluates it from probed values, not constants.

Three inputs are probed once at dataset open, costing a few seconds:

- `t_source` — decode+resize time per frame, sampled over random frames
- `t_cached` — read time per frame for each affordable rung
- GPU demand — from a short synthetic-input step-rate probe (§6, phase 1)

If measured GPU demand is already below `1/t_source`, the run is compute-bound
and **the policy declines to cache**, because the cache would buy nothing and
cost disk. This is the case the design must not get wrong: a cache that always
builds is not adaptive, it is just eager.

### 5. Fill, eviction and ordering

**Never evict within a run.** DNN training touches every item exactly once per
epoch, which is the pathological case for LRU: entries are evicted just before
they would be reused, and a cache sized at 35% of the dataset delivers far less
than a 35% hit rate. The MinIO result is that a cache which simply fills and then
stops replacing achieves hit rate equal to its capacity fraction. When the cache
cannot hold the dataset, it should hold a **fixed random subset**, chosen once,
and the remainder should stream from source.

This also means the design must not lean on the OS page cache as its cache. Page
cache is welcome as an accelerator underneath a memmap, but the residency
decision has to be ours, because the kernel's replacement policy is the wrong one
for this access pattern.

**Order accesses to respect locality when the cache does not fit.** Full random
shuffling maximises statistical quality and minimises I/O locality. FFCV's
quasi-random ordering samples a permutation of _pages_ and draws batches from the
resident pages, approximating a shuffle while keeping reads local. Adopt it only
on the partial-cache path; when the cache is fully resident, plain random
shuffling costs nothing and is statistically cleaner.

### 6. Instrumentation

Two layers, because they answer different questions.

**Offline: a stall-attribution probe.** A `lerobot-bench-input` command
implementing the DS-Analyzer three-phase differential:

1. **Ingestion ceiling** — train on synthetic pre-loaded tensors. No I/O, no
   preprocessing. Yields max GPU step rate `G`.
2. **Prep stalls** — train on a fully cached dataset with all cores available
   but GPU compute disabled. The shortfall against phase 1 is CPU preprocessing
   cost.
3. **Fetch stalls** — drop caches, train normally. The shortfall against phase 2
   is storage I/O.

The output is a three-way split of epoch time into compute, prep stall and fetch
stall. That is the number this whole design exists to move, and it is the
acceptance criterion for every change below.

**Online: add margin to the existing signal.** The training view already charts
`data_s` and `updt_s`, so the work here is not a new dashboard but fixing the
one property that makes the existing one unsafe to act on — it reports residual
stall, and reads healthiest just before the pipeline falls off its cliff.

**Prefetch queue depth** supplies the missing margin. A persistently full queue
means the loader is ahead and the GPU is the constraint; a queue hovering near
empty means the pipeline is at its limit even when `data_s` is still small. It
costs nothing to sample, needs no differential reasoning, and is the difference
between "we are fine" and "we are one camera from a cliff".

Both belong on the training view rather than behind flags — the operator runs
everything through the GUI, and an instrument nobody can reach closes no gap.
Charting queue depth beside the existing Data/Update series is a small change to
a panel that already exists, which is the cheapest half of this whole design.

### 7. Autotuning, in a fixed order

Four knobs are currently constants: worker count and prefetch depth
(`num_workers: 3`), batch size, and gradient checkpointing. All four are worth
tuning — both tf.data's AUTOTUNE and
[Plumber](https://anakli.inf.ethz.ch/papers/plumber_mlsys22.pdf) report large
speedups purely from correcting misconfiguration — but they are not independent,
and tuning them in the wrong order produces confident nonsense.

The order follows the dependency:

1. **Supply first** — cache decision, then workers and prefetch depth, until the
   loader is comfortably ahead of the current ceiling. Tuning anything else
   against a starved loader measures the loader.
2. **Then the VRAM budget** — sweep batch and checkpointing together, since they
   compete for the same memory and the better claimant is workload-specific.
   Optimise sustained samples/s end to end, never VRAM occupancy or steps/s.
3. **Then re-check supply**, because step 2 raises demand. One iteration is
   normally enough; the loop terminates when raising the ceiling stops producing
   throughput.

A hill-climb on measured samples/s over a few hundred steps is sufficient and far
simpler than an analytical model. Freeze afterwards rather than tuning
continuously: oscillating worker counts or batch sizes make step times noisy,
results incomparable, and — for batch — optimisation itself non-reproducible.

Batch carries a constraint the others do not. Larger batches change convergence,
so the tuner needs a ceiling it may not cross without being told, and the chosen
value must be recorded with the run. A run that silently retuned its own batch
size is not comparable to the one before it.

## Alternatives Considered and Rejected

**GPU decode (NVDEC).** The RTX 5090 exposes `av1_cuvid` and the hardware path is
the architecturally cleanest answer — it removes decode from the CPU entirely
rather than avoiding it. Rejected for now on two grounds: the installed
torchcodec build fails `validateDeviceInterface` for CUDA devices, so it is not
reachable without changing the stack; and the repository already documents that
CUDA decoding inside DataLoader workers raises initialisation errors, so adopting
it means restructuring the loader, not setting a flag. This is the strongest
future direction and should be revisited when either constraint lifts —
[DALI](https://developer.nvidia.com/dali) is the reference implementation.

**Preprocessed derived datasets.** The original proposal: a GUI operation that
writes a resized copy as a new dataset. Rejected because it makes the user manage
a second copy of the truth — naming, publishing, staleness, deletion — to solve
what is a caching problem. A cache keyed on `(dataset fingerprint, transform
signature)` is invisible, self-invalidating, and costs one 3.4-minute rebuild to
change your mind rather than a regeneration and republish. Resizing is not a
decision the operator should be asked to make.

**Data echoing.** Reusing each batch for several optimisation steps reclaims idle
accelerator time when input-bound, without touching the input pipeline, and
[the original study](https://arxiv.org/abs/1907.05550) found no quality
degradation on the workloads tested. Rejected as a primary remedy because it
treats the symptom: our input pipeline is 6× short of demand and fixable
directly. Worth keeping as a fallback for cases the cache cannot serve — a
dataset far larger than disk, for example.

**Fixing it at the encoder instead of the loader.** The recording path already
exposes `fast_decode`, a codec-specific tuning knob that trades file size for
decode speed, and it is off by default. Recording at a lower resolution, or with
`fast_decode` enabled, would reduce the problem at its source with no cache at
all. Rejected as _the_ answer for two reasons: it does nothing for the datasets
already recorded, which is all of them; and it couples a recording decision to a
training decision that may not exist yet — you would be choosing the model's
input resolution at the moment you press record. It is worth evaluating
separately as a default change for new recordings, where it is close to free.

**Caching encoded rather than decoded items.** CoorDL caches raw compressed items
because decoded data is 5–7× larger. That reasoning holds when decode is cheap
relative to fetch. Ours is the opposite case — decode _is_ the bottleneck — so
caching post-decode is what buys the win. The format ladder in §3 is where this
tension is resolved per-machine rather than by fiat.

## Verification

The design is not accepted until measured. Every claim above is falsifiable and
should be falsified before code is written against it.

**Benchmark harness.** Fixed and reported: dataset fingerprint, **codec, pixel
format, GOP**, resolution, camera count, batch size, worker count, model, and
whether page cache was dropped. Varying: cache rung, ordering strategy, worker
count. The harness must support dropping caches between runs, or every number
after the first is a lie.

**The matrix must span resolutions, and only sample codecs.** A suite run only on
720p AV1 would validate the design on the case it was designed for and never
exercise the branch where the policy declines. The axis that flips the decision
is resolution: at minimum 720p (caches) and a 224-native dataset (must decline).

Codec earns two confirmation runs rather than a full axis, since it was measured
at 1.11× between AV1 and H.264 — enough to catch a regression in that assumption,
not enough to justify crossing it with everything else. A depth feature belongs
in the matrix on different grounds: it is the only input that exercises the pyav
backend, and backend differences were never measured at all.

All of these are cheap to synthesise by re-encoding one episode.

**Acceptance gates.** The design ships only if all hold on the reference rig:

| #   | Gate                                                                                                           | Rationale                                                                                  |
| --- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1   | Fetch + prep stall falls below **10%** of epoch time, from a measured baseline                                 | The actual goal; measured by the phase probe, not inferred                                 |
| 2   | End-to-end epoch time improves **≥3×**                                                                         | Guards against moving the bottleneck without moving the clock                              |
| 3   | Cache build amortises within the measured `steps_breakeven`, and the probe agrees with the model to within 25% | If the cost model mispredicts, the adaptive policy is unsound even when the cache is fast  |
| 4   | A GOP-161 cache is **rejected by a test**, not merely avoided                                                  | The one configuration that silently regresses                                              |
| 5   | Compute-bound workload → policy declines to cache                                                              | Proves adaptivity rather than eagerness                                                    |
| 6   | Transform-chain change invalidates the cache                                                                   | Proves the key is correct; a stale cache serves wrong-sized frames silently                |
| 7   | GPU utilisation reaches **≥90%** at the tuned batch, measured, not inferred from VRAM                          | The stated objective. Occupancy and steps/s can both look healthy while throughput is flat |
| 8   | Batch tuning run against a **starved** loader reports no improvement                                           | The failure mode of tuning in the wrong order; a tuner that "succeeds" here is broken      |
| 9   | A large-model / small-demand workload declines to cache **and** declines to raise batch                        | The pi05-on-A100 case nobody on the reference rig will encounter                           |

**Negative controls.** Gates 4–6, 8 and 9 are deliberately failure cases. A
benchmark suite that only demonstrates the happy path will pass against a cache
that ignores its own policy, and against a tuner that maximises the wrong
quantity — which are precisely the defects worth catching.

**Gate 9 needs a workload nobody here has.** It can be approximated without an
A100 by holding the loader fixed and substituting a deliberately slow model —
a larger encoder, a longer chunk, or simply an injected delay — so that demand
falls below supply. That is a weaker test than the real thing and should be
labelled as such wherever its result is reported.

**Independent quick win, measured separately.** The timestamp-lookup path costs
~2.3× (458 vs 197 frames/s) and is unrelated to caching. It should be measured
and, if confirmed, fixed on its own merits — otherwise the cache will be credited
with a speedup it did not produce.

## Risks

**The cost model is calibrated on one rig and one dataset.** Every number here
comes from a single machine and a single 34-episode dataset. The design's
defence is that the policy probes rather than assumes; the risk is that probe
values are noisy enough to make it oscillate. Gate 3 exists to catch this.

**Disk growth is unbounded across datasets.** Per-dataset caches accumulate.
Never-evict is correct _within_ a run and wrong _across_ them; the design needs a
cross-run LRU over whole caches, plus a visible total and a way to clear it.

**A silent stale cache is the worst failure mode.** It cannot be detected from
loss curves and would corrupt every result produced while it persisted.
Fingerprint verification must be a startup assertion that fails loudly, not a
best-effort comparison.

**~~The decline branch has never been exercised.~~ Resolved.** It has now, against
a 224-native mirror of the same dataset: 1,529 frames/s, 2.66× the ceiling, where
the policy must decline. H.264 had been assumed to be the input that would
exercise this and was not — the two codecs are 1.11× apart. Resolution is the
discriminator, and it separates the branches by 11×.

**Nothing has been measured against a cache that is too large to hold.** Every
figure assumes page-cache residency: 8.8 GiB against 32 GiB available. The
partial-cache path — never-evict fill over a fixed random subset, quasi-random
ordering, the remainder streaming from source — is entirely unmeasured, and it is
the path with the most moving parts. A dataset an order of magnitude larger than
this one would exercise it, and none is on hand. This is now the largest hole.

**Silent per-machine divergence via `vcodec="auto"`.** Two datasets recorded by
the same script on the workstation and the rig can differ in codec, and therefore
in whether caching is worthwhile. Anything that reports "cached" or "declined"
must report the codec alongside, or the difference reads as nondeterminism.

**~~Backbone-only GPU demand may understate headroom.~~ Resolved.** The full S1
step was measured at 17.98 steps/s (575 frames/s), 20% slower than the
backbone-only estimate. The deficit narrows from 5.8× to 4.6× and the conclusion
holds. Kept here rather than deleted because it is the pattern to repeat: the
risk named a specific measurement, the measurement was cheap, and running it
before implementation cost nothing and moved a headline number.

**The ceiling itself is workload-specific — quantified.** 575 frames/s was one
encoder, one batch size, checkpointing on. Sweeping batch moves it to 841, and
turning checkpointing off moves it to 867; a larger model moves it down by an
order of magnitude. The span across the model × GPU × batch matrix is wider than
the effect the cache produces, which is why phase 1 ships as a sweeping probe
rather than as a number recorded here.

**An auto-tuner could optimise the wrong objective convincingly.** Batch tuned
against VRAM occupancy, or against steps/s rather than samples/s, will report
success while delivering none — and against a starved loader it will do so every
time. The tuner's objective must be sustained samples/s measured end to end, and
it must run after the cache decision, never before it.

## Sources

- [Analyzing and Mitigating Data Stalls in DNN Training](https://vldb.org/pvldb/vol14/p771-mohan.pdf) — MinIO never-evict cache, DS-Analyzer differential stall attribution
- [Cachew: Machine Learning Input Data Processing as a Service](https://www.usenix.org/conference/atc22/presentation/graur) — autocaching cut-point policy, autoscaling from domain metrics
- [FFCV: Accelerating Training by Removing Data Bottlenecks](https://arxiv.org/pdf/2306.12517) and [the Bottleneck Doctor](https://docs.ffcv.io/bottleneck_doctor.html) — quasi-random ordering, storage-format ladder, bottleneck-to-remedy mapping
- [Plumber: Diagnosing and Removing Performance Bottlenecks in ML Pipelines](https://anakli.inf.ethz.ch/papers/plumber_mlsys22.pdf) — automatic tuning of parallelism, prefetch and caching
- [tf.data: A Machine Learning Data Processing Framework](https://arxiv.org/pdf/2101.12127) — AUTOTUNE
- [Faster Neural Network Training with Data Echoing](https://arxiv.org/abs/1907.05550) — reclaiming idle accelerator time when input-bound
- [NVIDIA DALI](https://developer.nvidia.com/dali) — GPU-side decode and preprocessing, NVDEC
