# Plan: Adaptive Input Cache for Training

## Problem Statement

HVLA S1 training is input-bound: the GPU spends most of each step waiting for
frames. On the reference rig the loader delivers **124 frames/s** through the
repository's own decode path while a full S1 training step consumes
**575 frames/s**, so the accelerator runs at roughly **22% of its capacity** and
about four fifths of it is idle. Training time is set by the dataloader, not by
the model.

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

Three secondary findings shaped the design:

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
| Codec                  | `h264`, `hevc`, `libsvtav1`, `libaom-av1`, plus hardware encoders | H.264 decodes several times faster than AV1. An h264 dataset may already outrun the GPU, making a cache pure cost               |
| `vcodec="auto"`        | Resolves to a _hardware_ encoder when one exists                  | The same recording script produces different codecs on different machines. Codec is a property of the file, not of the pipeline |
| Resolution             | Per-feature, set at record time                                   | 720p is not a constant; a 224-native dataset needs no cache at all                                                              |
| GOP / `crf` / `preset` | Per-recording encoder settings                                    | GOP dominates seek cost — the finding that inverted our first conclusion                                                        |
| `fast_decode`          | Codec-specific tuning, **default 0 (off)**                        | An encoder-side decode-speed lever we are not currently using                                                                   |
| Feature type           | RGB video, **depth video**                                        | Depth is quantised and forced onto the pyav backend, so it has a different cost curve _within the same dataset_                 |
| Codebase version       | v2.1, v3.0, converted in place by `convert_dataset_v21_to_v30`    | Layout changes under a stable `repo_id`                                                                                         |

Three consequences bind the rest of the design:

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

### 7. Autotuning the remaining knobs

Worker count and prefetch depth are currently fixed constants
(`num_workers: 3`). Both tf.data's AUTOTUNE and
[Plumber](https://anakli.inf.ethz.ch/papers/plumber_mlsys22.pdf) show these are
worth tuning automatically, with Plumber reporting large speedups purely from
correcting misconfiguration. A hill-climb on measured samples/s, run for a few
hundred steps at the start of training and then frozen, is sufficient and far
simpler than a full analytical model. Freeze rather than tune continuously:
oscillating worker counts mid-run make step times noisy and results hard to
compare.

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

**The matrix must span codecs, not just cache rungs.** A suite run only on
720p AV1 would validate the design on its easiest case and tell us nothing about
where the policy declines to cache. At minimum: AV1 (the measured case), H.264
(the plausible already-fast case), and a depth feature (the pyav backend). These
are cheap to synthesise by re-encoding a single episode, and they are the inputs
that exercise gate 5.

**Acceptance gates.** The design ships only if all hold on the reference rig:

| #   | Gate                                                                                                           | Rationale                                                                                 |
| --- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1   | Fetch + prep stall falls below **10%** of epoch time, from a measured baseline                                 | The actual goal; measured by the phase probe, not inferred                                |
| 2   | End-to-end epoch time improves **≥3×**                                                                         | Guards against moving the bottleneck without moving the clock                             |
| 3   | Cache build amortises within the measured `steps_breakeven`, and the probe agrees with the model to within 25% | If the cost model mispredicts, the adaptive policy is unsound even when the cache is fast |
| 4   | A GOP-161 cache is **rejected by a test**, not merely avoided                                                  | The one configuration that silently regresses                                             |
| 5   | Compute-bound workload → policy declines to cache                                                              | Proves adaptivity rather than eagerness                                                   |
| 6   | Transform-chain change invalidates the cache                                                                   | Proves the key is correct; a stale cache serves wrong-sized frames silently               |

**Negative controls.** Gates 4–6 are deliberately failure cases. A benchmark
suite that only demonstrates the happy path will pass against a cache that
ignores its own policy, which is precisely the defect worth catching.

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

**The measured case may be the unrepresentative one.** All evidence here comes
from 720p AV1 — plausibly the most expensive configuration the format allows.
On an H.264 dataset the policy may correctly decline to cache, and a design
validated only against AV1 would never have exercised that path. Gate 5 is the
guard, and the codec matrix above is what makes it meaningful.

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

**The ceiling itself is workload-specific.** 575 frames/s is one encoder
(ViT-S/14), one batch size, one chunk length, with gradient checkpointing on.
ViT-B, a longer chunk, or disabling checkpointing all move it, and a large enough
move flips the caching decision. This is the reason phase 1 belongs in the
shipped probe rather than in this document as a constant.

## Sources

- [Analyzing and Mitigating Data Stalls in DNN Training](https://vldb.org/pvldb/vol14/p771-mohan.pdf) — MinIO never-evict cache, DS-Analyzer differential stall attribution
- [Cachew: Machine Learning Input Data Processing as a Service](https://www.usenix.org/conference/atc22/presentation/graur) — autocaching cut-point policy, autoscaling from domain metrics
- [FFCV: Accelerating Training by Removing Data Bottlenecks](https://arxiv.org/pdf/2306.12517) and [the Bottleneck Doctor](https://docs.ffcv.io/bottleneck_doctor.html) — quasi-random ordering, storage-format ladder, bottleneck-to-remedy mapping
- [Plumber: Diagnosing and Removing Performance Bottlenecks in ML Pipelines](https://anakli.inf.ethz.ch/papers/plumber_mlsys22.pdf) — automatic tuning of parallelism, prefetch and caching
- [tf.data: A Machine Learning Data Processing Framework](https://arxiv.org/pdf/2101.12127) — AUTOTUNE
- [Faster Neural Network Training with Data Echoing](https://arxiv.org/abs/1907.05550) — reclaiming idle accelerator time when input-bound
- [NVIDIA DALI](https://developer.nvidia.com/dali) — GPU-side decode and preprocessing, NVDEC
