# Plan: A Resolution Shadow for Training Datasets

## Problem

Training decodes full-resolution video every step and throws most of the pixels
away to reach whatever the model consumes. The cost is paid once per sample, every
epoch, forever, and belongs to how the data is stored rather than to any model.

Measured on the reference rig, sampling the device counter during real training:

| dataset                         |     CPU |      GPU |
| ------------------------------- | ------: | -------: |
| 4-camera 720p                   | **95%** |  **61%** |
| the same data stored at 224×224 |  **6%** | **100%** |

Input-bound and CPU-saturated at capture resolution; GPU-bound once stored at the
size the model consumes. A 20k-step run on the 274-episode reference dataset fell
from **252 to ~153 minutes** — 1.64× — for a one-time conversion of 70.7 minutes
that also took the dataset from 4.3 GB to 0.68 GB.

> An earlier draft put the accelerator at 14% of capacity. That was a throughput
> ratio, not a measurement, and it overstated the problem; the counter says 61%.
> The premise survives, the urgency does not.

**No ratio of `data_s` to `updt_s` measures utilisation.** Async CUDA keeps
kernels running while the loop blocks on the next batch, so the ratio understates
utilisation exactly when prefetching works — it read 62% on a run the counter put
at 100%. Anything built on it inherits the error.

## Evidence

Decode throughput, `thewisp/intervention_cylinder_ring_assembly`, 4 cameras 720p
AV1 GOP=2:

| Path                                          | frames/s | ms/frame |
| --------------------------------------------- | -------: | -------: |
| Repo decode path, 4 cameras — **today**       |  **124** |     8.06 |
| Repo decode path, 1 camera                    |      197 |     5.06 |
| Repo path, source stored at 256×256           |    1,067 |     0.94 |
| Repo path, source stored at 224×224           |    2,108 |     0.47 |
| Raw `uint8` memmap at 256×256 (page-resident) |     ~10⁶ |    0.001 |

GPU demand and the batch curve, measured on synthetic pre-loaded tensors:

| Batch | steps/s | frames/s demanded | VRAM GiB | % of peak |
| ----: | ------: | ----------------: | -------: | --------: |
|     8 |   17.97 |               575 |     2.95 |       68% |
|    32 |    6.23 |               797 |     7.43 |       95% |
|   128 |    1.64 |               841 |    25.37 |      100% |

Three findings that changed a decision:

- **GOP dominates seek cost, not resolution.** A naive downsize at the encoder's
  default GOP measured _slower_ than the 720p source. The recorder writes `g=2`,
  which is the only reason random access is cheap. **Any re-encode must pin
  `-g 2`** — the default produces a cache slower than no cache, a regression that
  presents as an optimisation. This belongs in a test.
- **GOP=2 flattens the codec axis.** AV1 197, H.264 219, HEVC 127 frames/s. At
  GOP=2 every second frame is intra-coded, so the inter-frame prediction where AV1
  and HEVC spend their complexity barely runs. Codec stays a storage decision
  (143/77/45 MB) and is nearly a non-decision for speed.
- **The largest batch that fits VRAM is not the optimum.** Batch 32 reaches 95% of
  peak on 29% of the memory, and gradient checkpointing — on by default — costs
  10–13%: batch 32 without it beats batch 128 with it.

## Model families read the data differently

A per-step policy samples one random frame per camera; a video JEPA or world model
reads contiguous clips, which `LeRobotDataset` already expresses via
`delta_timestamps`. Measured on the same file, 16-frame clips against single
random frames: at GOP=2, 759 vs 854 frames/s; at GOP=161, 35 vs 331.

So `g=2` is insurance against random access — roughly 3× the file size for 21× on
single-frame reads — excellent for a per-step policy and largely wasted on a clip
consumer. Demand is `batch × cameras × frames_per_sample`, and a clip consumer
should not inherit `g=2` blindly.

## Direction: an opt-in resolution shadow

The measurement collapses most of what an adaptive design would decide. Probing
`t_source` per feature at dataset open, a breakeven formula, a format ladder chosen
by page-cache residency, autotuning in a fixed order — all of it exists to decide
_whether_ and _how_ to cache without asking anyone. An operator who states the
resolution removes the question.

Opt-in also settles what this design could not. The cut position depends on a
policy's transform chain, which is declared nowhere shared — HVLA resizes inside
`FlowMatchingDataset`, other policies elsewhere — so inferring it needs a refactor
first. Being told does not.

**Shape.** A shadow of the dataset at reduced resolution, owned by it: living under
its parent, following it, rebuilt or dropped with it. Not a sibling dataset, which
is what a manual conversion produces and what someone then has to remember to
select. This is the lifecycle a game engine gives a level-of-detail asset — source
authoritative, derived levels disposable.

**A fixed ladder, not an arbitrary size.** 224 / 336 / 448 covers every encoder
input in practice and makes a shadow reusable across policies rather than tied to
one run. Storage favours this more than it does in games: a full mip chain costs
+33% over the source, where the 224 level measured **0.68 GB against 4.3 GB**, so
the whole ladder is a fraction of the original.

**Where the analogy fails is the point.** A mip level is a quality compromise
chosen per-sample by distance; a policy resizes to a fixed input every step
regardless. Storing that size is not an approximation of what the model sees, it
_is_ what it sees — 38.5 dB against the trainer-resized source, the difference being
one encode round trip, after which the trainer's own resize short-circuits to a
no-op (torchvision returns the identical object at matching size). There is no
quality trade-off to argue, only whether the resize is paid once or ~54 times.

**Populate asynchronously.** The decode and resize happen anyway on the first pass;
persisting the result costs only the write, and with a raw layout that is I/O
rather than the CPU which is already saturated. A pass is 373 steps at batch 128
and a 20k run is ~54 passes:

| approach               | first 20k run | later runs |
| ---------------------- | ------------: | ---------: |
| no shadow, 720p source |       252 min |    252 min |
| blocking pre-build     |       221 min |    153 min |
| **async shadow**       |  **~152 min** |   ~153 min |

A pre-build pays 70 minutes upfront and recovers nothing until pass 2; an async
shadow is warm for 53 of 54 passes and needs no wait before training starts.

## Three properties it must have

- **Invalidate on what produces frames, not on any change.** Most edits to these
  datasets are quality labels: parquet columns, videos untouched. A shadow keyed on
  "the dataset changed" is discarded by every labelling session for nothing. Key on
  video file identity, episode count and order, and source resolution; be
  indifferent to labels, tasks and statistics. Merges and episode deletion do
  invalidate.
- **Refuse on a resolution mismatch rather than falling back.** A 224 shadow used
  by a run configured for 336 is either wrong pixels or a silent performance cliff.
  The training view names the shadow it uses and treats a mismatch as an explicit
  choice.
- **Do not offer it where it cannot help.** At 224 the measurement is CPU 6%, GPU
  100%. A shadow on a dataset already at training resolution buys nothing, and the
  dataset view knows the source resolution.

## What exists, what is missing

`dataset_postprocess.resize_cameras` performs the transform: non-camera data
verbatim, per-episode statistics recomputed, codec unified. It has run end to end
on the 274-episode reference dataset with the quality-label columns verified
element-wise against the source.

Missing: lifecycle — the shadow following its parent rather than being a sibling —
and a place to opt in, from the dataset view or the training view.

## Prior art

Three points where published work changed a decision here; the rest is in Sources.

- **Latent caching is the industrial form of this.** Open-Sora Plan and CogVideoX
  train on precomputed VAE latents rather than re-encoding every epoch. The
  published account of that approach's drawbacks — _"substantial storage overhead,
  disables on-the-fly data augmentation, and limits the flexibility of frame
  sampling strategies"_ — restates this design's tensions, which means none can be
  engineered away, only traded. It also suggests the extension not taken: where the
  encoder is frozen, the cut moves past it and you cache features rather than
  pixels. Out of scope for HVLA, which finetunes DINOv2.
- **Web-scale infrastructure optimises the opposite problem.** WebDataset, MosaicML
  Streaming and Megatron-Energon abandon random access for sequential shard
  streaming because at PB scale it is not viable. This is the other regime — one
  machine, local NVMe, a sampler that wants random access — which is why a shadow
  rather than a stream.
- **Never evict within a run.** Training touches every item once per epoch, LRU's
  pathological case. MinIO's result is that a cache which fills and then stops
  replacing achieves a hit rate equal to its capacity fraction. A partial shadow
  should hold a fixed random subset, chosen once.

## Plan of work

1. **Lifecycle.** Shadow storage under the parent, the invalidation key above, and
   discovery: given a dataset and a resolution, is there a valid shadow.
2. **Opt-in surfaces.** Dataset view and training view, showing source resolution
   and what a shadow would cost.
3. **Async population**, so a run starts immediately and warms as it goes.
4. **Re-measure** with the device counter, not a step-time ratio.

Two cache-independent wins should be taken and measured _before_ any of this, or
the shadow will be credited with them: gradient checkpointing (10–13%) and the
batch-size curve above.

## Verification

- A converted dataset's quality-label columns match the source element-wise.
- Any video rung pins `-g 2`; a test asserts it, because the default is a
  regression that looks like an optimisation.
- A resolution mismatch is refused, not silently served.
- A labelling edit does not invalidate a shadow; a merge does.
- Before/after measured with `nvidia-smi`, not `data_s`/`updt_s`.

## Risks

**Calibrated on one rig and one dataset.** Every number here comes from a single
machine and a 34-episode dataset, with the 274-episode result as the only
end-to-end confirmation.

**Disk growth across datasets.** Per-dataset shadows accumulate. Never-evict is
correct within a run and wrong across them: this needs a visible total and a way to
clear it.

**A shadow that drifts from its parent is worse than none**, which is why
invalidation is a property above and a verification item, not an implementation
detail.

## Sources

- [Analyzing and Mitigating Data Stalls in DNN Training](https://vldb.org/pvldb/vol14/p771-mohan.pdf) — MinIO never-evict cache, DS-Analyzer differential stall attribution
- [Cachew: Machine Learning Input Data Processing as a Service](https://www.usenix.org/conference/atc22/presentation/graur) — autocaching cut-point policy, autoscaling from domain metrics
- [FFCV: Accelerating Training by Removing Data Bottlenecks](https://arxiv.org/pdf/2306.12517) and [the Bottleneck Doctor](https://docs.ffcv.io/bottleneck_doctor.html) — quasi-random ordering, storage-format ladder, bottleneck-to-remedy mapping
- [Plumber: Diagnosing and Removing Performance Bottlenecks in ML Pipelines](https://anakli.inf.ethz.ch/papers/plumber_mlsys22.pdf) — automatic tuning of parallelism, prefetch and caching
- [tf.data: A Machine Learning Data Processing Framework](https://arxiv.org/pdf/2101.12127) — AUTOTUNE
- [Faster Neural Network Training with Data Echoing](https://arxiv.org/abs/1907.05550) — reclaiming idle accelerator time when input-bound
- [NVIDIA DALI](https://developer.nvidia.com/dali) — GPU-side decode and preprocessing, NVDEC

Video-scale infrastructure, consulted for the prior-art section:

- [WebDataset](https://github.com/webdataset/webdataset) — tar-shard sequential streaming; the format the web-scale ecosystem settled on
- [MosaicML StreamingDataset](https://www.databricks.com/blog/mosaicml-streamingdataset) and [its shuffling design](https://docs.mosaicml.com/projects/streaming/en/latest/dataset_configuration/shuffling.html) — shuffle-within-shard to bound download demand, with random access retained
- [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) — NVIDIA's multimodal loader: TB-scale sharding, dataset blending, deterministic resume
- [Training Video Foundation Models with NVIDIA NeMo](https://arxiv.org/pdf/2503.12964) — end-to-end video FM pipeline, curation through training
- [Open-Sora Plan](https://arxiv.org/html/2412.00131v1) and [CogVideoX](https://www.emergentmind.com/papers/2408.06072) — latent-space training; precomputed VAE latents as the industrial form of caching a deterministic prefix
- [Survey of Video Diffusion Models](https://arxiv.org/pdf/2504.16081) — source of the latent-caching drawbacks quoted above: storage overhead, augmentation loss, sampling inflexibility
- [TorchCodec](https://pytorch.org/blog/torchcodec/) — the decoder this repository already uses; explicitly optimised for seek-heavy ML sampling and weaker on sequential reads
