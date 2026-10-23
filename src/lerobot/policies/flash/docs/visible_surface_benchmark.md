# Visible surface tracking benchmark

Status: protocol in progress. The initial harness, seven point frontends, six labelled public development clips, three exact-motion controls, a local dowel comparison, and a minimal paced 30 Hz consumer probe are documented in the [first implementation and evidence](../../../../../docs/proofs/flash-visible-surface-benchmark/README.md). The remaining phases below are not yet claimed as complete.

FLASH needs points that remain attached to the same physical surface as an object moves. Dots staying inside a segmentation mask, a smooth gizmo, or a high inference FPS do not establish this. The comparison must identify point drift, distinguish it from depth and fitting errors, and measure how quickly usable updates reach the consumer.

**Proposal.** Compare pretrained point trackers on identical RGB-D recordings and queries, first without pose fitting, then with a shared rigid estimator. Evaluate complete pose systems separately. Score continuously visible surface intervals, with independent references where available. Compare offline quality and causal performance separately, including the overhead of the full perception pipeline.

## Scope and requirements

The first objective is visible-surface consistency on moving rigid objects. Partial object occlusion is allowed: points on the visible part still count. Following hidden points, recovering their identities after occlusion, cross-episode registration, robot task success, and network fine-tuning are separate questions. No robot motion or replacement of recorded demo poses is needed for this benchmark.

Correct correspondence and honest failure accounting take priority over smooth appearance. Speed is measured alongside accuracy, rather than used to discard slower offline candidates.

| ID  | Priority | Requirement                                       | Target and rationale                                                                                           | Verification                                                    |
| --- | -------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| R1  | P0       | Score the same physical visible points            | All independently labelled visible query/frame pairs remain in the denominator, including tracker failures     | Audit eligibility masks and deliberately missing predictions    |
| R2  | P0       | Separate point errors from downstream corrections | Preserve raw tracks, common-fit outputs and native-system outputs independently                                | Recompute scores from saved arrays                              |
| R3  | P0       | Comparable data and inputs                        | Identical source frames, timestamps, depth, queries and shared masks within each comparison                    | Hashed manifests and adapter checks                             |
| R4  | P0       | Measure both speed and delay                      | Throughput, publication cadence, result age and stage costs on the same named hardware                         | Instrumented replay with source and publication timestamps      |
| R5  | P0       | Preserve the asynchronous runtime contract        | A simulated consumer runs at the configured 30 Hz independently of perception; report missed 33.3 ms deadlines | Loaded-versus-idle consumer timing trace                        |
| R6  | P1       | Generalize beyond one recording                   | Local manipulation examples, two public sensor RGB-D sources and exact-motion controls                         | Results by source, object and motion class; held-out recordings |
| R7  | P1       | Make failures reviewable                          | Fixed IDs, visible reference points, raw and fitted views, and uncropped failure intervals                     | Video inspection against saved per-point results                |

R5 is a scheduling test with a simulated consumer, not proof of hardware servo timing. No fixed perception FPS is assumed: the benchmark determines the accuracy/latency trade-off.

## Evidence behind the protocol

**O1.** The [drift audit](../../../../../docs/proofs/flash-drift-features/EVIDENCE.md) found changing pairwise distances in surface-supported TrackEverything tracks. Some common motion could also be slip or calibration error. Therefore, rigidity and gripper agreement are diagnostic measures, not independent material-point ground truth.

**O2.** The [implementation audit](../../../../../docs/proofs/flash-moving-pose/EVIDENCE.md#implementation-isolation-and-deterministic-fit-correction) found fitting instability on unchanged input points. Therefore, raw point tracking and rigid estimation need separate scores.

**O3.** The [earlier tracker comparison](../../../../../docs/proofs/flash-tracker-comparison/EVIDENCE.md) found query-timestamp integration issues and different update cadences. Therefore, adapters must be checked before model quality is ranked, and amortized FPS must be separated from update latency.

These observations permit different implementations behind one measurement contract. They do not justify filtering away inconvenient predictions or assuming one existing model is the winner.

## Candidate matrix

All candidates receive a feasibility check and a result or an explicit blocked/unsupported entry. A failed installation is not an accuracy result. Models with different input or temporal capabilities remain labelled as such.

| Candidate                                                             | Comparison role                            | Required checks                                                                        |
| --------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------- |
| [TrackEverything](https://github.com/ayushjain1144/trackeverything)   | Existing dense 3D baseline                 | Measured depth input, query births, window lookahead and repeated computation          |
| [TAPIP3D](https://github.com/zbw001/TAPIP3D)                          | Direct RGB-D challenger                    | Sensor-depth path, camera/world coordinate conversion and window timing                |
| [Track-On-R](https://github.com/gorkaydemir/track_on)                 | Streaming 2D challenger, lifted with depth | New-query support, memory growth and frame-by-frame output                             |
| [BootsTAPIR and TAPNext++](https://github.com/google-deepmind/tapnet) | Existing and newer causal 2D frontends     | Query timestamp and resolution; run outside Point2Pose as well                         |
| [CoTracker3](https://github.com/facebookresearch/co-tracker)          | Windowed point baseline                    | Output cadence and any future-frame dependency                                         |
| [LiteTracker](https://github.com/ImFusionGmbH/lite-tracker)           | Fast causal baseline                       | Accuracy versus point budget, memory and iterations                                    |
| [SpatialTrackerV2](https://github.com/henry123-boy/SpaTrackerV2)      | Additional RGB-D challenger                | Supplied depth/camera input and supported temporal mode                                |
| Existing classical optical-flow tracker + measured depth              | Low-cost control                           | Same queries; retain lost points in scoring                                            |
| [Point2Pose](https://github.com/tzuyuan/point-to-pose)                | Complete causal pose system                | Report raw frontend tracks and map-propagated points separately                        |
| [BundleSDF](https://github.com/NVlabs/BundleSDF)                      | Complete geometric system                  | Visible-only initialization; report online and offline refinement separately           |
| [Point4D](https://github.com/point-4d/Point4D)                        | Additional offline comparison              | Verify sensor-depth support; if unavailable, label as RGB-only with predicted geometry |

Point4D's projected 2D tracks can still be compared, but predicted-scale 3D output must not be silently mixed into a metric sensor-depth ranking. Supplied CAD geometry is reserved for evaluation, never provided to a model-free candidate. Learned geometry accumulated from input frames is allowed and its cost is recorded.

## Recordings and reference measurements

### Our examples

Target 12 object clips spanning rotation, translation, changing camera view, small image footprint and weak texture. Include different source episodes where available. Freeze selection before inspecting new model outputs.

| Required case                                 | Initial source                                   | What it tests                                                |
| --------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| Held dowel rotating                           | Insertion demo, frames 280–370                   | Low texture and substantial rotation                         |
| Visible dowel during insertion                | Insertion demo, approximately 380–500            | Small surface and fine motion; only co-visible points scored |
| Moving yellow Lego                            | Rollout `20261016_112149`, staged frames 0–74    | Flat appearance, independent object motion                   |
| Moving held gamepad                           | Rollout `20261012_194250`, staged frames 90–145  | More appearance structure and changing surfaces              |
| Ring moving with tray or after contact        | Insertion rollouts                               | Target motion, curved shape and camera-relative motion       |
| Further moving objects and alternate episodes | Inventory existing stacking/insertion recordings | Prevent tuning to the above intervals                        |

The [saved feature-audit configuration](../../../../../docs/proofs/flash-drift-features/config.json) identifies the initial sources. Staged frame numbers must be mapped back to original frames and capture times. Preserve source data referenced through `/tmp` in a durable benchmark cache before running. Do not present the nearly stationary late gamepad interval as evidence of tracking a moving object; retain stationary intervals only as noise controls.

Annotate identifiable surface features independently of model outputs, including visibility and annotation uncertainty. Use a second annotation pass to estimate labelling error. Smooth unmarked regions may lack exact real-world point ground truth; report their rigidity and depth consistency separately, without calling those absolute accuracy. Gripper motion is an auxiliary comparison, since grasp slip remains possible.

### Public RGB-D examples

| Source                                                                               | Planned selection                                                  | Reference and role                                                                                                                                                            |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [DexYCB](https://dex-ycb.github.io/)                                                 | 12 moving-object clips spanning shape, texture and hand motion     | Sensor RGB-D; use the [TAPIP3D data/annotation protocol](https://github.com/zbw001/TAPIP3D/blob/main/DATASET.md) or explicitly documented object-pose-derived material tracks |
| [YCBInEOAT](https://github.com/wenbowen123/BundleTrack#data)                         | 12 visible intervals across available robot-manipulation sequences | Raw depth, RGB, intrinsics and annotated object poses; evaluate material points transported by reference poses                                                                |
| [PointOdyssey](https://github.com/y-zheng18/point_odyssey) and exact-motion controls | Small rigid-object subset plus controlled flat/textured motion     | Known trajectories and visibility; separate synthetic results from sensor results                                                                                             |

Object meshes and annotated poses may construct evaluation tracks and visibility. They stay outside tracker inputs. Inspect reference alignment and quantify annotation/depth disagreement before treating millimetre differences as meaningful. Use YCBInEOAT's raw depth first; its filled depth is a separate ablation.

An optional camera-motion extension can use [Aria Digital Twin](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset). Its synthesized depth must be labelled separately. TAPVid-3D annotations alone do not supply a sensor-depth video, so that benchmark is not automatically a replacement for the two sensor datasets.

Use roughly one third of source recordings for adapter development/configuration and the remainder as a frozen holdout. Keep every view and interval from a source recording in the same split. Record known model-training overlap; a local holdout does not establish that a pretrained model never saw a public dataset. Exact source IDs and exclusions belong in the manifest before model comparison.

## Test protocol

### 1. Validate adapters and reference data

Check RGB/depth alignment, metric units, intrinsics after resizing/padding, pixel conventions, camera transforms, query birth times and returned frame indices. Use stationary, known translation and known rotation controls with exact reference trajectories. Include a point introduced after initialization to catch the existing class of query-timestamp bug. Run each official example once as a compatibility check.

Store model commit, checkpoint hash, precision, resolution, window length, lookahead, query count, support-query count, seed and all depth/mask settings. Reuse the existing benchmark utilities where sound; keep model dependencies in separate environments behind one result format. These checks implement [R2–R3](#scope-and-requirements) and address [O2–O3](#evidence-behind-the-protocol).

### 2. Compare a fixed set of surface points

Seed the same spatially distributed physical points at the same frame. Track each until its independently labelled first occlusion, image exit or clip end. Start later visible intervals as separately initialized trials; do not require identity recovery across the hidden interval. Visibility is per point, so a partially occluded object remains useful.

Never remove a visible query from scoring because the tracker calls it invisible, it leaves the SAM mask, its depth is invalid, or the rigid fitter rejects it. Those are failed or unsupported predictions. Use independent labels to determine eligibility. Where reference depth is unavailable, retain 2D scoring and report the unavailable 3D reference explicitly.

Save raw 2D output, native 3D output where available, and a common depth-lifted 3D version. This distinguishes image correspondence error from geometry error. For causal evaluation, publish only outputs computed from available past/current frames. Evaluate windowed/offline methods separately and expose how much future context each result uses. This enforces [R1](#scope-and-requirements).

### 3. Compare growing point sets

Add identical query cohorts at predetermined frames where new surface is visible, using the shared mask and measured depth. Preserve old point identities and score each cohort by its age. New points cannot replace the denominator of an older failed cohort.

Sweep configurable active budgets of 64, 256 and 1,024 points. These are experiment sizes to expose scaling, not runtime safety thresholds. Use the same eligible locations and retirement policy; report insufficient surface area or unsupported budgets. Record extra support points used internally.

Measure query insertion, retirement and memory growth. If a backend needs a new window or replay to accept queries, count all that work and flag the limitation. Full-prefix reruns must not masquerade as a single streaming pass. Native adaptive seeding is a separate system experiment, because its point sets differ across models.

### 4. Isolate the rigid estimator

Run three separate result tracks: raw point output; the same deterministic rigid estimator applied to each point model; and complete native Point2Pose/BundleSDF outputs. Use identical reference points and parameters for the common estimator. Record all rejection reasons and retain failed frames in coverage metrics.

Changing pair distances is informative only on raw tracks: a fitted rigid cloud is rigid by construction. Also measure common translation/rotation error against independent reference motion, which catches a whole cloud drifting together. Where independent truth is unavailable, label that quantity unknown. Use this separation to attribute improvement to tracking, depth treatment or fitting rather than to a smoother drawing.

### 5. Measure quality

Report per-object and per-source distributions before aggregate scores:

- 2D material-point error in pixels and 3D error in millimetres, including median, p90 and worst intervals.
- Correct-visible coverage: the fraction of all reference-visible query/frame pairs with a valid prediction within a stated error. Missing predictions count as failures. Plot coverage/error curves rather than choosing a favourable confidence cutoff.
- Drift versus track age and motion magnitude; show translation and rotation intervals separately.
- Raw pair-distance change and common-motion discrepancy, with their different meanings retained.
- Frame-to-frame error changes relative to reference motion, so real motion is not mistaken for jitter.
- For growth, surface coverage, old-cohort survival, new-cohort accuracy and memory use over time.

Use the same configurable error grid for all methods, initially 1/2/5/10/20 mm and 1/2/4/8 pixels. These are reporting scales, not claims that the sensor can resolve 1 mm. Show annotation/noise floors beside the curves. Aggregate equally by clip/object and estimate uncertainty by resampling recordings, not millions of correlated points.

## Frame rate and overhead comparison

Measure on the same recorded hardware/software configuration, initially the local rig. Record GPU, driver, clocks/power mode, CPU, precision and competing processes. Preserve native capture times; never pretend repeated frames are new observations.

| Measurement                                 | What it answers                                                                                          |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Tracker-only wall time and GPU time         | Model cost with inputs prepared and resident                                                             |
| Full perception wall time                   | Decode/preprocess + transfers + segmentation + tracking + depth/point management + fitting + publication |
| Distinct source frames processed per second | Actual throughput; exclude repeated computation from the numerator                                       |
| Unique fresh publications per second        | How often the robot can receive a new estimate; repeated holds do not count                              |
| Result age at publication                   | Publication time minus the source frame's scheduled capture time                                         |
| Effective age at each consumer tick         | Includes time spent holding the last estimate between updates                                            |
| Update-gap distribution                     | Random waits hidden by average FPS; report p50/p95/p99 and longest gap                                   |
| Query-growth overhead and memory            | Cost of adding/removing points, state maintenance and longer histories                                   |
| Cold start and first usable result          | Model load, compilation, initialization and first publication                                            |
| Consumer deadline misses                    | CPU/GPU contention affecting the independent 30 Hz loop                                                  |

Run three performance modes:

1. **Offline:** all frames processed, native temporal mode. Report total runtime and throughput, with lookahead/window information. This decides demo-processing cost.
2. **Causal replay:** release frames at recorded times to an asynchronous producer and a simulated 30 Hz consumer. Measure queue growth, frame drops, output age and accuracy on the actual processed sequence. Use a declared bounded latest-frame policy where supported; flag models requiring another policy. Report missed source frames explicitly.
3. **Integrated load:** run shared SAM3 segmentation, depth handling, fitting, recording and optional overlay alongside the same consumer. Compare cached-mask, live-segmentation, and overlay-on/off costs. An asynchronous overlay can still compete for resources.

Replay accuracy is scored both at observation time and at consumption time against available reference motion. This exposes a tracker whose output is accurate but arrives too late. Do not conceal age with extrapolation in the baseline.

For point models, compare one, two and three objects at the same total point budget, then at equal per-object budgets. Record shared-scene inference versus repeated per-object calls. Test model-recommended resolution first; use supported higher resolutions and point budgets to establish an accuracy/cost curve rather than forcing an unsuitable input shape.

Use separate profiling and clean timing runs. Collect synchronized wall timings and CUDA events without serializing every asynchronous stage in the throughput run. Report overlapping work as an execution trace, not as an additive sum. Repeat warmed runs at least three times and retain initialization measurements separately; extend repetitions if timing is unstable. Explain any thermal/background variation.

Deliver a stage breakdown, CPU flame graph, CPU/GPU timeline and quality-versus-latency plot. Stage attribution includes model scheduling/waits and otherwise unexplained wall time. This satisfies [R4–R5](#scope-and-requirements).

## Execution order and review outputs

| Phase | Work                                                                           | Reviewable output                                                                     |
| ----- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| A     | Freeze sources/visibility references and validate adapters                     | Manifest, reference overlays, exact-motion checks and compatibility table             |
| B     | Run every candidate on the same small local/public pilot                       | Raw point comparison, failures, initial timings and measured full-suite cost estimate |
| C     | Run all compatible candidates with frozen settings on the full selected corpus | Held-out quality/coverage tables and representative plus worst-case clips             |
| D     | Run fixed/growing budgets and causal/integrated timing                         | Accuracy/latency curves, memory-growth plots, flame graph and timeline                |
| E     | Test focused improvements on development data, then held-out data              | Paired evidence for depth filtering, seeding, fitting or bounded growth changes       |

The pilot establishes compute/storage cost before broad parameter sweeps. A slow candidate still receives offline quality evaluation; a candidate with no causal mode receives no invented live FPS. Additional tuning concentrates on promising candidates, with the untuned results preserved.

For each clip, save a synchronised raw-video comparison with persistent point IDs and short trails. Show the same A/B/C anchors and an axis gizmo in a separate fitted-motion view, with resets visibly marked. Use one colour for tracked points, a distinct reference colour when truth exists, large persistent model names, and source-time playback. Include full intervals and worst cases; export phone-playable H.264 MP4. Detailed numbers belong beside the video, not in small moving text.

The final table must identify the best offline option and the best causal accuracy/latency trade-off, including cases where neither is adequate. Prefer a method only when improvement persists on held-out moving objects and exceeds measurement/repeat variability. Do not average away failure on the low-texture rotation cases. Choosing a deployable backend remains a later decision informed by this report.

## Alternatives and costs

Reusing current videos alone is inexpensive but cannot quantify coherent drift. Public benchmarks alone provide stronger references but can miss our small, weakly textured objects. Smoothness, mask agreement and rigid consistency alone can all reward a wrong trajectory. The combined protocol costs annotation and adapter work to separate those failure modes.

A new neural tracker or task-specific fine-tuning would make attribution harder before the baselines are measured. First test small shared improvements, each as an explicit ablation. Depth filtering, corner-biased seeding and rigid corrections must earn their place through held-out accuracy and overhead, not appearance alone.
