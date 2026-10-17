# State checks for a demo-replay policy: what the field does

Research notes, 2026-10-17. Written to settle how a Milestone's state check (see [act_execution.md](act_execution.md))
can be correct and fast, before any of it is built. The question the user put: _for shirt folding, what tells us the
gripper pinched a corner rather than missing it, or the cloth next to it? What do we even check?_

## The answer in one paragraph

Nobody checks "the corner is pinched" with a fixed camera. Every cloth system that answers that question answers it
with touch: a tactile fingertip classifies what is between the fingers (edge, corner, fold, all fabric, nothing: 92%;
one layer vs two: 100% / 87%), and a fixed camera is used only for the coarse facts it can see — the cloth came up
with the gripper, the cloth now covers the template, the task is at stage 3 of 4. Grasp outcome from a fixed RGB
camera before lifting is barely better than chance (68.8% on 9,000 rigid-object grasps; chance 61.8%). After
lifting, a local geometric comparison next to the fingertips (what was under them before closing vs after the lift)
reaches 92.7% recall of failed cloth grasps at 4.5% false alarms, trained on successful demonstrations only. VLM
judges of success or failure are 70–96% accurate and take 10–20 s per query; they are a slow, global sanity layer,
not a gate. Runtime monitors built for generative policies (Sentinel, FIPER) watch the policy's own actions, not the
world; they exist because VLAs do not solve this problem either, they absorb it into training data (π0) and still
recover from only 3–31% of failures on a benchmark. The checks that work are per-subgoal predicates over an
estimate of local state, each taking the sensor that can answer it, with regression rather than stopping when a
later check contradicts an earlier one (ReKep, FoldBack, the ReSkin loop all do this).

## What each family of check reads, and how well

| Family                                | What it reads                                                                     | Mechanism in the literature                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Measured                                                                                                                                                                                                    | Latency                                                                                      | Fails when                                                                                                                                                                                                                                         |
| ------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Proprioception                        | Gripper width, servo load, lift-and-hold                                          | QT-Opt labels a grasp by image subtraction after dropping the object into the bin [[Q]](#sources); Calandra lifts, holds 2 s, labels by a contact classifier on the tactile image [[C]](#sources)                                                                                                                                                                                                                                                                                                                                                                                                                             | On the rig: width separates a hold from a miss by 1.2–1.6 units against 0.4 noise; load 128–240 holding vs ≤ 34 empty ([act_execution.md E9–E11](act_execution.md#e9))                                      | 0.1–0.25 s                                                                                   | Says _something_ is held, never _what_; thin objects and light squeezes narrow the margin; the servo's load is effort while it moves                                                                                                               |
| Touch                                 | A tactile image or skin reading between the fingers                               | GelSight edge/corner/fold/all-fabric/none classifier, 92% (98% for edge vs not) [[S]](#sources); ReSkin layer count 0/1/2/3 at 100 / 99.9 / 86.6 / 47.8% from a single reading, mode over ~160 readings per attempt [[T]](#sources); outcome prediction before lifting: vision 68.8%, touch 75.6%, both 77.8% [[C]](#sources); slip during transport from GelSight Mini, 99% on 10 objects [[G]](#sources)                                                                                                                                                                                                                    | Cloth: 1-layer pinch on an unseen towel 8/10 with touch vs 3/10 with an image classifier [[T]](#sources)                                                                                                    | One reading (ms); a decision over 0.2–0.5 s of readings                                      | Folds read as edges (the main false positive in [[S]](#sources)); corners were not classifiable well enough to end a slide, so shear on the other gripper was used instead; a classifier is specific to the training cloth's thickness and texture |
| Local geometry around the fingers     | A point cloud or depth patch under the fingertips before closing vs after lifting | FoldBack: two local clouds in the gripper frame, PointNet, trained on successful demos with negatives made by substituting a nearby pre-pick cloud; 92.7% failed-grasp recall, 4.5% arm-wise false positives, 93–100% on slip trials [[F]](#sources); a low-cost camera after lifting: 97% per frame, 100% per run [[L]](#sources)                                                                                                                                                                                                                                                                                            | On the rig the same idea from the fixed camera, the resting place reading as table: clean, but 4.5 s, since the arm covers the spot until the carry ([act_execution.md E12](act_execution.md#e12))          | One frame after the lift when the sensor rides with the gripper; seconds from a fixed camera | The gripper hides the patch; a fixed camera must wait for the arm to clear                                                                                                                                                                         |
| Keypoints and relations               | Tracked 3D keypoints, segmented parts, FK of the end-effector                     | ReKep: sub-goal constraint cost ≤ 0 within tolerance advances the stage; a path-constraint violation backtracks one stage at a time until one holds; keypoints tracked at 20 Hz, solve at 10 Hz; occlusion is the first failure cause [[R]](#sources). Code-as-Monitor: constraint elements (points, lines, surfaces) from masks and depth, tracked with CoTracker, a boolean check written per sub-goal by a VLM once; detects displacement and rotation failures, not force-driven ones [[M]](#sources)                                                                                                                     | On the rig: the object's own tracks move with the fingertip 0.84–1.0× when held, 0.01–0.03× at rest, no verdict when the gripper covers them (20 of 49 acts) ([act_execution.md E15](act_execution.md#e15)) | 50–100 ms per frame                                                                          | Occlusion by the arm; a thin object's pose from tracks alone is degenerate ([E17](act_execution.md#e17)); a check that needs force has no signal                                                                                                   |
| Learned stage and progress from video | A few frames over the last seconds, the task text, optionally joints              | SARM: 9 frames over 8 s from the top camera plus joints, a 60M-parameter transformer, outputs the stage (of 4, for T-shirt folding) and progress within it; 12/12, 11/12, 12/12 on successful, partial and failed rollouts; "rewind" augmentation is what lets it see a regression [[A]](#sources). GVL: a VLM orders shuffled frames by completion; success detection 0.71–0.75 against SuccessVQA 0.62 [[V]](#sources). SuccessVQA: Flamingo 3B fine-tuned on 101,789 labelled episodes; 94–96% on insertion in distribution, 82–87% on removal; robust to viewpoint changes where bespoke models collapse [[Q2]](#sources) | SARM's model and its annotation tool are in this repository (`lerobot/rewards/sarm`, `data_processing/sarm_annotations`)                                                                                    | SARM: a forward pass; VLMs: seconds (GPT-4o 14 s mean in [[N]](#sources))                    | Needs annotated demonstrations of the task; episode-level or several-second judgements; cannot say _which_ local fact failed                                                                                                                       |
| Failure explanation by LLM            | Scene graphs from detections, sound events, subgoal summaries                     | REFLECT: detection per subgoal from the summary, then explanation from the event timeline; 93.8% real-world localisation of execution failures [[E]](#sources); AHA: a VLM fine-tuned on procedurally perturbed demonstrations, +10.3% over GPT-4o [[H]](#sources)                                                                                                                                                                                                                                                                                                                                                            | —                                                                                                                                                                                                           | Seconds to tens of seconds                                                                   | Depends on the detector's object states; after the fact                                                                                                                                                                                            |
| Policy-side runtime monitors          | The policy's own action distributions, OOD scores, entropy                        | Sentinel: statistical temporal action consistency (STAC) plus a VLM progress judge; 97% of unknown failures, STAC 5–9 s, VLM 13–21 s [[N]](#sources); FIPER: OOD + action-chunk entropy with thresholds calibrated by conformal prediction on successful rollouts only [[P]](#sources)                                                                                                                                                                                                                                                                                                                                        | —                                                                                                                                                                                                           | 5–20 s                                                                                       | Says the policy is confused, not what the world is; calibration must match deployment                                                                                                                                                              |
| Symbolic predicates for planning      | Classifiers over images or perception APIs                                        | STRIPS-WM learns propositional models from image transitions; VisualPredicator and InterPreT invent predicates (as Python over perception APIs) for preconditions and effects [[W]](#sources)                                                                                                                                                                                                                                                                                                                                                                                                                                 | —                                                                                                                                                                                                           | A classifier call                                                                            | The predicates are only as good as the perception underneath; most work is in simulation                                                                                                                                                           |
| Industry pick verification            | Weight, suction pressure, in-hand or arm cameras, barcode                         | Weight against the expected item; on-arm cameras that see the grasp; a scale under the pick point; disagreement sends the item to an exception station; double picks caught by vision or weight [[I]](#sources); Vulcan picks with suction and force sensing [[U]](#sources)                                                                                                                                                                                                                                                                                                                                                  | —                                                                                                                                                                                                           | Weight and pressure within a second                                                          | Needs the item's expected weight; cameras need the gripper to pass in front of them                                                                                                                                                                |

## What the shirt corner needs

The corner question has three parts, and they are answered by different sensors:

1. **Is cloth held at all?** Width and load say so at once ([E9–E10](act_execution.md#e9)); the local geometry under the
   fingers confirms it at the lift (FoldBack, our resting-place test). A fixed camera also sees the towel come up
   with the gripper, which [[S]](#sources) found sufficient for the first pick: "picking the lowest point and using
   vision to confirm that the towel is being held".
2. **Is it the corner, an edge, a fold, or the middle?** Only touch answers this in the literature, at 92%, and
   corners were the hardest class: Sunil et al. could not classify a corner well enough to use it as a stop
   condition and read shear on the other gripper instead [[S]](#sources). How many layers are pinched is likewise a
   touch question (ReSkin, [[T]](#sources)). The rig's SO-107 gripper has no tactile sensing; a ReSkin-class skin,
   a GelSight Mini or an infrared fingertip (UnfoldIR, used by the ICRA cloth-competition winner [[K]](#sources)) is
   a hardware decision before any corner Milestone can be checked directly.
3. **Did the fold happen?** Global and slow: coverage or IoU against the folded template from the top camera, a
   wrinkle penalty, a graded neatness score [[B]](#sources); or a learned stage and progress estimate (SARM, which
   was built on exactly this task).

Without touch, the corner Milestone can only be checked by its consequences: the cloth held (fast), and the fold's
outcome at the next global check (slow). That is the same shape as the dowel's _attached_ check today: pass on the
fast signal, fail later on the consequence, regress.

## What this changes in the Milestone design

- **A check is a predicate over a local state estimate, and each kind names the sensor that answers it.** Width,
  load, touch, geometry under the fingers, tracked keypoints, a stage model: the check layer must take any of them.
  The ReKep form is the right contract, a function of the state estimate that is satisfied at or below zero, with
  a tolerance.
- **Regress, do not stop, has precedent.** ReKep backtracks a stage when a path constraint breaks; FoldBack reopens
  the failed gripper, reverses the recorded motion to the pre-grasp pose and resamples away from failed locations;
  the ReSkin loop releases and moves ±2 mm. All three are the Progress update of [act_execution.md](act_execution.md).
- **Train checks on successful records with synthetic negatives.** FoldBack's negatives substitute a nearby pre-pick
  cloud for the post-lift one; FIPER and Sentinel calibrate on successful rollouts only. The rig has 264 act records,
  nearly all holds; that is the data a learned local check would train on, with misses made by substitution.
- **A sensor on the gripper cuts the attachment check from seconds to a frame.** The fixed camera's 4.5 s is the
  arm covering the resting place; FoldBack's local cloud rides with the gripper. A wrist camera, or using the depth
  camera's view of the fingertips rather than of the table, is the cheapest path to a sub-second geometric check.
- **The learned stage model is the global layer.** SARM in this repository, trained on the demos with the existing
  subtask annotation tool, gives a stage and a progress value every few seconds; its rewind training is what lets
  it say the shirt was crumpled again. It cannot gate a grasp; it can contradict a Milestone late and feed the
  Progress update's "which stage are we at".
- **VLM judges are not in the loop.** 10–20 s per query and 70–96% accuracy put them where REFLECT puts them: after
  the fact, explaining, or as the slowest contradiction.

## Open decisions

1. **Touch on the gripper.** Without it, cloth Milestones about _what_ is pinched are unverifiable; with it, the
   literature's numbers are 87–100% on layers and 92% on edge/corner/fold. Which sensor is a rig decision.
2. **A camera that rides with the gripper**, for the local geometric check, versus living with the fixed camera's
   seconds and regressing late.
3. **Whether the SARM stage model becomes the global layer now** (its data is the demos already recorded, with
   annotations to add) or after the per-Milestone checks exist.

## Sources

- [C] Calandra et al., _The Feeling of Success: Does Touch Sensing Help Predict Grasp Outcomes?_, CoRL 2017.
  https://arxiv.org/abs/1710.05512 — 9,000+ grasps; vision 68.8 ± 1.0%, touch 75.6 ± 0.8%, both 77.8 ± 0.3%, chance
  61.8%; label: lift, hold 2 s; runtime use as a rejection step (regrasp until p > 0.9).
- [S] Sunil et al., _Visuotactile Affordances for Cloth Manipulation with Local Control_, CoRL 2022.
  https://arxiv.org/abs/2212.05108 — GelSight; edge/corner/fold/all-fabric/no-fabric 92%, edge vs not 98%; edge
  grasps 90%; folds as false positives; corner detection insufficient, shear used instead.
- [T] Tirumala et al., _Learning to Singulate Layers of Cloth using Tactile Feedback_, IROS 2022.
  https://arxiv.org/abs/2207.11196 — ReSkin; 0/1/2/3 layers at 1.00/0.999/0.866/0.478; ~160 readings per attempt;
  ±2 mm regrasp loop; 180 trials; unseen cloth 8/10 touch vs 3/10 image.
- [F] _FoldBack: Self-Correcting Masked Generative Policy for Long-Horizon Garment Folding_, 2026.
  https://arxiv.org/abs/2610.10462 — local pre-pick vs post-lift clouds, PointNet, success-only training with
  substituted negatives; 92.7% recall, 4.5% false-positive rate; verified rollback and repair.
- [L] _Performance Evaluation of Low-Cost Machine Vision Cameras for Image-Based Grasp Verification_, 2020.
  https://arxiv.org/abs/2003.10167 — 97% per frame, 100% per run after lifting.
- [Q] Kalashnikov et al., _QT-Opt_, CoRL 2018. https://arxiv.org/abs/1806.10293 — success by image subtraction
  after dropping; 3 of 28 "misgrasps" were mislabelled successes.
- [R] Huang et al., _ReKep_, CoRL 2024. https://arxiv.org/abs/2409.01652 — sub-goal and path constraints over
  keypoints; advance on tolerance, backtrack on violation; 20 Hz tracking; occlusion the main failure.
- [M] Zhou et al., _Code-as-Monitor_, CVPR 2025. https://arxiv.org/abs/2412.04455 — constraint elements, CoTracker,
  VLM-written boolean checks; displacement and rotation failures only.
- [A] _SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation_, 2025.
  https://arxiv.org/abs/2509.25358 — 9 frames over 8 s plus joints, 60M parameters, 4-stage T-shirt folding,
  12/12, 11/12, 12/12 rollout classification, rewind augmentation. In this repository: `src/lerobot/rewards/sarm`.
- [V] Ma et al., _Generative Value Learning_, 2024. https://generative-value-learning.github.io/ — success detection
  0.71 zero-shot, 0.75 one-shot; SuccessVQA 0.62.
- [Q2] Du et al., _Vision-Language Models as Success Detectors_, 2023. https://arxiv.org/abs/2303.07280 —
  Flamingo 3B, 101,789 episodes; 94–96% insertion, 82–87% removal; viewpoint-robust.
- [E] Liu et al., _REFLECT_, CoRL 2023. https://arxiv.org/abs/2306.15724 — RGB-D, audio, state; 93.8% real-world
  failure localisation.
- [H] Duan et al., _AHA_, ICLR 2025.
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/70a06501001e1820fd1eb9ee821302d2-Abstract-Conference.html
- [N] Agia et al., _Unpacking Failure Modes of Generative Policies_ (Sentinel), 2024. https://arxiv.org/abs/2410.04640
  — STAC 5–9 s, VLM 13–21 s, GPT-4o 14.0 s mean; 97% of unknown failures.
- [P] Römer et al., _Failure Prediction at Runtime for Generative Robot Policies_ (FIPER), NeurIPS 2025.
  https://arxiv.org/abs/2510.09459 — conformal thresholds from successful rollouts only.
- [W] _STRIPS-WM_ https://arxiv.org/abs/2606.06832; _VisualPredicator_ https://arxiv.org/abs/2410.23156;
  _InterPreT_ https://arxiv.org/abs/2405.19758; Wang et al., _Learning compositional models of robot skills_
  https://arxiv.org/abs/2006.06444.
- [B] Fold metrics: UniFolding https://arxiv.org/abs/2311.01267 (coverage; a grasp-failure detector with
  re-grasp, unspecified); SSFold https://arxiv.org/abs/2411.02608; RoboFolDeX https://arxiv.org/abs/2609.10243
  (six-level neatness, FoldScore).
- [K] De Gusseme et al., _Insights for robotic cloth manipulation: a comprehensive analysis of a competition-winning
  system_, IJARS 2025. https://journals.sagepub.com/doi/10.1177/17298806251322582 — infrared fingertip (UnfoldIR).
- [G] Dong, Yuan, Adelson, _Improved GelSight Tactile Sensor for Measuring Geometry and Slip_, IROS 2017.
  https://arxiv.org/abs/1708.00922; GelSight Mini real-time slip detection, 99% on ten objects (via search).
- [I] Pick verification practice: https://blog.zivid.com/eliminating-double-picks-in-robotic-parcel-handling-cells;
  https://physicalailab.io/en/warehouse-picking-robot-guide/; US patent 11839983 (grasp verification by on-arm
  cameras); US patent 10099391 (weight against the expected item).
- [U] Amazon Vulcan (press): https://www.therobotreport.com/amazon-vulcan-robot-uses-force-sensing-to-stow-items/
- VLA recovery: π0 https://arxiv.org/abs/2410.24164 (recovery from the data mix); π0.5 https://www.pi.website/download/pi05.pdf
  (high-level sub-task inference "easily distracted"); SO-101 benchmark https://arxiv.org/abs/2606.08881
  (recovery rates π0.5 30.8%, Wall-X 20.5%, ACT 6.5%, SmolVLA 3.2%); RePO-VLA https://arxiv.org/abs/2605.09410.
