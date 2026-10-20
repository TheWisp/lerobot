# FLASH implementation status

FLASH lives at **Run → FLASH**. The Run panel inspects a regular local LeRobot
dataset episode and discovers spatial skill annotations and prepared references
under that dataset's `flash/v1` extension. A source episode can have multiple
skills. With a prepared skill and calibrated local trial configuration, the
panel can launch a bounded live trial and record another ordinary LeRobot
dataset. Startup requires each needed object to register into the _same stable
object frame_ used by the demonstration before robot motion begins.

The executable core already separates:

- demonstrated contact intervals, sampled on their own 1× local clock;
- sample-bound semantic events that survive replanning, with an optional
  milestone gate before the associated gripper change;
- generated connections, timed from live path length and configured speed;
- explicit pose-relation milestone tolerances, observed true/false/unknown
  verdicts from the perception collector (including expiry when frames stop),
  and bounded recovery transitions;
- metric RGB-D and pose evidence with source and availability times;
- one aligned RGB-D frameset from LeRobot's RealSense camera with an atomic
  host capture-time estimate and capture-time camera calibration;
- held-object world pose from the gripper-to-object attachment;
- capture-time interpolation of measured end-effector poses before accepting
  delayed held-object evidence into that attachment; a held attachment updater
  can seed a pregrasp fallback, marks whether it has a postgrasp observation,
  and withdraws the attachment on a large observed disagreement;
- demonstrated gripper events can declare an attach or detach transition with
  an object role and frame version. A lifecycle adapter applies those only
  after the motor accepts the event, seeds a recent observed pregrasp pose as
  an assumption, and retires that assumption on release. Release success still
  requires independent milestone evidence;
- skill compilation from ordinary LeRobot state/action/timestamp columns plus
  versioned object-pose annotations;
- metric cross-episode object-frame registration from RGB-D correspondences,
  with a checked binding from a live tracker's frame to the demo frame;
- a matcher-neutral RGB-D pose backend that fits visible matched points from
  any number of annotated demo views and reports loss when depth or geometry
  cannot support a pose;
- an initial SAM3-mask/DINOv2-patch matcher that constructs a metric object
  point-and-appearance bank from annotated RGB-D demo views, then submits
  live correspondences to the common pose backend;
- a versioned prepared-skill extension that saves the spatial skill, stable
  multi-view object references and derived metric point/descriptors beside
  the regular dataset. Run → FLASH discovers either a raw skill annotation
  under `flash/v1/skills/*.json` or a prepared reference under
  `flash/v1/prepared/*/bundle.json`, and reports what an episode still lacks;
- a spawn-isolated perception process that loads GPU models in the child and
  accepts the latest aligned camera snapshot. With causal tracking enabled it
  submits timestamped source-frame proposals to the scene estimator; otherwise
  the legacy path publishes its pose results to a short-lock store outside the
  motor loop;
- a live perception service that owns camera capture and result collection,
  waits for actual seeded object observations before motion, and exposes
  camera/estimator failures instead of reusing stale poses; its latest aligned
  RGB-D frame is retained separately for a nonblocking motor-tick recorder;
- kinematics-neutral short IK chunks and atomic future-queue replacement;
- an asynchronous IK producer that supersedes pending replan requests while
  the motor owner keeps consuming the already accepted queue; a solve made
  stale by a newer request is discarded before publication. The producer
  can snapshot the committed prefix and seed IK at a future splice tick;
- a planning coordinator that fills that queue ahead of the motor, retains a
  generated connection's clock across ordinary refills, retargets from the
  committed joint prefix when a reference changes, and interrupts if a pose
  is lost or moves beyond explicit replanning limits;
- a synchronous attempt host that composes those services outside the GUI
  event loop: it waits for an observed first object pose in the demonstrated
  frame before resetting the motor epoch, passes pose changes and milestone
  verdicts to the session, runs bounded steps, and saves an explicit outcome.
  Finishing motion without an independent verifier remains `unknown`;
- a selected-joint adapter that runs the SO-107's seven-joint FK/IK model
  while the planner controls six arm joints and the gripper separately;
- a camera-independent motor read/dispatch tick through `Robot`;
- the separately configured SO-107 gravity-compensated robot controller used
  in the sealed rig measurements, with its measured gains and URDF inertias;
- a session state machine that consumes asynchronous evidence before each tick
  and gates semantic actions on current milestone verdicts;
- standard LeRobot episode frames, with a `flash/v1` trace joined by episode
  and frame index. Every executed motor tick is retained, including waits; the trace
  distinguishes a new command, the servo's held goal and an observed hold.
  The regular dataset stores replayable robot-API actions; the trace also
  records the accepted actuator target, which may include gravity compensation
  or a robot-side limit. The pose layer and candidate/prior disagreements can
  be read back by episode and frame index for replay and skill compilation.
  Missed motor deadlines are explicit in the trace; such an episode has
  `timing_regular=false` because LeRobot's ordinary dataset timestamp remains
  based on its nominal FPS and must not be treated as exact physical timing.
  Motor observations also retain their actual read time for capture-time
  gripper interpolation.
  A tick assembler joins the latest available RGB-D frame and pose evidence
  before the recorder worker performs disk/video encoding.

The annotation points to a source LeRobot dataset episode. Dense transform
samples are derived geometry; the original RGB-D, robot observations and
actions remain in the regular episode. Each interval identifies its object
role and reference-frame version. A held-to-target interval also declares the
held object's reference-frame version. Missing versions are shown in skill
inspection and rejected by the joint planner. A pose-relation milestone can
select an observed demo frame to define its target object relationship, while
the task tolerances and maximum evidence age are supplied explicitly.
Held-to-target intervals also name the held role; skill inspection reports a
missing role or a missing matching attach event before live integration. It
also counts observed and predicted demo poses and warns when a held interval
has no independently observed held pose, rather than presenting predicted
contact geometry as a measurement.

The first matcher is measured on a recorded insertion scene in
[the RGB-D replay note](docs/rgbd_replay.md). It follows a static target well,
but the single tray view loses the dowel after lift and turn. Neither this
measurement nor a fit to matched points establishes independent pose accuracy
under occlusion. Multi-view reference preparation, evidence/observability
gates and asynchronous publication to the controller remain necessary.
Other milestone types, such as grasp verification, still need independent
evidence. The SO-107 trial launcher accepts a prepared skill, calibrated rig
configuration and ordinary LeRobot source dataset. It records each rollout as
an ordinary LeRobot episode with a FLASH decision trace. Startup waits for the
first action's observed object and a usable pose for every later referenced
object before the motor epoch begins; a poor instance seed must fail before
the arm moves.

The active absolute-pose estimator uses SAM3 masks, DINO appearance matches
and aligned RGB-D. A geometry-only shape fit can propose a pose when appearance
matches are sparse. The 2026-10-20 live trial showed that a low surface residual
can still place the dowel's object frame centimetres away from its previous
pose; the recorded fixture and consistency rejection cover that failure, but
do not establish accurate pose under occlusion. The optional causal service
receives every captured camera frame, tracks object and shared scene points,
and can predict group motion after observed co-motion verifies a relationship.
When causal tracking is enabled, the heavy pose process submits
source-frame proposals to the causal service, which alone publishes accepted
poses and catches delayed proposals up to its retained frame history. That is
the first part of the [single scene-estimator contract](docs/architecture.md#one-scene-estimator-several-evidence-sources), not the complete arbitration:
partial observability, alternative poses, and calibrated uncertainty are still
missing. Consequently the group path is experimental, and no successful physical
insertion has been established for this redesign.

The next architecture milestone is the [scene-estimator contract](docs/architecture.md#one-scene-estimator-several-evidence-sources), independent of which tracker is best on a particular view. The current code has one publisher when causal tracking is enabled, but it still treats a backend's candidate `PoseEstimate` as an accepted six-axis observation. In particular, the RGB-D backend keeps its own previous candidate for shape-continuity checks, and the scene service has no general rule for comparing a new candidate with visible depth, point motion, a group prediction and unresolved symmetry at the same source time. The accepted pose trace does not yet retain each raw candidate and its rejection reason.

Implementation should proceed in this order:

1. Introduce typed source-frame evidence for identity, visible geometry, relative motion and relationship hypotheses. Allow a producer to abstain or constrain only an observable subspace; retain source/calibration/reference versions and raw support.
2. Move candidate acceptance and event-time replay into the scene-estimator owner. Keep a bounded event history, compare proposals to the state at their original frame, replay later measurements as well as motion, publish one versioned belief, and record each decision. The interim service conservatively discards a late proposal older than a newer direct observation; that guard is not a substitute for full replay. Backends must not maintain a private notion of what the scene accepted.
3. Make the policy check the accepted belief at its contact points against the active interval's requirements. Unknown or conflicting dimensions must lead to a declared observation/recovery choice, not a fabricated pose or a model-name-specific override.
4. Run the same recorded disagreement, occlusion, moved-support and held-slip cases against alternate backends before another physical insertion claim. Report both pose error and source-to-decision age; a better mask alone is insufficient.
