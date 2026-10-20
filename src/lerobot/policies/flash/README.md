# FLASH implementation status

FLASH lives at **Run → FLASH**. The current Run panel inspects a regular local
LeRobot dataset episode and discovers spatial skill annotations and prepared
references under that dataset's `flash/v1` extension. A source episode can
have multiple skills. The panel deliberately does not command a robot yet: the
live tracker must register each object into the _same stable object frame_ used
by the demonstration before the spatial equation can produce safe goals.

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
- a spawn-isolated perception process that loads GPU models in the child,
  accepts the latest aligned camera snapshot and publishes timestamped poses
  to a short-lock store outside the motor loop;
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

The active absolute-pose estimator is SAM3 instance masks, DINO appearance
matches and a robust aligned RGB-D fit. An optional causal point tracker has
its own bounded handoff for **every** captured camera frame; the heavy pose
process may still skip superseded frames. After a depth-backed absolute pose
and aligned instance mask seed it, the causal tracker can propagate that
object's motion through a temporary appearance-matching dropout. It reports
the resulting pose as **predicted**, with the new RGB-D frame's timestamp. It
cannot initialize an object whose absolute pose was never established, and
it is not an environment group tracker or a recovery mechanism after its
point tracks are lost. The pose-gap wrapper that simply reused an unmoving
last pose has been removed. Causal tracking remains an explicitly configured
experimental live option pending a physical rollout; no successful insertion
has been established with it.
