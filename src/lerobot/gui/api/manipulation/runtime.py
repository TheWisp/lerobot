#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared manipulation session state, limits and dedicated executors."""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import os
import pathlib
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

from lerobot.p2p_runtime import checkout_path
from lerobot.showservo import motion as core, pose_verification

if TYPE_CHECKING:
    from . import recording


_REPO = pathlib.Path(__file__).resolve().parents[5]


_WORKER = _REPO / "benchmarks" / "manipulation_worker.py"


@dataclass
class _Teach:
    at: str
    box: tuple[int, int, int, int]
    rgb: np.ndarray
    depth_m: np.ndarray
    intr: dict[str, float]
    keypoints: dict[str, Any]
    tip_pose: np.ndarray | None = (
        None  # the demo's first fingertip pose, base frame, 4x4 (set when a demo is loaded)
    )
    gripper: float | None = None  # its gripper opening, the follower's 0..100 units
    stamp: float = 0.0  # when its frame was read (wall clock), as the camera's frames are stamped; 0 unknown


@dataclass
class _Test:
    at: str
    rgb: np.ndarray
    result: dict[str, Any]
    transported: np.ndarray | None = None  # base frame, 4x4


@dataclass
class _Job:
    """One frame handed to the worker, and what came back."""

    id: str
    kind: str  # "teach" | "find"
    concept: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intr: dict[str, float]
    created: float
    camera_stamp_mono: float | None = None  # host publication time of the source RGB-D frame
    result_received_at: float | None = None  # request body arrived from the worker
    result_decoded_at: float | None = None  # NPZ and metadata decoded by the GUI
    verified_at: float | None = None  # geometry checks finished
    rendered_at: float | None = None  # live overlay finished
    worker_timing_ms: dict[str, float] = field(default_factory=dict)
    taken: bool = False
    result: dict[str, Any] | None = None
    camera_check: dict[str, Any] | None = None  # marker drift on this frame; None without a calibration
    algo: str | None = None  # a track job's algorithm
    compress: bool = True  # the frame's NPZ: compressed for one-off jobs, raw for the tracking stream
    click: list[int] | None = None  # a teach by a clicked pixel instead of by name
    extra: dict[str, Any] = field(
        default_factory=dict
    )  # kind-specific fields handed to the worker as they are
    arrays: dict[str, np.ndarray] = field(default_factory=dict)  # arrays sent with the job's frame
    progress: float = 0.0  # what the worker reported of a long job, 0..1
    log: dict[str, Any] | None = (
        None  # a find's why and where its click came from, for the find log; kept here
    )


@dataclass
class _Worker:
    """The SAM3 + DINO process: spawned here, fed by long-polled jobs, results posted back."""

    proc: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)
    jobs: dict[str, _Job] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)  # job ids not yet taken, in order
    wake: asyncio.Event | None = None  # created on the loop, set when a job is queued
    started_at: float = 0.0

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


@dataclass
class _GroupsView:
    """The point groups' live view (benchmarks/group_live.py --live --record), started and finished from the
    Approach tab's Groups panel. It holds the camera's recording and TAPIR on the GPU while it runs, so it is
    not left running between sessions."""

    proc: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)
    last: dict[str, Any] = field(
        default_factory=dict
    )  # what the view said of its recording as it was finished

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


@dataclass
class _GroupsFeed:
    """The point groups as the act and its camera view use them, taken from the view while acts use them: each
    designated object's pose frame by frame (frame time, camera <- object) from the frame after its designation, and
    the newest frame's picture (lerobot.showservo.groups_scene.drawing). Written on the event loop under the lock."""

    task: asyncio.Task | None = None
    frames: dict[str, collections.deque] = field(default_factory=dict)
    floor: dict[str, float] = field(
        default_factory=dict
    )  # name -> its frames count after this time (designation)
    since: float = 0.0  # the newest frame time taken
    fresh: float = 0.0  # when it came, monotonic
    drawing: dict[str, Any] | None = None
    reason: str = ""  # why the objects found last are not in the point groups, when they are not


@dataclass
class _PointDepths:
    """Where an object's tracked points were when a view last placed it, for the depth check (:func:`core.depth_seen`):
    by track number, camera frame, with when that view's frame was read. A new tracker session numbers its tracks
    afresh, so the places kept belong to ``session`` only."""

    session: Any = None
    t: float | None = None
    at: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class _Track:
    """The live tracker: one frame at a time goes to the worker, the newest answer is the object's pose."""

    on: bool = False
    algo: str = "p2p"  # the one tracker; the DINO modes stay as a comparison
    follow: bool = False  # the jog's target follows the transported pre-grasp
    hover_mm: float = 20.0
    job: str | None = None  # the track job in flight
    last: dict[str, Any] = field(default_factory=dict)  # the newest result's readout, no arrays
    overlay: bytes | None = None
    fps: float = 0.0
    t_prev: float = 0.0
    done: asyncio.Event | None = None  # set when the in-flight job's result has been applied
    task: asyncio.Task | None = None
    # (when the frame was read, certified, delta_cam) per frame, bounded. The frame's time, not the answer's: the
    # point groups' poses are stamped the same way, and the act carries a view by them from when it was seen.
    history: list = field(default_factory=list)
    depths: _PointDepths = field(default_factory=_PointDepths)  # the picked object's
    # (capture time, control delta_cam) of accepted observations. Held correction
    # must not bypass visibility or geometry decisions through raw P2P history.
    hold_views: list = field(default_factory=list)
    # The newest frame's object model as the session has it (the points the live view draws yellow, every one the
    # session adopted) with what places it: GET /track/model.npz, to inspect what the model took in
    model: dict[str, Any] | None = None
    verification: dict[str, pose_verification.GroupReference] = field(default_factory=dict)
    corrected_verification: dict[str, pose_verification.GroupMotionReference] = field(default_factory=dict)
    verification_conflicts: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class _TargetTrack:
    """The live track of one of the act's objects from its last find, other than the one whose find started the
    Point2Pose session: each trusted frame moves that find, so whatever reads the find (the live view's outline, the
    act's aim) follows the object. Its share of the session comes back with every tracked frame."""

    obj: str | None = None
    anchor: np.ndarray | None = None  # the find's motion from the demo's view to the frame the track began on
    n_points: int = 0  # the tracks it began with, for the trust test
    last: dict[str, Any] = field(default_factory=dict)  # the newest frame's readout
    depths: _PointDepths = field(default_factory=_PointDepths)
    hold_views: list = field(
        default_factory=list
    )  # (stamp, accepted motion from the demo's view), including geometry correction


@dataclass
class _Demo:
    """A recorded demonstration: the fingertip's path in the base frame, the gripper, and the object's pose while it ran."""

    name: str
    concept: str
    fps: float
    t: np.ndarray  # (N,) seconds from the start
    tips: np.ndarray  # (N, 4, 4) fingertip poses, base frame, from the follower's observed joints
    grippers: np.ndarray  # (N,) openings, the follower's 0..100 units
    q_obs: np.ndarray  # (N, J) observed joints
    q_cmd: np.ndarray  # (N, J) commanded joints
    deltas: np.ndarray  # (N, 4, 4) the object's motion since teach, camera frame; identity where unseen
    seen: np.ndarray  # (N,) whether the tracker had the object at that sample
    delta0: np.ndarray  # (4, 4) the object's motion since teach when the demo began
    t0: float = 0.0  # wall-clock start of the recording, to pair samples with camera frames
    camera: str = "camera"  # the camera the frames came from, as the dataset names its image feature
    frames: list | None = None  # the top camera while recording, (t, small rgb), when tracking ran
    root: str | None = None  # the dataset folder once saved
    intr: dict[str, float] | None = None  # the camera intrinsics the frames were taken with
    recording: str | None = (
        None  # the camera stream while recording: rgb/%06d.jpg, depth/%06d.png (mm), cam_K.txt, times.txt
    )
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)  # designated on the stream, by name
    keypoints: list[dict[str, Any]] = field(default_factory=list)  # the operator's marks: t, name, anchor
    landing: str = (
        "exact"  # the turns the place may land at on its object (core.LANDINGS); kept with the marks
    )
    landing_axis: str = "surface_normal"  # task turn axis; independent of visual pose symmetry
    video: list[bytes] | None = None  # the saved video decoded once for the editor, one JPEG per sample
    taught: bool = False  # recorded with an object taught first; unnamed marks follow that object
    hold_index: int = 0  # transient execution context; each grasp has its own attachment correction
    interaction_span: tuple[int, int] | None = None  # transient range of key frames being executed
    holds: dict[str, Any] = field(
        default_factory=dict
    )  # how the demo held the picked object, by the marks it was measured for
    view_points: dict[str, Any] = field(
        default_factory=dict
    )  # each object's surface on its clicked frame, read once
    surface_points: dict[str, Any] = field(default_factory=dict)  # eroded, dense reference observations
    reference_constraints: dict[str, dict] = field(
        default_factory=dict
    )  # planning references, raw and checked


@dataclass
class _Act:
    """One replay of the demo on the object where it is now."""

    on: bool = False
    step: str = ""
    ok: bool | None = None
    reason: str = ""
    stop_requested: bool = False
    progress: float = 0.0
    speed: float = 1.0
    task: asyncio.Task | None = None
    plan: dict[str, Any] | None = (
        None  # what the last act judged before moving: per mark reach, the path's worst
    )
    place: dict[str, Any] | None = (
        None  # what the last act measured for its place: the grasp check, the holds
    )
    inject: dict[str, Any] | None = None  # an error the operator asked this act to inject, for testing
    find_error: np.ndarray | None = (
        None  # while it runs: the injected error in the object's found pose, base frame
    )
    correct_hold: bool = True  # False replays the place uncorrected for the hold, as a baseline
    allow_grasp_prediction: bool = False  # explicitly allow an unchanged-grip prediction when vision fails
    force_recovered_pose: bool = False  # operator confirmed the current ambiguous grasp assignment
    recovered_pose_token: str | None = None
    recovered_pose_object: str | None = None
    finds: list[str] = field(
        default_factory=list
    )  # the find log's rows of the finds it asked for (_log_find)


@dataclass
class _StreamRecorder:
    """The camera's colour and depth written to disk while a demo is recorded, for designating objects afterwards."""

    out: pathlib.Path
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    n: int = 0
    error: str = ""


@dataclass
class _FrameShare:
    """The camera's frames copied into a shared-memory ring as they arrive (lerobot.showservo.frame_ring), so a
    reader in another process takes the newest one with no disk, encoder or wait for a file to be complete."""

    ring: Any
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    n: int = 0
    error: str | None = None


# A view moves an object's pose only while at least this share of its tracked points is seen, for the object picked
# and the one placed onto alike (the first half of the act's rule, :func:`core.view_places`): under the gripper its
# points drift onto the arm while the frame is still trusted. Replayed on five recorded acts, 0.97 kept the place
# within 1.2-3.9 mm of where the cube lay (one act 9 mm for a frame, then 3) against 4-163 mm without it. A run-time
# option (the page's slider) while borrowed points are not there yet.
TRUST_SHARE_DEFAULT = 0.97


@dataclass
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    teach: _Teach | None = None
    test: _Test | None = None
    worker: _Worker = field(default_factory=_Worker)
    groups: _GroupsView = field(default_factory=_GroupsView)
    teach_job: str | None = None  # a features teach awaiting its result
    find_job: str | None = None
    flat: bool = (
        False  # opt-in resting prior: the fit's motion as a turn about the surface the object rests on
    )
    trust_share: float = TRUST_SHARE_DEFAULT  # the place object's track is followed only with this share seen
    # The depth check (on by default): a tracked point counts as seen only while the depth under it agrees, within
    # depth_tol_m, with where it was when a view last placed its object (core.depth_seen), for both objects.
    depth_check: bool = True
    depth_tol_m: float = core.DEPTH_AGREE_M
    # Whether acts run with the point groups (src/lerobot/showservo/docs/act_loop.md): the act starts the view when it
    # is not running, designates the objects it follows in it before the arm moves, and moves each object with what it
    # rests on while no view places it. Off, an act finishes a running view, and an object no view places stays where
    # it was last placed.
    groups_with_acts: bool = True
    groups_feed: _GroupsFeed = field(default_factory=_GroupsFeed)
    server_url: str | None = (
        None  # this server's own address, as the worker and the point groups view are told it
    )
    track: _Track = field(default_factory=_Track)
    act: _Act = field(default_factory=_Act)
    demo: _Demo | None = None  # the demo recorded or loaded last
    recording: dict[str, Any] | None = None  # while a demo is being recorded: its start and the frames so far
    stream: _StreamRecorder | None = None  # the camera stream being written while a demo is recorded
    run: recording._Run | None = None  # the act being recorded: its tracker frames, its targets, the arm
    camera_recording: _StreamRecorder | None = (
        None  # the camera recorded on its own, to replay the tracker over
    )
    camera_share: _FrameShare | None = (
        None  # the camera's frames in shared memory, for the point groups' view
    )
    located: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # objects found against the demo's view of them, by name, each moved by its track
    targets: dict[str, _TargetTrack] = field(default_factory=dict)  # the session's other objects, by name
    reach_landing: dict[str, Any] = field(
        default_factory=dict
    )  # the reach preview's landing turn and what it was searched for: one search per find of the target


_state = _State()


_RENDER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="manipulation-render"
)


_ACT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="manipulation-act")


_SEEN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="manipulation-seen")


# An act's recording goes to disk on its own thread, in order; the tracker never waits for it.
_RUN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="manipulation-run")


TRACK_JOB_TIMEOUT_S = 5.0


TRACK_FRAME_RETRY_S = 0.5  # how soon the live track asks the camera again after a frame it could not read


TRACK_HISTORY_MAX = 3000  # about a hundred seconds at the camera's rate: longer than any demo


_HEAVY = ("delta_cam", "teach_uv", "live_uv", "live_mask")


JOB_TIMEOUT_S = 120.0  # the worker's first job loads the models; longer than that and no answer is coming


def _expire_jobs_locked(s: _State) -> None:
    """A pending teach or find the worker has not answered in time stops pending, with a log line saying so.

    Seen once: a worker idle for minutes never received a queued job, and the
    tab said "teaching…" until the page was reloaded. Pre: ``s.lock`` held.
    """
    now = time.time()
    for attr in ("teach_job", "find_job"):
        jid = getattr(s, attr)
        if jid is None:
            continue
        job = s.worker.jobs.get(jid)
        if job is not None and (job.result is not None or now - job.created <= JOB_TIMEOUT_S):
            continue
        setattr(s, attr, None)
        what = (
            "a job"
            if job is None
            else f"{job.kind} {jid} ({'taken' if job.taken else 'never taken'} by the worker)"
        )
        s.worker.log.append(
            f"{what} timed out after {JOB_TIMEOUT_S:.0f} s; try again, restart the worker if it repeats"
        )


class TeachBody(BaseModel):
    box: list[int] = []  # x0, y0, x1, y1 in frame pixels (box mode)
    mode: str = "box"  # "box" | "features" (SAM3 by concept or by a clicked pixel, in the worker)
    concept: str = ""
    click: list[int] = []  # x, y in frame pixels: SAM3 segments what is under it instead of a name
    ref_object: str = (
        ""  # a demo's designated object: the live view is registered against the demo's view of it
    )


class GoBody(BaseModel):
    hover_mm: float = 20.0


class OptionsBody(BaseModel):
    flat: bool | None = None
    groups_with_acts: bool | None = None
    trust_share: float | None = Field(None, ge=0.0, le=1.0)
    depth_check: bool | None = None
    depth_tol_mm: float | None = Field(None, ge=1.0, le=100.0)


# The act's frames are read beside the camera's other readers, not queued behind them on the camera executor: there
# the frame share's read, always waiting for the next frame, held each of the act's reads back a frame period, and its
# clearing of the camera's new-frame event cost the act's read another (2026-10-10 acts: 50-58 ms per read, the
# tracker's frames 3-5 a second).
_GRAB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="manipulation-grab")


_grab_lock = threading.Lock()


_last_grab: tuple[Any, float] | None = (
    None  # (camera, the capture time of the last frame _grab returned from it)
)


_PAGE = pathlib.Path(__file__).resolve().parents[2] / "static" / "index.html"


_page_seen: tuple[float, str | None] = (-1.0, None)


LAST_SEEN_FILE = (
    "last_seen.json"  # beside a saved demo: where each of its objects was last seen live, as a click
)


LAST_SEEN_EVERY_S = 2.0  # a tracked object's last-seen point is written at most this often


REFIND_WAIT_S = (
    120.0  # after a load, how long the finds at the last-seen points wait for the worker and the camera
)


_last_seen_written: dict[str, float] = {}


_refinds: set[asyncio.Task] = set()  # the finds a load started, kept until they end


FINDS_DIR = (
    "finds"  # beside a saved demo: each find of its objects (:func:`_log_find`) and each last-seen write
)


FINDS_LOG = "finds.jsonl"


SEEN_LOG = "seen.jsonl"


_FIND_LOG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="manipulation-find-log"
)


JAW_HALF_M = 0.025


APPROACH_M = 0.030


FRAME_AXIS_M = 0.03  # the drawn object frame's axis length


FOUND_COLOUR = (
    0,
    140,
    255,
)  # an object as a find placed it: orange, apart from the tracked object's magenta and yellow


POSE_ESTIMATE_COLOUR = (255, 160, 100)  # blue: the current tracker estimate, not an accepted control target


# ── trials: one row per run, with the operator's verdict; the milestone's evidence ─────────────

TRIALS_PATH = _REPO / "captures" / "trials.jsonl"


TRIAL_VERDICTS = ("lifted", "missed", "collided", "other")


_trials: list[dict[str, Any]] | None = None


ACTS_DIR = "acts"  # beside a saved demo: one folder per act, all it saw and did


RUN_FRAME_FIELDS = (
    "software",
    "ok",
    "state",
    "reason",
    "ms",
    "n_matches",
    "lost_streak",
    "n_tracks",
    "n_model_points",
    "fit_points",
    "fit_inliers",
    "jump_guard_rejected",
    "pose_jump_guard_info",
    "mask_pixels",
    "fit_support",
)


# ── the replay: an act's recorded frames with what the act had then, every value as the run recorded it ─

_REPLAY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="manipulation-replay"
)


_replay_meta: dict[
    str, tuple[float, dict[str, Any]]
] = {}  # an act's folder -> (its act.json's mtime, its record)


# BGR, the act's objects in order: the one whose find starts the session, then each other joining it
REPLAY_COLOURS = ((0, 220, 255), (255, 255, 0), (120, 230, 120), (240, 110, 200), (255, 230, 90))


REPLAY_TOLD = (0, 0, 255)


REPLAY_VIDEO_FPS = 10  # an exported replay's rate: each recorded frame shown until the next one's time


REPLAY_VIDEO_TIMEOUT_S = 120.0


_REPLAY_EXPORT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="manipulation-replay-export"
)


# ── the demo: a recorded path saved as a LeRobot dataset; the act: that path on the object where it is now ─

DEMOS_NAMESPACE = "demos"


DEMO_FILE = "showservo_demo.npz"


HISTORY_MATCH_S = 0.2  # a tracker frame this close in time to a joint sample is that sample's object pose


ACT_ARRIVE_M = 0.004  # within this of a target counts as arrived: the servo's stiction band


ACT_STEP_TIMEOUT_S = 20.0


ACT_SETTLE_DEG = 2.0  # arm joints this close to the last target have arrived: the servo's own band


# A joint held short under load stops there and waits do not close it: right after the grasp's lift, with the gamepad
# in the gripper, the worst joint of five pick-and-place acts sat 1.2 to 3.4 deg off its last target, the 3.4 holding
# for 20 s (2026-10-07). Stopped (no joint moving ACT_STILL_DEG over ACT_STILL_S) within this, the arm has arrived.
ACT_STALL_DEG = 6.0


ACT_STILL_DEG = 0.3


ACT_STILL_S = 0.5


ACT_TICK_S = 0.05


LANDING_EVERY_S = (
    0.25  # a landing turn is judged on the place's samples this far apart, as well as its pre-places
)


LANDING_TOP_M = 0.004  # the top of the object placed onto: its surface within this of its highest points


GRIP_SETTLE_S = 1.0  # the grasp check waits at most this long for the gripper to stop closing


HOLD_VIEW_GAP_S = (
    0.1  # views of the held object at the grip are this far apart, so each is a new camera frame
)


HOLD_TRACK_WAIT_S = 5.0  # the live track's views of a held object are waited for this long, about 25 frames


GROUPS_VIEW = (
    "127.0.0.1",
    9141,
)  # the point groups' live view's own server (benchmarks/group_live.py --live)


_GROUPS_SCRIPT = _REPO / "benchmarks" / "group_live.py"


# The view runs in Point2Pose's environment, where TAPIR and its checkpoint live, as the worker's bridge does.
P2P_PYTHON = os.environ.get(
    "LEROBOT_P2P_PYTHON", str(pathlib.Path.home() / ".cache/point2pose/venv/bin/python")
)


P2P_REPO = str(checkout_path())


_GROUPS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="manipulation-groups"
)


GROUPS_PAGE = """<html><head><title>point groups</title></head><body style='margin:0;background:#111'>
<img src='/api/pregrasp/groups/stream' style='width:100%'
 onerror="setTimeout(()=>{this.src='/api/pregrasp/groups/stream?'+Date.now()},1000)"></body></html>"""


GROUPS_READY_S = 180.0  # a view started for the objects loads TAPIR, then SAM 2.1 for each designation


GROUPS_POLL_S = 0.05  # how often the feed takes the view's newest frames; it draws one every 70-300 ms


GROUPS_STALE_S = 5.0  # a view that drew no frame for this long has stopped carrying the objects: an act stops


GROUPS_FRAMES_KEPT = 1800  # each object's poses kept in the feed: minutes at the view's rate, many acts' span


GROUPS_DRAW_MAX_AGE_S = 1.0  # the camera view paints the groups' picture only while it is this fresh


GROUPS_HELD_S = 0.1  # how often an act tells the point groups where the arm has the object it holds


_groups_follows: set[asyncio.Task] = set()  # the designations a find started, kept until they end


DEMO_RECORDING = "recording"


KEYPOINTS_FILE = "keypoints.json"


OBJECTS_FILE = "objects.npz"


OBJECT_COLOURS = (
    (255, 170, 0),
    (80, 200, 255),
    (120, 230, 120),
    (240, 110, 200),
    (255, 230, 90),
)  # BGR, by order


POSE_AXIS_M = 0.04  # the drawn x and y axes' length


POSE_UP_M = 0.08  # up and true vertical, longer: seen from above, a tilt barely moves a short one


INJECT_MAX_MM = 30.0  # an injected error stays a test of the grasp, not a way to drive the arm elsewhere


INJECT_MAX_DEG = 20.0
