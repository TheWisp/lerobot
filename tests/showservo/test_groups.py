"""The point groups: which points move together, and an object's pose from its group while its own points are
hidden. Synthetic clouds in the camera frame, RealSense-like noise, at 30 fps."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.showservo.groups import GroupTracker

RNG = np.random.default_rng(3)
NOISE_M = 0.0015  # RealSense depth noise at half a metre, per axis


def _cloud(n, centre, size):
    return np.asarray(centre) + (RNG.random((n, 3)) - 0.5) * np.asarray(size)


def _moved(pts, rotvec_deg, trans, about):
    rot = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    return (pts - about) @ rot.T + about + np.asarray(trans)


def _pose(centre):
    m = np.eye(4)
    m[:3, 3] = centre
    return m


def _run(tracker, frames):
    """Feed (xyz, seen) frames; return the tracker."""
    for xyz, seen in frames:
        tracker.update(xyz + RNG.normal(0.0, NOISE_M, xyz.shape), seen)
    return tracker


def test_a_still_scene_is_one_group_and_a_body_that_moves_leaves_it_with_its_pose():
    """A tray of 200 points and a cube of 30 on it: one group once the window fills. The cube slides 40 mm and turns
    20 deg from frame 20: its points leave the tray's group and found their own, and its pose follows the slide."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    cube = _cloud(30, (0.1, 0.05, 0.43), (0.03, 0.03, 0.03))
    t = GroupTracker()
    t.add_object("cube", np.arange(200, 230), _pose((0.1, 0.05, 0.43)))
    frames = []
    for k in range(60):
        s = min(max(k - 20, 0), 20) / 20.0  # the slide, over frames 20-40
        c = _moved(cube, (0.0, 0.0, 20.0 * s), (0.04 * s, 0.0, 0.0), np.array([0.1, 0.05, 0.43]))
        frames.append((np.vstack([tray, c]), np.ones(230, bool)))
    _run(t, frames[:20])
    assert len(t.groups) == 1 and (t.group_of >= 0).mean() > 0.9, "one group for a still scene"
    _run(t, frames[20:])
    kinds = [t.group_of[:200], t.group_of[200:]]
    assert len(set(kinds[0])) == 1 and len(set(kinds[1])) == 1 and kinds[0][0] != kinds[1][0], (
        "the tray and the cube are two groups"
    )
    pose = t.pose("cube")
    assert np.linalg.norm(pose[:3, 3] - [0.14, 0.05, 0.43]) < 0.004, pose[:3, 3]
    assert abs(np.degrees(Rotation.from_matrix(pose[:3, :3]).magnitude()) - 20.0) < 2.0


def test_a_hidden_object_follows_the_group_it_rests_on():
    """The cube's points vanish from frame 15 (the gripper over it). From frame 25 the whole tray turns 15 deg and
    moves 50 mm (the camera moved, or the tray did). The cube's pose follows the tray; held where it was last seen,
    as the trust gate holds it today, it would be 50 mm off."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    cube_c = np.array([0.1, 0.05, 0.43])
    cube = _cloud(30, cube_c, (0.03, 0.03, 0.03))
    about = np.array([0.0, 0.0, 0.45])
    t = GroupTracker()
    t.add_object("cube", np.arange(200, 230), _pose(cube_c))
    frames = []
    for k in range(60):
        s = min(max(k - 25, 0), 20) / 20.0
        pts = _moved(np.vstack([tray, cube]), (0.0, 0.0, 15.0 * s), (0.05 * s, 0.0, 0.0), about)
        seen = np.ones(230, bool)
        if k >= 15:
            seen[200:] = False
        frames.append((pts, seen))
    _run(t, frames)
    truth = _moved(cube_c[None], (0.0, 0.0, 15.0), (0.05, 0.0, 0.0), about)[0]
    pose = t.pose("cube")
    assert t.objects["cube"].n_seen == 0 and t.objects["cube"].group is not None
    assert np.linalg.norm(pose[:3, 3] - truth) < 0.004, (pose[:3, 3], truth)
    assert abs(np.degrees(Rotation.from_matrix(pose[:3, :3]).magnitude()) - 15.0) < 2.0
    assert np.linalg.norm(cube_c - truth) > 0.04, "holding the last seen pose would be far off"


def test_a_body_that_stops_merges_back_and_keeps_its_pose():
    """The cube slides, then rests: its group merges into the tray's after merge_frames of moving alike, and its
    pose does not jump at the merge."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    cube_c = np.array([0.1, 0.05, 0.43])
    cube = _cloud(30, cube_c, (0.03, 0.03, 0.03))
    t = GroupTracker(merge_frames=20)
    t.add_object("cube", np.arange(200, 230), _pose(cube_c))
    frames = []
    for k in range(80):
        s = min(max(k - 10, 0), 10) / 10.0
        c = _moved(cube, (0.0, 0.0, 0.0), (0.04 * s, 0.0, 0.0), cube_c)
        frames.append((np.vstack([tray, c]), np.ones(230, bool)))
    _run(t, frames[:30])
    assert len(t.groups) == 2
    before = t.pose("cube")[:3, 3]
    _run(t, frames[30:])
    assert len(t.groups) == 1, "merged back once it rested"
    after = t.pose("cube")[:3, 3]
    assert np.linalg.norm(after - before) < 0.004 and np.linalg.norm(after - [0.14, 0.05, 0.43]) < 0.004


def test_drifting_points_leave_without_moving_the_group():
    """A few tracks slide away on their own (points that latched onto the arm): they get struck out and leave, and
    the tray's motion stays identity."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    t = GroupTracker()
    ways = RNG.normal(size=(5, 3)) * [1.0, 1.0, 0.0]  # 5 sliders, each its own way, 4 mm a frame
    ways = 0.004 * ways / np.linalg.norm(ways, axis=1, keepdims=True)
    frames = []
    for k in range(40):
        pts = tray.copy()
        if k >= 10:
            pts[:5] += ways * (k - 9)
        frames.append((pts, np.ones(200, bool)))
    _run(t, frames)
    (g,) = t.groups.values()
    assert (t.group_of[:5] != g.id).all(), "the sliders left"
    assert (t.group_of[5:] == g.id).all()
    assert np.linalg.norm(g.motion.trans) < 0.003 and np.degrees(g.motion.angle) < 0.5


def test_a_new_track_joins_the_group_that_explains_it():
    """Tracks appear over time (the sampler adds them). A new point on the still tray joins the tray's group; one on
    the sliding cube joins the cube's."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    cube_c = np.array([0.1, 0.05, 0.43])
    cube = _cloud(30, cube_c, (0.03, 0.03, 0.03))
    t = GroupTracker()
    frames = []
    for k in range(50):
        s = min(max(k - 5, 0), 30) / 30.0
        c = _moved(cube, (0.0, 0.0, 0.0), (0.06 * s, 0.0, 0.0), cube_c)
        pts, seen = np.vstack([tray, c]), np.ones(230, bool)
        if k >= 25:  # two late tracks: one on the tray, one on the cube
            late = np.vstack([[0.2, 0.1, 0.45], c.mean(axis=0) + [0.0, 0.01, 0.0]])
            pts, seen = np.vstack([pts, late]), np.ones(232, bool)
        frames.append((pts, seen))
    _run(t, frames)
    assert t.group_of[230] == t.group_of[0] and t.group_of[231] == t.group_of[200]
    assert t.group_of[230] != t.group_of[231]


@pytest.mark.parametrize("seed", [1, 2])
def test_pose_error_while_hidden_is_within_the_fits_noise(seed):
    """Over 40 hidden frames with the scene drifting slowly, the composed pose stays within a few mm of the truth."""
    rng = np.random.default_rng(seed)
    tray = rng.random((150, 3)) * [0.5, 0.35, 0.002] + [-0.25, -0.175, 0.45]
    cube_c = np.array([0.05, -0.02, 0.43])
    cube = rng.random((25, 3)) * 0.03 + cube_c - 0.015
    t = GroupTracker()
    t.add_object("cube", np.arange(150, 175), _pose(cube_c))
    worst = 0.0
    for k in range(60):
        s = k / 60.0
        pts = _moved(np.vstack([tray, cube]), (0.0, 0.0, 10.0 * s), (0.03 * s, 0.01 * s, 0.0), np.zeros(3))
        seen = np.ones(175, bool)
        if k >= 20:
            seen[150:] = False
        t.update(pts + rng.normal(0.0, NOISE_M, pts.shape), seen)
        if k >= 20:
            truth = _moved(cube_c[None], (0.0, 0.0, 10.0 * s), (0.03 * s, 0.01 * s, 0.0), np.zeros(3))[0]
            worst = max(worst, float(np.linalg.norm(t.pose("cube")[:3, 3] - truth)))
    assert worst < 0.004, worst
