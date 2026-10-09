"""The point groups: which points move together, and an object's pose from its group while its own points are
hidden. Synthetic clouds in the camera frame, RealSense-like noise, at 30 fps."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.showservo.groups import GroupTracker

RNG = np.random.default_rng(3)
NOISE_M = 0.0015  # RealSense depth noise at half a metre, per axis


@pytest.fixture(autouse=True)
def _fresh_rng():
    """Every test draws from the same fresh generator: what one test draws never changes what the next one sees."""
    global RNG
    RNG = np.random.default_rng(3)


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
    _run(t, frames[20:50])  # the slide and its first still frames, before the cube merges back
    kinds = [t.group_of[:200], t.group_of[200:]]
    assert len(set(kinds[0])) == 1 and len(set(kinds[1])) == 1 and kinds[0][0] != kinds[1][0], (
        "the tray and the cube are two groups"
    )

    def check_pose():
        pose = t.pose("cube")
        assert np.linalg.norm(pose[:3, 3] - [0.14, 0.05, 0.43]) < 0.004, pose[:3, 3]
        assert abs(np.degrees(Rotation.from_matrix(pose[:3, :3]).magnitude()) - 20.0) < 2.0

    check_pose()
    _run(t, frames[50:])  # at rest beside the tray long enough to merge back: the pose must not move
    check_pose()


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
    for k in range(42):  # the slide ends at 35; judged before the cube, at rest again, merges back
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


def test_a_point_that_keeps_slipping_is_retired_and_a_steady_one_earns_its_tenure():
    """Four tracks jump 15 mm off and back every 8 frames (corners on a depth edge, or on texture that slides):
    struck out three times, they are retired and never rejoin, and the tray's motion stays identity. The steady
    points have held their place for the whole run; a point that left once holds no tenure."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    t = GroupTracker()
    frames = []
    for k in range(90):
        pts = tray.copy()
        if k >= 10 and (k // 8) % 2 == 1:
            pts[:4] += [0.0, 0.0, 0.015]
        frames.append((pts, np.ones(200, bool)))
    _run(t, frames)
    (g,) = t.groups.values()
    assert t.retired[:4].all() and (t.group_of[:4] == -1).all(), "the slippers are retired"
    assert not t.retired[4:].any()
    assert (t.leaves[:4] >= 3).all() and (t.tenure[4:] >= 60).all()
    assert np.linalg.norm(g.motion.trans) < 0.003 and np.degrees(g.motion.angle) < 0.5


def test_a_point_no_group_ever_explains_is_retired():
    """Six tracks wander on their own from the start (texture that is not a surface: a reflection, a shadow's edge):
    never explained, never enough to found a group of their own, they are retired after retire_unexplained frames."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    t = GroupTracker(retire_unexplained=40)
    walk = RNG.normal(size=(6, 3)) * [0.004, 0.004, 0.0]
    frames = []
    for k in range(70):
        pts = tray.copy()
        pts[:6] += walk * k + RNG.normal(0.0, 0.003, (6, 3))
        frames.append((pts, np.ones(200, bool)))
    _run(t, frames)
    assert t.retired[:6].all() and not t.retired[6:].any()
    assert len(t.groups) == 1 and (t.group_of[6:] >= 0).mean() > 0.9


def test_old_points_outvote_a_young_crowd_that_slides_together():
    """Sixty tracks join the tray late and then slide away together before they are established (a sheet laid on
    the tray and pulled): the tray's forty established points keep the group's motion, and the sliders leave as
    one. Established members define a group's frame; newcomers only agree or leave, as SLAM's map outlives its
    pending points."""
    tray = _cloud(40, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    sheet = _cloud(60, (0.1, 0.0, 0.448), (0.2, 0.15, 0.001))
    t = GroupTracker(full_tenure=20)
    frames = []
    for k in range(60):  # the slide ends at 55; judged before the sheet, at rest again, merges back
        s = min(max(k - 35, 0), 20) / 20.0
        pts = np.vstack([tray, sheet + [0.04 * s, 0.0, 0.0]])
        seen = np.ones(100, bool)
        if k < 25:
            seen[40:] = False  # the sheet's points appear late
        frames.append((pts, seen))
    _run(t, frames)
    tray_group = t.group_of[0]
    assert (t.group_of[:40] == tray_group).all(), "the tray held its group"
    assert np.linalg.norm(t.groups[tray_group].motion.trans) < 0.004, "and its motion"
    assert (t.group_of[40:] != tray_group).all(), "the sliders left it together"


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


def test_a_field_track_hidden_for_long_is_retired_but_an_objects_own_track_is_not():
    """A sheet of paper over part of the tray: the tracks under it are not seen. A field track hidden for
    retire_unseen frames is retired (it is probably gone for good); an object's own track is not, since the
    object is placed by its own points again the moment they show (the roll under the paper, 250 frames)."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    t = GroupTracker(retire_unseen=40)
    t.add_object("roll", np.arange(190, 200), _pose(tray[190:].mean(axis=0)))
    frames = []
    for k in range(80):
        seen = np.ones(200, bool)
        if k >= 20:
            seen[180:] = False  # ten field tracks and the roll's ten, under the paper
        frames.append((tray, seen))
    _run(t, frames)
    assert (t.unseen[180:] == 60).all() and (t.unseen[:180] == 0).all()
    assert t.retired[180:190].all() and (t.group_of[180:190] == -1).all(), "the field tracks under the paper"
    assert not t.retired[190:].any() and (t.group_of[190:] >= 0).all(), "the roll's own tracks, still carried"
    assert not t.retired[:180].any()


def test_two_groups_merge_only_when_every_member_of_the_smaller_would_fit():
    """A ring of forty desk points 300 mm out around a tray of eighty. The ring slides off 40 mm: two groups. The
    tray then turns slowly, 0.3 deg a frame, for 40 frames: at the ring's centre, 40 mm from the tray's, the
    relative motion over a window is 3 mm and under 5 deg, which a judgement at the centre would have merged, and
    then struck the ring's points, 23 mm off, out again. Judged at its points, the ring does not merge while the
    tray turns, and no third group is ever founded; once the tray stops it merges, every ring point in."""
    tray = _cloud(80, (0.0, 0.0, 0.45), (0.1, 0.1, 0.002))
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.stack([0.3 * np.cos(ang), 0.3 * np.sin(ang), np.full(40, 0.45)], axis=1)
    t = GroupTracker()
    frames = []
    for k in range(130):
        s = min(max(k - 20, 0), 10) / 10.0  # the ring slides off over frames 20-30
        turn = 0.3 * min(max(k - 32, 0), 40)  # the tray turns from frame 32 to 72, then stops
        tr = _moved(tray, (0.0, 0.0, turn), (0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.45]))
        frames.append((np.vstack([tr, ring + [0.04 * s, 0.0, 0.0]]), np.ones(120, bool)))
    _run(t, frames[:40])
    assert len(t.groups) == 2 and len(set(t.group_of[80:])) == 1, "the ring is a group of its own"
    ring_group = t.group_of[80]
    _run(t, frames[40:72])  # the tray turning: at the ring's centre little moves, at its points 23 mm
    assert len(t.groups) == 2 and (t.group_of[80:] == ring_group).all(), "no merge while the tray turns"
    assert t._next_group == 2, "and no third group founded by points struck out of a merge"
    _run(t, frames[72:])  # both still: merged, and every ring point stays in the merged group
    assert len(t.groups) == 1 and len(set(t.group_of)) == 1 and (t.group_of >= 0).all()
    assert t._next_group == 2
