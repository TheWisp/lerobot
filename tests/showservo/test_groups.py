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


def test_strays_struck_out_of_a_group_found_nothing_when_they_come_to_rest_with_it():
    """Twenty-four tray points jitter 15 mm an axis for four frames (a rim's depth while the tray moved) and are
    struck out. Banned from the tray's group for a while, they come to rest where the tray puts them: they found
    no group of their own, since a new group must move differently from the group its founders came from, and
    they rejoin the tray once the ban ends. On the recording this was the still group born a second after a merge."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    t = GroupTracker()
    frames = []
    for k in range(70):
        pts = tray.copy()
        if 20 <= k < 24:
            pts[:24] += RNG.normal(0.0, 0.015, (24, 3))
        frames.append((pts, np.ones(200, bool)))
    _run(t, frames[:40])
    assert t._next_group == 1, "the strays founded no group"
    _run(t, frames[40:])
    assert len(t.groups) == 1 and (t.group_of == 0).all(), "all back in the tray's group"


def _split_scene(k_motion, n_frames):
    """A tray of 200 points and a cube of 30 on it; ``k_motion(k)`` gives the cube's offset at frame k."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    cube = _cloud(30, (0.1, 0.05, 0.43), (0.03, 0.03, 0.03))
    return [(np.vstack([tray, cube + k_motion(k)]), np.ones(230, bool)) for k in range(n_frames)]


def test_a_body_that_starts_to_move_splits_off_within_frames():
    """The cube slides at 2 mm a frame from frame 40. Weighed as a body, it is its own group within five frames
    of the start, at this scene's noise (1.5 mm an axis, the split's trigger three times the group's median
    offset, 8 mm); point by point (the split off) each of its points first has to clear 10 mm for 3 frames,
    four frames or more later."""
    frames = _split_scene(lambda k: [0.002 * max(k - 40, 0), 0.0, 0.0], 70)

    def split_frame(tracker):
        for k, (xyz, seen) in enumerate(frames):
            tracker.update(xyz + RNG.normal(0.0, NOISE_M, xyz.shape), seen)
            cube = tracker.group_of[200:]
            if (
                k >= 40
                and (cube >= 0).mean() > 0.5
                and np.bincount(cube[cube >= 0]).argmax() != tracker.group_of[0]
            ):
                return k - 40
        return None

    fast = split_frame(GroupTracker())
    slow = split_frame(GroupTracker(split_trigger_m=0.0))
    assert fast is not None and fast <= 5, fast
    assert slow is not None and slow >= fast + 4, (fast, slow)


def test_a_body_displaced_once_does_not_split_off():
    """The cube's points jump 15 mm in one frame and stay there (a tracker glitch, a hand brushing past), well past
    the split's trigger: they hold their new offset rather than keep moving, so no body splits off at once (without
    the velocity rule they would, at the jump's frame)."""
    frames = _split_scene(lambda k: [0.015 if k >= 40 else 0.0, 0.0, 0.0], 46)
    t = GroupTracker()
    _run(t, frames)
    assert not any(src == t.group_of[0] and f <= 45 for f, src, *_ in t.split_log), t.split_log


def test_a_hidden_object_leaves_with_what_it_rests_on_when_that_splits_off():
    """A desk of 300 points around a tray of 100, one group while still; a cube of 30 on the tray hides under a
    sheet at frame 15. From frame 30 the tray slides 50 mm: the tray splits off the desk's group, and the cube's
    hidden tracks go with it, the side most of their nearest seen neighbours took, so the cube is carried with the
    tray. Left in the desk's group, it would stay 50 mm behind."""
    ang = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    desk = np.stack([0.32 * np.cos(ang), 0.24 * np.sin(ang), np.full(300, 0.47)], axis=1)
    tray = _cloud(100, (0.0, 0.0, 0.45), (0.3, 0.2, 0.002))
    cube_c = np.array([0.05, 0.03, 0.43])
    cube = _cloud(30, cube_c, (0.03, 0.03, 0.03))
    t = GroupTracker()
    t.add_object("cube", np.arange(400, 430), _pose(cube_c))
    frames = []
    for k in range(70):
        s = min(max(k - 30, 0), 25) / 25.0
        shift = [0.05 * s, 0.0, 0.0]
        seen = np.ones(430, bool)
        if k >= 15:
            seen[400:] = False
        frames.append((np.vstack([desk, tray + shift, cube + shift]), seen))
    _run(t, frames[:50])  # mid-slide
    assert t.group_of[400] == t.group_of[300] != t.group_of[0], "the cube's tracks went with the tray"
    _run(t, frames[50:])  # the tray stops and merges back into the desk's group: the cube keeps its place
    assert np.linalg.norm(t.pose("cube")[:3, 3] - (cube_c + [0.05, 0.0, 0.0])) < 0.006, t.pose("cube")[:3, 3]


def test_a_corner_of_an_object_left_in_view_does_not_turn_it_its_group_carries_it():
    """With the wrist over all but a corner of a gamepad, its own points there placed it 23 degrees off while it lay
    still (2026-10-09): enough of them to fit, too few and too bunched to pin it. Under the act's rule they do not
    place it; its group carries it where it lay. Without the rule (any share, no noise to carry to its middle) the same
    corner tilts it."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    xs, ys = np.linspace(-0.036, 0.036, 6), np.linspace(-0.020, 0.020, 5)
    centre = np.array([0.10, 0.05, 0.43])
    pad = np.array([centre + [x, y, 0.0] for x in xs for y in ys])  # 30 own points over its top face
    corner = np.flatnonzero((pad[:, 0] > centre[0]) & (pad[:, 1] > centre[1] - 0.005))  # a quarter of it
    covered = pad.copy()
    # What the corner's points say near the wrist: a patch turned 15 degrees about its own middle, every point
    # within a few millimetres of where it lay, so none leaves the group.
    covered[corner] = _moved(pad[corner], (15.0, 0.0, 0.0), (0.0, 0.0, 0.0), pad[corner].mean(axis=0))

    def run(**rule):
        t = GroupTracker(**rule)
        t.add_object("gamepad", np.arange(200, 230), _pose(centre))
        frames = []
        for k in range(60):
            seen = np.ones(230, bool)
            pts = np.vstack([tray, pad])
            if k >= 20:  # all but the corner hidden from here
                seen[200:] = False
                seen[200 + corner] = True
                pts = np.vstack([tray, covered])
            frames.append((pts, seen))
        return _run(t, frames)

    t = run()
    obj = t.objects["gamepad"]
    assert len(corner) >= 6 and not obj.own_ok and "points are seen" in obj.why, (len(corner), obj.why)
    pose = t.pose("gamepad")
    # Where it lay, to the act's reach tolerance (3 mm, 3 deg): its fully seen pose already carries the noise.
    assert np.linalg.norm(pose[:3, 3] - centre) < 0.003, pose[:3, 3]
    assert np.degrees(Rotation.from_matrix(pose[:3, :3]).magnitude()) < 3.0, "it lies where it lay"
    loose = run(own_share=0.0, point_noise_m=1e-6).pose("gamepad")
    assert np.degrees(Rotation.from_matrix(loose[:3, :3]).magnitude()) > 10.0, (
        "the corner alone would tilt it"
    )


def test_tracks_of_a_covered_object_that_slide_onto_the_wrist_do_not_carry_it_off():
    """The wrist comes over a gamepad at frame 30 and moves on at 4 mm a frame: 18 of its 30 tracks stick to the
    wrist and move with it, still seen, the rest hidden (Point2Pose's tracks slid onto the gripper so on 2026-10-09).
    The slid ones split off as a body into the arm's group, where most of its grouped tracks now are, which took the
    gamepad along before; another group takes it only on a view that places it there, and 60% of it seen is none. Its
    hidden tracks stay with the tray: where a hidden track goes in a split is the support's vote around it, not the
    object's own tracks, which may be the ones sliding."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    centre = np.array([0.10, 0.05, 0.43])
    pad = _cloud(30, centre, (0.072, 0.040, 0.010))
    arm = _cloud(40, (0.10, 0.05, 0.36), (0.05, 0.05, 0.05))
    slid = np.arange(18)  # the gamepad's tracks that stick to the wrist; the other 12 are hidden

    def run(**rule):
        t = GroupTracker(**rule)
        t.add_object("gamepad", np.arange(240, 270), _pose(centre))
        frames = []
        for k in range(75):
            off = np.array([0.004 * max(k - 20, 0), 0.0, 0.0])  # the arm moves on from frame 20
            pts, seen = pad.copy(), np.ones(270, bool)
            if k >= 30:  # from where they were as the wrist came over
                pts[slid] += off - [0.040, 0.0, 0.0]
                seen[240 + len(slid) :] = False
            frames.append((np.vstack([tray, arm + off, pts]), seen))
        return _run(t, frames)

    t = run()
    arm_group = int(np.bincount(t.group_of[200:240][t.group_of[200:240] >= 0]).argmax())
    assert arm_group != t.group_of[0] and (t.group_of[240 + slid] == arm_group).mean() > 0.5, (
        "the slid tracks went to the arm's group"
    )
    assert (t.group_of[240 + len(slid) : 270] == t.group_of[0]).all(), "the hidden ones stayed with the tray"
    pose = t.pose("gamepad")
    assert t.objects["gamepad"].group == t.group_of[0], "it rests in the tray's group"
    assert np.linalg.norm(pose[:3, 3] - centre) < 0.003, pose[:3, 3]
    assert np.degrees(Rotation.from_matrix(pose[:3, :3]).magnitude()) < 3.0
    loose = run(own_share=0.0, point_noise_m=1e-6).pose("gamepad")
    assert np.linalg.norm(loose[:3, 3] - centre) > 0.03, "a view of the slid tracks alone would carry it off"


def test_an_object_at_rest_stays_with_its_group_until_its_own_points_show_it_moved():
    """A 36 mm cube seen only by the 25 points of its top face: each fit of them alone tilts it a degree or more from
    the last (depth noise over a short, flat base), and re-placing it with every fit made it wobble while it lay
    still, which an act aiming above it carries to the fingertip. It stays where its group, fitted on the tray's 200
    points, carries it until its own points put it beyond the place tolerance from there; then they move it, here a
    20 mm slide over the tray."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    centre = np.array([0.10, 0.05, 0.41])
    xs = np.linspace(-0.018, 0.018, 5)
    top = np.array([centre + [x, y, 0.0] for x in xs for y in xs])
    t = GroupTracker()
    t.add_object("cube", np.arange(200, 225), _pose(centre))
    poses = []
    for k in range(90):
        slide = np.array([0.020 * min(max(k - 60, 0), 10) / 10, 0.0, 0.0])  # frames 60-70
        _run(t, [(np.vstack([tray, top + slide]), np.ones(225, bool))])
        poses.append(t.pose("cube"))
    rest = poses[10:60]
    turn = [np.degrees(Rotation.from_matrix(p[:3, :3] @ rest[0][:3, :3].T).magnitude()) for p in rest]
    shift = [np.linalg.norm(p[:3, 3] - rest[0][:3, 3]) for p in rest]
    assert max(turn) < 0.5 and max(shift) < 0.001, (max(turn), max(shift))
    moved = poses[-1][:3, 3] - rest[-1][:3, 3]
    assert abs(moved[0] - 0.020) < 0.003 and np.linalg.norm(moved[1:]) < 0.003, moved


def test_an_object_in_the_gripper_is_where_the_arm_has_it_and_rests_where_it_is_let_go():
    """The gripper closes on the gamepad at frame 20 and lifts it 50 mm, then sets it down 60 mm along the tray at
    frame 40. The fingers hide most of it and its hidden tracks stay with the tray, so no group carries it; while
    held, its pose is the one the arm's joints give (hold), and let go (release) it rests in the world there, and its
    group carries it from there."""
    tray = _cloud(200, (0.0, 0.0, 0.45), (0.5, 0.35, 0.002))
    centre = np.array([0.10, 0.05, 0.43])
    pad = _cloud(30, centre, (0.072, 0.040, 0.010))
    t = GroupTracker()
    t.add_object("gamepad", np.arange(200, 230), _pose(centre))
    _run(t, [(np.vstack([tray, pad]), np.ones(230, bool))] * 20)
    rest = t.pose("gamepad")
    for k in range(20):  # lifted with the gripper: only a few of its points seen, moving with it
        arm = _pose(np.zeros(3))
        arm[:3, 3] = [0.06 * k / 19, 0.0, -0.05 * min(k, 10) / 10 + 0.05 * max(k - 10, 0) / 9]
        t.hold("gamepad", arm @ rest)
        pts, seen = np.vstack([tray, pad]), np.ones(230, bool)
        pts[200:205] += arm[:3, 3]
        seen[205:] = False
        _run(t, [(pts, seen)])
        assert np.allclose(t.pose("gamepad"), arm @ rest), "where the arm has it"
    t.release("gamepad")
    placed = pad + [0.06, 0.0, 0.0]
    _run(t, [(np.vstack([tray, placed]), np.ones(230, bool))] * 20)
    pose = t.pose("gamepad")
    assert t.objects["gamepad"].group == t.group_of[0], "it rests in the world"
    assert np.linalg.norm(pose[:3, 3] - (centre + [0.06, 0.0, 0.0])) < 0.003, pose[:3, 3]
