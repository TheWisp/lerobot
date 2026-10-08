# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The requirement, stated as tests: a top camera at an ARBITRARY angle and height,
an object with no model and no ground truth, and an error that is exact anyway.

The scene is built in world coordinates, viewed through a randomly placed camera, and
every 3D quantity the system uses is obtained the way the rig will obtain it — project
to pixels, read a depth, deproject. Nothing is handed the world frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.showservo.pose import (
    CameraIntrinsics,
    Rigid3,
    fit_rigid,
    fit_surface,
    fold_turn,
    main_plane_normal,
    ransac_fit_rigid,
    rotation_matrix,
    rotation_vector,
    sample_depth,
)
from lerobot.showservo.servo import servo_error_3d

INTR = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0)

# A target patch on a low object, and held-end features 8 cm ABOVE it — the exact
# configuration that cost the image-plane method 16 mm.
TARGET_W = np.array([[-0.03, -0.03, 0.04], [0.03, -0.03, 0.04], [0.03, 0.03, 0.045], [-0.03, 0.03, 0.04]])
HELD_W = np.array([[-0.012, 0.0, 0.12], [0.012, 0.0, 0.12], [0.0, 0.015, 0.128], [0.0, -0.014, 0.126]])


def _look_at(eye, at=(0.0, 0.0, 0.05), up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera rotation whose columns are the camera axes in world frame."""
    eye, at = np.asarray(eye, float), np.asarray(at, float)
    z = at - eye
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _random_camera(rng) -> tuple[np.ndarray, np.ndarray]:
    """An arbitrary viewpoint: random azimuth, random elevation, random distance."""
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(np.deg2rad(25), np.deg2rad(85))  # 25 deg is a steeply oblique view
    dist = rng.uniform(0.45, 0.9)
    eye = np.array([dist * np.cos(el) * np.cos(az), dist * np.cos(el) * np.sin(az), dist * np.sin(el)])
    return eye, _look_at(eye)


def _observe(points_w: np.ndarray, eye: np.ndarray, rot_c: np.ndarray) -> np.ndarray:
    """World -> what the rig measures -> camera-frame 3D, through pixels and depth."""
    cam = (np.asarray(points_w, float) - eye) @ rot_c
    assert (cam[:, 2] > 1e-6).all(), "scene must be in front of the camera"
    uv = INTR.project(cam)
    return INTR.deproject(uv, cam[:, 2])


def _world_move(points_w: np.ndarray, rotvec, t) -> np.ndarray:
    return Rigid3.from_rotvec(rotvec, t).apply(points_w)


# --- the algebra --------------------------------------------------------------------


def test_compose_and_inverse_are_consistent():
    rng = np.random.default_rng(0)
    a = Rigid3.from_rotvec(rng.normal(size=3) * 0.4, rng.normal(size=3))
    b = Rigid3.from_rotvec(rng.normal(size=3) * 0.4, rng.normal(size=3))
    pts = rng.normal(size=(9, 3))
    np.testing.assert_allclose(a.compose(b).apply(pts), a.apply(b.apply(pts)), atol=1e-12)
    np.testing.assert_allclose(a.inverse().apply(a.apply(pts)), pts, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.3, 1.5, np.pi - 1e-6, np.pi])
def test_rotation_vector_round_trips_including_the_pi_singularity(angle):
    axis = np.array([0.3, -0.7, 0.65])
    axis /= np.linalg.norm(axis)
    rot = rotation_matrix(axis * angle)
    back = rotation_vector(rot)
    assert np.isfinite(back).all()
    np.testing.assert_allclose(rotation_matrix(back), rot, atol=1e-7)


def test_fit_recovers_a_clean_rigid_motion_exactly():
    truth = Rigid3.from_rotvec([0.2, -0.4, 0.1], [0.05, -0.02, 0.11])
    fit, scale = fit_rigid(TARGET_W, truth.apply(TARGET_W), estimate_scale=True)
    np.testing.assert_allclose(fit.apply(TARGET_W), truth.apply(TARGET_W), atol=1e-12)
    assert scale == pytest.approx(1.0, abs=1e-9)


def test_a_reflection_never_fits_better_than_a_rotation():
    mirrored = TARGET_W * np.array([1.0, 1.0, -1.0])
    fit, _ = fit_rigid(TARGET_W, mirrored)
    assert np.linalg.det(fit.rot) == pytest.approx(1.0, abs=1e-9)


def test_coplanar_points_are_not_degenerate():
    # The trap on the monocular path: coplanar points break the essential matrix. With
    # depth-lifted points there is no such degeneracy, which is why flat objects viewed
    # from above are fine here and would not be there.
    flat = np.array([[0.0, 0.0, 0.05], [0.06, 0.0, 0.05], [0.06, 0.05, 0.05], [0.0, 0.05, 0.05]])
    truth = Rigid3.from_rotvec([0.0, 0.0, 0.5], [0.1, 0.05, 0.0])
    fit = ransac_fit_rigid(flat, truth.apply(flat))
    assert fit.ok and fit.n_inliers == 4
    np.testing.assert_allclose(fit.transform.apply(flat), truth.apply(flat), atol=1e-9)


def test_a_scaled_fit_is_reported_because_a_rigid_object_cannot_change_size():
    truth = Rigid3.from_rotvec([0.1, 0.0, 0.2], [0.02, 0.0, 0.0])
    bad_depth = truth.apply(TARGET_W) * 1.25  # a mis-scaled depth unit
    fit = ransac_fit_rigid(TARGET_W, bad_depth, inlier_m=1.0)
    assert fit.ok
    assert not fit.scale_is_plausible(), f"scale {fit.scale:.3f} should have been flagged"


def test_the_fit_survives_a_minority_of_flying_pixels():
    truth = Rigid3.from_rotvec([0.05, 0.1, -0.2], [0.03, 0.04, 0.02])
    src = np.vstack([TARGET_W, HELD_W])
    dst = truth.apply(src)
    dst[2] += [0.09, -0.07, 0.05]  # depth landed on the background behind the silhouette
    dst[5] += [-0.06, 0.08, -0.04]

    fit = ransac_fit_rigid(src, dst)
    assert fit.ok and fit.n_inliers == 6
    assert not fit.inliers[2] and not fit.inliers[5]


def test_the_fit_abstains_rather_than_guessing_below_four_points():
    truth = Rigid3.from_rotvec([0.1, 0.0, 0.0], [0.01, 0.0, 0.0])
    valid = np.array([True, True, True, False])
    fit = ransac_fit_rigid(TARGET_W, truth.apply(TARGET_W), valid)
    assert not fit.ok, "three points fit exactly and can never be caught being wrong"


def _sliding_majority(rng, angle_deg: float = 40.0):
    """The white-plug failure, distilled: a self-similar surface whose matches slide.

    20 distinctive points report the TRUE motion; 80 sliders' matches landed on
    look-alike texture at their OLD positions, so they coherently report "no motion".
    The z-translation of 30 mm keeps the two stories cleanly separated: under either
    hypothesis the other camp's points are off by centimetres, never millimetres.
    """
    truth = Rigid3.from_rotvec((0.0, 0.0, np.deg2rad(angle_deg)), (0.02, -0.01, 0.03))
    distinct = rng.uniform(-0.03, 0.03, size=(20, 3))
    sliders = rng.uniform(-0.03, 0.03, size=(80, 3))
    src = np.vstack([distinct, sliders])
    dst = np.vstack([truth.apply(distinct), sliders]) + rng.normal(0.0, 0.0005, size=(100, 3))
    ballot = np.zeros(100)
    ballot[:20] = 1.0
    return truth, src, dst, ballot


def test_a_sliding_majority_outvotes_the_truth_without_a_ballot():
    truth, src, dst, _ = _sliding_majority(np.random.default_rng(3))
    fit = ransac_fit_rigid(src, dst, inlier_m=0.004)
    assert fit.ok and fit.n_inliers >= 60, "the sliders' coherent lie wins the headcount"
    assert np.rad2deg(fit.transform.angle) < 5.0 < np.rad2deg(truth.angle)


@pytest.mark.parametrize("seed", range(4))
def test_the_ballot_recovers_the_rotation_the_sliding_majority_buried(seed):
    truth, src, dst, ballot = _sliding_majority(np.random.default_rng(seed))
    fit = ransac_fit_rigid(src, dst, inlier_m=0.004, hypo_weights=ballot)
    assert fit.ok
    assert np.rad2deg(fit.transform.angle) == pytest.approx(np.rad2deg(truth.angle), abs=2.0)
    assert fit.inliers[:20].sum() >= 16, "the fit must rest on the points that nominated it"


def test_a_ballot_too_thin_to_form_a_triple_is_plain_ransac():
    truth = Rigid3.from_rotvec([0.05, 0.1, -0.2], [0.03, 0.04, 0.02])
    src = np.vstack([TARGET_W, HELD_W])
    dst = truth.apply(src)
    thin = np.zeros(len(src))
    thin[:2] = 1.0  # two nominated points cannot propose a 3-point hypothesis
    plain = ransac_fit_rigid(src, dst)
    balloted = ransac_fit_rigid(src, dst, hypo_weights=thin)
    assert balloted.ok and plain.ok
    np.testing.assert_allclose(balloted.transform.rot, plain.transform.rot)
    np.testing.assert_allclose(balloted.transform.trans, plain.transform.trans)


def _mirrored_majority(rng, flip_deg: float = 175.0, true_deg: float = 8.0):
    """The mustard-bottle failure, distilled: a surface that looks like its own mirror.

    40 points' matches report the TRUE small turn; 60 landed on look-alike texture on
    the far side and coherently report the object flipped round. Under either story the
    other camp is off by centimetres, so RANSAC must pick one, and headcount picks the lie.
    """
    truth = Rigid3.from_rotvec((0.0, 0.0, np.deg2rad(true_deg)), (0.01, 0.0, 0.02))
    flipped = Rigid3.from_rotvec((0.0, 0.0, np.deg2rad(flip_deg)), (0.01, 0.0, 0.02))
    honest = rng.uniform(-0.04, 0.04, size=(40, 3))
    mirrored = rng.uniform(-0.04, 0.04, size=(60, 3))
    src = np.vstack([honest, mirrored])
    dst = np.vstack([truth.apply(honest), flipped.apply(mirrored)]) + rng.normal(0.0, 0.0005, size=(100, 3))
    return truth, src, dst


def test_a_mirrored_majority_flips_the_fit_without_a_prior():
    truth, src, dst = _mirrored_majority(np.random.default_rng(5))
    fit = ransac_fit_rigid(src, dst, inlier_m=0.004)
    assert fit.ok and np.rad2deg(fit.transform.angle) > 150.0, "headcount takes the flipped story"


@pytest.mark.parametrize("seed", range(4))
def test_the_last_motion_keeps_the_fit_among_poses_a_rigid_body_could_reach(seed):
    truth, src, dst = _mirrored_majority(np.random.default_rng(seed))
    # A moment ago the object had not moved: the flipped pose is 175 degrees away, unreachable.
    fit = ransac_fit_rigid(src, dst, inlier_m=0.004, prior=Rigid3.identity(), prior_rot_deg=45.0)
    assert fit.ok
    assert np.rad2deg(fit.transform.angle) == pytest.approx(np.rad2deg(truth.angle), abs=2.0)
    assert fit.inliers[:40].sum() >= 34, "the fit rests on the honest points"


def test_a_prior_nothing_reaches_is_an_abstention_not_a_guess():
    truth, src, dst = _mirrored_majority(np.random.default_rng(1))
    far = Rigid3.from_rotvec((np.deg2rad(90.0), 0.0, 0.0), (0.0, 0.0, 0.0))
    fit = ransac_fit_rigid(src, dst, inlier_m=0.004, prior=far, prior_rot_deg=20.0)
    assert not fit.ok
    # The translation bound is about where the points' centroid went, not the raw offset.
    fit = ransac_fit_rigid(
        src, dst, inlier_m=0.004, prior=Rigid3.identity(), prior_rot_deg=45.0, prior_trans_m=0.005
    )
    assert not fit.ok, "the truth carries the centroid 22 mm, past a 5 mm bound"


# --- the shape sets what a plain object's matches cannot ---------------------------

RIG = CameraIntrinsics(fx=604.2, fy=604.2, cx=419.2, cy=250.6)  # the rig's D435 at 848x480


def _plain_cube(rng, tilt_deg: float = 8.0, turn_deg: float = 40.0):
    """The lime cube of 2026-10-07, distilled: a 30 mm plain cube 0.62 m from an oblique camera looking at one of its
    corners, rendered pixel by pixel as the depth sees it (one point per pixel it covers, 0.3 mm along the ray), slid
    9 mm and turned 6 deg on the table since the reference view. Two wrong motions such as its matches gave: one tilted
    ``tilt_deg`` about the top face's centre, one turned ``turn_deg`` more about the cube's vertical axis.
    Post: (truth, tilted, turned, reference surface, live surface), camera frame."""
    el, az = np.radians(55.0), np.radians(-60.0)  # above, and off the front face toward the right one
    eye = np.array([0.62 * np.cos(el) * np.cos(az), 0.62 * np.cos(el) * np.sin(az), 0.62 * np.sin(el)])
    rot_c = _look_at(eye, (0.0, 0.0, 0.015))
    lo, hi = np.array([-0.015, -0.015, 0.0]), np.array([0.015, 0.015, 0.03])

    def render(pose: Rigid3) -> np.ndarray:
        """The camera-frame points of the cube placed by ``pose`` (cube frame -> world), one per pixel it covers."""
        centre = (pose.apply(np.array([[0.0, 0.0, 0.015]]))[0] - eye) @ rot_c
        u0, v0 = RIG.project(centre[None])[0]
        us, vs = np.meshgrid(np.arange(u0 - 70, u0 + 70), np.arange(v0 - 70, v0 + 70))
        rays = np.stack([(us - RIG.cx) / RIG.fx, (vs - RIG.cy) / RIG.fy, np.ones_like(us)], -1).reshape(-1, 3)
        o = pose.rot.T @ (eye - pose.trans)
        d = (rays @ rot_c.T) @ pose.rot  # each ray in the cube's frame
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        near, far = np.nanmax(np.minimum(t1, t2), axis=1), np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (far >= near) & (near > 0)
        pts = rays[hit] * near[hit, None]
        return pts * (1 + rng.normal(0.0, 0.0003, (len(pts), 1)) / np.linalg.norm(pts, axis=1, keepdims=True))

    def in_camera(m: Rigid3) -> Rigid3:
        corners = np.array([[x, y, z] for x in (-0.015, 0.015) for y in (-0.015, 0.015) for z in (0.0, 0.03)])
        return fit_rigid((corners - eye) @ rot_c, (m.apply(corners) - eye) @ rot_c)[0]

    def about(point, rotvec) -> Rigid3:
        turn = Rigid3.from_rotvec(rotvec, (0.0, 0.0, 0.0))
        return Rigid3(turn.rot, point - turn.rot @ point)

    def then(a: Rigid3, b: Rigid3) -> Rigid3:  # b after a
        return Rigid3(b.rot @ a.rot, b.rot @ a.trans + b.trans)

    slide = Rigid3.from_rotvec((0.0, 0.0, np.deg2rad(6.0)), (0.008, -0.005, 0.0))
    tilt = about(np.array([0.0, 0.0, 0.03]), (np.deg2rad(tilt_deg), 0.0, 0.0))
    spin = about(np.zeros(3), (0.0, 0.0, np.deg2rad(turn_deg)))
    reference, live = render(Rigid3.identity()), render(slide)
    return in_camera(slide), in_camera(then(tilt, slide)), in_camera(then(spin, slide)), reference, live


def _degrees_apart(a: Rigid3, b: Rigid3) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.rot @ b.rot.T) - 1.0) / 2.0, -1.0, 1.0))))


# A surface pins its position along a face only through that face's edges: on this cube the fit's cost is flat over
# a few millimetres there (lower at the fit than at the truth), so the position is held to 5 mm and the turn to 2 deg.


@pytest.mark.parametrize("seed", range(3))
def test_the_surface_fit_finds_the_turn_a_plain_cubes_matches_got_wrong(seed):
    truth, _tilted, turned, reference, live = _plain_cube(np.random.default_rng(seed))
    assert _degrees_apart(turned, truth) > 35.0
    motion, cost = fit_surface(reference, live, turned, near_m=0.003)
    assert _degrees_apart(motion, truth) < 2.0
    assert np.linalg.norm(motion.trans - truth.trans) < 0.005
    assert cost < 0.001


@pytest.mark.parametrize("seed", range(3))
def test_the_surface_fit_sets_the_tilt_the_matches_left_loose(seed):
    truth, tilted, _turned, reference, live = _plain_cube(np.random.default_rng(seed))
    motion, _cost = fit_surface(reference, live, tilted, near_m=0.003)
    assert _degrees_apart(motion, truth) < 2.0


def test_a_surface_fit_started_at_the_truth_stays_there():
    truth, _tilted, _turned, reference, live = _plain_cube(np.random.default_rng(7))
    motion, _cost = fit_surface(reference, live, truth, near_m=0.003)
    assert _degrees_apart(motion, truth) < 1.0
    assert np.linalg.norm(motion.trans - truth.trans) < 0.005


def test_the_main_plane_is_the_one_most_points_lie_on():
    rng = np.random.default_rng(3)
    big = np.column_stack([rng.uniform(-0.02, 0.02, (600, 2)), np.full(600, 0.5)])  # facing the camera
    small = np.column_stack([rng.uniform(-0.02, 0.02, 200), np.full(200, 0.02), rng.uniform(0.48, 0.52, 200)])
    n = main_plane_normal(np.vstack([big, small]))
    assert abs(n[2]) > 0.999


# --- depth sampling -----------------------------------------------------------------


def test_depth_is_sampled_nearest_so_silhouettes_do_not_invent_surfaces():
    # Foreground 0.40 m, background 0.90 m, a hard edge between them. Bilinear would
    # return ~0.65 m at the boundary — a surface that exists nowhere.
    depth = np.full((20, 20), 0.90)
    depth[:, :10] = 0.40
    z, valid = sample_depth(depth, np.array([[9.4, 5.0], [9.6, 5.0]]))
    assert valid.all()
    assert set(np.round(z, 2)) <= {0.40, 0.90}


def test_invalid_depth_is_reported_not_silently_zero():
    depth = np.full((20, 20), 0.5)
    depth[5, 5] = 0.0  # RealSense "no return"
    depth[6, 6] = 8.0  # beyond range
    z, valid = sample_depth(depth, np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [100.0, 3.0]]))
    assert list(valid) == [False, False, True, False]
    assert z[2] == pytest.approx(0.5)


# --- the requirement ----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_the_error_is_zero_when_the_taught_relation_is_reproduced_from_any_viewpoint(seed):
    """Arbitrary camera, no object model, held features 8 cm off the target's plane."""
    rng = np.random.default_rng(seed)
    eye, rot_c = _random_camera(rng)

    taught_target = _observe(TARGET_W, eye, rot_c)
    taught_held = _observe(HELD_W, eye, rot_c)

    # The object is somewhere else entirely, and the held end came with it.
    move = (rng.normal(size=3) * np.array([0.02, 0.02, 0.6]), rng.uniform(-0.12, 0.12, size=3))
    live_target = _observe(_world_move(TARGET_W, *move), eye, rot_c)
    live_held = _observe(_world_move(HELD_W, *move), eye, rot_c)

    target_fit = ransac_fit_rigid(taught_target, live_target)
    held_fit = ransac_fit_rigid(taught_held, live_held)
    err = servo_error_3d(taught_held, target_fit, held_fit)

    assert err.ok, err.reason
    assert err.norm < 1e-9, f"{err.norm * 1000:.4f} mm"
    assert err.angle < 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_the_measured_error_is_the_same_physical_quantity_from_every_viewpoint(seed):
    """Camera pose must not enter the difference — the claim, tested directly."""
    rng = np.random.default_rng(100 + seed)
    move = (rng.normal(size=3) * np.array([0.02, 0.02, 0.5]), rng.uniform(-0.10, 0.10, size=3))
    # The held end did NOT follow the object: a real, non-zero error to measure.
    slip = (np.zeros(3), np.array([0.013, -0.009, 0.021]))
    moved_target = _world_move(TARGET_W, *move)
    moved_held = _world_move(_world_move(HELD_W, *move), *slip)

    world_errors = []
    for cam_seed in range(5):
        eye, rot_c = _random_camera(np.random.default_rng(1000 + cam_seed))
        taught_held = _observe(HELD_W, eye, rot_c)
        err = servo_error_3d(
            taught_held,
            ransac_fit_rigid(_observe(TARGET_W, eye, rot_c), _observe(moved_target, eye, rot_c)),
            ransac_fit_rigid(taught_held, _observe(moved_held, eye, rot_c)),
        )
        assert err.ok, err.reason
        world_errors.append(rot_c @ err.e_t)  # camera frame -> world, for comparison

    spread = np.abs(np.array(world_errors) - np.mean(world_errors, axis=0)).max()
    assert spread < 1e-9, f"viewpoint changed the answer by {spread * 1000:.5f} mm"
    np.testing.assert_allclose(np.mean(world_errors, axis=0), -slip[1], atol=1e-9)


def test_the_case_that_broke_the_image_plane_method_is_now_exact():
    # Held features 8 cm above the target's plane, an 11 cm displacement, and a camera
    # that is deliberately NOT perpendicular. The 2D transport was wrong by 16 mm here.
    eye, rot_c = np.array([0.35, 0.20, 0.42]), None
    rot_c = _look_at(eye)
    move = (np.array([0.0, 0.0, np.deg2rad(30)]), np.array([0.10, 0.05, 0.0]))

    taught_held = _observe(HELD_W, eye, rot_c)
    err = servo_error_3d(
        taught_held,
        ransac_fit_rigid(_observe(TARGET_W, eye, rot_c), _observe(_world_move(TARGET_W, *move), eye, rot_c)),
        ransac_fit_rigid(taught_held, _observe(_world_move(HELD_W, *move), eye, rot_c)),
    )
    assert err.ok
    assert err.norm * 1000 < 0.001, f"{err.norm * 1000:.5f} mm"


def _patch(rng, n: int, centre, spread=(0.035, 0.035, 0.004)) -> np.ndarray:
    return np.asarray(centre, float) + rng.uniform(-1, 1, size=(n, 3)) * np.asarray(spread, float)


def test_realistic_depth_noise_degrades_gracefully_rather_than_lying():
    # §4 sizes a team at ~8 points, which is what makes a fit noise-tolerant: the
    # residual averages down while the consensus set stays large enough to certify.
    rng = np.random.default_rng(7)
    eye, rot_c = _random_camera(rng)
    target_w = _patch(rng, 8, [0.0, 0.0, 0.042])
    held_w = _patch(rng, 6, [0.0, 0.0, 0.122], spread=(0.014, 0.014, 0.004))
    move = (np.array([0.0, 0.0, 0.4]), np.array([0.08, -0.05, 0.0]))

    def noisy(pts):  # ~2 mm RMS, roughly RealSense at half a metre
        return pts + rng.normal(scale=0.002, size=pts.shape)

    taught_held = noisy(_observe(held_w, eye, rot_c))
    err = servo_error_3d(
        taught_held,
        ransac_fit_rigid(
            noisy(_observe(target_w, eye, rot_c)), noisy(_observe(_world_move(target_w, *move), eye, rot_c))
        ),
        ransac_fit_rigid(taught_held, noisy(_observe(_world_move(held_w, *move), eye, rot_c))),
    )
    assert err.ok, err.reason
    assert err.norm < 0.006, f"{err.norm * 1000:.2f} mm from 2 mm depth noise"


def test_a_minimum_size_team_under_noise_abstains_rather_than_fitting_badly():
    """Fail-closed, and a sizing rule: a 4-point team has no margin for depth noise.

    With ``min_points`` equal to the team size, every point must land inside the
    consensus band, so ordinary RealSense noise makes the fit fail. That is the
    certificate behaving correctly — and the reason the compiler must extract enough
    points per team rather than the algebraic minimum.
    """
    rng = np.random.default_rng(3)
    eye, rot_c = _random_camera(rng)
    move = (np.array([0.0, 0.0, 0.3]), np.array([0.06, -0.04, 0.0]))
    noisy = lambda p: p + rng.normal(scale=0.002, size=p.shape)  # noqa: E731

    fit = ransac_fit_rigid(
        noisy(_observe(TARGET_W, eye, rot_c)), noisy(_observe(_world_move(TARGET_W, *move), eye, rot_c))
    )
    assert not fit.ok, "a 4-point team should not certify itself under 2 mm noise"


def _mean_transport_error(probe_height: float, n_points: int, sigma: float, seeds: int = 12) -> float:
    """Error, in metres, when the target's fit is used to transport a point at
    ``probe_height`` — averaged over noise draws and viewpoints so the comparison is
    about geometry rather than about one lucky seed."""
    move = (np.array([0.0, 0.0, 0.4]), np.array([0.08, -0.05, 0.0]))
    probe_w = np.array([[0.0, 0.0, probe_height]])
    errors = []
    for s in range(seeds):
        rng = np.random.default_rng(500 + s)
        eye, rot_c = _random_camera(rng)
        pts = _patch(rng, n_points, [0.0, 0.0, 0.042])

        def noisy(p, r=rng):
            return p + r.normal(scale=sigma, size=p.shape)

        fit = ransac_fit_rigid(
            noisy(_observe(pts, eye, rot_c)), noisy(_observe(_world_move(pts, *move), eye, rot_c))
        )
        if not fit.ok:
            continue
        predicted = fit.transform.apply(_observe(probe_w, eye, rot_c))
        errors.append(float(np.linalg.norm(predicted - _observe(_world_move(probe_w, *move), eye, rot_c))))
    assert errors, "no fit succeeded; the comparison would be vacuous"
    return float(np.mean(errors))


def test_more_points_per_team_buy_accuracy():
    """The compiler's sizing rule, measured rather than asserted."""
    few = _mean_transport_error(0.122, n_points=8, sigma=0.002)
    many = _mean_transport_error(0.122, n_points=16, sigma=0.002)
    assert many < few, f"16 points ({many * 1000:.2f} mm) should beat 8 ({few * 1000:.2f} mm)"


def test_error_grows_with_the_lever_arm_from_the_patch():
    """Where to put the held features, correctly this time.

    Residual rotation error in the target's fit is amplified by the distance from the
    patch to whatever the fit transports. This is the honest version of an earlier,
    wrong intuition about coplanarity: what matters is not sharing a plane in the
    image, it is the LEVER ARM in 3D. Tracking the fingertips still turns out to be
    right, for this reason instead of that one.
    """
    near = _mean_transport_error(0.06, n_points=10, sigma=0.002)
    far = _mean_transport_error(0.30, n_points=10, sigma=0.002)
    assert far > 2.0 * near, f"near {near * 1000:.2f} mm vs far {far * 1000:.2f} mm"


@pytest.mark.parametrize("broken", ["target", "held"])
def test_a_failed_fit_abstains(broken):
    from lerobot.showservo.pose import RigidFit

    good = ransac_fit_rigid(TARGET_W, TARGET_W)
    good_held = ransac_fit_rigid(HELD_W, HELD_W)
    dead = RigidFit(ok=False)
    err = servo_error_3d(
        HELD_W, dead if broken == "target" else good, dead if broken == "held" else good_held
    )
    assert not err.ok and err.reason


def test_implausible_depth_scale_abstains_instead_of_servoing_on_a_lie():
    stretched = ransac_fit_rigid(TARGET_W, TARGET_W * 1.3, inlier_m=1.0)
    err = servo_error_3d(HELD_W, stretched, ransac_fit_rigid(HELD_W, HELD_W))
    assert not err.ok
    assert "implausible" in err.reason


def _turned(deg: float, about=(0.02, -0.01, 0.4)) -> Rigid3:
    """A motion that turns an object resting at ``about`` by ``deg`` about the vertical, slid 5 mm and tilted 3 deg."""
    c = np.asarray(about)
    rot = rotation_matrix(np.array([np.deg2rad(3.0), 0.0, 0.0])) @ rotation_matrix(
        np.array([0.0, 0.0, np.deg2rad(deg)])
    )
    return Rigid3(rot, c + np.array([0.005, 0.0, 0.0]) - rot @ c)


def _vertical_turn(m: Rigid3) -> float:
    return float(np.degrees(np.arctan2(m.rot[1, 0], m.rot[0, 0])))


@pytest.mark.parametrize(
    ("order", "deg", "folded"),
    [
        (1, 172.0, 172.0),
        (2, 172.0, -8.0),
        (2, -8.0, -8.0),
        (4, 131.0, 41.0),
        (4, 41.0, 41.0),
        (4, -139.0, 41.0),
    ],
)
def test_a_symmetric_objects_turn_is_folded_to_the_least(order, deg, folded):
    """A cube turned 131 deg looks the same as one turned 41; a gamepad's outline turned 172 the same as one turned -8.
    The find reports the least of the turns it cannot tell apart, and where the object is stays put."""
    m = _turned(deg)
    centre = m.apply(np.array([[0.02, -0.01, 0.4]]))[0]  # where the motion puts the object's centre
    out = fold_turn(m, np.array([0.0, 0.0, 1.0]), centre, order)
    assert abs((_vertical_turn(out) - folded + 180.0) % 360.0 - 180.0) < 1.5
    assert np.allclose(out.apply(np.array([[0.02, -0.01, 0.4]]))[0], centre, atol=1e-9), (
        "the object stays where it is"
    )


def _slab(rng):
    """The gamepad of 2026-10-08, distilled: a 64 x 32 x 12 mm slab with a 4 mm rise over its last 20 mm, 0.62 m from
    a camera 80 deg above the table, slid 9 mm and turned 6 deg since the reference view, rendered pixel by pixel. The
    rise sits at the other end in the live view, so the live surface is laid on best turned end to end, as the real
    gamepad's was. Post: (truth, reference surface, live surface), camera frame."""
    el, az = np.radians(80.0), np.radians(-60.0)
    eye = np.array([0.62 * np.cos(el) * np.cos(az), 0.62 * np.cos(el) * np.sin(az), 0.62 * np.sin(el)])
    rot_c = _look_at(eye, (0.0, 0.0, 0.007))
    slab = (np.array([-0.032, -0.016, 0.0]), np.array([0.032, 0.016, 0.012]))

    def rise(end):
        return (np.array([0.022 * end - 0.010, -0.016, 0.0]), np.array([0.022 * end + 0.010, 0.016, 0.016]))

    def render(pose: Rigid3, boxes) -> np.ndarray:
        centre = (pose.apply(np.array([[0.0, 0.0, 0.007]]))[0] - eye) @ rot_c
        u0, v0 = RIG.project(centre[None])[0]
        us, vs = np.meshgrid(np.arange(u0 - 90, u0 + 90), np.arange(v0 - 90, v0 + 90))
        rays = np.stack([(us - RIG.cx) / RIG.fx, (vs - RIG.cy) / RIG.fy, np.ones_like(us)], -1).reshape(-1, 3)
        o = pose.rot.T @ (eye - pose.trans)
        d = (rays @ rot_c.T) @ pose.rot
        best = np.full(len(rays), np.inf)
        for lo, hi in boxes:
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            near, far = np.nanmax(np.minimum(t1, t2), axis=1), np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (far >= near) & (near > 0)
            best = np.where(hit, np.minimum(best, near), best)
        hit = np.isfinite(best)
        pts = rays[hit] * best[hit, None]
        return pts * (1 + rng.normal(0.0, 0.0003, (len(pts), 1)) / np.linalg.norm(pts, axis=1, keepdims=True))

    slide = Rigid3.from_rotvec((0.0, 0.0, np.deg2rad(6.0)), (0.008, -0.005, 0.0))
    corners = np.array([[x, y, z] for x in (-0.032, 0.032) for y in (-0.016, 0.016) for z in (0.0, 0.012)])
    truth = fit_rigid((corners - eye) @ rot_c, (slide.apply(corners) - eye) @ rot_c)[0]
    return truth, render(Rigid3.identity(), [slab, rise(+1)]), render(slide, [slab, rise(-1)])


def test_a_gamepad_declared_order_2_is_found_the_same_whichever_end_its_surface_prefers():
    """Taught four times on one still scene, the gamepad's find came out 7, 7, 173 and 8 deg from the demo view; on 12
    real frames its live surface was laid on best turned end to end every time. Declared order 2, the end-to-end
    turn is folded away and the find's turn about the axis the slab rests on agrees with the truth; undeclared, it
    stays turned round. (The reversed fit also tilts to lay the moved rise, 8-13 deg here: that is the synthetic rise,
    not the fold.)"""
    for seed in range(4):
        truth, reference, live = _slab(np.random.default_rng(seed))
        fit, _ = fit_surface(reference, live, truth, near_m=0.004)
        assert _degrees_apart(fit, truth) > 150.0, "precondition: the surface alone turns the slab end to end"
        axis, centre = main_plane_normal(live), live.mean(axis=0)
        assert _degrees_apart(fold_turn(fit, axis, centre, 1), truth) > 150.0, (
            "no symmetry declared: as fitted"
        )
        rest = rotation_vector(fold_turn(fit, axis, centre, 2).rot @ truth.rot.T)
        assert abs(np.degrees(rest @ axis)) < 2.0, f"seed {seed}: order 2 folds the turn back"
