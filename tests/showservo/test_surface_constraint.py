"""Measured tilt, real rotations, missing surfaces and unobserved degrees of freedom."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.showservo import surface_constraint as sc

LIMITS = {"noise_m": 0.003, "tol_m": 0.003, "tol_deg": 3.0}


def scene(tilt=0.0, yaw=0.0):
    k = {"fx": 600.0, "fy": 600.0, "cx": 80.0, "cy": 72.0}
    centre = np.array([0.0, 0.0, 0.45])
    x, y = np.meshgrid(np.linspace(-0.025, 0.025, 32), np.linspace(-0.018, 0.018, 24))
    ref = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, centre[2])))
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xz", [tilt, yaw], degrees=True).as_matrix()
    pose[:3, 3] = centre - pose[:3, :3] @ centre
    cloud = ref @ pose[:3, :3].T + pose[:3, 3]
    uv = np.round(cloud[:, :2] / cloud[:, 2:] * 600 + [80, 72]).astype(np.int32)
    mask = np.zeros((144, 160), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(uv), 1)
    v, u = np.indices(mask.shape)
    rays = np.stack(((u - 80) / 600, (v - 72) / 600, np.ones_like(u)), axis=-1)
    n = pose[:3, :3] @ [0.0, 0.0, 1.0]
    depth = (n @ centre) / (rays @ n)
    depth[mask == 0] = 0
    return ref, depth, mask, k, pose, centre


def test_corrects_false_tilt_and_offset_but_keeps_tangential_position_and_twist():
    ref, depth, mask, k, _, centre = scene()
    raw = np.eye(4)
    raw[:3, :3] = Rotation.from_euler("xz", [15, 12], degrees=True).as_matrix()
    raw[:3, 3] = centre - raw[:3, :3] @ centre + [0.002, -0.001, 0.002]
    before = raw.copy()
    r = sc.constrain(raw, ref, depth, mask, k, **LIMITS)
    assert r["state"] == "corrected" and r["observed_dof"] == 3
    out = np.array(r["pose"])
    np.testing.assert_allclose(
        out[:3, :3], Rotation.from_euler("z", 12, degrees=True).as_matrix(), atol=1e-10
    )
    # Visibility changes the supported patch's centre slightly. Preserve the
    # tangential position at that measured anchor, not at an unseen cloud centre.
    anchor = np.array(r["reference_point"])
    expected = raw[:3, :3] @ anchor + raw[:3, 3]
    expected[2] = centre[2]
    np.testing.assert_allclose(out[:3, :3] @ anchor + out[:3, 3], expected, atol=1e-10)
    assert r["validation_normal_after_deg"] < 1e-5
    np.testing.assert_array_equal(raw, before)


def test_a_real_tilt_is_preserved_without_world_up_or_an_object_category():
    ref, depth, mask, k, pose, _ = scene(28, 19)
    r = sc.constrain(pose, ref, depth, mask, k, **LIMITS)
    assert r["pose"] is not None
    np.testing.assert_allclose(r["pose"], pose, atol=1e-9)


def test_hidden_dominant_face_cannot_override_the_visible_material_patch():
    ref, depth, mask, k, pose, centre = scene()
    y, z = np.meshgrid(np.linspace(-0.08, 0.08, 40), np.linspace(0.4, 0.5, 40))
    hidden_face = np.column_stack((np.full(y.size, 0.08), y.ravel(), z.ravel()))
    # This reference has more samples on a perpendicular face that is outside
    # the current object mask. The visible face itself did not rotate.
    r = sc.constrain(pose, np.concatenate((ref, hidden_face)), depth, mask, k, **LIMITS)
    assert r["pose"] is not None, r
    np.testing.assert_allclose(r["pose"], pose, atol=1e-10)
    assert r["visibility"]["visible_reference_points"] < r["visibility"]["reference_points"] / 2
    assert np.linalg.norm(np.array(r["reference_point"]) - centre) < 0.003


@pytest.mark.parametrize("name", ["early", "later"])
def test_recorded_visible_held_patch_survives_a_change_in_occluded_surface(name):
    with np.load(Path(__file__).parent / "fixtures/held_overlap_20261014.npz") as z:
        r = sc.resolve(
            z[f"{name}_pose"],
            z[f"{name}_ref"],
            z[f"{name}_depth"],
            z[f"{name}_mask"],
            json.loads(str(z["intr"])),
            **LIMITS,
        )
        point = np.array(r["reference_point"])
        raw_point = z[f"{name}_pose"][:3, :3] @ point + z[f"{name}_pose"][:3, 3]
    assert r["pose"] is not None and r["observed_dof"] == 3
    assert r["local_support_before"] > 0.9 and r["local_support_after"] > 0.9
    corrected = np.array(r["pose"])
    assert np.linalg.norm(corrected[:3, :3] @ point + corrected[:3, 3] - raw_point) < 0.003


def test_newly_visible_live_face_cannot_replace_the_matched_reference_face():
    with np.load(Path(__file__).parent / "fixtures/held_new_face_20261014.npz") as z:
        r = sc.constrain(z["pose"], z["ref"], z["depth"], z["mask"], json.loads(str(z["intr"])), **LIMITS)
    assert r["pose"] is not None, r
    # Whole-live fitting selected another face and rotated a good appearance
    # registration by 72 degrees. The shared patch agrees within reach tolerance.
    assert r["rotation_deg"] < LIMITS["tol_deg"]
    assert abs(r["normal_shift_mm"]) < LIMITS["tol_m"] * 1000
    assert r["visibility"]["common_live_points"] < r["visibility"]["live_points"]


@pytest.mark.parametrize("case", ["missing", "foreground", "narrow"])
def test_insufficient_or_foreground_geometry_cannot_certify_a_pose(case):
    ref, depth, mask, k, pose, _ = scene()
    if case == "missing":
        depth[:] = 0
    elif case == "foreground":
        depth[mask > 0] -= 0.06
    else:
        mask[:, :78] = 0
        mask[:, 85:] = 0
    r = sc.resolve(pose, ref, depth, mask, k, **LIMITS)
    assert r["state"] == "rejected" and r["pose"] is None


def test_nonplanar_surface_is_not_forced_to_its_dominant_viewing_direction():
    ref, depth, mask, k, pose, _ = scene()
    rng = np.random.default_rng(22)
    sphere = rng.normal(size=(3000, 3))
    sphere = sphere / np.linalg.norm(sphere, axis=1)[:, None] * 0.03 + [0, 0, 0.45]
    r = sc.constrain(pose, sphere, depth, mask, k, **LIMITS)
    assert r["pose"] is None


def test_small_curved_cap_is_not_mistaken_for_an_observable_rigid_face():
    # A 2 cm sphere's visible cap passes a simple majority-plane residual check.
    # Rotating a textured sphere need not rotate that cap's measured normal.
    k = {"fx": 600.0, "fy": 600.0, "cx": 80.0, "cy": 72.0}
    v, u = np.indices((144, 160))
    rays = np.stack(((u - 80) / 600, (v - 72) / 600, np.ones_like(u)), axis=-1)
    a = np.sum(rays * rays, axis=-1)
    discriminant = 0.9**2 - 4 * a * (0.45**2 - 0.01**2)
    mask = discriminant > 0
    depth = np.where(mask, (0.9 - np.sqrt(np.maximum(0, discriminant))) / (2 * a), 0)
    ref = sc.depth_points(depth, mask, k)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    pose[:3, 3] = np.array([0, 0, 0.45]) - pose[:3, :3] @ [0, 0, 0.45]
    assert sc.constrain(pose, ref, depth, mask, k, **LIMITS)["pose"] is None


def test_nonplanar_agreement_is_retained_but_a_conflict_is_not_silently_accepted():
    ref, depth, mask, k, pose, _ = scene()
    ref = ref[:2]  # no supported reference plane
    for state in ["consistent", "disagreement", "inconclusive", "unavailable"]:
        r = sc.resolve(pose, ref, depth, mask, k, surface={"state": state}, **LIMITS)
        assert (r["pose"] is not None) == (state == "consistent")


def cylinder_scene():
    k = {"fx": 600.0, "fy": 600.0, "cx": 80.0, "cy": 72.0}
    y, x = np.indices((144, 160))
    ray_y = (y - k["cy"]) / k["fy"]
    a = 1 + ray_y**2
    discriminant = 0.9**2 - 4 * a * (0.45**2 - 0.006**2)
    depth = (0.9 - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
    mask = (discriminant > 0) & (abs((x - k["cx"]) / k["fx"] * depth) < 0.05)
    depth[~mask] = 0
    return sc.depth_points(depth, mask, k), depth, mask, k


def test_narrow_curved_surface_constrains_direction_without_inventing_axial_roll():
    ref, depth, mask, k = cylinder_scene()
    centre = ref.mean(axis=0)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xz", [20, 2], degrees=True).as_matrix()
    pose[:3, 3] = centre - pose[:3, :3] @ centre
    r = sc.resolve(pose, ref, depth, mask, k, **LIMITS)
    assert r["observed_dof"] == 2 and r["pose"] is not None, r
    assert r["unobserved_rotation"] == "about the principal axis"
    out = np.array(r["pose"])
    # Rasterized depth and bounded subsampling perturb the fitted axis slightly.
    np.testing.assert_allclose(out[:3, :3], Rotation.from_euler("x", 20, degrees=True).as_matrix(), atol=1e-4)
    np.testing.assert_allclose(out[:3, :3] @ centre + out[:3, 3], centre, atol=1e-10)


@pytest.mark.parametrize("change", ["partial", "foreground", "missing"])
def test_axis_measurement_rejects_missing_or_different_surfaces(change):
    ref, depth, mask, k = cylinder_scene()
    if change == "partial":
        mask[:, :80] = False
    elif change == "foreground":
        depth[mask] -= 0.03
    else:
        depth[:] = 0
    assert sc.constrain_axis(np.eye(4), ref, depth, mask, k, **LIMITS)["pose"] is None


def test_an_isotropic_surface_has_no_axis_to_constrain():
    y, x = np.meshgrid(np.linspace(-0.02, 0.02, 50), np.linspace(-0.02, 0.02, 50))
    assert sc._long_axis(np.column_stack((x.ravel(), y.ravel(), np.ones(x.size))), 0.003) is None


def test_recorded_elongated_surface_has_supported_direction_despite_icp_roll_disagreement():
    with np.load(Path(__file__).parent / "fixtures/axis_surface_20261013.npz") as z:
        r = sc.resolve(z["pose"], z["ref"], z["depth"], z["mask"], json.loads(str(z["intr"])), **LIMITS)
    assert r["pose"] is not None and r["observed_dof"] == 2, r
    assert r["rotation_deg"] < 1
    assert r["axis_support_after"] > 0.9


@pytest.mark.parametrize("name", ["latest", "earlier", "clear", "baseline"])
def test_recorded_wrong_tilts_improve_on_held_out_geometry(name):
    with np.load(Path(__file__).parent / "fixtures/pose_disagreement_20261012.npz") as z:
        r = sc.resolve(
            z[f"{name}_pose"],
            z["reference"],
            z[f"{name}_depth"],
            z[f"{name}_mask"],
            json.loads(str(z[f"{name}_intr"])),
            **LIMITS,
        )
    # Even the baseline has a supported common patch once newly visible live
    # material is excluded. Require improvement on the split held-out geometry,
    # not the old whole-cloud rejection or just a successful return value.
    assert r["pose"] is not None, r
    assert r["reference_planar"] and r["live_planar"]
    assert r["validation_normal_after_deg"] < LIMITS["tol_deg"]
    assert r["validation_normal_after_deg"] < r["validation_normal_before_deg"]
    assert r["corrected_plane_rms_mm"] < r["raw_plane_rms_mm"]
    assert r["local_support_after"] >= r["local_support_before"]
