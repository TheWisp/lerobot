"""Recorded false tilts, actual object motion, missing evidence and fixed-frame comparisons."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.showservo import pose_verification as pv

LIMITS = {"tol_m": 0.003, "tol_deg": 3.0}
CENTRE = np.array([0.12, -0.05, 0.45])


def turn(degrees, centre=CENTRE):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("x", degrees, degrees=True).as_matrix()
    t[:3, 3] = centre - t[:3, :3] @ centre
    return t


def test_rotation_is_measured_even_when_object_centre_does_not_move():
    d = pv.difference(np.eye(4), turn(15), CENTRE, **LIMITS)
    assert d["state"] == "disagreement" and d["rotation_deg"] == pytest.approx(15)
    assert d["centre_mm"] < 1e-10


def test_group_reference_does_not_rebase_on_small_p2p_updates():
    check = pv.GroupReference()
    args = {"centre": CENTRE, "epoch": ("session", "designation"), **LIMITS}
    assert check.compare(np.eye(4), np.eye(4), stamp=0, **args)["state"] == "baseline"
    for i in range(1, 7):
        r = check.compare(turn(i), np.eye(4), stamp=i, **args)
        assert r["reference_t"] == 0
    assert r["rotation_deg"] == pytest.approx(6) and r["state"] == "disagreement"


def test_real_group_and_p2p_motion_agree_despite_different_reference_frames():
    g0, p0 = turn(-30), turn(25)
    g0[:3, 3] += [0.2, 0.1, -0.1]
    p0[:3, 3] += [0.02, 0, 0]
    check = pv.GroupReference()
    check.compare(p0, g0, CENTRE, 0, "one", **LIMITS)
    motion = turn(35)
    motion[:3, 3] += [0.1, -0.1, 0.03]
    r = check.compare(motion @ p0, motion @ g0, CENTRE, 1, "one", **LIMITS)
    assert r["state"] == "consistent" and r["centre_mm"] < 1e-10
    assert r["rotation_deg"] < 1e-5


def test_missing_groups_are_unknown_and_redesignation_starts_a_new_reference():
    c = pv.GroupReference()
    c.compare(np.eye(4), np.eye(4), CENTRE, 1, "old", **LIMITS)
    assert c.compare(turn(10), None, CENTRE, 2, "old", **LIMITS)["state"] == "unavailable"
    r = c.compare(turn(10), np.eye(4), CENTRE, 3, "new", **LIMITS)
    assert r["state"] == "baseline" and r["reference_t"] == 3


def scene(degrees=0):
    """A measured plane truly rotated, with no world-up assumption in the check."""
    intr = {"fx": 600.0, "fy": 600.0, "cx": 80.0, "cy": 72.0}
    centre = np.array([0.0, 0.0, 0.45])
    x, y = np.meshgrid(np.linspace(-0.022, 0.022, 20), np.linspace(-0.017, 0.017, 20))
    ref = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.45)])
    pose = turn(degrees, centre)
    cloud = ref @ pose[:3, :3].T + pose[:3, 3]
    uv = np.round(cloud[:, :2] / cloud[:, 2:] * 600 + [80, 72]).astype(np.int32)
    mask = np.zeros((144, 160), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(uv), 1)
    v, u = np.indices(mask.shape)
    rays = np.stack([(u - 80) / 600, (v - 72) / 600, np.ones_like(u)], axis=-1)
    n = pose[:3, :3] @ [0.0, 0.0, 1.0]
    depth = (n @ centre) / (rays @ n)
    depth[mask == 0] = 0
    return ref, depth, mask, intr, pose


def test_surface_detects_false_tilt_with_a_stationary_centre():
    ref, depth, mask, intr, _ = scene()
    wrong = turn(15, np.array([0.0, 0.0, 0.45]))
    before = wrong.copy()
    r = pv.surface(wrong, ref, depth, mask, intr, noise_m=0.003, **LIMITS)
    assert r["state"] == "disagreement" and r["rotation_deg"] > 12
    assert r["refit_rms_mm"] < r["raw_rms_mm"]
    np.testing.assert_array_equal(wrong, before)  # diagnostic does not align the control pose


def test_surface_does_not_force_a_truly_tilted_object_level():
    ref, depth, mask, intr, pose = scene(25)
    r = pv.surface(pose, ref, depth, mask, intr, noise_m=0.003, **LIMITS)
    assert r["state"] == "consistent" and r["rotation_deg"] < 1


@pytest.mark.parametrize("missing", ["mask", "depth", "foreground"])
def test_missing_or_occluded_surface_is_not_reported_as_agreement(missing):
    ref, depth, mask, intr, pose = scene()
    if missing == "mask":
        mask[:] = 0
    elif missing == "depth":
        depth[:] = 0
    else:
        depth[mask > 0] -= 0.1
    r = pv.surface(pose, ref, depth, mask, intr, noise_m=0.003, **LIMITS)
    assert r["state"] == "inconclusive"


@pytest.mark.parametrize("name", ["latest", "earlier", "clear", "baseline"])
def test_recorded_clear_cube_failures_and_initial_find(name):
    with np.load(Path(__file__).parent / "fixtures/pose_disagreement_20261012.npz") as z:
        r = pv.surface(
            z[f"{name}_pose"],
            z["reference"],
            z[f"{name}_depth"],
            z[f"{name}_mask"],
            json.loads(str(z[f"{name}_intr"])),
            noise_m=0.003,
            **LIMITS,
        )
    assert r["state"] == ("consistent" if name == "baseline" else "disagreement")
    if name != "baseline":
        assert r["rotation_deg"] > 6 and r["centre_mm"] < 3


def test_invalid_pose_is_not_a_surface_measurement():
    ref, depth, mask, intr, pose = scene()
    pose[0, 0] = np.nan
    assert pv.surface(pose, ref, depth, mask, intr, noise_m=0.003, **LIMITS)["state"] == "unavailable"
