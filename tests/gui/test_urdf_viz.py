"""Tests for the URDF state visualization (``lerobot.gui.urdf_viz``).

Covers the correctness-critical pieces: the motor->URDF angle conversion,
robot resolution from an observation's motor set, the per-arm angle
computation, the ``/api/run/urdf-viz`` endpoint glue, and the integrity of
every vendored ``*_description`` package (URDF parses, declared joints
exist, referenced meshes are on disk).
"""

import asyncio
import importlib
import math
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from lerobot.gui.urdf_viz import (
    JointCalibration,
    _discover_descriptions,
    compute_joint_angles,
    obs_to_urdf_rad,
    resolve_robot,
)

# Motor sets of the two vendored robots. SO-101's six motors are a subset of
# SO-107's seven (SO-107 adds forearm_roll) — which is exactly what the
# "most specific match wins" rule in resolve_robot must disambiguate.
SO101_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SO107_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _pos_keys(motors, prefix=""):
    """Observation ``<motor>.pos`` keys for the given motors, optionally prefixed."""
    return [f"{prefix}{m}.pos" for m in motors]


# ============================================================================
# obs_to_urdf_rad — the motor->URDF angle conversion
# ============================================================================


class TestObsToUrdfRad:
    """The layer-2 conversion ``urdf_rad = deg2rad(sign * pos + offset_deg)``."""

    def test_identity_calibration(self):
        assert obs_to_urdf_rad(90.0, JointCalibration()) == pytest.approx(math.radians(90.0))

    def test_zero_stays_zero(self):
        assert obs_to_urdf_rad(0.0, JointCalibration()) == 0.0

    def test_sign_flip(self):
        assert obs_to_urdf_rad(90.0, JointCalibration(sign=-1.0)) == pytest.approx(math.radians(-90.0))

    def test_offset_only(self):
        assert obs_to_urdf_rad(0.0, JointCalibration(offset_deg=-90.0)) == pytest.approx(math.radians(-90.0))

    def test_sign_and_offset_combined(self):
        # urdf_deg = -1 * 45 + 78.87 = 33.87
        got = obs_to_urdf_rad(45.0, JointCalibration(sign=-1.0, offset_deg=78.87))
        assert got == pytest.approx(math.radians(33.87))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_input_raises(self, bad):
        with pytest.raises(AssertionError):
            obs_to_urdf_rad(bad, JointCalibration())


class TestJointCalibration:
    def test_defaults_are_identity(self):
        c = JointCalibration()
        assert c.sign == 1.0
        assert c.offset_deg == 0.0


# ============================================================================
# resolve_robot — matching a live robot to a description package
# ============================================================================


class TestResolveRobot:
    def test_resolves_so101_from_six_motors(self):
        spec = resolve_robot(_pos_keys(SO101_MOTORS))
        assert spec is not None
        assert spec.name == "SO-101"
        assert spec.urdf_url_path.startswith("so101_description/")
        assert len(spec.arms) == 1
        assert spec.arms[0].obs_prefix == ""

    def test_resolves_so107_from_seven_motors(self):
        spec = resolve_robot(_pos_keys(SO107_MOTORS))
        assert spec is not None
        assert spec.name == "SO-107"
        assert len(spec.arms) == 1

    def test_most_specific_match_wins(self):
        # The seven-motor set is a superset of SO-101's six; SO-107 must win.
        spec = resolve_robot(_pos_keys(SO107_MOTORS))
        assert spec is not None
        assert spec.name == "SO-107"

    def test_bimanual_detected_from_left_prefix(self):
        keys = _pos_keys(SO107_MOTORS, "left_") + _pos_keys(SO107_MOTORS, "right_")
        spec = resolve_robot(keys)
        assert spec is not None
        assert len(spec.arms) == 2
        assert {a.obs_prefix for a in spec.arms} == {"left_", "right_"}

    def test_extra_non_motor_keys_ignored(self):
        keys = _pos_keys(SO101_MOTORS) + ["observation.images.cam", "timestamp", "task"]
        spec = resolve_robot(keys)
        assert spec is not None
        assert spec.name == "SO-101"

    def test_no_pos_keys_returns_none(self):
        assert resolve_robot(["observation.images.cam", "task"]) is None
        assert resolve_robot([]) is None

    def test_unknown_motor_set_returns_none(self):
        assert resolve_robot(["mystery_a.pos", "mystery_b.pos"]) is None

    def test_partial_motor_set_returns_none(self):
        # A strict subset of a known robot's motors is not enough to match it.
        assert resolve_robot(_pos_keys(SO101_MOTORS[:3])) is None


# ============================================================================
# compute_joint_angles — per-arm URDF angles from an observation
# ============================================================================


class TestComputeJointAngles:
    def test_so101_identity_mapping(self):
        spec = resolve_robot(_pos_keys(SO101_MOTORS))
        obs = {f"{m}.pos": 10.0 for m in SO101_MOTORS}
        angles = compute_joint_angles(spec, obs)
        assert set(angles.keys()) == {""}
        # SO-101's URDF joint names equal its motor names; alignment is identity.
        for m in SO101_MOTORS:
            assert angles[""][m] == pytest.approx(math.radians(10.0))

    def test_so107_alignment_is_applied(self):
        spec = resolve_robot(_pos_keys(SO107_MOTORS))
        obs = {f"{m}.pos": 0.0 for m in SO107_MOTORS}
        joints = compute_joint_angles(spec, obs)[""]
        # Unimanual SO-107 reuses the right-arm alignment; shoulder_lift there
        # is (sign +1, offset -90), so pos 0 -> -90 deg. Its URDF joint is S2.
        assert joints["S2"] == pytest.approx(math.radians(-90.0))

    def test_partial_observation_omits_missing_joint(self):
        spec = resolve_robot(_pos_keys(SO101_MOTORS))
        obs = {f"{m}.pos": 5.0 for m in SO101_MOTORS if m != "gripper"}
        joints = compute_joint_angles(spec, obs)[""]
        assert "gripper" not in joints
        assert "shoulder_pan" in joints

    def test_bimanual_arms_use_distinct_alignments(self):
        keys = _pos_keys(SO107_MOTORS, "left_") + _pos_keys(SO107_MOTORS, "right_")
        spec = resolve_robot(keys)
        obs = dict.fromkeys(keys, 0.0)
        angles = compute_joint_angles(spec, obs)
        assert set(angles.keys()) == {"left_", "right_"}
        # gripper (URDF joint S7): left is (+1, -90), right is (-1, 0).
        assert angles["left_"]["S7"] == pytest.approx(math.radians(-90.0))
        assert angles["right_"]["S7"] == pytest.approx(math.radians(0.0))


# ============================================================================
# /api/run/urdf-viz endpoint
# ============================================================================


def _call_endpoint():
    from lerobot.gui.api.run import urdf_viz_state

    return asyncio.run(urdf_viz_state())


class TestUrdfVizEndpoint:
    def test_unavailable_when_no_reader(self):
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=None):
            assert _call_endpoint() == {"available": False}

    def test_unavailable_when_no_observation(self):
        reader = MagicMock()
        reader.read_obs.return_value = []
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            assert _call_endpoint() == {"available": False}

    def test_unavailable_for_unknown_robot(self):
        reader = MagicMock()
        reader.read_obs.return_value = [{"mystery_a.pos": 1.0, "mystery_b.pos": 2.0}]
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            assert _call_endpoint() == {"available": False}

    def test_returns_urdf_and_angles_for_known_robot(self):
        reader = MagicMock()
        reader.read_obs.return_value = [{f"{m}.pos": 0.0 for m in SO107_MOTORS}]
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            result = _call_endpoint()
        assert result["available"] is True
        assert result["name"] == "SO-107"
        assert result["urdf"].startswith("/urdf-assets/so107_description/")
        assert result["bimanual"] is False
        assert len(result["arms"]) == 1
        assert result["arms"][0]["prefix"] == ""
        # Angles are radians, keyed by URDF joint name.
        assert result["arms"][0]["joints"]["S2"] == pytest.approx(math.radians(-90.0))


# ============================================================================
# Vendored *_description packages — URDF + mesh integrity
# ============================================================================

_DESCRIPTIONS = _discover_descriptions()


def test_vendored_descriptions_are_discovered():
    names = {pkg for pkg, _ in _DESCRIPTIONS}
    assert "so101_description" in names
    assert "so107_description" in names


def _urdf_path(pkg_name):
    return importlib.import_module(f"lerobot.robots.{pkg_name}").get_urdf_path()


@pytest.mark.parametrize(
    ("pkg_name", "viz_spec"),
    _DESCRIPTIONS,
    ids=[d[0] for d in _DESCRIPTIONS],
)
class TestDescriptionAssets:
    """Integrity checks run against every discovered ``*_description`` package."""

    def test_viz_spec_shape(self, pkg_name, viz_spec):
        assert isinstance(viz_spec["name"], str) and viz_spec["name"]
        assert len(viz_spec["motors"]) == len(viz_spec["urdf_joints"]), (
            "motors and urdf_joints must be parallel sequences"
        )
        assert len(set(viz_spec["motors"])) == len(viz_spec["motors"]), "duplicate motor name"
        assert len(set(viz_spec["urdf_joints"])) == len(viz_spec["urdf_joints"]), "duplicate URDF joint"

    def test_urdf_exists_and_parses(self, pkg_name, viz_spec):
        urdf = _urdf_path(pkg_name)
        assert urdf.name == viz_spec["urdf_file"], "get_urdf_path disagrees with VIZ_SPEC['urdf_file']"
        ET.parse(urdf)  # raises ParseError on malformed XML

    def test_declared_joints_exist_in_urdf(self, pkg_name, viz_spec):
        urdf = _urdf_path(pkg_name)
        joint_names = {j.get("name") for j in ET.parse(urdf).getroot().iter("joint")}
        for jn in viz_spec["urdf_joints"]:
            assert jn in joint_names, f"VIZ_SPEC joint {jn!r} is absent from {urdf.name}"

    def test_referenced_meshes_exist(self, pkg_name, viz_spec):
        urdf = _urdf_path(pkg_name)
        mesh_files = [m.get("filename") for m in ET.parse(urdf).getroot().iter("mesh")]
        assert mesh_files, "URDF references no meshes"
        for fn in mesh_files:
            resolved = (urdf.parent / fn).resolve()
            assert resolved.exists(), f"mesh {fn!r} referenced by {urdf.name} is missing at {resolved}"

    def test_resolve_robot_round_trips(self, pkg_name, viz_spec):
        """A robot exposing exactly this description's motors resolves back to it."""
        spec = resolve_robot([f"{m}.pos" for m in viz_spec["motors"]])
        assert spec is not None
        assert spec.name == viz_spec["name"]
