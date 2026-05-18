#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Live URDF visualization for SO-107 (unimanual or bimanual).

The viewer is :class:`pinocchio.visualize.MeshcatVisualizer`, which serves
a three.js viewer at ``http://127.0.0.1:7000/static/``. For bimanual, two
copies of the SO-107 URDF are loaded under separate root nodes and
offset along world +X / -X so they don't overlap.

State-based: the viz reflects the OBSERVED joint angles each tick, not
the commanded ones. This makes it useful for any workflow — teleop
(any kind), policy rollout, replay, even a robot just streaming
observations — and surfaces motor lag / collisions / unreachable IK
targets as a divergence between what you commanded and what the URDF
shows.

Use as a safe-testing backstop for teleop: set ``dry_run=True`` on the
robot config to lock out motor writes, then add ``--display-urdf=true``
to ``lerobot-teleoperate``. The motors don't move, but the observation
stream still reports the current static joint positions, which the viz
renders. (For dry-run teleop, what you see is whatever pose the arm
was already in — useful for verifying the URDF base orientation / world
yaw mapping without any motion risk.)

This module avoids importing pinocchio / meshcat at module load — both
are heavy and only needed when ``--display-urdf`` is actually on.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
)

logger = logging.getLogger(__name__)

# Per-arm world-frame X offset so the two URDF copies don't collide visually.
# Roughly matches the physical center-to-center distance of a side-by-side
# SO-107 setup so the scene looks proportional to reality.
_BIMANUAL_X_OFFSET_M = 0.165

# Motor names in URDF joint order. The SO-107 URDF has 7 revolute joints
# (S1..S7) mapping 1:1 to these motors; the gripper IS a URDF joint, not a
# trailing controller-only DOF, so it has to be included for pinocchio's
# nq=7 configuration vector to validate.
_SO107_ARM_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class BimanualUrdfViz:
    """Shared MeshCat scene with two SO-107 URDF copies (left + right).

    Preconditions:
        * pinocchio and meshcat are importable (declared via the
          ``viz`` extras group in pyproject).
        * Port 7000 is free (or override via ``viewer_url``).

    Postconditions on :meth:`set_arm_joints_deg`:
        * The MeshCat scene's named arm (``"left"`` or ``"right"``) renders
          at the supplied joint angles. No motor traffic occurs.
    """

    def __init__(self, viewer_url: str | None = None, open_browser: bool = True) -> None:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        from . import get_meshes_dir, get_urdf_path

        urdf = str(get_urdf_path())
        package_dirs = [str(get_meshes_dir()), str(get_meshes_dir().parent)]

        # Shared underlying MeshCat server; both arms get their own viz root.
        master_model, master_coll, master_visual = pin.buildModelsFromUrdf(urdf, package_dirs)
        master = MeshcatVisualizer(master_model, master_coll, master_visual)
        if viewer_url is not None:
            import meshcat

            master.initViewer(viewer=meshcat.Visualizer(zmq_url=viewer_url), open=open_browser)
        else:
            master.initViewer(open=open_browser)

        # Place each arm so its position in the URDF scene matches what
        # the user sees in the physical setup. With the default camera
        # at (+0.7, -0.5, +0.5) looking at origin, +X world projects
        # toward the LEFT of the rendered image, so the "right" arm has
        # to live at -X and "left" at +X to feel natural. (An earlier
        # revision only reordered these two function calls without
        # changing the base_xyz coordinates — looked symmetric on read,
        # was still wrong on screen.)
        self._left_viz, self._left_model = self._add_arm(
            master, urdf, package_dirs, root="left", base_xyz=(+_BIMANUAL_X_OFFSET_M, 0.0, 0.0)
        )
        self._right_viz, self._right_model = self._add_arm(
            master, urdf, package_dirs, root="right", base_xyz=(-_BIMANUAL_X_OFFSET_M, 0.0, 0.0)
        )
        self._master_viewer = master.viewer

        # Position the camera at a sensible default so the two arms fill the
        # view on first load. MeshCat's stock camera is wide and far; for a
        # ~30 cm-baseline pair of 50 cm arms a closer position makes the
        # scene immediately readable. The user can still orbit / zoom freely.
        try:
            import meshcat.transformations as tf

            cam_xyz = (0.7, -0.5, 0.5)  # diagonal-front, slightly elevated
            master.viewer["/Cameras/default"].set_transform(tf.translation_matrix([0.0, 0.0, 0.2]))
            master.viewer["/Cameras/default/rotated/<object>"].set_property("position", list(cam_xyz))
        except Exception:
            pass  # Camera defaults are aesthetic; failure here is harmless.

        self.url: str = (
            master.viewer.url() if hasattr(master.viewer, "url") else "http://127.0.0.1:7000/static/"
        )
        logger.info(f"BimanualUrdfViz live at {self.url}")

    @staticmethod
    def _add_arm(master, urdf: str, package_dirs: list[str], root: str, base_xyz: tuple[float, float, float]):
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        model, coll, visual = pin.buildModelsFromUrdf(urdf, package_dirs)
        viz = MeshcatVisualizer(model, coll, visual)
        viz.initViewer(viewer=master.viewer)
        viz.loadViewerModel(rootNodeName=root)
        # Translate this arm's root node to its base position.
        T = np.eye(4)
        T[:3, 3] = base_xyz
        master.viewer[root].set_transform(T)
        return viz, model

    def set_arm_joints_deg(self, arm: str, joint_deg_7: np.ndarray) -> None:
        """Update the ``"left"`` or ``"right"`` arm to the given joint angles (motor degrees, 7 values).

        Input is in MOTOR space (what the robot's observation reports). We
        apply the per-arm motor->URDF map (sign + offset_deg, defined in
        :mod:`lerobot.robots.so107_description.kinematics`) before handing
        the vector to pinocchio. Without this, the URDF renders the right
        arm with shoulder_pan/forearm_roll inverted and shoulder_lift /
        elbow_flex / wrist_roll offset by ~90 deg from the actual robot —
        because the SO-107 right-arm mounting is asymmetric relative to
        the URDF's zero pose.
        """
        from .kinematics import LEFT_ARM_MAP, MOTOR_NAMES, RIGHT_ARM_MAP

        assert arm in ("left", "right"), f"arm must be 'left' or 'right', got {arm!r}"
        viz = self._left_viz if arm == "left" else self._right_viz
        joint_map = LEFT_ARM_MAP if arm == "left" else RIGHT_ARM_MAP
        motor_deg = np.asarray(joint_deg_7, dtype=float)
        # urdf_deg = sign * motor_deg + offset_deg
        urdf_deg = np.empty_like(motor_deg)
        for i, name in enumerate(MOTOR_NAMES):
            jm = joint_map[name]
            urdf_deg[i] = jm.sign * motor_deg[i] + jm.offset_deg
        q_rad = np.deg2rad(urdf_deg)
        # Pinocchio model.nq should be 7 (revolute joints S1..S7).
        viz.display(q_rad[: viz.model.nq])


# ── ProcessorStep that taps the observation stream and pushes to the viz ──


@ProcessorStepRegistry.register("urdf_viz_mirror")
@dataclass
class UrdfVizMirrorStep(ObservationProcessorStep):
    """Side-effect step: render the robot's CURRENT joint state in the MeshCat scene.

    Sits in the observation pipeline so it works regardless of teleop type —
    leader arms, Cartesian VR teleop, joint-space replay, policy rollout, even
    a robot just streaming observations with no controller attached. The
    observation is returned unchanged so it never perturbs downstream steps.

    For bimanual, both arms are updated each tick using ``left_<motor>.pos``
    and ``right_<motor>.pos`` observation keys. For unimanual, only the
    configured arm is rendered; the other arm stays at its initial zero pose.

    State (observation) vs commanded (action): we deliberately render OBSERVED
    joints because that's the universal signal — every robot's
    ``get_observation()`` returns ``<motor>.pos``. The commanded action is
    only available downstream of any IK step, and not all workflows have one.
    Showing the actual robot pose is also what the user usually wants to see
    when debugging motor lag, collisions, or unreachable IK targets.

    Attributes:
        viz: Shared BimanualUrdfViz instance (built outside the step).
        bimanual: True if the observation carries ``left_*`` / ``right_*``
            prefixed keys; False for unprefixed unimanual observations.
        unimanual_arm: Which arm to render when ``bimanual=False``.
    """

    viz: Any  # BimanualUrdfViz; Any because viz import is deferred.
    bimanual: bool = False
    unimanual_arm: str = "right"
    _arm_motor_names: tuple[str, ...] = field(default=_SO107_ARM_MOTOR_NAMES, init=False, repr=False)
    _warned_arms: set[str] = field(default_factory=set, init=False, repr=False)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        if self.bimanual:
            self._update_arm("left", "left_", observation)
            self._update_arm("right", "right_", observation)
        else:
            self._update_arm(self.unimanual_arm, "", observation)
        return observation

    def _update_arm(self, arm: str, prefix: str, observation: RobotObservation) -> None:
        joints_deg = []
        for name in self._arm_motor_names:
            key = f"{prefix}{name}.pos"
            v = observation.get(key)
            if v is None:
                # Observation doesn't have this arm's joints yet (early tick).
                return
            joints_deg.append(float(v))
        try:
            self.viz.set_arm_joints_deg(arm, np.asarray(joints_deg, dtype=float))
        except Exception as e:
            # Viz errors must never break the run loop, but they shouldn't be
            # totally silent either — DEBUG-only swallowing previously masked
            # a complete render failure (wrong joint vector size) during a
            # real-hardware replay. Warn ONCE per arm so the user knows the
            # scene isn't tracking, then drop to DEBUG to avoid log spam at
            # 30-200 Hz.
            if arm not in self._warned_arms:
                self._warned_arms.add(arm)
                logger.warning(
                    "UrdfVizMirrorStep: viz update failed for %s (%s). "
                    "The MeshCat scene won't track this arm. Further errors "
                    "for this arm logged at DEBUG.",
                    arm,
                    e,
                )
            else:
                logger.debug("UrdfVizMirrorStep: viz update failed for %s: %s", arm, e)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Pass-through: doesn't add or remove any observation keys.
        return features


# ── Periodic commanded-joints logger (debug companion to the URDF viz) ────


@ProcessorStepRegistry.register("commanded_joints_log")
@dataclass
class CommandedJointsLogStep(RobotActionProcessorStep):
    """Periodically log the post-IK commanded ``<motor>.pos`` values per arm.

    Sits on the ACTION pipeline (after IK) and prints one INFO line per
    ``log_interval_s`` seconds with the commanded joints. Complements
    :class:`UrdfVizMirrorStep` for dry-run testing: the URDF reflects the
    OBSERVED state (motors held still under dry_run), while this log
    shows what the IK *would* have commanded — together they tell you
    whether the teleop -> IK math is doing the right thing without
    requiring motors to move.

    Throttled by wall clock so the log is sane across teleop rates
    (60 Hz unimanual, 200 Hz predictive). Action is returned unchanged.

    Attributes:
        log_interval_s: Minimum seconds between log lines.
        bimanual: When True, log left_ and right_ separately.
    """

    log_interval_s: float = 1.0
    bimanual: bool = False
    _last_log_t: float = field(default=0.0, init=False, repr=False)
    _arm_motor_names: tuple[str, ...] = field(default=_SO107_ARM_MOTOR_NAMES, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        now = time.monotonic()
        if now - self._last_log_t < self.log_interval_s:
            return action
        self._last_log_t = now
        if self.bimanual:
            for prefix in ("left_", "right_"):
                line = self._format_arm(action, prefix)
                if line:
                    logger.info("commanded[%s] %s", prefix.rstrip("_"), line)
        else:
            line = self._format_arm(action, "")
            if line:
                logger.info("commanded %s", line)
        return action

    def _format_arm(self, action: RobotAction, prefix: str) -> str:
        parts = []
        for name in self._arm_motor_names:
            v = action.get(f"{prefix}{name}.pos")
            if v is not None:
                parts.append(f"{name}={float(v):+.1f}")
        return " ".join(parts)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Pass-through: doesn't add or remove any action keys.
        return features


# ── Convenience: one-call viz attachment for any run script ──────────────


def maybe_attach_urdf_viz(
    obs_processor_steps: list[Any],
    robot: Any,
    logger_: logging.Logger | None = None,
) -> Any:
    """Build a BimanualUrdfViz and append a UrdfVizMirrorStep to the steps list.

    Shared helper used by lerobot-teleoperate / lerobot-replay /
    lerobot-record / HVLA launch so the wiring lives in one place. Detects
    bimanual via the robot's observation_features (any ``left_<motor>.pos``
    key). Failures are caught and downgraded to a warning — the
    visualization is a debug aid, never load-bearing.

    Returns the BimanualUrdfViz instance (so the caller can log the URL),
    or ``None`` if construction failed.
    """
    log = logger_ or logger
    try:
        obs_features = getattr(robot, "observation_features", {})
        bimanual = any(
            isinstance(k, str) and k.startswith("left_") and k.endswith(".pos") for k in obs_features
        )
    except Exception:
        bimanual = False

    try:
        # open_browser=False: the GUI's iframe is the intended viewer.
        # When run from the CLI without the GUI, the user can still open
        # the URL manually from the log line below.
        viz = BimanualUrdfViz(open_browser=False)
    except Exception as e:
        log.warning(f"display_urdf=True but viz construction failed: {type(e).__name__}: {e}")
        return None

    obs_processor_steps.append(UrdfVizMirrorStep(viz=viz, bimanual=bimanual))
    log.info(
        f"display_urdf: live MeshCat scene at {viz.url} ({'bimanual' if bimanual else 'unimanual'} mode)."
    )
    return viz
