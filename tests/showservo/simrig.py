# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A rendered bench for the end-to-end test: real pixels, real depth, awkward camera.

This exists because the unit tests have a seam in them. The 3D tests hand exact
correspondences straight to Kabsch, and the tracker tests never touch 3D — so the
join, which is where tracking error becomes geometric error, was never exercised.
Here the system sees only what the rig would: an RGB frame and a depth map.

No physics and no arm. The held end is a textured marker moved directly in Cartesian
space, because the first thing to prove is that perception drives the loop the right
way, not that an IK solver works. Contact and joints come later.

``SimRig.self_check`` is not optional decoration. A camera-convention error (MuJoCo's
frame is +y up, +z BACKWARD; OpenCV's is +y down, +z forward) would silently corrupt
every metre measured here while leaving the images looking perfectly reasonable.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from lerobot.showservo.pose import CameraIntrinsics  # noqa: E402

# A camera deliberately off the perpendicular: 0.42 m up, well off to one side, tilted.
_SCENE = """
<mujoco>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>
    <texture name="tabletex" type="2d" builtin="checker" rgb1="0.25 0.28 0.33"
             rgb2="0.62 0.64 0.68" width="512" height="512"/>
    <material name="tablemat" texture="tabletex" texrepeat="10 10"/>
    <!-- Coarse textures on purpose: a 256 px random pattern mapped onto a face only
         ~50 px wide aliases into mush, and SIFT then finds nothing. 64 px keeps each
         speckle several pixels across at the working distance. -->
    <texture name="blocktex" type="2d" builtin="flat" rgb1="0.85 0.45 0.18"
             width="64" height="64" mark="random" markrgb="0.05 0.05 0.05" random="0.55"/>
    <material name="blockmat" texture="blocktex"/>
    <texture name="marktex" type="2d" builtin="flat" rgb1="0.25 0.6 0.85"
             width="64" height="64" mark="random" markrgb="0.98 0.98 0.98" random="0.55"/>
    <material name="markmat" texture="marktex"/>
  </asset>
  <worldbody>
    <light pos="0.2 0.2 1.2" dir="-0.2 -0.2 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.4 0.1 0.8" dir="0.4 -0.1 -0.8" diffuse="0.4 0.4 0.4"/>
    <geom name="table" type="plane" size="0.6 0.6 0.01" material="tablemat"/>
    <body name="block" mocap="true" pos="0 0 0.04">
      <geom name="block" type="box" size="0.05 0.05 0.04" material="blockmat"/>
    </body>
    <body name="held" mocap="true" pos="0.10 0.10 0.14">
      <geom name="held" type="box" size="0.032 0.032 0.026" material="markmat"/>
    </body>
    <camera name="rig" pos="0.34 0.26 0.40" xyaxes="-0.609 0.793 0 -0.487 -0.374 0.789"/>
  </worldbody>
</mujoco>
"""


class SimRig:
    """Renders the scene. Pre: MuJoCo importable with a working EGL context."""

    def __init__(self, width: int = 960, height: int = 720):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(_SCENE)
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._depth_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._depth_renderer.enable_depth_rendering()

        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rig")
        fovy = float(self.model.cam_fovy[self._cam_id])
        f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        self.intrinsics = CameraIntrinsics(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)
        self._body = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("block", "held")
        }
        self._forward()

    # --- scene control -----------------------------------------------------------

    def place(self, name: str, pos, yaw: float = 0.0) -> None:
        """Pre: ``name`` in {"block", "held"}; ``pos`` is (3,) world metres."""
        assert name in self._body, f"unknown body {name!r}"
        idx = self.model.body_mocapid[self._body[name]]
        self.data.mocap_pos[idx] = np.asarray(pos, dtype=np.float64)
        half = yaw / 2.0
        self.data.mocap_quat[idx] = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        self._forward()

    def pose_of(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Ground truth, for ASSERTIONS ONLY. Post: ``(pos (3,), rot (3,3))`` in world."""
        bid = self._body[name]
        return self.data.xpos[bid].copy(), self.data.xmat[bid].reshape(3, 3).copy()

    def _forward(self) -> None:
        self._mj.mj_forward(self.model, self.data)

    # --- what the rig sees -------------------------------------------------------

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """Post: ``(rgb HxWx3 uint8, depth HxW float64 metres)``.

        Depth is the distance along the camera's viewing axis, which is what
        :meth:`CameraIntrinsics.deproject` expects — not radial range.
        """
        self._renderer.update_scene(self.data, camera="rig")
        rgb = self._renderer.render().copy()
        self._depth_renderer.update_scene(self.data, camera="rig")
        depth = self._depth_renderer.render().astype(np.float64)
        return rgb, depth

    def silhouette(self, name: str) -> np.ndarray:
        """Pixels body ``name`` actually covers. Post: HxW bool, occlusion-correct.

        The designation stand-in. A projected bounding box was the earlier stand-in and
        is a poor one: it hands whatever tier is under test a rectangle containing table,
        which a corner detector largely ignores and a dense patch grid samples in full.
        That difference is a property of the stand-in, not of the tiers, and it is
        exactly the kind of confound that makes a benchmark answer the wrong question.
        SAM3 returns silhouettes on real frames, so the stand-in returns one too.

        Segmentation is rendered by toggling the colour renderer rather than holding a
        third GL context, because contexts are the scarce resource here.
        """
        mj = self._mj
        self._renderer.enable_segmentation_rendering()
        try:
            self._renderer.update_scene(self.data, camera="rig")
            seg = self._renderer.render()
        finally:
            self._renderer.disable_segmentation_rendering()
        obj_id, obj_type = seg[..., 0], seg[..., 1]
        is_geom = obj_type == mj.mjtObj.mjOBJ_GEOM
        mask = np.zeros(obj_id.shape, dtype=bool)
        wanted = self._body[name]
        # Map every visible geom back to its body; a body may own several geoms.
        for gid in np.unique(obj_id[is_geom]):
            if int(self.model.geom_bodyid[int(gid)]) == wanted:
                mask |= is_geom & (obj_id == gid)
        assert mask.any(), f"{name} is not visible from the rig camera"
        return mask

    def world_to_camera(self, points_w: np.ndarray) -> np.ndarray:
        """Ground truth conversion, for ASSERTIONS ONLY. Post: (N, 3) in OpenCV camera frame.

        MuJoCo's camera looks down -z with +y up; OpenCV looks down +z with +y down.
        Flipping y and z is the whole of the conversion, and getting it wrong is
        exactly the silent error :meth:`self_check` exists to catch.
        """
        cam_pos = self.data.cam_xpos[self._cam_id]
        cam_rot = self.data.cam_xmat[self._cam_id].reshape(3, 3)
        local = (np.asarray(points_w, dtype=np.float64).reshape(-1, 3) - cam_pos) @ cam_rot
        return local * np.array([1.0, -1.0, -1.0])

    def self_check(self, tol_m: float = 2e-3) -> None:
        """Prove the harness before trusting anything measured through it.

        Renders, deprojects the pixel at each body's centre, and compares against that
        body's known position in the camera frame. Any convention error — depth units,
        y/z flips, a principal point off by half the image — shows up here as
        centimetres rather than as a subtly wrong servo three modules later.
        """
        _, depth = self.render()
        for name in ("block", "held"):
            pos_w, _ = self.pose_of(name)
            truth_cam = self.world_to_camera(pos_w)[0]
            uv = self.intrinsics.project(truth_cam[None, :])[0]

            col, row = int(round(uv[0])), int(round(uv[1]))
            assert 0 <= col < self.width and 0 <= row < self.height, f"{name} is out of frame"
            z = float(depth[row, col])
            measured = self.intrinsics.deproject(uv[None, :], np.array([z]))[0]

            # Depth returns the visible SURFACE, the truth is the body CENTRE, and the
            # two sit at different z on the same ray — where x and y scale with z. So
            # the comparison is against the true ray sampled at the measured depth, not
            # against the centre itself.
            assert 0.0 < z <= truth_cam[2] + 1e-6, (
                f"{name}: surface depth {z:.4f} m should be in front of centre {truth_cam[2]:.4f} m"
            )
            expected = truth_cam * (z / truth_cam[2])
            off = float(np.linalg.norm(measured - expected))
            assert off < tol_m, f"{name}: camera model disagrees with the sim by {off * 1000:.1f} mm"
