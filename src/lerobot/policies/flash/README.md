# FLASH perception experiments

**Run → FLASH** offers two views:

- **Video examples:** select an RGB-D recording, play or scrub it, compare saved
  methods and scores, inspect the 3D scene, or rerun one/all examples.
- **Live camera:** open an RGB-D camera, name an object or freeze and click it,
  then inspect the mask, fitted surface, axes and supporting groups in image/3D mode.
  This inspector never moves the robot.

Both use `SurfaceFrameSession.step`. Only acquisition, segmentation and recording
adapters differ. Fresh sessions use LK image matching, the single-loop surface
fitter, supporting groups, a growing measured surface and Kalman pose estimation.
UFM remains selectable; small_gicp/Open3D are explicit research comparisons.
The old SciPy solver and separate editor/camera-debugging UI have been retired.

Geometry checks reject poses that contradict reliable measured depth. Texture
and outline coverage rank compatible poses; they cannot excuse protrusion into
measured empty space. Missing depth and occlusion are unknown evidence.
Tracking remains imperfect, especially through occlusion and recovery. The new
session is **not connected to robot policy execution yet**.

Start with:

- [Pipeline and limitations](docs/tracking_pipeline.md): data flow, geometry,
  groups, uncertainty and surface growth.
- [Configuration reference](docs/pipeline_settings.md): settings shared by
  recorded benchmarks and live camera inspection, including iteration limits.
- [Research evidence](../../../../docs/proofs/flash-tracker-restart/RESEARCH.md):
  dated experiments, rejected approaches, comparisons and measured trade-offs.

Old saved playback retains its actual solver/settings and never reruns inference.
Rerunning an old SciPy configuration selects the maintained fitter and records
that change; the previous result is unchanged.

The [sealed handover](docs/SEALED_20261031.md) describes the previous demo compiler,
robot engine and editor design. Those research APIs and stored recordings remain;
the current GUI focuses on perception. The [old implementation map](https://github.com/TheWisp/lerobot/blob/18fa9d9f40d230ca2fa8d8bead75d24b573a8316/src/lerobot/policies/flash/README.md)
is historical, not the current UI specification.
