# IK end-to-end proof artifacts

Supporting material for **PR #9** (`kinematics: Pink-based IK`). This branch
is **not** part of the PR — it only hosts the proof GIFs and the one-off
harness that produced them, so the PR diff stays scoped to the IK feature.

## GIFs

`ik_<robot>_<shape>.gif` — the Pink IK tracing each scripted Cartesian shape
(30 mm and 60 mm circles, a 50 mm square) on SO-101 and SO-107, rendered
through the GUI's actual URDF renderer (the vendored three.js + urdf-loader
from PR #8). Each GIF's on-screen `three.js vs pinocchio FK` readout shows
the renderer reproduces the pinocchio FK to 0.0 mm.

## How they were made

A throwaway harness, run from a scratch directory:

1. **`gen_traj.py`** — computes the IK trajectories (the same shapes
   `tests/model/test_pink_ik_trajectory.py` exercises) and dumps joint
   angles + pinocchio FK per frame to JSON.
2. **`harness.html`** — a standalone three.js + urdf-loader page that loads a
   URDF, plays a trajectory JSON, and reads the rendered end-effector
   position back out of the scene graph to cross-check it against the
   pinocchio FK.
3. **`capture.sh`** — screenshots each frame with headless Chrome, then
   `ffmpeg` stitches the frames into a GIF.

It expects the vendored renderer, URDFs and meshes served over HTTP from the
scratch directory (see the paths in the scripts). This is a **one-off visual
cross-check** — the rigorous, reproducible, CI-run proof is the pytest
`tests/model/test_pink_ik_trajectory.py`.
