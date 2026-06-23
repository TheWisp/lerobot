"""FoundationPose sidecar worker — runs in the SAM3D venv, NOT the lerobot env.

Invoked as a standalone script (NOT `python -m lerobot...`, which would trigger the
lerobot package import that the SAM3D venv can't satisfy):

    ~/.cache/sam3d/venv/bin/python .../debug_vision/foundationpose_worker.py [--mesh X.glb]

Attaches to the FoundationPoseIPC shared memory the debug-vision adapter created,
then loops: REGISTER (first frame, with mask) anchors the mesh to the depth and runs
`register`; TRACK runs `track_one`; each frame it renders the posed mesh to an RGBA
overlay (mesh + magenta where it extends past the visible mask = amodal/hidden
geometry) and writes it back. v1 hardcodes the ring mesh; the adapter will pass the
mesh path once we generalize.
"""

import argparse
import logging
import os
import sys
import time
import warnings

# FoundationPose's internals are noisy with torch deprecations; they read like errors in
# the GUI debug panel but are harmless. Real failures still surface via logging.exception.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*set_default_tensor_type.*")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for foundationpose_ipc
sys.path.insert(0, os.path.expanduser("~/.cache/foundationpose"))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import nvdiffrast.torch as dr  # noqa: E402
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # noqa: E402
from foundationpose_ipc import (  # noqa: E402
    CMD_REGISTER,
    CMD_RESET,
    CMD_TRACK,
    ST_FAIL,
    ST_OK,
    FoundationPoseIPC,
)
from Utils import make_mesh_tensors, nvdiffrast_render, set_logging_format, set_seed  # noqa: E402

DEFAULT_MESH = os.path.expanduser("~/.cache/huggingface/lerobot/gui/scan3d/object.glb")


def anchor_scale(mesh, depth, mask, K):  # noqa: N803  (K = camera intrinsics)
    """Scale the (fake-metric) mesh to the depth-implied object size.

    Uses the back-projected mask bbox SIDE (diameter), not the diagonal (which
    overestimates a round object by √2 and makes FoundationPose over-extend).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mv = mask & (depth > 0.05)
    ys, xs = np.where(mv)
    if len(xs) < 20:
        return mesh.copy()
    dd = depth[ys, xs]
    p = np.stack([(xs - cx) * dd / fx, (ys - cy) * dd / fy], 1)
    diam = float(max(np.percentile(p, 98, 0) - np.percentile(p, 2, 0)))
    m = mesh.copy()
    m.apply_scale(diam / max(m.extents))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=DEFAULT_MESH)
    args = ap.parse_args()

    set_logging_format()
    set_seed(0)
    logging.getLogger().setLevel(logging.ERROR)
    glctx = dr.RasterizeCudaContext()
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    base_mesh = trimesh.load(args.mesh, force="mesh")
    print(f"[fp-worker] loaded FoundationPose + mesh {os.path.basename(args.mesh)}", flush=True)

    # Attach to the IPC the adapter created (retry until it exists).
    ipc = None
    for _ in range(200):
        try:
            ipc = FoundationPoseIPC(create=False)
            break
        except FileNotFoundError:
            time.sleep(0.05)
    if ipc is None:
        print("[fp-worker] IPC never appeared; exiting", flush=True)
        return
    print("[fp-worker] attached to IPC; ready", flush=True)

    est = None
    mesh_tensors = None
    pose = None
    last_seq = 0
    while True:
        req = ipc.poll_request(last_seq)
        if req is None:
            time.sleep(0.003)
            continue
        last_seq = req["seq"]
        cmd, rgb, depth, mask, K = req["cmd"], req["rgb"], req["depth"], req["mask"], req["K"]  # noqa: N806
        h, w = rgb.shape[:2]
        try:
            if cmd == CMD_RESET:
                est, pose = None, None
                ipc.send_response(req["seq"], ST_OK)
                continue
            if cmd == CMD_REGISTER and mask is not None:
                m = anchor_scale(base_mesh, depth, mask, K)
                if est is None:
                    est = FoundationPose(
                        model_pts=m.vertices,
                        model_normals=m.vertex_normals,
                        mesh=m,
                        scorer=scorer,
                        refiner=refiner,
                        glctx=glctx,
                        debug=0,
                        debug_dir="/tmp/fp_debug",  # nosec B108  FoundationPose debug sink, unused at debug=0
                    )
                else:
                    est.reset_object(model_pts=m.vertices, model_normals=m.vertex_normals, mesh=m)
                mesh_tensors = make_mesh_tensors(m)
                pose = np.asarray(est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)).reshape(
                    4, 4
                )
            elif cmd == CMD_TRACK and est is not None:
                pose = np.asarray(est.track_one(rgb=rgb, depth=depth, K=K, iteration=2)).reshape(4, 4)

            if est is None or pose is None or mesh_tensors is None:
                ipc.send_response(req["seq"], ST_FAIL)
                continue

            c, d, _ = nvdiffrast_render(
                K=K,
                H=h,
                W=w,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                use_light=True,
                ob_in_cams=torch.as_tensor(pose[None], device="cuda", dtype=torch.float),
            )
            ren = (c[0].cpu().numpy() * 255).astype(np.uint8)
            rmask = d[0].cpu().numpy() > 0.001
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[rmask, :3] = ren[rmask]
            rgba[rmask, 3] = 150
            if mask is not None:
                amodal = rmask & ~mask
                rgba[amodal] = (255, 0, 255, 185)  # uncovered hidden geometry
            cnts, _ = cv2.findContours(rmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgba, cnts, -1, (0, 255, 255, 255), 2)
            ipc.send_response(req["seq"], ST_OK, rgba, pose)
        except Exception:
            logging.exception("[fp-worker] frame failed")
            ipc.send_response(req["seq"], ST_FAIL)


if __name__ == "__main__":
    main()
