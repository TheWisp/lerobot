"""FoundationPose sidecar worker — runs in the SAM3D venv, NOT the lerobot env.

Invoked as a standalone script (NOT `python -m lerobot...`, which would trigger the
lerobot package import that the SAM3D venv can't satisfy):

    ~/.cache/sam3d/venv/bin/python .../overlays/foundationpose_worker.py [--mesh X.glb]

Attaches to the FoundationPoseIPC shared memory the debug-vision adapter created,
then loops: REGISTER (first frame, with mask) anchors the mesh to the depth and runs
`register`; TRACK runs `track_one`. Each frame it draws an RGBA debug overlay — the
posed mesh as a white silhouette + faint front-face wireframe, a pose gizmo, and the
SAM mask outline (cyan) — and writes it back. v1 is ring-specific: the default mesh is
the hand-measured tube (a short annular cylinder) and its rotational symmetry is declared
to FoundationPose; pass ``--mesh`` to track a different object.
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
from Utils import (  # noqa: E402
    make_mesh_tensors,
    nvdiffrast_render,
    set_logging_format,
    set_seed,
    symmetry_tfs_from_info,
)

# v1 tracked object: a short tube (annular cylinder), hand-measured ground truth in mm.
# A single-view SAM-3D reconstruction got the tube ~2x too fat; the measured geometry is
# exact, correctly thin, and rotationally symmetric.
RING_OUTER_DIA, RING_HOLE_DIA, RING_HEIGHT = 48.0, 23.0, 19.0


def _project(P, R, t, K):  # noqa: N803  P:(N,3) object frame -> (uv:(N,2) px, z:(N,) camera)
    Pc = P @ R.T + t  # noqa: N806  object->camera (OpenCV convention)
    z = Pc[:, 2]
    uv = Pc @ K.T
    return uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None), z


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
    ap.add_argument("--mesh", default=None, help="mesh path; default = the ground-truth tube")
    args = ap.parse_args()

    set_logging_format()
    set_seed(0)
    logging.getLogger().setLevel(logging.ERROR)
    glctx = dr.RasterizeCudaContext()
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    if args.mesh:
        base_mesh = trimesh.load(args.mesh, force="mesh")
        mesh_name = os.path.basename(args.mesh)
    else:
        base_mesh = trimesh.creation.annulus(
            r_min=RING_HOLE_DIA / 2000.0,  # mm diameter -> m radius
            r_max=RING_OUTER_DIA / 2000.0,
            height=RING_HEIGHT / 1000.0,
            sections=64,
        )
        mesh_name = f"ground-truth tube {RING_OUTER_DIA:.0f}x{RING_HOLE_DIA:.0f}x{RING_HEIGHT:.0f}mm"
    # The tube is an annular cylinder: continuous rotational symmetry about its axis (mesh +Z)
    # plus a 180-degree end-flip. Declare it so FoundationPose doesn't chase the unobservable
    # about-axis rotation (which otherwise makes the pose spin frame-to-frame).
    sym_tfs = symmetry_tfs_from_info(
        {
            "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}],
            "symmetries_discrete": [np.diag([1.0, -1.0, -1.0, 1.0]).tolist()],  # 180 deg about X
        },
        rot_angle_discrete=15,
    )
    print(f"[fp-worker] loaded FoundationPose + mesh: {mesh_name}", flush=True)

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
    gizmo_center = None  # debug-overlay gizmo origin, set at register
    axis_len = 0.0
    wire_verts = wire_faces = wire_fnormals = None  # interior wireframe geom, set at register
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
                        symmetry_tfs=sym_tfs,
                        scorer=scorer,
                        refiner=refiner,
                        glctx=glctx,
                        debug=0,
                        debug_dir="/tmp/fp_debug",  # nosec B108  FoundationPose debug sink, unused at debug=0
                    )
                else:
                    est.reset_object(
                        model_pts=m.vertices, model_normals=m.vertex_normals, symmetry_tfs=sym_tfs, mesh=m
                    )
                mesh_tensors = make_mesh_tensors(m)
                gizmo_center = np.asarray(m.vertices, np.float32).mean(0)  # gizmo origin
                axis_len = 0.5 * float(max(m.extents))
                # interior wireframe: moderate decimation keeps the ring shape; back/inside
                # faces are culled at render time so only the front-facing wires show.
                try:
                    wm = m.simplify_quadric_decimation(face_count=400) if len(m.faces) > 400 else m
                except Exception:  # nosec B110  best-effort; fall back to full mesh
                    wm = m
                wire_verts = np.asarray(wm.vertices, np.float32)
                wire_faces = np.asarray(wm.faces)
                wire_fnormals = np.asarray(wm.face_normals, np.float32)
                pose = np.asarray(est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)).reshape(
                    4, 4
                )
            elif cmd == CMD_TRACK and est is not None:
                pose = np.asarray(est.track_one(rgb=rgb, depth=depth, K=K, iteration=2)).reshape(4, 4)

            if est is None or pose is None or mesh_tensors is None:
                ipc.send_response(req["seq"], ST_FAIL)
                continue

            # Debug overlay: FP mesh SILHOUETTE (outer ring + hole contours, no interior
            # clutter) + pose gizmo + SAM-mask outline. The silhouette comes from the posed
            # render so it keeps the true ring shape; outlines (not a fill) keep the pose
            # readable and stop amodal mesh-growth under occlusion from looking like a desync.
            c, d, _ = nvdiffrast_render(
                K=K,
                H=h,
                W=w,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                use_light=True,
                ob_in_cams=torch.as_tensor(pose[None], device="cuda", dtype=torch.float),
            )
            rmask = (d[0].cpu().numpy() > 0.001).astype(np.uint8)  # filled FP silhouette
            rgba = np.zeros((h, w, 4), np.uint8)
            R, t = pose[:3, :3], pose[:3, 3]  # noqa: N806
            # interior mesh wireframe (front-facing triangles only), faint, under the silhouette
            uv, z = _project(wire_verts, R, t, K)
            uvi = np.round(uv).astype(np.int32)
            fn_cam = wire_fnormals @ R.T  # face normals in camera frame
            for fi in range(len(wire_faces)):
                if fn_cam[fi, 2] >= 0:  # back/inside face -> cull
                    continue
                a, b, cc = wire_faces[fi]
                if z[a] <= 0 or z[b] <= 0 or z[cc] <= 0:
                    continue
                pa, pb, pc = (
                    (int(uvi[a, 0]), int(uvi[a, 1])),
                    (int(uvi[b, 0]), int(uvi[b, 1])),
                    (int(uvi[cc, 0]), int(uvi[cc, 1])),
                )
                cv2.line(rgba, pa, pb, (255, 255, 255, 110), 1)
                cv2.line(rgba, pb, pc, (255, 255, 255, 110), 1)
                cv2.line(rgba, pc, pa, (255, 255, 255, 110), 1)
            # FP silhouette (outer ring + hole), bright, on top
            fp_cnts, _ = cv2.findContours(rmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgba, fp_cnts, -1, (255, 255, 255, 240), 2)
            # pose gizmo. The tube's about-axis rotation is unobservable, so show the real
            # symmetry axis (mesh +Z, blue) and DERIVE the perpendicular axes (red/green) from
            # a fixed reference (camera up) — the gizmo cannot spin about the axis by design.
            oc = R @ gizmo_center + t
            z_cam = R @ np.array([0.0, 0.0, 1.0])
            z_cam /= np.linalg.norm(z_cam) + 1e-9
            x_cam = np.cross(np.array([0.0, 1.0, 0.0]), z_cam)
            x_cam = x_cam if np.linalg.norm(x_cam) > 1e-6 else np.cross(np.array([1.0, 0.0, 0.0]), z_cam)
            x_cam /= np.linalg.norm(x_cam) + 1e-9
            y_cam = np.cross(z_cam, x_cam)
            axes_cam = np.stack([oc, oc + axis_len * x_cam, oc + axis_len * y_cam, oc + axis_len * z_cam])
            ap = axes_cam @ K.T
            auv = ap[:, :2] / np.clip(ap[:, 2:3], 1e-6, None)
            origin = (int(auv[0, 0]), int(auv[0, 1]))
            for k, col in ((1, (255, 0, 0, 255)), (2, (0, 255, 0, 255)), (3, (0, 0, 255, 255))):
                if axes_cam[k, 2] > 0:
                    cv2.line(rgba, origin, (int(auv[k, 0]), int(auv[k, 1])), col, 2)
            # SAM mask outline (cyan)
            if mask is not None:
                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgba, cnts, -1, (0, 255, 255, 255), 2)
            ipc.send_response(req["seq"], ST_OK, rgba, pose)
        except Exception:
            logging.exception("[fp-worker] frame failed")
            ipc.send_response(req["seq"], ST_FAIL)


if __name__ == "__main__":
    main()
