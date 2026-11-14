"""Render the captured-reference motor-free test; no physical action or contact simulation."""

import base64
import json
import subprocess
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import cv2
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from lerobot.gui.api.robot import _make_robot_from_profile
from lerobot.policies.flash.demo_live_worker import ExecutionLoop

D = Path("docs/proofs/flash-demo-editor/apply-20261114")
p = json.loads((D / "compiled-plan.json").read_text())
e = json.loads((D / "live-execution.json").read_text())
r = _make_robot_from_profile(
    json.loads(Path("/home/feit/.cache/flash-demo-editor-preview/config/robots/white_left.json").read_text())
)
r.connect = Mock(side_effect=AssertionError("No motors"))
r.send_action = Mock(side_effect=AssertionError("No motors"))
a1 = np.array(
    [
        [-0.9992393055258567, -0.035027346002018574, -0.017143375521168066, -0.17230311004117477],
        [-0.027711368482377603, 0.9470686729162621, -0.31983278261845616, -0.08094086091001063],
        [0.027438847443702757, -0.3191144211919702, -0.9473188987021619, 0.4050805254648025],
        [0, 0, 0, 1],
    ]
)
a2 = np.array(e["evidence"]["base_from_reference"])
# Read an existing captured preview; this does not create a temporary file.
preview = json.loads(Path("/tmp/lerobot-surface-debug-pr7cs9tq/preview.json").read_text())  # nosec B108
bg = cv2.imdecode(np.frombuffer(base64.b64decode(preview["image"].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
K = np.array(preview["intrinsics"])
cam = np.linalg.inv(p["base_from_camera"])
video = "/home/feit/.cache/flash-demo-editor-preview/config/flash/editor/recordings/39eef9776a7442e1bb96e9cbf4a979b0/videos/observation.images.overhead/chunk-000/file-000.mp4"
raw = subprocess.check_output(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "2",
        "-i",
        video,
        "-vf",
        r"select=between(n\,420\,570)",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
)
frames = np.frombuffer(raw, np.uint8).reshape(-1, 480, 848, 3)
cache = {j + 420: im for j, im in enumerate(frames)}
loop = ExecutionLoop(p, r, Path(__file__).parent, Event(), simulate=True)
m = loop.model
sel = m.kinematics
mapped = sel.inner
pink = mapped._inner._inner
out = Path("src/lerobot/gui/static/flash_apply_stall_20261114.mp4")
W, H = 1696, 670
proc = subprocess.Popen(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{W}x{H}",
        "-framerate",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ],
    stdin=subprocess.PIPE,
)
orange = (40, 165, 255)
green = (90, 235, 115)
white = (240, 242, 245)
muted = (175, 185, 200)
red = (95, 110, 250)


def text(im, s, x, y, color=white, scale=0.63):
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def project(xyz):
    c = xyz @ cam[:3, :3].T + cam[:3, 3]
    uv = c @ K.T
    return np.rint(uv[:, :2] / uv[:, 2, None]).astype(int)


def marker(im, pose, col):
    xyz = np.vstack(
        [
            pose[:3, 3],
            pose[:3, 3] + pose[:3, 0] * 0.02,
            pose[:3, 3] + pose[:3, 1] * 0.02,
            pose[:3, 3] + pose[:3, 2] * 0.02,
        ]
    )
    uv = project(xyz)
    for v in uv[1:]:
        cv2.line(im, tuple(uv[0]), tuple(v), col, 2, cv2.LINE_AA)
    cv2.circle(im, tuple(uv[0]), 5, col, -1, cv2.LINE_AA)
    return uv[0]


trace = []
stopped = None
for i in range(900):
    now = 1 + i / 30
    idx = loop.executor.index
    k = p["keyframes"][idx]
    anchor = a1 if k["reference"] == "object_1" else a2
    loop.observations = {
        k["reference"]: {
            "status": "observed_candidate",
            "pose": np.linalg.inv(p["base_from_camera"]) @ anchor,
            "source_time_s": now,
            "available_time_s": now,
        }
    }
    loop.tick(now, 1 / 30)
    tool = m.forward(loop.state)
    target = loop.executor.target
    q = mapped._motor_to_urdf(sel._full(m.degrees(loop.state)))
    pin.forwardKinematics(pink.robot.model, pink.robot.data, np.deg2rad(q))
    xyz = np.array([pink.robot.data.oMi[j].translation.copy() for j in range(7)] + [tool[:3, 3]])
    frame = k["frame"]
    assert frame in cache
    right = bg.copy()
    uv = project(xyz)
    for a, b in zip(uv[:-1], uv[1:], strict=True):
        cv2.line(right, tuple(a), tuple(b), orange, 3, cv2.LINE_AA)
    for a in uv:
        cv2.circle(right, tuple(a), 4, orange, -1, cv2.LINE_AA)
    marker(right, target, green)
    marker(right, tool, orange)
    im = np.full((H, W, 3), (28, 22, 17), np.uint8)
    im[85:565, :848] = cache[frame]
    im[85:565, 848:] = right
    text(im, "YOUR RECORDED DEMO", 20, 30, scale=0.78)
    text(im, "Matched to the keyframe being attempted (not continuous playback)", 20, 60, muted, 0.55)
    text(im, "NO-MOTION TEST - RECONSTRUCTION", 868, 30, scale=0.78)
    text(im, "Orange: simulated arm / tip    Green: requested tip", 868, 60, muted, 0.55)
    phase = "PICKUP" if idx < 131 else "PLACEMENT APPROACH"
    pos = float(np.linalg.norm(tool[:3, 3] - target[:3, 3]) * 1000)
    ang = float(np.rad2deg(Rotation.from_matrix(tool[:3, :3].T @ target[:3, :3]).magnitude()))
    text(im, f"Demo frame {frame}  |  {phase}", 20, 600, scale=0.7)
    text(
        im,
        f"{loop.executor.index}/342 keyframes completed  |  simulated time {i / 30:.1f}s",
        20,
        631,
        muted,
        0.6,
    )
    stalled = loop.executor.status.startswith("waiting: IK made no progress")
    if stalled and stopped is None:
        stopped = i
    text(im, "STOPPED BEFORE PLACEMENT" if stalled else phase, 868, 598, red if stalled else white, 0.72)
    text(
        im,
        f"Tip error: {pos:.2f} mm   |   Angle error: {ang:.2f} deg",
        868,
        628,
        red if stalled else muted,
        0.66,
    )
    text(
        im,
        "Recomputed from captured references. No motors, grasp contacts or object motion simulated.",
        20,
        658,
        muted,
        0.53,
    )
    proc.stdin.write(im.tobytes())
    trace.append(
        {
            "tick": i,
            "frame": frame,
            "completed": loop.executor.index,
            "status": loop.executor.status,
            "position_error_mm": pos,
            "angle_error_deg": ang,
        }
    )
    if stopped is not None and i >= stopped + 90:
        break
proc.stdin.close()
assert proc.wait() == 0
r.connect.assert_not_called()
r.send_action.assert_not_called()
(D / "video-trace.json").write_text(json.dumps(trace, indent=2))
cv2.imwrite(str(D / "apply-stall-video-poster.jpg"), im)
print(out, "frames", len(trace), "final", trace[-1])
