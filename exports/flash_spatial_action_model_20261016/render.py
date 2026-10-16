"""Reproducible offline edit; no camera, robot, web server, or model inference.

Run from the repository with PYTHONPATH=src:benchmarks in the lerobot environment.
Original recordings are read-only. Every cut and timing decision is in manifest.json.
"""

import bisect
import json
import pathlib
import subprocess

import cv2
import numpy as np
from act_replay_bench import _demo
from PIL import Image, ImageDraw, ImageFont

from lerobot.gui.api.manipulation import rendering, replay

cv2.setNumThreads(2)
OUT = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path("/home/feit/.cache/huggingface/lerobot/demos")
FPS, W, H = 30, 1920, 1080
BG = (10, 16, 24)
WHITE, MUTED, ACCENT = (242, 245, 248), (168, 183, 197), (112, 234, 207)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
RUN = ROOT / "insertion/acts/20261016_112149"
META = json.loads((RUN / "act.json").read_text())
TIMELINE = json.loads((RUN / "camera/timeline.json").read_text())
STACKS = [
    ("20261011_080249", 13.4),
    ("20261011_233111", 20.7),
    ("20261012_041448", 19.0),
    ("20261012_042101", 15.2),
    ("20261012_042338", 19.3),
    ("20261012_042642", 15.6),
    ("20261012_192802", 20.0),
    ("20261012_194250", 33.5),
    ("20261013_211511", 18.7),
]
PLACEMENT_CENTRES = {
    "20261011_080249": (290, 271),
    "20261011_233111": (362, 251),
    "20261012_041448": (306, 273),
    "20261012_042101": (315, 305),
    "20261012_042338": (291, 314),
    "20261012_042642": (282, 324),
    "20261012_192802": (343, 294),
    "20261012_194250": (330, 292),
    "20261013_211511": (334, 282),
}
GRID_SECONDS = 7
GRID_CROP_SIZE = (320, 164)

# Keep the actual recorded overlay geometry. Only the diagnostic prose and the
# appended verification panel are omitted in this presentation, in this process.
replay._replay_text = lambda *args, **kwargs: None
rendering._pose_verification_panel = lambda im, *args, **kwargs: im


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def text(im, xy, value, size, fill=WHITE, bold=False, anchor=None):
    ImageDraw.Draw(im).text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def chrome(title, detail=""):
    im = Image.new("RGB", (W, H), BG)
    text(im, (36, 22), title, 35, bold=True)
    if detail:
        text(im, (1884, 28), detail, 25, MUTED, anchor="ra")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def scene_frame(bgr, base, progress):
    im = base.copy()
    # Retain the complete image, with its original aspect ratio.
    height = 960
    width = round(bgr.shape[1] * height / bgr.shape[0])
    x = (W - width) // 2
    im[88 : 88 + height, x : x + width] = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    return im


class Encoder:
    def __init__(self, path):
        self.path = path
        self.p = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{W}x{H}",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "3",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        assert self.p.wait() == 0, self.path


def encode(name, frames, count):
    path = OUT / name
    if path.exists() and path.stat().st_size > 10000:
        print(f"Already rendered: {name}", flush=True)
        return
    enc = Encoder(path)
    for i, frame in enumerate(frames):
        enc.write(frame)
        if i % 150 == 0:
            print(f"{name}: {i}/{count}", flush=True)
    assert i + 1 == count, (name, i + 1, count)
    enc.close()
    print(f"Complete: {name}", flush=True)


class Stack:
    def __init__(self, name, release, demo):
        self.run = ROOT / "pick_place/acts" / name
        self.meta = json.loads((self.run / "act.json").read_text())
        self.points = replay._replay_points(self.meta, demo)
        assert self.points.get("cube") is not None
        self.times = [f["t_frame"] - self.meta["t_started"] for f in self.meta["frames"]]
        self.start = 35.0 if name == "20261012_194250" else release - 0.5
        self.centre = PLACEMENT_CENTRES[name]
        self.last = None
        self.raw = self.drawn = None

    def frame(self, t, overlay):
        i = max(0, bisect.bisect_right(self.times, self.start + t) - 1)
        i = min(i, len(self.times) - 1)
        if i != self.last:
            raw = cv2.imread(str(self.run / "frames" / f"{i:06d}.jpg"))
            drawn = raw.copy()
            # Keep just the recorded cube silhouette in the grid. Removing the
            # point cloud, axes, wireframe and labels makes the actual result legible.
            frame = self.meta["frames"][i]
            with np.load(self.run / "frames" / f"{i:06d}.npz") as z:
                pose = z.get("onto_held")
                for k, held in enumerate(frame.get("held") or []):
                    if held["object"] == "cube":
                        pose = z.get(f"held_{k}")
                if pose is not None:
                    mask = rendering._found_mask(self.meta["intr"], self.points["cube"], pose, raw.shape)
                    if mask is not None:
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(drawn, contours, -1, rendering.runtime.FOUND_COLOUR, 1, cv2.LINE_AA)
            cx, cy = self.centre
            cw, ch = GRID_CROP_SIZE
            crop = np.s_[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2]
            self.raw = cv2.resize(raw[crop], (616, 316))
            self.drawn = cv2.resize(drawn[crop], (616, 316))
            self.last = i
        alpha = float(np.clip(4.0 - t, 0, 1)) if overlay else 0.0
        return cv2.addWeighted(self.drawn, alpha, self.raw, 1 - alpha, 0)


def grid(stacks, overlay):
    base = chrome("Different rollouts", "Autonomous, 1x speed")
    for n in range(GRID_SECONDS * FPS):
        canvas = base.copy()
        for k, stack in enumerate(stacks):
            x, y = 24 + (k % 3) * 628, 84 + (k // 3) * 328
            canvas[y : y + 316, x : x + 616] = stack.frame(n / FPS, overlay)
        yield canvas


def demo_frames(demo):
    times = np.loadtxt(ROOT / "insertion/recording/times.txt")
    base = chrome("Single demo")
    for n in range(13 * FPS):
        t = demo.t0 + 4.5 + n / FPS
        i = max(0, int(np.searchsorted(times, t, side="right") - 1))
        bgr = cv2.imread(str(ROOT / "insertion/recording/rgb" / f"{i:06d}.jpg"))
        assert bgr is not None
        for k, obj in enumerate(demo.objects.values()):
            masks = obj.get("masks")
            if masks is None or i >= len(masks):
                continue
            mask = cv2.resize(
                masks[i].astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            colour = np.array([(207, 234, 112), (80, 180, 250)][k % 2])
            bgr[mask] = (0.84 * bgr[mask] + 0.16 * colour).astype(np.uint8)
            rendering._outline(bgr, mask, tuple(int(x) for x in colour))
        yield scene_frame(bgr, base, (n + 1) / (13 * FPS))


def act_frames(demo, overlay):
    points = replay._replay_points(META, demo)
    rows = TIMELINE["rows"]
    if overlay:
        rows = rows[round(5.5 * FPS) : round(24.5 * FPS)]
    stamps = [f.get("t_result", f["t_frame"]) for f in META["frames"]]
    rawbase = chrome("Autonomous, 1x speed", "Raw")
    olddetail, base = None, None
    for n, row in enumerate(rows):
        i = row.get("image")
        raw = cv2.imread(str(RUN / "camera/rgb" / f"{i:06d}.jpg")) if i is not None else None
        if raw is None:
            raw = np.zeros((480, 848, 3), np.uint8)
        if overlay:
            at = float(row["t_wall"])
            source = bisect.bisect_right(stamps, at) - 1
            if source >= 0:
                raw = replay._replay_draw(
                    RUN,
                    META,
                    source,
                    points,
                    camera_bgr=raw,
                    at=at,
                    mono_at=row.get("camera_mono") or row["t_mono"],
                )
            t = at - META["t_started"]
            event = (
                "Manual ring displacement"
                if t < 8.5
                else "Approach and rotation"
                if t < 11
                else "Contact and tray perturbations"
                if t < 17
                else "Insertion"
                if t < 21.5
                else "Release and retreat"
            )
            detail = event
            if detail != olddetail:
                base = chrome(
                    "Autonomous, 1x speed",
                    detail,
                )
                olddetail = detail
        else:
            base = rawbase
        yield scene_frame(raw, base, (n + 1) / len(rows))


def title_frames(summary=False):
    cap = cv2.VideoCapture(str(OUT / "stacking_grid_raw.mp4"))
    count = 105 if summary else 45
    if summary:
        cap.set(cv2.CAP_PROP_POS_FRAMES, GRID_SECONDS * FPS - count)
    for _n in range(count):
        ok, bgr = cap.read()
        assert ok
        # Reuse only the footage area, without its labels, as an animated backdrop.
        bgr = cv2.resize(bgr[84:1056, 24:1896], (W, H))
        bgr = cv2.GaussianBlur(bgr, (0, 0), 9)
        bgr = (bgr.astype(float) * 0.32 + np.asarray(BG[::-1]) * 0.35).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if summary:
            text(im, (160, 330), "Single-demo generalization", 68, bold=True)
            text(im, (160, 425), "Object poses · Camera viewpoints", 52, ACCENT)
            text(im, (160, 620), "Robust to occlusion & perturbations", 64, bold=True)
        else:
            text(im, (W // 2, 320), "FLASH", 172, bold=True, anchor="ma")
            text(im, (W // 2, 550), "A Spatial-Action Model", 70, anchor="ma")
        yield cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    cap.release()


def main():
    insertion, stacking = _demo(ROOT / "insertion"), _demo(ROOT / "pick_place")
    stacks = [Stack(n, r, stacking) for n, r in STACKS]
    manifest = {
        "title": "FLASH: A Spatial-Action Model",
        "fps": FPS,
        "resolution": [W, H],
        "autonomous_run": str(RUN),
        "autonomous_camera_slots": len(TIMELINE["rows"]),
        "playback_speed": 1.0,
        "controller_speed_during_recording": 2.0,
        "demo_cut_seconds": [4.5, 17.5],
        "overlay_cut_seconds": [5.5, 24.5],
        "edit_revision": 2,
        "grid_seconds": GRID_SECONDS,
        "grid_layout": [3, 3],
        "grid_crop_size": GRID_CROP_SIZE,
        "grid_overlay_fade_seconds": [3, 4],
        "title_seconds": 1.5,
        "summary_seconds": 3.5,
        "grid_sources": [
            {
                "run": str(s.run),
                "start_seconds": s.start,
                "end_seconds": s.start + GRID_SECONDS,
                "crop_centre": s.centre,
            }
            for s in stacks
        ],
        "timing": "Camera timeline at 30 Hz; latest available recorded async result. Older grid clips use recorded t_frame timestamps and hold frames, with no optical-flow interpolation. Last frame held where a source ends.",
        "overlay": "Insertion uses existing replay drawing without diagnostic prose/panel. Grid shows only the recorded cube silhouette, fading to raw footage during retreat. Cube pose is a recorded accepted/predicted estimate, not ground truth.",
        "audit": {
            "completed": True,
            "visually_observed_insertion": True,
            "tracking_hz": 4.34177,
            "longest_motor_tick_ms": 603.44,
            "queue_empty_time_over_100ms_s": 3.150,
            "verified_post_lift_held_pose": False,
            "group_transport_withheld_updates": 10,
            "caveat": "Successful task, with remaining timing and held-pose verification limitations.",
        },
        "wording_sources": [
            "https://proceedings.mlr.press/v229/vitiello23a.html",
            "https://publications.ri.cmu.edu/view-generalizable-manipulation-policies",
        ],
        "no_robot_or_server_access": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    encode("stacking_grid_overlay.mp4", grid(stacks, True), GRID_SECONDS * FPS)
    encode("stacking_grid_raw.mp4", grid(stacks, False), GRID_SECONDS * FPS)
    encode("01_single_demo.mp4", demo_frames(insertion), 390)
    encode("02_autonomous_raw.mp4", act_frames(insertion, False), len(TIMELINE["rows"]))
    encode("03_autonomous_overlay.mp4", act_frames(insertion, True), 570)
    encode("00_title.mp4", title_frames(), 45)
    encode("05_summary.mp4", title_frames(True), 105)
    names = [
        "00_title.mp4",
        "01_single_demo.mp4",
        "02_autonomous_raw.mp4",
        "03_autonomous_overlay.mp4",
        "stacking_grid_overlay.mp4",
        "05_summary.mp4",
    ]
    concat = OUT / "concat.txt"
    concat.write_text("".join(f"file '{name}'\n" for name in names))
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(OUT / "FLASH_A_Spatial_Action_Model.mp4"),
        ],
        check=True,
    )
    print("MASTER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
