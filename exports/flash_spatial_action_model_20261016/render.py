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

# Keep the actual recorded overlay geometry. Only the diagnostic prose and the
# appended verification panel are omitted in this presentation, in this process.
replay._replay_text = lambda *args, **kwargs: None
rendering._pose_verification_panel = lambda im, *args, **kwargs: im


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def text(im, xy, value, size, fill=WHITE, bold=False, anchor=None):
    ImageDraw.Draw(im).text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def chrome(title, detail, footer=""):
    im = Image.new("RGB", (W, H), BG)
    text(im, (36, 25), title, 35, bold=True)
    text(im, (1884, 28), "FLASH", 29, ACCENT, True, "ra")
    text(im, (36, 73), detail, 23, MUTED)
    if footer:
        text(im, (36, 1046), footer, 20, MUTED)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def scene_frame(bgr, base, progress):
    im = base.copy()
    # Retain the complete image, with its original aspect ratio.
    height = 912
    width = round(bgr.shape[1] * height / bgr.shape[0])
    x = (W - width) // 2
    im[120 : 120 + height, x : x + width] = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(im, (x, 1034), (x + int(width * progress), 1037), ACCENT[::-1], -1)
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
        self.start = release - 6.0
        self.last = None
        self.raw = self.drawn = None

    def frame(self, t, overlay):
        i = max(0, bisect.bisect_right(self.times, self.start + t) - 1)
        i = min(i, len(self.times) - 1)
        if i != self.last:
            raw = cv2.imread(str(self.run / "frames" / f"{i:06d}.jpg"))
            drawn = replay._replay_draw(self.run, self.meta, i, self.points)
            # A fixed crop, identical throughout each clip: retain the placement
            # region, finger approach, release and retreat. No synthetic view.
            self.raw = cv2.resize(raw[88:432, 48:744], (616, 304))
            self.drawn = cv2.resize(drawn[88:432, 48:744], (616, 304))
            self.last = i
        return self.drawn if overlay else self.raw


def grid(stacks, overlay):
    detail = (
        "Nine placements · orange outline: estimated cube pose, including when occluded"
        if overlay
        else "Nine placements · raw camera views"
    )
    base = chrome(
        "Autonomous, 1x speed",
        detail,
        "Independent runs aligned around release · original capture cadence · final-frame holds",
    )
    for n in range(12 * FPS):
        canvas = base.copy()
        for k, stack in enumerate(stacks):
            x, y = 24 + (k % 3) * 628, 110 + (k // 3) * 316
            canvas[y : y + 304, x : x + 616] = stack.frame(n / FPS, overlay)
            cv2.rectangle(canvas, (x + 7, y + 7), (x + 45, y + 33), (20, 25, 30), -1)
            cv2.putText(
                canvas,
                f"{k + 1:02d}",
                (x + 13, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (235, 240, 245),
                1,
                cv2.LINE_AA,
            )
        yield canvas


def demo_frames(demo):
    times = np.loadtxt(ROOT / "insertion/recording/times.txt")
    base = chrome("Single demo", "One human demonstration · segmentation masks only")
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
    rawbase = chrome("Autonomous, 1x speed", "Complete rollout · raw camera video")
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
            detail = "Tracking overlay  /  " + event
            if detail != olddetail:
                base = chrome(
                    "Autonomous, 1x speed",
                    detail,
                    "Recorded estimates: segmentation · object poses · rigid motion groups",
                )
                olddetail = detail
        else:
            base = rawbase
        yield scene_frame(raw, base, (n + 1) / len(rows))


def title_frames(summary=False):
    cap = cv2.VideoCapture(str(OUT / "stacking_grid_raw.mp4"))
    count = 9 * FPS if summary else 105
    for _n in range(count):
        ok, bgr = cap.read()
        assert ok
        # Reuse only the footage area, without its labels, as an animated backdrop.
        bgr = cv2.resize(bgr[110:1046, 24:1896], (W, H))
        bgr = cv2.GaussianBlur(bgr, (0, 0), 18)
        bgr = (bgr.astype(float) * 0.28 + np.asarray(BG[::-1]) * 0.35).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(im)
        draw.rectangle((116, 180, 198, 188), fill=ACCENT)
        text(im, (110, 224), "FLASH", 112, bold=True)
        text(im, (116, 361), "A Spatial-Action Model", 54)
        if summary:
            text(im, (116, 535), "01", 29, ACCENT, True)
            text(im, (196, 526), "Object-pose and camera-viewpoint generalization", 38, bold=True)
            text(im, (196, 582), "from a single demonstration", 34, MUTED)
            text(im, (116, 704), "02", 29, ACCENT, True)
            text(im, (196, 695), "Robustness to occlusion and perturbations", 38, bold=True)
            text(im, (116, 939), "Recorded real-world examples", 24, MUTED)
        else:
            text(im, (116, 537), "One demonstration. New object configurations.", 34, MUTED)
            text(im, (116, 935), "Real robot · real-time playback", 26, ACCENT)
        yield cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    cap.release()


def main():
    insertion, stacking = _demo(ROOT / "insertion"), _demo(ROOT / "pick_place")
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
        "grid_seconds": 12.0,
        "grid_layout": [3, 3],
        "grid_release_alignment_seconds": 6.0,
        "grid_crop_xyxy": [48, 88, 744, 432],
        "grid_sources": [
            {
                "run": str(ROOT / "pick_place/acts" / n),
                "start_seconds": r - 6,
                "end_seconds": r + 6,
                "release_marker_seconds": r,
            }
            for n, r in STACKS
        ],
        "timing": "Camera timeline at 30 Hz; latest available recorded async result. Older grid clips use recorded t_frame timestamps and hold frames, with no optical-flow interpolation. Last frame held where a source ends.",
        "overlay": "Existing replay drawing, with diagnostic prose and verification panel omitted. Cube outline uses recorded accepted/predicted pose, not a newly inferred ground truth.",
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
    stacks = [Stack(n, r, stacking) for n, r in STACKS]
    encode("stacking_grid_overlay.mp4", grid(stacks, True), 360)
    encode("stacking_grid_raw.mp4", grid(stacks, False), 360)
    encode("01_single_demo.mp4", demo_frames(insertion), 390)
    encode("02_autonomous_raw.mp4", act_frames(insertion, False), len(TIMELINE["rows"]))
    encode("03_autonomous_overlay.mp4", act_frames(insertion, True), 570)
    encode("00_title.mp4", title_frames(), 105)
    encode("05_summary.mp4", title_frames(True), 270)
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
