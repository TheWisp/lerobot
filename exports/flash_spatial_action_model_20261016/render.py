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
GRID_SECONDS = 9
GRID_CROP_SIZE = (320, 164)
INSERTIONS = [
    ("20261015_165734", (364, 250)),
    ("20261015_184144", (278, 224)),
    ("20261015_233027", (328, 250)),
    ("20261016_005114", (318, 234)),
    ("20261016_022104", (417, 337)),
    ("20261016_075037", (361, 299)),
    ("20261016_100628", (367, 338)),
    ("20261016_110310", (331, 302)),
    ("20261016_112149", (366, 294)),
]

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


def mask_outline(drawn, mask, colour):
    if mask is not None:
        contours, _ = cv2.findContours(
            np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(drawn, contours, -1, colour, 1, cv2.LINE_AA)


def pose_outline(drawn, intr, points, pose, _label, colour=rendering.runtime.FOUND_COLOUR):
    mask_outline(drawn, rendering._found_mask(intr, points, pose, drawn.shape), colour)


def object_outlines(raw, run, meta, i, points):
    """Retain all recorded object silhouettes and scene wireframes, without point dots or labels."""
    from lerobot.showservo import groups_scene

    drawn = raw.copy()
    if i < 0:
        return drawn
    frame = meta["frames"][i]
    names = replay._replay_objects(meta)
    colours = {
        n: rendering.runtime.REPLAY_COLOURS[k % len(rendering.runtime.REPLAY_COLOURS)]
        for k, n in enumerate(names)
    }
    with np.load(run / "frames" / f"{i:06d}.npz") as z:
        recorded_groups = frame.get("groups") or {}
        if "groups_links" in z and "groups_points" in z:
            picture = {"depth_range": recorded_groups.get("depth_range")}
            picture.update(
                points=z["groups_points"],
                links=z["groups_links"],
                depth=z.get("groups_depth"),
                joins=z.get("groups_joins"),
            )
            drawn = groups_scene.paint_wire(drawn, picture, bar=False)
        mask = cv2.imread(str(run / "frames" / f"{i:06d}_mask.png"), cv2.IMREAD_GRAYSCALE)
        mask_outline(drawn, None if mask is None else mask > 0, colours[names[0]])
        others = frame.get("others") if frame.get("others") is not None else names[1:2]
        for k, name in enumerate(others):
            mask_outline(
                drawn, z.get(f"other_mask_{k}"), colours.get(name, rendering.runtime.REPLAY_COLOURS[-1])
            )
        # Reuse the replay's recorded-model validity rules, drawing only its
        # silhouette in this presentation. Restore the live renderer immediately.
        original = rendering._draw_found
        try:
            rendering._draw_found = pose_outline
            rendering._draw_track_estimate(drawn, meta["intr"], frame, z, names[0])
        finally:
            rendering._draw_found = original
        held_poses = {}
        if "onto_held" in z and len(names) > 1:
            held_poses[names[1]] = z["onto_held"]
        for k, held in enumerate(frame.get("held") or []):
            held_poses[held["object"]] = z.get(f"held_{k}")
        for name, pose in held_poses.items():
            if pose is not None and points.get(name) is not None:
                pose_outline(drawn, meta["intr"], points[name], pose, name)
    return drawn


def tile_crop(frame, centre):
    cx, cy = centre
    cw, ch = GRID_CROP_SIZE
    return cv2.resize(frame[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2], (616, 316))


class Stack:
    def __init__(self, name, release, demo):
        self.run = ROOT / "pick_place/acts" / name
        self.meta = json.loads((self.run / "act.json").read_text())
        self.points = replay._replay_points(self.meta, demo)
        assert self.points.get("cube") is not None
        self.times = [f["t_frame"] - self.meta["t_started"] for f in self.meta["frames"]]
        self.start = 33.0 if name == "20261012_194250" else release - 2.5
        self.centre = PLACEMENT_CENTRES[name]
        self.last = None
        self.raw = self.drawn = None

    def frame(self, t, overlay):
        i = max(0, bisect.bisect_right(self.times, self.start + t) - 1)
        i = min(i, len(self.times) - 1)
        if i != self.last:
            raw = cv2.imread(str(self.run / "frames" / f"{i:06d}.jpg"))
            drawn = object_outlines(raw, self.run, self.meta, i, self.points)
            self.raw = tile_crop(raw, self.centre)
            self.drawn = tile_crop(drawn, self.centre)
            self.last = i
        return self.drawn if overlay else self.raw


class Insertion:
    def __init__(self, name, centre, demo):
        self.run = ROOT / "insertion/acts" / name
        self.meta = json.loads((self.run / "act.json").read_text())
        self.rows = json.loads((self.run / "camera/timeline.json").read_text())["rows"]
        self.times = [r["t_wall"] - self.meta["t_started"] for r in self.rows]
        self.result_times = [f.get("t_result", f["t_frame"]) for f in self.meta["frames"]]
        self.points = replay._replay_points(self.meta, demo)
        assert self.points.get("ring") is not None
        release = next(f["t_frame"] for f in self.meta["frames"] if f["step"] == "key frame 8")
        self.start = release - self.meta["t_started"] - 4.5
        self.centre = centre
        self.last = None
        self.raw = self.drawn = None

    def frame(self, t, overlay):
        row = self.rows[max(0, bisect.bisect_right(self.times, self.start + t) - 1)]
        source = bisect.bisect_right(self.result_times, row["t_wall"]) - 1
        key = (row["image"], source)
        if key != self.last:
            raw = cv2.imread(str(self.run / "camera/rgb" / f"{row['image']:06d}.jpg"))
            assert raw is not None
            drawn = object_outlines(raw, self.run, self.meta, source, self.points)
            self.raw = tile_crop(raw, self.centre)
            self.drawn = tile_crop(drawn, self.centre)
            self.last = key
        return self.drawn if overlay else self.raw


def grid(stacks, overlay, subject="pick and place"):
    base = chrome(f"Different rollouts: {subject}", "Autonomous, 1x speed")
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
            text(im, (160, 245), "Single-demo generalization", 68, bold=True)
            text(im, (160, 340), "Object poses · Camera viewpoints", 52, ACCENT)
            text(im, (160, 520), "Robust to occlusion & perturbations", 64, bold=True)
            text(im, (160, 700), "Real time", 64, bold=True)
        else:
            text(im, (W // 2, 350), "Preview: FLASH", 128, bold=True, anchor="ma")
            text(im, (W // 2, 550), "A Spatial-Action Model", 70, anchor="ma")
        yield cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    cap.release()


def main():
    insertion, stacking = _demo(ROOT / "insertion"), _demo(ROOT / "pick_place")
    stacks = [Stack(n, r, stacking) for n, r in STACKS]
    insertions = [Insertion(n, c, insertion) for n, c in INSERTIONS]
    manifest = {
        "title": "Preview: FLASH — A Spatial-Action Model",
        "fps": FPS,
        "resolution": [W, H],
        "autonomous_run": str(RUN),
        "autonomous_camera_slots": len(TIMELINE["rows"]),
        "playback_speed": 1.0,
        "controller_speed_during_recording": 2.0,
        "demo_cut_seconds": [4.5, 17.5],
        "overlay_cut_seconds": [5.5, 24.5],
        "edit_revision": 4,
        "grid_seconds": GRID_SECONDS,
        "grid_layout": [3, 3],
        "grid_crop_size": GRID_CROP_SIZE,
        "grid_overlay_fade_seconds": None,
        "summary_real_time_claim": "Real time",
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
        "insertion_grid_sources": [
            {
                "run": str(s.run),
                "start_seconds": s.start,
                "end_seconds": s.start + GRID_SECONDS,
                "crop_centre": s.centre,
            }
            for s in insertions
        ],
        "insertion_grid_selection": "Nine camera-timeline runs visually checked for a seated dowel and visible retreat. All use the same crop dimensions as the placement grid, centred on insertion. Completed runs with a missed ring were excluded.",
        "timing": "Camera timeline at 30 Hz; latest available recorded async result. Older grid clips use recorded t_frame timestamps and hold frames, with no optical-flow interpolation. Last frame held where a source ends.",
        "overlay": "Main insertion replay uses existing drawing without diagnostic prose/panel. Both grids retain every available recorded segmentation outline, raw-model silhouette, accepted/predicted target silhouette and scene wireframe through the last frame, without fading. Missing recorded geometry is not fabricated. Poses are estimates, not ground truth.",
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
    encode("insertion_grid_overlay.mp4", grid(insertions, True, "dowel insertion"), GRID_SECONDS * FPS)
    encode("insertion_grid_raw.mp4", grid(insertions, False, "dowel insertion"), GRID_SECONDS * FPS)
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
        "insertion_grid_overlay.mp4",
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
