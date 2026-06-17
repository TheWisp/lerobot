"""Build the customer highlight reel from assets/ -> highlight_reel.mp4.

Run `bash fetch_assets.sh` first (downloads the two VR clips). Then:
    /home/feit/miniforge3/envs/lerobot/bin/python build_reel.py

Edit the segment list at the bottom to retune (durations, VR windows, order).
1080p / CRF 18 / silent. Captions via drawtext (DejaVu Bold).
"""

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
D = ROOT / "assets"
SEG = Path(tempfile.mkdtemp(prefix="reel_segs_"))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = "0x0f1117"
W, H = 1920, 1080


def norm(caption, crop=None):
    cap = caption.replace(":", "\\:").replace("'", "")
    pre = f"crop={crop}," if crop else ""
    return (
        f"{pre}scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG},setsar=1,fps=30,format=yuv420p,"
        f"drawtext=fontfile={FONT}:text='{cap}':fontcolor=white:fontsize=44:"
        f"box=1:boxcolor=0x000000@0.62:boxborderw=22:x=(w-text_w)/2:y=h-104"
    )


def card(title, sub):
    return (
        f"drawtext=fontfile={FONT}:text='{title}':fontcolor=0x34d399:fontsize=96:"
        f"x=(w-text_w)/2:y=h/2-110,"
        f"drawtext=fontfile={FONT}:text='{sub}':fontcolor=0xcfd6e2:fontsize=44:"
        f"x=(w-text_w)/2:y=h/2+30,format=yuv420p"
    )


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print("FFMPEG ERR:", " ".join(args))
        print(r.stderr[-1500:])
        raise SystemExit(1)


def enc(out):
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-an",
        str(out),
    ]


segs = []


def title_card(name, title, sub, dur):
    out = SEG / f"{name}.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={BG}:s={W}x{H}:d={dur}:r=30",
            "-vf",
            card(title, sub),
            *enc(out),
        ]
    )
    segs.append(out)


def clip(name, src, vf, dur, ss=None):
    out = SEG / f"{name}.mp4"
    pre = ["ffmpeg", "-y", "-loglevel", "error"]
    if ss is not None:
        pre += ["-ss", str(ss)]
    pre += ["-t", str(dur), "-i", str(src), "-vf", vf, *enc(out)]
    run(pre)
    segs.append(out)


def loopclip(name, src, vf, dur, loops):
    out = SEG / f"{name}.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-stream_loop",
            str(loops),
            "-t",
            str(dur),
            "-i",
            str(src),
            "-vf",
            vf,
            *enc(out),
        ]
    )
    segs.append(out)


def still(name, src, vf, dur):
    out = SEG / f"{name}.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-t",
            str(dur),
            "-i",
            str(src),
            "-vf",
            vf,
            *enc(out),
        ]
    )
    segs.append(out)


def still_crop(name, src, crop, caption, dur):
    still(name, src, norm(caption, crop=crop), dur)


# ── reel timeline (edit me) ─────────────────────────────────────────────────
title_card("00_intro", "LeRobot Studio", "feature highlights", 2.5)
clip("01_vr_real", D / "vr_real.mp4", norm("VR teleoperation - Quest 3 to real bimanual SO-107"), 3, ss=12)
clip(
    "02_vr_virtual", D / "vr_virtual.mp4", norm("Quest 3 to a virtual robot - no hardware needed"), 2.5, ss=5
)
clip(
    "03_training",
    D / "training_dashboard.mp4",
    norm("Live training dashboard - loss descending in real time"),
    12,
)
still_crop(
    "035_rlt",
    D / "rlt.png",
    "1600:233:0:780",
    "HVLA RLT - online RL fine-tuning of a frozen policy, live telemetry",
    5,
)
loopclip("04_ik", D / "ik_circle.gif", norm("Cartesian IK - precise end-effector tracking"), 4, 4)
loopclip("05_replay", D / "record_replay.gif", norm("Record then replay on a virtual robot"), 8, 2)
title_card("06_nebius_title", "Cloud GPUs on Nebius", "train on a Nebius GPU straight from the GUI", 2.5)
still("07_nebius_shot", D / "nebius.png", norm("Spawn on demand - auto-destroy when the run ends"), 4.5)
# ────────────────────────────────────────────────────────────────────────────

listfile = SEG / "list.txt"
listfile.write_text("".join(f"file '{s}'\n" for s in segs))
out = ROOT / "highlight_reel.mp4"
run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listfile),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
)
dur = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out)],
    capture_output=True,
    text=True,
).stdout.strip()
print(f"OK -> {out}  ({float(dur):.1f}s, {out.stat().st_size // 1024} KiB)")
