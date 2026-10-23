"""Show two inferred hidden-end locations over a recorded RGB frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def tip_pixel(trace: Path, frame: int, point: np.ndarray, intrinsics: np.ndarray) -> tuple[int, int]:
    with np.load(trace) as poses:
        found = np.flatnonzero(poses["frames"] == frame)
        if len(found) != 1 or bool(poses["lost"][found[0]]):
            raise ValueError(f"no inferred pose for frame {frame} in {trace}")
        pose = poses["poses"][found[0]]
    projected = pose[:3, :3] @ point + pose[:3, 3]
    if projected[2] <= 0:
        raise ValueError("predicted end projects behind the camera")
    pixels = intrinsics @ (projected / projected[2])
    return round(float(pixels[0])), round(float(pixels[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--frame", type=int, default=499)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/feit/.cache/huggingface/lerobot/demos/insertion/recording"),
    )
    parser.add_argument("--recording", type=Path, default=root / "pose_traces.npz")
    parser.add_argument("--gripper", type=Path, default=root / "gripper_dowel_seed340_340_500.npz")
    parser.add_argument("--axis", type=Path, default=root / "gripper340_visible_axis_340_500.npz")
    parser.add_argument("--output", type=Path, default=root / "hidden_tip_axis_ablation_499.png")
    args = parser.parse_args()

    with np.load(args.recording) as recorded:
        center = recorded["dowel_reference_center_camera_m"].copy()
        axis = recorded["dowel_reference_shaft_camera"].copy()
        far_offset = float(recorded["dowel_reference_end_offsets_m"][1])
        intrinsics = recorded["intrinsics"].copy()
    point = center + far_offset * axis
    predictions = (
        ("Rigid grasp", args.gripper, "#47b4ff"),
        ("Depth-axis update", args.axis, "#ffab4e"),
    )
    points = [
        (name, tip_pixel(path, args.frame, point, intrinsics), color) for name, path, color in predictions
    ]
    image = Image.open(args.dataset / "rgb" / f"{args.frame:06d}.jpg").convert("RGB")
    crop = image.crop((175, 65, 330, 260)).resize((465, 585), Image.Resampling.NEAREST)
    canvas = Image.new(
        "RGB", (image.width + crop.width + 30, max(image.height, crop.height) + 115), "#10151c"
    )
    canvas.paste(image, (0, 48))
    canvas.paste(crop, (image.width + 30, 48))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    title = ImageFont.truetype("DejaVuSans.ttf", 20)
    draw.text((16, 12), f"Recorded RGB • frame {args.frame}", font=title, fill="white")
    draw.text((image.width + 45, 12), "Insertion close-up", font=title, fill="white")
    for _, (x, y), color in points:
        draw.ellipse((x - 5, y + 48 - 5, x + 5, y + 48 + 5), outline=color, width=3)
        zoom_x = image.width + 30 + (x - 175) * 3
        zoom_y = 48 + (y - 65) * 3
        draw.ellipse((zoom_x - 9, zoom_y - 9, zoom_x + 9, zoom_y + 9), outline=color, width=4)
    footer_y = max(image.height, crop.height) + 58
    draw.text(
        (16, footer_y), "Blue: rigid grasp     Orange: current depth-axis correction", font=font, fill="white"
    )
    draw.text(
        (16, footer_y + 24),
        "Both are predictions of a hidden end, not measured tip positions.",
        font=font,
        fill="#c5cbd4",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
