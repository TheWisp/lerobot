"""Render source RGB and supplied masks at first visibility, first pose loss, and clip end."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lerobot_dataset import LeRobotDataset

COLORS = ((0, 255, 255), (255, 0, 255))


def render(results: list[dict], foundry: dict, output: Path) -> list[Path]:
    episode_info = {value["episode_index"]: value for value in foundry["episodes"].values()}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    written = []
    for page in range(0, len(results), 5):
        canvas = Image.new("RGB", (840, min(5, len(results) - page) * 420), "white")
        for row_index, summary in enumerate(results[page : page + 5]):
            root = Path(summary["dataset"])
            source = json.loads((root / "flash/v1/source.json").read_text())
            episode = int(source["source"].rsplit(" ", 1)[1])
            info = episode_info[episode]
            names = info["contact_slot_object_labels"]
            draw = ImageDraw.Draw(canvas)
            labels = ", ".join(f"{chr(65 + index)}={name}" for index, name in enumerate(names))
            draw.text(
                (8, row_index * 420 + 4),
                f"Episode {episode}: {info['task']} — {labels}",
                font=font,
                fill="black",
            )
            roles = list(summary["roles"])
            visible_frames = [
                index
                for index in range(summary["frames"])
                if any(
                    np.any(cv2.imread(str(root / f"flash/v1/imported-masks/{role}/{index:06d}.png"), 0))
                    for role in roles
                )
            ]
            first, last = visible_frames[0], visible_frames[-1]
            failures = [
                item["first_lost_while_visible"]
                for item in summary["roles"].values()
                if item["first_lost_while_visible"] is not None
            ]
            failure = min(failures) if failures else None
            middle = failure if failure is not None else visible_frames[len(visible_frames) // 2]
            dataset = LeRobotDataset(source["repo_id"], root=root, video_backend="pyav")
            for column, frame_id in enumerate((first, middle, last)):
                rgb = (
                    np.rint(dataset[frame_id]["observation.images.rgb"].numpy().transpose(1, 2, 0) * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                rgb = np.ascontiguousarray(rgb)
                evidence_path = root / f"flash/v1/observation-evidence/episode_000000/{frame_id:06d}.json.gz"
                entry = json.loads(gzip.decompress(evidence_path.read_bytes()))
                states = []
                for role_index, role in enumerate(roles):
                    mask = cv2.imread(str(root / f"flash/v1/imported-masks/{role}/{frame_id:06d}.png"), 0)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(rgb, contours, -1, COLORS[role_index], 2)
                    states.append(f"{chr(65 + role_index)}:{entry['estimates'][role]['evidence']}")
                image = Image.fromarray(cv2.resize(rgb, (280, 360)))
                offset = (column * 280, row_index * 420 + 60)
                canvas.paste(image, offset)
                draw.text(
                    (column * 280 + 6, row_index * 420 + 30),
                    f"Frame {frame_id} | {'; '.join(states)}",
                    font=small,
                    fill="black",
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        target = output.with_name(f"{output.stem}_{page // 5 + 1}{output.suffix}")
        canvas.save(target)
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--foundry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.metrics.read_text())
    foundry = json.loads(args.foundry.read_text())
    for path in render(results, foundry, args.output):
        print(path)


if __name__ == "__main__":
    main()
