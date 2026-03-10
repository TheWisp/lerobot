#!/usr/bin/env python
"""Batch extraction of S2 latents from Pi0.5 for dual-system VLA training.

Iterates over a LeRobot dataset, sends each frame to the Pi0.5 WebSocket server
with mode="extract_latent", and saves all latents as a .npy file.

Usage:
    python scripts/extract_s2_latents.py \
        --dataset-path /path/to/lerobot/dataset \
        --server-uri ws://localhost:8765 \
        --high-level-prompt "do the task" \
        --output-path s2_latents.npy \
        --image-keys observation.images.front,observation.images.wrist_left,observation.images.wrist_right
"""

import argparse
import asyncio
import base64
import json
import time

import numpy as np
import websockets

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def encode_image_base64(image_tensor) -> dict:
    """Convert a CHW float tensor [0,1] or uint8 to base64-encoded dict."""
    if image_tensor.dtype != np.uint8:
        img = (image_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img = image_tensor.numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
    raw = img.tobytes()
    return {
        "base64": base64.b64encode(raw).decode("ascii"),
        "shape": list(img.shape),
    }


# Map LeRobot image keys to Pi0.5 model keys
IMAGE_KEY_MAP = {
    "observation.images.front": "base_0_rgb",
    "observation.images.top": "base_0_rgb",
    "observation.images.wrist_left": "left_wrist_0_rgb",
    "observation.images.left_wrist": "left_wrist_0_rgb",
    "observation.images.wrist_right": "right_wrist_0_rgb",
    "observation.images.right_wrist": "right_wrist_0_rgb",
}


async def extract_latents(
    dataset_path: str,
    server_uri: str,
    high_level_prompt: str,
    output_path: str,
    image_keys: list[str],
    state_key: str | None = "observation.state",
    batch_size: int = 1,
):
    """Extract S2 latents for all frames in a dataset."""
    print(f"Loading dataset from {dataset_path}...")
    dataset = LeRobotDataset(dataset_path)
    n_frames = len(dataset)
    print(f"Dataset has {n_frames} frames")

    latents = []
    timings = []

    async with websockets.connect(server_uri, max_size=50 * 1024 * 1024) as ws:
        # Read server metadata
        metadata = json.loads(await ws.recv())
        print(f"Connected to server: {metadata.get('model', 'unknown')}")

        for idx in range(n_frames):
            item = dataset[idx]

            # Prepare images
            images_data = {}
            for key in image_keys:
                if key in item:
                    model_key = IMAGE_KEY_MAP.get(key, key.split(".")[-1])
                    images_data[model_key] = encode_image_base64(item[key])

            if not images_data:
                raise ValueError(f"No images found for keys {image_keys} at index {idx}")

            # Prepare state
            state = None
            if state_key and state_key in item:
                state = item[state_key].numpy().tolist()

            # Build request
            request = {
                "mode": "extract_latent",
                "images": images_data,
                "high_level_prompt": high_level_prompt,
            }
            if state is not None:
                request["state"] = state

            # Send and receive
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())

            if response.get("status") != "success":
                raise RuntimeError(f"Server error at frame {idx}: {response.get('error')}")

            latent = np.array(response["s2_latent"], dtype=np.float32)
            latents.append(latent)
            timings.append(response["timing"]["total_ms"])

            if (idx + 1) % 100 == 0 or idx == n_frames - 1:
                avg_ms = np.mean(timings[-100:])
                print(f"  [{idx+1}/{n_frames}] avg latent extraction: {avg_ms:.1f}ms")

    # Stack and save
    latents_array = np.stack(latents, axis=0)  # [N_frames, 2048]
    print(f"Saving latents with shape {latents_array.shape} to {output_path}")
    np.save(output_path, latents_array)

    print(f"Done. Mean extraction time: {np.mean(timings):.1f}ms")
    return latents_array


def main():
    parser = argparse.ArgumentParser(description="Extract S2 latents from Pi0.5 for dual-system VLA")
    parser.add_argument("--dataset-path", required=True, help="Path or repo_id of LeRobot dataset")
    parser.add_argument("--server-uri", default="ws://localhost:8765", help="Pi0.5 WebSocket server URI")
    parser.add_argument("--high-level-prompt", default="do the task", help="High-level task prompt")
    parser.add_argument("--output-path", default="s2_latents.npy", help="Output .npy file path")
    parser.add_argument(
        "--image-keys",
        default="observation.images.front,observation.images.wrist_left,observation.images.wrist_right",
        help="Comma-separated LeRobot image keys",
    )
    parser.add_argument("--state-key", default="observation.state", help="State key (empty to skip)")
    args = parser.parse_args()

    image_keys = [k.strip() for k in args.image_keys.split(",")]
    state_key = args.state_key if args.state_key else None

    asyncio.run(
        extract_latents(
            dataset_path=args.dataset_path,
            server_uri=args.server_uri,
            high_level_prompt=args.high_level_prompt,
            output_path=args.output_path,
            image_keys=image_keys,
            state_key=state_key,
        )
    )


if __name__ == "__main__":
    main()
