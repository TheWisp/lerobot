#!/usr/bin/env python
"""Batch extraction of S2 latents from Pi0.5 for dual-system VLA training.

Iterates over a LeRobot dataset, sends each frame to the Pi0.5 WebSocket server
with mode="extract_latent", and saves all latents as a .npy file.

Uses /dev/shm for zero-copy image transfer when running on localhost (same as
the real-time inference client), falling back to base64 with --no-shm.

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
import os
import time

import numpy as np
import websockets

from lerobot.datasets.lerobot_dataset import LeRobotDataset


SHM_DIR = "/dev/shm/s2_extract"


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


def encode_image_shm(image_tensor, model_key: str) -> dict:
    """Save image to /dev/shm and return shm_path reference."""
    if image_tensor.dtype != np.uint8:
        img = (image_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img = image_tensor.numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
    shm_path = os.path.join(SHM_DIR, f"{model_key}.npy")
    np.save(shm_path, img)
    return {"shm_path": shm_path}


# Map LeRobot image keys to Pi0.5 model keys
IMAGE_KEY_MAP = {
    "observation.images.front": "base_0_rgb",
    "observation.images.top": "base_1_rgb",
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
    use_shm: bool = True,
):
    """Extract S2 latents for all frames in a dataset."""
    print(f"Loading dataset from {dataset_path}...")
    dataset = LeRobotDataset(dataset_path)
    n_frames = len(dataset)
    print(f"Dataset has {n_frames} frames")

    # Set up shared memory directory
    if use_shm:
        os.makedirs(SHM_DIR, exist_ok=True)
        print(f"Using shared memory at {SHM_DIR}")
        encode_fn = encode_image_shm
    else:
        print("Using base64 encoding")
        encode_fn = None  # will use encode_image_base64

    # Resume support: auto-detect from existing output file
    latents = []
    timings = []
    resume_from = 0
    output_path_expanded = os.path.expanduser(output_path)
    if os.path.exists(output_path_expanded):
        existing = np.load(output_path_expanded)
        if existing.shape[0] >= n_frames:
            print(f"Output file already complete ({n_frames} frames), nothing to do")
            return existing
        resume_from = existing.shape[0]
        latents = list(existing)
        print(f"Found existing {output_path} with {resume_from}/{n_frames} frames, resuming")

    async with websockets.connect(server_uri, max_size=50 * 1024 * 1024) as ws:
        # Read server metadata
        metadata = json.loads(await ws.recv())
        print(f"Connected to server: {metadata.get('model', 'unknown')}")

        for idx in range(resume_from, n_frames):
            item = dataset[idx]

            # Prepare images
            images_data = {}
            for key in image_keys:
                if key in item:
                    model_key = IMAGE_KEY_MAP.get(key, key.split(".")[-1])
                    if use_shm:
                        images_data[model_key] = encode_image_shm(item[key], model_key)
                    else:
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
                eta_s = avg_ms * (n_frames - idx - 1) / 1000
                eta_h = eta_s / 3600
                print(f"  [{idx+1}/{n_frames}] avg: {avg_ms:.1f}ms | ETA: {eta_h:.1f}h")

            # Periodic save every 10000 frames
            if (idx + 1) % 10000 == 0:
                partial = np.stack(latents, axis=0)
                np.save(output_path, partial)
                print(f"  Checkpoint saved: {partial.shape}")

    # Clean up shared memory
    if use_shm:
        for f in os.listdir(SHM_DIR):
            os.remove(os.path.join(SHM_DIR, f))
        os.rmdir(SHM_DIR)

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
        default="observation.images.front,observation.images.wrist_left,observation.images.right_wrist",
        help="Comma-separated LeRobot image keys",
    )
    parser.add_argument("--state-key", default="observation.state", help="State key (empty to skip)")
    parser.add_argument("--no-shm", action="store_true", help="Disable shared memory, use base64 instead")
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
            use_shm=not args.no_shm,
        )
    )


if __name__ == "__main__":
    main()
