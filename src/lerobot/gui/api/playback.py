# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WebSocket playback endpoint for streaming video frames."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["playback"])

# Will be set by server.py
_app_state: "AppState" = None  # type: ignore


def set_app_state(state: "AppState") -> None:
    """Set the application state for WebSocket handlers."""
    global _app_state
    _app_state = state


@router.websocket("/ws/playback/{dataset_id:path}")
async def playback_stream(websocket: WebSocket, dataset_id: str):
    """WebSocket endpoint for streaming video playback.

    Protocol:
        Client sends JSON commands:
            {"cmd": "start", "episode": int, "camera": str, "fps": int}
            {"cmd": "pause"}
            {"cmd": "resume"}
            {"cmd": "seek", "frame": int}
            {"cmd": "stop"}

        Server sends JSON messages:
            {"type": "frame", "frame_idx": int, "data": base64_jpeg}
            {"type": "status", "playing": bool, "frame_idx": int, "total_frames": int}
            {"type": "error", "message": str}
    """
    await websocket.accept()

    if dataset_id not in _app_state.datasets:
        await websocket.send_json({"type": "error", "message": f"Dataset not found: {dataset_id}"})
        await websocket.close()
        return

    dataset = _app_state.datasets[dataset_id]

    # Playback state
    playing = False
    episode_idx = 0
    camera_key = list(dataset.meta.camera_keys)[0] if dataset.meta.camera_keys else None
    current_frame = 0
    total_frames = 0
    target_fps = dataset.fps
    play_task = None

    # Load episode info
    episodes = dataset.meta.episodes
    if episodes is None:
        from lerobot.datasets.utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    async def send_frame(frame_idx: int):
        """Send a single frame to the client."""
        nonlocal current_frame
        if frame_idx < 0 or frame_idx >= total_frames:
            return

        current_frame = frame_idx
        ep = episodes[episode_idx]
        global_idx = ep["dataset_from_index"] + frame_idx

        def decode_frame():
            item = dataset[global_idx]
            return item[camera_key]

        try:
            jpeg_bytes = _app_state.frame_cache.get_or_decode(
                dataset_id=dataset_id,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                camera_key=camera_key,
                decode_fn=decode_frame,
            )

            await websocket.send_json(
                {
                    "type": "frame",
                    "frame_idx": frame_idx,
                    "data": base64.b64encode(jpeg_bytes).decode("ascii"),
                }
            )
        except Exception as e:
            logger.exception(f"Error sending frame {frame_idx}: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})

    async def send_status():
        """Send current playback status."""
        await websocket.send_json(
            {
                "type": "status",
                "playing": playing,
                "frame_idx": current_frame,
                "total_frames": total_frames,
                "episode_idx": episode_idx,
                "camera": camera_key,
                "fps": target_fps,
            }
        )

    async def play_loop():
        """Playback loop that sends frames at the target FPS."""
        nonlocal current_frame, playing
        frame_interval = 1.0 / target_fps

        while playing:
            start_time = asyncio.get_event_loop().time()

            if current_frame >= total_frames - 1:
                # Loop back to start
                current_frame = 0
            else:
                current_frame += 1

            await send_frame(current_frame)

            # Maintain frame rate
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    try:
        await send_status()

        while True:
            try:
                message = await websocket.receive_text()
                cmd = json.loads(message)
            except WebSocketDisconnect:
                break

            command = cmd.get("cmd")

            if command == "start":
                # Start or switch playback
                new_episode = cmd.get("episode", episode_idx)
                new_camera = cmd.get("camera", camera_key)
                target_fps = cmd.get("fps", dataset.fps)

                if new_episode != episode_idx:
                    episode_idx = new_episode
                    current_frame = 0

                    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
                        await websocket.send_json({"type": "error", "message": f"Invalid episode: {episode_idx}"})
                        continue

                    total_frames = episodes[episode_idx]["length"]

                if new_camera and new_camera in dataset.meta.camera_keys:
                    camera_key = new_camera

                playing = True

                # Cancel existing play task
                if play_task and not play_task.done():
                    play_task.cancel()
                    try:
                        await play_task
                    except asyncio.CancelledError:
                        pass

                await send_status()
                play_task = asyncio.create_task(play_loop())

            elif command == "pause":
                playing = False
                if play_task and not play_task.done():
                    play_task.cancel()
                    try:
                        await play_task
                    except asyncio.CancelledError:
                        pass
                await send_status()

            elif command == "resume":
                if total_frames > 0 and not playing:
                    playing = True
                    await send_status()
                    play_task = asyncio.create_task(play_loop())

            elif command == "seek":
                frame_idx = cmd.get("frame", 0)
                was_playing = playing

                # Pause during seek
                if playing:
                    playing = False
                    if play_task and not play_task.done():
                        play_task.cancel()
                        try:
                            await play_task
                        except asyncio.CancelledError:
                            pass

                await send_frame(frame_idx)
                await send_status()

                # Resume if was playing
                if was_playing:
                    playing = True
                    play_task = asyncio.create_task(play_loop())

            elif command == "stop":
                playing = False
                if play_task and not play_task.done():
                    play_task.cancel()
                    try:
                        await play_task
                    except asyncio.CancelledError:
                        pass
                current_frame = 0
                await send_status()

            elif command == "status":
                await send_status()

            elif command == "frame":
                # Request single frame without changing playback state
                frame_idx = cmd.get("frame", current_frame)
                await send_frame(frame_idx)

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        playing = False
        if play_task and not play_task.done():
            play_task.cancel()
