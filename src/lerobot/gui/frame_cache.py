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
"""
Frame cache with LRU eviction based on memory budget.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# Still-frame profiles, mirroring the video PLAYBACK_PROFILES: (max_width, quality).
# ``full`` keeps the source resolution, which is what every caller got before
# the profile existed.
STILL_PROFILES: dict[str, tuple[int, int]] = {
    "low": (640, 70),
    "medium": (1280, 82),
    "full": (0, 85),
}


def encode_frame_to_jpeg(frame: torch.Tensor, quality: int = 85, max_width: int = 0) -> bytes:
    """Convert a torch tensor frame to JPEG bytes.

    Uses torchvision.io.encode_jpeg for fast encoding (libjpeg-turbo),
    avoiding the numpy/PIL conversion overhead.

    Args:
        frame: Tensor of shape (C, H, W) or (H, W, C) with values in [0, 255] or [0, 1]
        quality: JPEG quality (1-100)
        max_width: Downscale so the width is at most this, preserving aspect.
            0 (default) keeps the source resolution. Scaling before encoding is
            what actually saves bytes — quality alone barely moves a 640×480
            frame, and the wire cost is dominated by pixel count.

    Returns:
        JPEG encoded bytes
    """
    import torch
    from torchvision.io import encode_jpeg

    # Ensure C, H, W format (encode_jpeg requires CHW)
    if frame.dim() == 3 and frame.shape[0] not in (1, 3, 4):
        frame = frame.permute(2, 0, 1)  # H, W, C -> C, H, W
    elif frame.dim() == 2:
        frame = frame.unsqueeze(0)  # H, W -> 1, H, W

    # Convert to uint8 if needed
    if frame.is_floating_point():
        frame = (frame * 255).clamp(0, 255).to(torch.uint8)
    elif frame.dtype != torch.uint8:
        frame = frame.to(torch.uint8)

    frame = frame.cpu().contiguous()

    if max_width and frame.shape[-1] > max_width:
        import torch.nn.functional as F  # noqa: N812

        height = max(1, round(frame.shape[-2] * max_width / frame.shape[-1]))
        frame = (
            F.interpolate(
                frame.unsqueeze(0).float(),
                size=(height, max_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            .squeeze(0)
            .clamp(0, 255)
            .to(torch.uint8)
            .contiguous()
        )

    return encode_jpeg(frame, quality=quality).numpy().tobytes()


def encode_frame_for_profile(frame: torch.Tensor, profile: str = "full") -> bytes:
    """Encode at the named still profile. Unknown names fall back to source."""
    max_width, quality = STILL_PROFILES.get(profile, STILL_PROFILES["full"])
    return encode_frame_to_jpeg(frame, quality=quality, max_width=max_width)


class FrameCache:
    """LRU cache for decoded video frames with memory budget.

    Thread-safe cache that stores JPEG-encoded frames and evicts oldest
    entries when the memory budget is exceeded.
    """

    def __init__(self, max_bytes: int = 1_000_000_000):
        """Initialize the frame cache.

        Args:
            max_bytes: Maximum memory budget in bytes (default 1 GB)
        """
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.cache: OrderedDict[tuple, bytes] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(
        self, dataset_id: str, episode_idx: int, frame_idx: int, camera_key: str, variant: str = ""
    ) -> tuple:
        """Create a cache key from frame identifiers.

        ``variant`` distinguishes different renderings of the SAME frame, and
        the cache holds *encoded* bytes, so two things decide it: the saved-mask
        recipe composited in, and the quality profile it was encoded at. Without
        it a composited frame and a raw one collide, and whichever was decoded
        first is served for both -- which also made the bandwidth setting appear
        to do nothing once the cache was warm. With the recipe's fingerprint and
        the profile in the key, an edit invalidates by construction rather than
        by remembering to clear anything.
        """
        return (dataset_id, episode_idx, frame_idx, camera_key, variant)

    def contains(
        self, dataset_id: str, episode_idx: int, frame_idx: int, camera_key: str, variant: str = ""
    ) -> bool:
        """Check if a frame is cached without affecting LRU order.

        Args:
            dataset_id: Unique identifier for the dataset
            episode_idx: Episode index
            frame_idx: Frame index within the episode
            camera_key: Camera/image key

        Returns:
            True if the frame is cached, False otherwise
        """
        key = self._make_key(dataset_id, episode_idx, frame_idx, camera_key, variant)
        with self.lock:
            return key in self.cache

    def is_episode_cached(self, dataset_id: str, episode_idx: int, ep_length: int, camera_key: str) -> bool:
        """Check if all frames of an episode are cached for a given camera.

        Uses a single lock acquisition to check all frames efficiently.
        """
        with self.lock:
            return all((dataset_id, episode_idx, fi, camera_key, "") in self.cache for fi in range(ep_length))

    def get(
        self, dataset_id: str, episode_idx: int, frame_idx: int, camera_key: str, variant: str = ""
    ) -> bytes | None:
        """Get a cached frame if available.

        Args:
            dataset_id: Unique identifier for the dataset
            episode_idx: Episode index
            frame_idx: Frame index within the episode
            camera_key: Camera/image key (e.g., "observation.images.front")

        Returns:
            JPEG bytes if cached, None otherwise
        """
        key = self._make_key(dataset_id, episode_idx, frame_idx, camera_key, variant)

        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]

            self.misses += 1
            return None

    def put(
        self,
        dataset_id: str,
        episode_idx: int,
        frame_idx: int,
        camera_key: str,
        jpeg_bytes: bytes,
        variant: str = "",
    ) -> None:
        """Store a frame in the cache.

        Args:
            dataset_id: Unique identifier for the dataset
            episode_idx: Episode index
            frame_idx: Frame index within the episode
            camera_key: Camera/image key
            jpeg_bytes: JPEG-encoded frame data
        """
        key = self._make_key(dataset_id, episode_idx, frame_idx, camera_key, variant)
        size = len(jpeg_bytes)

        with self.lock:
            # If already cached, update and move to end
            if key in self.cache:
                old_size = len(self.cache[key])
                self.current_bytes -= old_size
                self.cache[key] = jpeg_bytes
                self.current_bytes += size
                self.cache.move_to_end(key)
                return

            # Add new entry
            self.cache[key] = jpeg_bytes
            self.current_bytes += size

            # Evict oldest entries until under budget
            while self.current_bytes > self.max_bytes and len(self.cache) > 1:
                _, evicted = self.cache.popitem(last=False)
                self.current_bytes -= len(evicted)

    def get_or_decode(
        self,
        dataset_id: str,
        episode_idx: int,
        frame_idx: int,
        camera_key: str,
        decode_fn,
        jpeg_quality: int = 85,
        profile: str = "full",
    ) -> bytes:
        """Get a cached frame or decode it if not cached.

        Args:
            dataset_id: Unique identifier for the dataset
            episode_idx: Episode index
            frame_idx: Frame index within the episode
            camera_key: Camera/image key
            decode_fn: Function that returns a torch tensor frame when called
            jpeg_quality: JPEG quality for encoding (1-100)

        Returns:
            JPEG bytes of the frame
        """
        # Check cache first
        cached = self.get(dataset_id, episode_idx, frame_idx, camera_key, profile)
        if cached is not None:
            return cached

        # Decode and encode
        frame_tensor = decode_fn()
        max_width = STILL_PROFILES.get(profile, STILL_PROFILES["full"])[0]
        jpeg_bytes = encode_frame_to_jpeg(frame_tensor, quality=jpeg_quality, max_width=max_width)

        # Cache it
        self.put(dataset_id, episode_idx, frame_idx, camera_key, jpeg_bytes)

        return jpeg_bytes

    def clear(self) -> None:
        """Clear all cached frames."""
        with self.lock:
            self.cache.clear()
            self.current_bytes = 0

    def invalidate_dataset(self, dataset_id: str) -> int:
        """Invalidate all cached frames for a specific dataset.

        Args:
            dataset_id: The dataset identifier to invalidate

        Returns:
            Number of entries removed
        """
        with self.lock:
            keys_to_remove = [k for k in self.cache if k[0] == dataset_id]
            for key in keys_to_remove:
                self.current_bytes -= len(self.cache[key])
                del self.cache[key]
            return len(keys_to_remove)

    def stats(self) -> dict:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            return {
                "entries": len(self.cache),
                "current_bytes": self.current_bytes,
                "max_bytes": self.max_bytes,
                "usage_percent": (self.current_bytes / self.max_bytes) * 100 if self.max_bytes > 0 else 0,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }
