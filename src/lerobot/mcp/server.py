"""FastMCP server exposing LeRobot datasets to AI agents over stdio.

Tools (v1):
  list_datasets               — discover datasets under $HF_LEROBOT_HOME
  get_dataset_info            — schema, feature shapes, length stats
  list_episodes               — paginated episode listing with summary fields
  get_episode_summary         — length, tasks, terminal-state hints
  get_frame                   — one image at (episode, frame_idx, camera)
  tag_episode                 — write a sidecar annotation
  get_episode_tags            — read sidecar annotations

Design intent: every tool takes logical IDs (``repo_id``, ``episode_id``),
never file paths. The server resolves to disk internally so that the same
surface works locally today and over a network later.
"""

from __future__ import annotations

import io
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.server import AuthSettings
from PIL import Image as PILImage

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.constants import HF_LEROBOT_HOME

from .auth import (
    SCOPE_COMMENT,
    SCOPE_READ,
    LeRobotTokenVerifier,
    TokenStore,
    requires_scope,
)
from .bridge_tools import register_bridge_tools
from .store import AnnotationStore

logger = logging.getLogger("lerobot.mcp")

_META_CACHE_MAX = 32
_DEFAULT_DB_PATH = HF_LEROBOT_HOME / "_mcp_annotations.sqlite"


class ServerState:
    """Holds per-process state: dataset root, metadata cache, annotation store.

    A single instance is bound to the FastMCP app at startup.
    """

    def __init__(self, dataset_root: Path | None = None, db_path: Path | None = None):
        self.dataset_root = Path(dataset_root) if dataset_root is not None else HF_LEROBOT_HOME
        self.store = AnnotationStore(db_path if db_path is not None else _DEFAULT_DB_PATH)
        self._meta_cache: OrderedDict[str, LeRobotDatasetMetadata] = OrderedDict()

    def discover_datasets(self) -> list[tuple[str, Path]]:
        """Return ``[(repo_id, local_root), ...]`` for datasets under ``dataset_root``."""
        out: list[tuple[str, Path]] = []
        root = self.dataset_root
        if not root.is_dir():
            return out
        for org_dir in sorted(root.iterdir()):
            if not org_dir.is_dir():
                continue
            for ds_dir in sorted(org_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                if (ds_dir / "meta" / "info.json").is_file():
                    out.append((f"{org_dir.name}/{ds_dir.name}", ds_dir))
        return out

    def resolve(self, repo_id: str) -> Path:
        root = self.dataset_root / repo_id
        if not (root / "meta" / "info.json").is_file():
            raise ValueError(f"Dataset not found locally: {repo_id!r} (looked in {root})")
        return root

    def get_meta(self, repo_id: str) -> LeRobotDatasetMetadata:
        """Cached metadata loader; LRU-evicts past ``_META_CACHE_MAX``."""
        if repo_id in self._meta_cache:
            self._meta_cache.move_to_end(repo_id)
            return self._meta_cache[repo_id]
        root = self.resolve(repo_id)
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        self._meta_cache[repo_id] = meta
        while len(self._meta_cache) > _META_CACHE_MAX:
            self._meta_cache.popitem(last=False)
        return meta


# Bound at server build time; tools close over this via the global ``state``.
state: ServerState | None = None


def _state() -> ServerState:
    assert state is not None, "ServerState not initialized (call build_server first)"
    return state


def _episode_length(meta: LeRobotDatasetMetadata, ep_id: int) -> int:
    """Frame count of an episode, derived from the episodes metadata table."""
    row = meta.episodes[ep_id]
    if "length" in row and row["length"] is not None:
        return int(row["length"])
    return int(row["dataset_to_index"] - row["dataset_from_index"])


def _normalize_frame_idx(frame_idx: int, ep_length: int) -> int:
    """Resolve negative indices (``-1`` → last frame). Bounds-checked."""
    if frame_idx < 0:
        frame_idx = ep_length + frame_idx
    if not (0 <= frame_idx < ep_length):
        raise ValueError(f"frame_idx {frame_idx} out of range [0, {ep_length})")
    return frame_idx


def _tensor_to_image(frame: torch.Tensor) -> Image:
    """Encode a single decoded frame (C,H,W float[0,1] or uint8) as a PNG Image."""
    if frame.ndim == 4:
        # decode_video_frames returns (T, C, H, W); take the single frame
        frame = frame[0]
    assert frame.ndim == 3, f"unexpected frame shape {tuple(frame.shape)}"
    if frame.dtype != torch.uint8:
        frame = (frame.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    # (C, H, W) → (H, W, C)
    arr = frame.permute(1, 2, 0).cpu().numpy()
    pil = PILImage.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return Image(data=buf.getvalue(), format="png")


# ── Build & register tools ─────────────────────────────────────────────────


def build_server(
    dataset_root: Path | None = None,
    db_path: Path | None = None,
    token_store: TokenStore | None = None,
    host: str = "127.0.0.1",
    port: int = 7861,
    streamable_http_path: str | None = None,
) -> FastMCP:
    """Construct a FastMCP server with the tool surface registered.

    Args:
        dataset_root: Override the dataset discovery root. Defaults to
            ``$HF_LEROBOT_HOME``.
        db_path: Override the annotation sidecar location. Defaults to
            ``$HF_LEROBOT_HOME/_mcp_annotations.sqlite``.
        token_store: If provided, the server runs with bearer-token auth
            (required for HTTP transport). The verifier checks every
            request and rejects with 401 if the bearer is missing or
            unknown, and 403 if it lacks the floor ``read`` scope. When
            ``None`` (stdio mode), no auth is enforced.
        host, port: Bind address used by HTTP transports. Ignored by stdio.
        streamable_http_path: Path the streamable-http transport listens on
            inside the FastMCP ASGI app. Default ``None`` keeps FastMCP's
            built-in ``/mcp``. Set to ``"/"`` when mounting the app under a
            FastAPI route (e.g., the GUI mounts at ``/mcp`` and wants the
            inner transport at its own root so the final URL stays
            ``<host>/mcp``, not ``<host>/mcp/mcp``).
    """
    global state
    state = ServerState(dataset_root=dataset_root, db_path=db_path)

    fastmcp_kwargs: dict[str, Any] = {"host": host, "port": port}
    if streamable_http_path is not None:
        fastmcp_kwargs["streamable_http_path"] = streamable_http_path
    if token_store is not None:
        base_url = f"http://{host}:{port}"
        fastmcp_kwargs["token_verifier"] = LeRobotTokenVerifier(token_store)
        fastmcp_kwargs["auth"] = AuthSettings(
            issuer_url=base_url,  # we issue our own tokens out-of-band
            resource_server_url=base_url,
            required_scopes=[SCOPE_READ],  # floor; per-tool decorators enforce annotate/operate
        )

    mcp = FastMCP("lerobot", **fastmcp_kwargs)

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def list_datasets() -> dict[str, Any]:
        """List datasets discoverable under the LeRobot dataset cache.

        Returns each dataset's ``repo_id``, episode count, fps, and camera keys
        (cheap — reads only ``meta/info.json``).
        """
        s = _state()
        items: list[dict[str, Any]] = []
        for repo_id, root in s.discover_datasets():
            try:
                info_path = root / "meta" / "info.json"
                import json as _json

                info = _json.loads(info_path.read_text())
                features = info.get("features", {})
                cameras = [k for k, ft in features.items() if ft.get("dtype") in ("video", "image")]
                items.append(
                    {
                        "repo_id": repo_id,
                        "n_episodes": int(info.get("total_episodes", 0)),
                        "n_frames": int(info.get("total_frames", 0)),
                        "fps": int(info.get("fps", 0)),
                        "robot_type": info.get("robot_type"),
                        "cameras": cameras,
                    }
                )
            except Exception as e:  # noqa: BLE001 — surface to agent
                items.append({"repo_id": repo_id, "error": f"{type(e).__name__}: {e}"})
        return {"datasets": items, "root": str(s.dataset_root)}

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_dataset_info(repo_id: str) -> dict[str, Any]:
        """Detailed schema and stats for one dataset.

        Args:
            repo_id: e.g. ``'thewisp/cylinder_ring_assembly'``.
        """
        meta = _state().get_meta(repo_id)
        features = {
            k: {"dtype": ft.get("dtype"), "shape": list(ft.get("shape", [])), "names": ft.get("names")}
            for k, ft in meta.features.items()
        }
        ep_lengths = [_episode_length(meta, i) for i in range(min(meta.total_episodes, 256))]
        length_stats = (
            {
                "min": int(min(ep_lengths)),
                "max": int(max(ep_lengths)),
                "mean": float(sum(ep_lengths) / len(ep_lengths)),
                "sampled": len(ep_lengths),
            }
            if ep_lengths
            else {}
        )
        return {
            "repo_id": repo_id,
            "robot_type": meta.robot_type,
            "fps": meta.fps,
            "total_episodes": meta.total_episodes,
            "total_frames": meta.total_frames,
            "total_tasks": meta.total_tasks,
            "cameras": meta.camera_keys,
            "image_keys": meta.image_keys,
            "video_keys": meta.video_keys,
            "features": features,
            "episode_length_stats": length_stats,
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def list_episodes(repo_id: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List episodes with per-episode summary fields (paginated).

        Each entry includes ``episode_id``, ``length``, ``tasks``, and any
        sidecar tags previously written via ``tag_episode``.

        Args:
            repo_id: Dataset id.
            limit: Max episodes to return (1..500). Default 50.
            offset: Starting episode index.
        """
        assert 1 <= limit <= 500, "limit must be in [1, 500]"
        assert offset >= 0, "offset must be >= 0"
        meta = _state().get_meta(repo_id)
        total = meta.total_episodes
        end = min(offset + limit, total)
        episodes: list[dict[str, Any]] = []
        for i in range(offset, end):
            row = meta.episodes[i]
            ep_summary: dict[str, Any] = {
                "episode_id": i,
                "length": _episode_length(meta, i),
            }
            if "tasks" in row:
                ep_summary["tasks"] = row["tasks"]
            ep_summary["tags"] = _state().store.get_tags(repo_id, i)
            episodes.append(ep_summary)
        return {
            "repo_id": repo_id,
            "total": total,
            "offset": offset,
            "returned": len(episodes),
            "episodes": episodes,
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_episode_summary(repo_id: str, episode_id: int) -> dict[str, Any]:
        """Per-episode detail: length, tasks, terminal action/state, tags.

        Cheap relative to ``get_frame`` — reads only the episode metadata row
        plus the sidecar.
        """
        meta = _state().get_meta(repo_id)
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        row = meta.episodes[episode_id]
        length = _episode_length(meta, episode_id)
        # Surface anything available without doing heavy parquet reads.
        keys_of_interest = {"tasks", "dataset_from_index", "dataset_to_index"}
        extras = {k: row[k] for k in keys_of_interest if k in row}
        return {
            "repo_id": repo_id,
            "episode_id": episode_id,
            "length": length,
            "duration_s": length / meta.fps if meta.fps else None,
            "cameras": meta.camera_keys,
            "tags": _state().store.get_tags(repo_id, episode_id),
            **extras,
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_frame(repo_id: str, episode_id: int, frame_idx: int, camera: str) -> Image:
        """One image from a video stream of a single episode.

        Args:
            repo_id: Dataset id.
            episode_id: Zero-based episode index.
            frame_idx: Zero-based frame index within the episode. Negative
                values count from the end (``-1`` = last frame).
            camera: One of the dataset's camera keys
                (see ``get_dataset_info(...).cameras``).

        Returns:
            A PNG image of that frame.
        """
        meta = _state().get_meta(repo_id)
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        if camera not in meta.camera_keys:
            raise ValueError(f"unknown camera {camera!r}; valid: {meta.camera_keys}")
        length = _episode_length(meta, episode_id)
        frame_idx = _normalize_frame_idx(frame_idx, length)

        if camera in meta.video_keys:
            rel_path = meta.get_video_file_path(episode_id, camera)
            video_path = meta.root / rel_path
            ts = frame_idx / meta.fps
            tol = 1.0 / max(meta.fps, 1)
            frames = decode_video_frames(video_path, [ts], tolerance_s=tol, return_uint8=True)
            return _tensor_to_image(frames)
        # image-feature path: not common in current datasets, defer to parquet
        raise NotImplementedError(
            f"Camera {camera!r} is an image-feature, not a video-feature. "
            "Image-feature frame access is not yet implemented in v1."
        )

    @mcp.tool()
    @requires_scope(SCOPE_COMMENT)
    def tag_episode(repo_id: str, episode_id: int, key: str, value: Any) -> dict[str, Any]:
        """Write a sidecar comment on an episode. Does not modify the canonical dataset.

        Comments live in a sidecar SQLite shared across all AI tools on this
        host and persist across sessions — leave a note today and the next AI
        session can read it. Comments are eventually surfaced in the GUI too.

        Args:
            repo_id: Dataset id.
            episode_id: Episode to comment on.
            key: Comment name (e.g. ``'outcome'``).
            value: JSON-serializable value (string, number, bool, list, dict).

        Returns the full tag dict for the episode after the write.
        """
        s = _state()
        meta = s.get_meta(repo_id)  # also validates dataset exists
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        s.store.set_tag(repo_id, episode_id, key, value)
        return {"repo_id": repo_id, "episode_id": episode_id, "tags": s.store.get_tags(repo_id, episode_id)}

    @mcp.tool()
    @requires_scope(SCOPE_COMMENT)
    def delete_episode_tag(repo_id: str, episode_id: int, key: str) -> dict[str, Any]:
        """Delete one sidecar comment from an episode.

        Returns ``{deleted: bool, tags: <remaining>}`` — ``deleted=False`` if
        the comment didn't exist (idempotent no-op, no error).
        """
        s = _state()
        meta = s.get_meta(repo_id)
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        deleted = s.store.delete_tag(repo_id, episode_id, key)
        return {
            "repo_id": repo_id,
            "episode_id": episode_id,
            "deleted": deleted,
            "tags": s.store.get_tags(repo_id, episode_id),
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_episode_tags(repo_id: str, episode_id: int) -> dict[str, Any]:
        """Read all sidecar comments for an episode."""
        s = _state()
        meta = s.get_meta(repo_id)
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        return {"repo_id": repo_id, "episode_id": episode_id, "tags": s.store.get_tags(repo_id, episode_id)}

    # Bridge tools (navigate_to / notify_user / highlight_in_viewer / set_filter)
    # are attached unconditionally; whether they actually deliver depends on
    # whether the GUI dispatch URL was configured via configure_bridge().
    register_bridge_tools(mcp)

    return mcp
