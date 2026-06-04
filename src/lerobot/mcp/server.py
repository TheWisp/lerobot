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
    SCOPE_EDIT,
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

    A single instance is bound to the FastMCP app at startup. ``app_state``
    is a getter (zero-arg callable) returning the GUI's ``AppState`` when
    MCP is mounted inside the GUI process (unified deployment); edit-tier
    tools that write to the in-memory ``PendingEdit`` queue use it as the
    shared surface. Returns ``None`` in standalone ``lerobot-mcp serve``
    mode and during the GUI startup window before the AppState is
    constructed — edit-tier tools raise a descriptive error there.

    A getter rather than a value because in the unified mode, ``_mount_mcp``
    runs in ``run_server`` *before* the FastAPI ``startup_event`` allocates
    the AppState. Resolving lazily lets MCP attach early and pick up the
    live AppState on the first tool call.
    """

    def __init__(
        self,
        dataset_root: Path | None = None,
        db_path: Path | None = None,
        app_state: Any = None,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root is not None else HF_LEROBOT_HOME
        self.store = AnnotationStore(db_path if db_path is not None else _DEFAULT_DB_PATH)
        self._meta_cache: OrderedDict[str, LeRobotDatasetMetadata] = OrderedDict()
        # Accept either a callable (preferred — handles startup ordering) or a
        # bare AppState value (test convenience). Normalised to a getter.
        if app_state is None:
            self._app_state_getter = lambda: None
        elif callable(app_state):
            self._app_state_getter = app_state
        else:
            self._app_state_getter = lambda: app_state

    @property
    def app_state(self):
        return self._app_state_getter()

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
    app_state: Any = None,
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
    state = ServerState(dataset_root=dataset_root, db_path=db_path, app_state=app_state)

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
        if not 1 <= limit <= 500:
            raise ValueError(f"limit must be in [1, 500]; got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0; got {offset}")
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

        Last-write-wins on the same ``(episode_id, key)``. The response
        carries ``overwrote`` (bool) and ``previous_value`` so the agent
        can tell whether it just clobbered an existing comment.

        Args:
            repo_id: Dataset id.
            episode_id: Episode to comment on.
            key: Comment name (e.g. ``'outcome'``).
            value: JSON-serializable value (string, number, bool, list, dict).

        Returns the full tag dict for the episode plus an explicit
        ``overwrote`` / ``previous_value`` so the actual effect is visible.
        """
        s = _state()
        meta = s.get_meta(repo_id)  # also validates dataset exists
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        existing = s.store.get_tags(repo_id, episode_id)
        had_key = key in existing
        previous_value = existing.get(key) if had_key else None
        s.store.set_tag(repo_id, episode_id, key, value)
        return {
            "repo_id": repo_id,
            "episode_id": episode_id,
            "key": key,
            "overwrote": had_key,
            "previous_value": previous_value,
            "tags": s.store.get_tags(repo_id, episode_id),
        }

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

    # ── Edit-tier: dataset pending-edit pipeline ───────────────────────────
    # These tools mirror the GUI's PendingEdit model. The AI proposes
    # edits (delete / trim / set-feature), the GUI's existing UI shows
    # them inline (badge in the header + range highlights in the
    # Inspector), and apply_pending_edits writes them through to disk in
    # one transaction. The same /api/edits routes the operator drives
    # the GUI with are reading and writing the same queue — proposals
    # via MCP and via the GUI are interchangeable.
    #
    # Requires unified deployment (`app_state` is set). Standalone
    # ``lerobot-mcp serve`` raises a descriptive error.

    def _require_app_state():
        s = _state()
        if s.app_state is None:
            raise ValueError(
                "Edit-tier tools require unified GUI deployment "
                "(open the dataset in the LeRobot GUI; MCP must be mounted in the same process)."
            )
        return s.app_state

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def list_pending_edits(repo_id: str | None = None) -> dict[str, Any]:
        """List currently staged (unsaved) dataset edits.

        Edits are staged into a per-process queue that the GUI also reads.
        The same queue is used whether the proposal came from the GUI's
        UI or from another MCP call. ``apply_pending_edits`` flushes them
        to disk; ``discard_pending_edits`` drops them.

        Args:
            repo_id: If given, only edits for this dataset. Otherwise all.

        Returns ``{"edits": [...], "total": N}``. Each edit carries its
        ``index`` (stable while the queue isn't mutated), ``edit_type``
        (``delete`` / ``trim`` / ``feature_set``), the target episode,
        and the type-specific ``params``.
        """
        from lerobot.gui.api._edits_core import list_pending

        return list_pending(_require_app_state(), repo_id)

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    def propose_delete_episode(repo_id: str, episode_id: int) -> dict[str, Any]:
        """Stage the deletion of one episode from a dataset.

        Doesn't touch disk — adds to the pending-edit queue. The episode
        stays visible in the GUI's tree with a deletion badge until the
        operator (or you, via ``apply_pending_edits``) commits.

        Args:
            repo_id: Dataset id.
            episode_id: Episode to delete.

        Returns the staged-edit confirmation. Raises if the episode is
        already marked for deletion or out of range. Idempotent only on
        the failure side — proposing the same delete twice is an error.
        """
        from lerobot.gui.api._edits_core import propose_delete

        return propose_delete(_require_app_state(), repo_id, episode_id)

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    def propose_trim_episode(
        repo_id: str,
        episode_id: int,
        keep_from: int,
        keep_to: int,
    ) -> dict[str, Any]:
        """Stage a trim of one episode — keep frames ``[keep_from, keep_to)``, drop the rest.

        Param names anchor the semantics: you specify the window to
        **keep**, not the window to remove. Frames outside the kept
        window — ``[0, keep_from)`` on the head and ``[keep_to,
        episode_length)`` on the tail — are dropped at apply time. The
        episode does NOT split: one episode in, one (shorter) episode
        out, with length ``keep_to - keep_from``.

        There is no "remove a middle range" operation. If you wanted
        that, you'd have to delete the episode and re-record, or split
        manually offline; this tool can't produce two episodes from one.

        Replaces any prior trim on the same episode (last-write-wins).
        Passing the full range (``keep_from=0``, ``keep_to=episode_length``)
        clears an existing trim — useful for undoing a stage.

        Args:
            repo_id: Dataset id.
            episode_id: Episode to trim.
            keep_from: First frame to keep (inclusive). Set to 0 to only chop the tail.
            keep_to: First frame to drop (exclusive). Set to ``episode_length`` to only chop the head.

        Returns the staged-edit confirmation with an explicit "kept M of
        N frames, dropped (N − M)" message so the actual outcome is
        visible regardless of how the call was phrased.
        """
        from lerobot.gui.api._edits_core import propose_trim

        result = propose_trim(_require_app_state(), repo_id, episode_id, keep_from, keep_to)
        # _edits_core keeps the historical start_frame/end_frame names
        # for FastAPI compat. Translate to keep_from/keep_to for the MCP
        # response so the schema and the field names line up.
        result["keep_from"] = result.pop("start_frame")
        result["keep_to"] = result.pop("end_frame")
        return result

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    def propose_set_feature(
        repo_id: str,
        episode_id: int,
        feature: str,
        frame_from: int,
        frame_to: int,
        value: Any,
        confirm_large: bool = False,
        confirm_overlap: bool = False,
    ) -> dict[str, Any]:
        """Stage a per-frame feature-value edit over the range ``[frame_from, frame_to)``.

        For tagging episode outcomes, labelling subtasks, correcting per-frame
        signals like ``reward`` or ``success``. The edit is staged; nothing
        is written to disk until ``apply_pending_edits``. Per-episode
        features (e.g. ``success``) are coerced to the full episode range
        automatically.

        Two retry-able conflicts surface via ``{"status": "conflict", "detail": {...}}``:
        - ``code=large_edit_confirmation_required`` — touches > 10k frames;
          retry with ``confirm_large=True``.
        - ``code=overlapping_edit`` — overlaps prior staged edit(s);
          retry with ``confirm_overlap=True`` to clip them (last-write-wins).

        Other validation failures (unknown feature, out-of-range, bounds
        violation) raise the tool with an error message.

        Args:
            repo_id: Dataset id.
            episode_id: Target episode.
            feature: Feature name (e.g. ``'reward'``, ``'success'``, ``'subtask'``).
                Read-only features (`action`, `observation.*`, video / image
                features) are rejected.
            frame_from: Inclusive lower frame (in episode-local indices).
            frame_to: Exclusive upper frame.
            value: JSON-serializable value. Categorical features accept
                string or int representations; declared min/max bounds
                are enforced.
            confirm_large: Acknowledge an edit touching > 10k frames.
            confirm_overlap: Clip prior overlapping edits on retry.

        Returns the staged-edit confirmation, possibly including
        ``coerced_range`` when the range was widened (per-episode feature).
        """
        from lerobot.gui.api._edits_core import EditConflictError, propose_feature_set

        try:
            return propose_feature_set(
                _require_app_state(),
                repo_id,
                episode_id,
                feature,
                frame_from,
                frame_to,
                value,
                confirm_large=confirm_large,
                confirm_overlap=confirm_overlap,
            )
        except EditConflictError as e:
            # Overlap / large-edit conflicts surface as structured data so
            # the AI can read ``detail.code`` and retry with the right
            # confirm_* flag. Validation errors (unknown feature, bad range,
            # bounds violation) still propagate as tool errors.
            return {"status": "conflict", "detail": e.detail}

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    def discard_pending_edits(repo_id: str | None = None) -> dict[str, Any]:
        """Drop pending edits without applying them.

        Use to abandon a staging session. Does not touch on-disk data —
        only the in-memory queue and the per-dataset edit-state file are
        cleared.

        Args:
            repo_id: If given, scope the discard to one dataset.
                Otherwise discards every dataset's queue.

        Returns ``{"status": "ok", "discarded": N}``.
        """
        from lerobot.gui.api._edits_core import discard_pending

        return discard_pending(_require_app_state(), repo_id)

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def apply_pending_edits(repo_id: str) -> dict[str, Any]:
        """Apply all staged edits for a dataset to disk in one transaction.

        Acquires the dataset lock, writes feature-set edits first (cell
        rewrites in existing rows), then trims (episode-bound changes),
        then deletes (row removal). Re-aggregates stats once at the end.
        Reloads the in-memory dataset object to reflect the new
        ``total_episodes`` / ``total_frames``. Clears the pending-edit
        queue + on-disk state file on success.

        This is the destructive step. Reversibility for dataset-on-disk
        comes from a clean Hub push beforehand or a filesystem-level
        snapshot — there's no built-in undo at the MCP layer.

        Args:
            repo_id: Dataset id.

        Returns ``{"status": "ok"|"partial", "applied": N, "errors": [...],
        "warnings": [...]}``. ``partial`` means data writes landed but
        stats recompute or verification surfaced issues — the user's
        edits ARE on disk.
        """
        app_state = _require_app_state()
        if repo_id not in app_state.datasets:
            raise ValueError(f"Dataset not opened in GUI: {repo_id}")

        # The lock is shared with the GUI's apply route — concurrent
        # apply attempts from MCP and the GUI's "Save" button serialize
        # cleanly without either having to know the other exists.
        lock = app_state.get_lock(repo_id)
        if lock.locked():
            raise ValueError(f"Dataset {repo_id} is busy (operation in progress)")
        async with lock:
            from lerobot.gui.api.edits import _apply_edits_locked

            return await _apply_edits_locked(repo_id)

    # ── Read-tier: Hugging Face Hub introspection ──────────────────────────
    # Read-only Hub surface. Tells the AI whether the user is logged in,
    # what a remote repo looks like, and how any in-flight transfers
    # (kicked off via the GUI's "Push to Hub" or `hub_start_upload` once
    # it lands) are progressing. No uploads / downloads here — those are
    # edit-tier and ship in a follow-up PR.
    #
    # ``hub_auth_status`` and ``hub_repo_info`` work even in standalone
    # ``lerobot-mcp serve`` mode (they don't need AppState). The two
    # job-tracking tools require unified deployment because the job
    # registry lives on the GUI's AppState.

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def hub_auth_status() -> dict[str, Any]:
        """Whether the host process has a working HF Hub login.

        Probes via ``HfApi().whoami()`` on every call (cheap; HF caches
        the token resolution). Any failure path — no token, expired
        token, network down — collapses to ``logged_in=False``. The
        agent should treat all of these uniformly: ask the operator to
        ``huggingface-cli login`` or set ``HF_TOKEN`` on the host.

        Returns ``{"logged_in": bool, "username": str | None}``.
        """
        from lerobot.gui.api._hub_core import get_auth_status

        return get_auth_status()

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def hub_repo_info(repo_id: str) -> dict[str, Any]:
        """Look up a dataset repo on the Hub.

        Reads basic repo metadata (visibility, last commit, file count,
        total size) plus best-effort ``meta/info.json`` enrichment for
        episode + frame counts. Useful before proposing a sync — the
        agent can compare local vs remote sizes / episode counts and
        warn the user before kicking off a big transfer.

        Args:
            repo_id: e.g. ``'thewisp/cylinder_ring_assembly'``.

        Returns ``{"exists": bool, ...}``. When ``exists=False`` (repo
        missing, private with no access, network down) only ``repo_id``
        and ``error`` are filled; the AI should branch on ``"exists"``
        rather than parse error text.
        """
        from lerobot.gui.api._hub_core import get_repo_info

        return get_repo_info(repo_id)

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def hub_list_jobs() -> dict[str, Any]:
        """List all Hub transfers the GUI is tracking, newest-first.

        Same source the GUI's Transfers tray reads — pending / running
        / complete / cancelled / failed jobs from all the operator's
        sessions on this host. Terminal jobs older than 30 minutes are
        opportunistically GC'd on this call.

        Returns ``{"jobs": [...], "total": N, "active": N_active}``.
        ``active`` is jobs in ``pending`` or ``running`` — the
        outcome-transparent count so the agent doesn't have to filter
        the array.

        Requires unified GUI deployment (job registry lives on AppState).
        """
        from lerobot.gui.api._hub_core import list_hub_jobs

        return list_hub_jobs(_require_app_state())

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def hub_job_progress(job_id: str) -> dict[str, Any]:
        """Snapshot of one Hub job — for polling a long-running transfer.

        For active jobs the snapshot is refreshed from the worker's
        progress JSON before being returned, so an agent that just
        kicked off a transfer can poll this without staleness.

        Args:
            job_id: From ``hub_list_jobs`` or from the response of the
                (future) ``hub_start_upload`` / ``hub_start_download``
                tool.

        Raises if ``job_id`` is unknown.
        """
        from lerobot.gui.api._hub_core import HubJobNotFoundError, get_job_progress

        try:
            return get_job_progress(_require_app_state(), job_id)
        except HubJobNotFoundError as e:
            raise ValueError(str(e)) from e

    # Bridge tools (navigate_to / notify_user / highlight_in_viewer / set_filter)
    # are attached unconditionally; whether they actually deliver depends on
    # whether the GUI dispatch URL was configured via configure_bridge().
    register_bridge_tools(mcp)

    return mcp
