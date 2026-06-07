"""Transcript proof for the dataset-merge MCP tools.

Demonstrates the real disk-mutating ``merge_into_dataset`` against
TWO randomly-named clones of the throwaway dataset
``thewisp/test_leader_follower_do_not_use``. Each run:

  1. Copies the throwaway into ``$HF_LEROBOT_HOME/_mcp_merge_<hex>/source``
     and ``.../target`` (two independent clones)
  2. Opens both in the GUI via the dataset-open API
  3. Calls ``validate_dataset_merge`` to demonstrate the compat check
  4. Calls ``merge_into_dataset`` to grow the target with the source's episodes
  5. Probes the live target via ``get_dataset_info`` to confirm growth
  6. Deletes both clones in a ``finally`` block so the operator's home
     directory is left clean — verified after the run.

If the script dies between clone and cleanup, the directories at:

  ~/.cache/huggingface/lerobot/_mcp_merge_<hex>/

can be removed by hand. The random suffix means concurrent runs don't
collide.

Output: ``mcp/docs/proofs/dataset_merge_transcript.md``.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/home/feit/Documents/lerobot-ai-native/src")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

from lerobot.mcp.auth import (  # noqa: E402
    SCOPE_EDIT,
    SCOPE_READ,
    TokenStore,
    default_token_store_path,
)
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402

GUI_URL = "http://127.0.0.1:8000"
MCP_URL = "http://127.0.0.1:8000/mcp/"
TEMPLATE_REPO_ID = "thewisp/test_leader_follower_do_not_use"
ARTIFACT_DIR = Path(__file__).parent / "proofs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _suffix() -> str:
    return secrets.token_hex(3)


THROWAWAY_TAG = f"_mcp_merge_{_suffix()}"
SOURCE_REPO_ID = f"{THROWAWAY_TAG}/source"
TARGET_REPO_ID = f"{THROWAWAY_TAG}/target"


async def _issue_token(name: str) -> str:
    store = TokenStore(default_token_store_path())
    for row in store.list_tokens():
        if row["name"] == name and row["revoked_at"] is None:
            store.revoke(name)
    try:
        return store.issue(name, [SCOPE_READ, SCOPE_EDIT])
    except ValueError:
        return store.issue(f"{name}-{secrets.token_hex(4)}", [SCOPE_READ, SCOPE_EDIT])


async def _revoke_token(name: str) -> None:
    TokenStore(default_token_store_path()).revoke(name)


def _clone_throwaway_to(dest_repo_id: str) -> Path:
    """Copy the template dataset to a new repo_id location under HF_LEROBOT_HOME.

    Returns the destination path. Raises if the template isn't present
    or the destination already exists (random-suffix names make that
    nearly impossible in practice).
    """
    src = Path(HF_LEROBOT_HOME) / TEMPLATE_REPO_ID
    if not src.is_dir() or not (src / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Template throwaway dataset {TEMPLATE_REPO_ID!r} not found at {src}. "
            "Pull it from Hub before running this proof."
        )
    dest = Path(HF_LEROBOT_HOME) / dest_repo_id
    if dest.exists():
        raise FileExistsError(f"Clone destination already exists: {dest}")
    shutil.copytree(src, dest)
    return dest


def _cleanup_clones() -> list[Path]:
    """Remove every clone directory created by this run + any orphans matching the tag.

    Returns the list of removed paths so the caller can log them.
    """
    home = Path(HF_LEROBOT_HOME)
    removed: list[Path] = []
    for d in home.iterdir():
        if d.is_dir() and d.name == THROWAWAY_TAG:
            shutil.rmtree(d)
            removed.append(d)
    return removed


def _open_dataset_via_api(repo_id: str) -> None:
    """Use the GUI's HTTP API to open the cloned dataset so AppState picks it up.

    The MCP tools require the dataset to be present in AppState.datasets,
    not just on disk — same lifecycle as opening from the GUI's Data tab.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps({"repo_id": repo_id}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 — admin-supplied URL, no user input
        f"{GUI_URL}/api/datasets",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310 — see above
            resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to open {repo_id!r} in GUI: {e}") from e


def _close_dataset_via_api(repo_id: str) -> None:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(  # noqa: S310 — admin URL
        f"{GUI_URL}/api/datasets/{repo_id}",
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            resp.read()
    except Exception:  # noqa: BLE001 — close is best-effort
        pass


class Transcript:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def heading(self, text: str) -> None:
        self.lines.append(f"\n## {text}\n")

    def note(self, text: str) -> None:
        self.lines.append(f"\n{text}\n")

    def intent(self, text: str) -> None:
        self.lines.append(f"\n_Intent: {text}_\n")

    async def call(self, session: ClientSession, name: str, args: dict):
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        self.lines.append(f"\n**→** `{name}({args_str})`\n")
        result = await session.call_tool(name, args)
        if result.isError:
            err = result.content[0].text if result.content else "(no message)"
            self.lines.append(f"\n**← error** — `{err}`\n")
        elif result.structuredContent is not None:
            body = json.dumps(result.structuredContent, indent=2)
            self.lines.append(f"\n**←**\n```json\n{body}\n```\n")
        else:
            text = result.content[0].text if result.content else "(empty)"
            self.lines.append(f"\n**←** {text}\n")
        return result

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines))


async def main() -> None:
    token_name = "proofs_dataset_merge"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    cloned_paths: list[Path] = []
    try:
        # Clone the throwaway dataset twice — source + target — under
        # a clearly-marked random-suffix tag.
        cloned_paths.append(_clone_throwaway_to(SOURCE_REPO_ID))
        cloned_paths.append(_clone_throwaway_to(TARGET_REPO_ID))

        # Open both in the GUI so they're in AppState.datasets.
        _open_dataset_via_api(SOURCE_REPO_ID)
        _open_dataset_via_api(TARGET_REPO_ID)

        headers = {"Authorization": f"Bearer {bearer}"}
        async with (
            streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tx = Transcript()
            tx.heading("Dataset merge MCP — transcript")
            tx.note(
                "Captured live against the unified GUI's MCP at "
                "`127.0.0.1:8000/mcp/`. The transcript clones a "
                f"throwaway dataset (`{TEMPLATE_REPO_ID}`) twice to "
                f"`{SOURCE_REPO_ID}` and `{TARGET_REPO_ID}` under "
                "`$HF_LEROBOT_HOME`, opens both in the GUI, runs the "
                "validate + merge tool pair, then deletes both clones. "
                "The operator's original throwaway dataset is **never** "
                "modified."
            )

            tx.heading("validate_dataset_merge — compat check")
            tx.intent(
                "Always run this before proposing a real merge. Same "
                "schema (cloned from the same template) → empty "
                "mismatches list."
            )
            await tx.call(
                session,
                "validate_dataset_merge",
                {"source_repo_id": SOURCE_REPO_ID, "target_repo_id": TARGET_REPO_ID},
            )

            tx.heading("get_dataset_info — target BEFORE merge")
            tx.intent("Snapshot the target's episode/frame counts for before/after comparison.")
            await tx.call(session, "get_dataset_info", {"repo_id": TARGET_REPO_ID})

            tx.heading("merge_into_dataset — the destructive write")
            tx.intent(
                "Copy the source's episodes into the target on disk. "
                "Response carries before/after counts so the AI can "
                "see exactly how much the target grew without a "
                "separate follow-up read."
            )
            await tx.call(
                session,
                "merge_into_dataset",
                {"source_repo_id": SOURCE_REPO_ID, "target_repo_id": TARGET_REPO_ID},
            )

            tx.heading("get_dataset_info — target AFTER merge")
            tx.intent(
                "Confirm the on-disk metadata reflects the merge "
                "(total_episodes / total_frames doubled because we "
                "merged a clone of the same dataset into itself)."
            )
            await tx.call(session, "get_dataset_info", {"repo_id": TARGET_REPO_ID})

            tx.heading("Error path — self-merge rejected")
            tx.intent(
                "Try to merge a dataset into itself — clean error so "
                "the AI doesn't accidentally double-write."
            )
            await tx.call(
                session,
                "merge_into_dataset",
                {"source_repo_id": SOURCE_REPO_ID, "target_repo_id": SOURCE_REPO_ID},
            )

            tx.heading("Error path — unknown source")
            tx.intent("Typo or unopened dataset.")
            await tx.call(
                session,
                "validate_dataset_merge",
                {"source_repo_id": "nope/missing", "target_repo_id": TARGET_REPO_ID},
            )

            out = ARTIFACT_DIR / "dataset_merge_transcript.md"
            tx.write(out)
            print(f"transcript: written to {out}")
    finally:
        # Close the cloned datasets in the GUI's AppState before
        # removing them from disk.
        _close_dataset_via_api(SOURCE_REPO_ID)
        _close_dataset_via_api(TARGET_REPO_ID)
        removed = _cleanup_clones()
        for d in removed:
            print(f"cleaned up clone: {d}")
        await _revoke_token(token_name)


if __name__ == "__main__":
    asyncio.run(main())
