"""Transcript proof for the edit-tier Hub MCP tools.

Exercises the real upload pipeline against a throwaway HF Hub repo
that this script creates AND deletes. Each run:

  1. Probes ``hub_auth_status`` to confirm the host is logged in.
  2. Clones the local throwaway dataset
     ``thewisp/test_leader_follower_do_not_use`` into a random-suffix
     destination under ``$HF_LEROBOT_HOME``.
  3. Opens the clone in the GUI so MCP can see it.
  4. Calls ``hub_start_upload`` with a Hub repo name like
     ``<your-namespace>/_mcp_upload_proof_<hex>``; the worker creates
     the repo if missing and pushes through the standard PR pipeline.
  5. Polls ``hub_job_progress`` until the job's status is
     ``complete``, ``failed``, or ``cancelled``.
  6. **Deletes the Hub repo** via ``HfApi().delete_repo()``.
  7. Closes + deletes the local clone.

Belt-and-suspenders cleanup in a ``finally`` block — if the script
dies mid-flight, both the local clone and the remote repo get cleaned
up. The random suffix means concurrent or aborted runs don't collide.

This is the FIRST proof script that touches the operator's real HF
account state. To run it you need a working ``HF_TOKEN`` / ``hf auth
login`` with write permission to your namespace.

Output: ``mcp/docs/proofs/hub_edit_transcript.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import shutil
import sys
import time
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

# Distinct random suffix per run.
THROWAWAY_SUFFIX = secrets.token_hex(3)
LOCAL_TAG = f"_mcp_upload_proof_{THROWAWAY_SUFFIX}"
LOCAL_REPO_ID = f"{LOCAL_TAG}/dataset"


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


def _hf_username() -> str:
    """Get the operator's HF Hub username via whoami()."""
    from huggingface_hub import HfApi

    info = HfApi().whoami()
    name = info.get("name") or info.get("fullname")
    if not name:
        raise RuntimeError(
            "Could not resolve HF Hub username via whoami(). Run `huggingface-cli login` and try again."
        )
    return name


def _clone_throwaway(dest_repo_id: str) -> Path:
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


def _cleanup_local() -> list[Path]:
    home = Path(HF_LEROBOT_HOME)
    removed: list[Path] = []
    for d in home.iterdir():
        if d.is_dir() and d.name == LOCAL_TAG:
            # safe-destruct: clone we just created under a random-suffix tag
            shutil.rmtree(d)
            removed.append(d)
    return removed


def _cleanup_hub_repo(repo_id: str) -> None:
    """Delete the throwaway repo from HF Hub. Idempotent — repo missing is fine."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    with contextlib.suppress(RepositoryNotFoundError):
        api.delete_repo(repo_id, repo_type="dataset", missing_ok=True)


def _open_dataset_via_api(repo_id: str) -> None:
    import urllib.error
    import urllib.request

    payload = json.dumps({"repo_id": repo_id}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310  # nosec B310 — admin URL
        f"{GUI_URL}/api/datasets",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310  # nosec B310
            resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to open {repo_id!r} in GUI: {e}") from e


def _close_dataset_via_api(repo_id: str) -> None:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(  # noqa: S310  # nosec B310 — admin URL
        f"{GUI_URL}/api/datasets/{repo_id}",
        method="DELETE",
    )
    with (
        contextlib.suppress(Exception),
        urllib.request.urlopen(req, timeout=10.0) as resp,  # noqa: S310  # nosec B310
    ):
        resp.read()


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


async def _poll_until_terminal(
    session: ClientSession,
    tx: Transcript,
    job_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 3.0,
) -> str | None:
    """Poll hub_job_progress until status is terminal. Records final state."""
    start = time.monotonic()
    last_status: str | None = None
    while time.monotonic() - start < timeout_s:
        await asyncio.sleep(poll_interval_s)
        result = await session.call_tool("hub_job_progress", {"job_id": job_id})
        if result.isError or result.structuredContent is None:
            continue
        snap = result.structuredContent
        status = snap.get("status")
        if status != last_status:
            tx.note(
                f"poll: status={status!r} milestone={snap.get('milestone')!r} "
                f"files={snap.get('files_done_estimate')}/{snap.get('files_total')}"
            )
            last_status = status
        if status in ("complete", "failed", "cancelled"):
            tx.heading(f"hub_job_progress — final snapshot ({status})")
            tx.lines.append(f"\n**→** `hub_job_progress(job_id={job_id!r})`\n")
            body = json.dumps(snap, indent=2)
            tx.lines.append(f"\n**←**\n```json\n{body}\n```\n")
            return status
    return last_status


async def main() -> None:
    token_name = "proofs_hub_edit"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    hub_repo_id: str | None = None
    try:
        # Pre-flight: confirm HF auth and resolve username for the repo path.
        username = _hf_username()
        hub_repo_id = f"{username}/{LOCAL_TAG}"
        print(f"Hub auth OK: {username}; will upload to {hub_repo_id}")

        # Clone the throwaway dataset.
        _clone_throwaway(LOCAL_REPO_ID)
        _open_dataset_via_api(LOCAL_REPO_ID)

        headers = {"Authorization": f"Bearer {bearer}"}
        async with (
            streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tx = Transcript()
            tx.heading("Hub edit-tier MCP — transcript")
            tx.note(
                "Captured live against the unified GUI's MCP at "
                "`127.0.0.1:8000/mcp/`. The transcript clones the "
                f"throwaway dataset `{TEMPLATE_REPO_ID}` locally to "
                f"`{LOCAL_REPO_ID}`, uploads it to a throwaway HF Hub "
                f"repo `{hub_repo_id}`, polls until the worker reaches "
                "a terminal state, then DELETES both the Hub repo and "
                "the local clone. The operator's original throwaway "
                "dataset is **never** modified."
            )

            tx.heading("hub_auth_status — pre-flight")
            tx.intent(
                "Any responsible agent calls this before kicking off "
                "an upload. If logged_in=false the upload would 401."
            )
            await tx.call(session, "hub_auth_status", {})

            tx.heading("hub_start_upload — kick off the real upload")
            tx.intent(
                "Spawn the worker subprocess that creates the repo, "
                "opens a PR, pushes the dataset, and merges. Returns "
                "a job_id immediately; we poll for completion."
            )
            start = await tx.call(
                session,
                "hub_start_upload",
                {"dataset_id": LOCAL_REPO_ID, "hub_repo_id": hub_repo_id},
            )

            structured = start.structuredContent or {}
            job_id = structured.get("job_id")
            if not job_id:
                tx.note(
                    "No job_id in response — upload did not start. "
                    "See the error above; cleanup will still run."
                )
            else:
                tx.heading("Polling hub_job_progress until terminal")
                final = await _poll_until_terminal(session, tx, job_id, timeout_s=180.0)
                tx.note(f"Final status: {final!r}")

            tx.heading("Error path — unknown job_id")
            tx.intent(
                "Confirm the cancel path's error shape — same clean "
                "tool-error pattern the read-tier hub_job_progress uses."
            )
            await tx.call(session, "hub_cancel_job", {"job_id": "does-not-exist"})

            out = ARTIFACT_DIR / "hub_edit_transcript.md"
            tx.write(out)
            print(f"transcript: written to {out}")
    finally:
        # Always clean up — local clone first (close in GUI then rmtree)
        # and the Hub repo. Both are best-effort; manual cleanup of
        # ``<your-namespace>/_mcp_upload_proof_*`` on Hub may be needed
        # if the delete API call itself fails.
        with contextlib.suppress(Exception):
            _close_dataset_via_api(LOCAL_REPO_ID)
        removed = _cleanup_local()
        for d in removed:
            print(f"cleaned up local clone: {d}")
        if hub_repo_id is not None:
            try:
                _cleanup_hub_repo(hub_repo_id)
                print(f"cleaned up Hub repo: {hub_repo_id}")
            except Exception as e:  # noqa: BLE001
                print(f"WARN: Hub repo cleanup failed for {hub_repo_id}: {e}")
        await _revoke_token(token_name)


if __name__ == "__main__":
    asyncio.run(main())
