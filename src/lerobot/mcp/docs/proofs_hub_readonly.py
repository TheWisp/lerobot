"""Transcript-only proof for the read-only Hub MCP tools.

These four tools (``hub_auth_status``, ``hub_repo_info``,
``hub_list_jobs``, ``hub_job_progress``) are pure reads — the GUI
doesn't visibly react to them the way ``propose_*`` makes a pending
badge appear. The right proof shape is the wire transcript: what the
AI sent, what came back. No screenshots; per the standard "two
complementary proofs" rule, this tool family only earns one.

Captures:

  1. ``hub_auth_status`` against the operator's HF login (or absence
     thereof).
  2. ``hub_repo_info`` against a known existing public dataset and a
     deliberately-missing one — shows the ``exists: true/false``
     branching.
  3. ``hub_list_jobs`` against the live GUI's job registry.
  4. ``hub_job_progress`` — covered indirectly: if no jobs are live we
     show the not-found error wire shape instead.

Outputs ``mcp/docs/proofs/hub_readonly_transcript.md``. Re-run after
changes to the tool surface or response shapes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/feit/Documents/lerobot-ai-native/src")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

from lerobot.mcp.auth import SCOPE_READ, TokenStore, default_token_store_path  # noqa: E402

MCP_URL = "http://127.0.0.1:8000/mcp/"
ARTIFACT_DIR = Path(__file__).parent / "proofs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Throwaway repo names to probe with hub_repo_info — one known to exist
# on the Hub (a small public dataset), one guaranteed to not exist.
KNOWN_EXISTING_REPO = "lerobot/aloha_sim_insertion_human"
KNOWN_MISSING_REPO = "nope/this_does_not_exist_for_sure_2026"


async def _issue_token(name: str) -> str:
    """Issue a fresh read-scope token; suffix on collision."""
    store = TokenStore(default_token_store_path())
    for row in store.list_tokens():
        if row["name"] == name and row["revoked_at"] is None:
            store.revoke(name)
    try:
        return store.issue(name, [SCOPE_READ])
    except ValueError:
        import secrets

        return store.issue(f"{name}-{secrets.token_hex(4)}", [SCOPE_READ])


async def _revoke_token(name: str) -> None:
    TokenStore(default_token_store_path()).revoke(name)


class Transcript:
    """Mirror of mcp/docs/proofs_dataset_edit.py's Transcript class so the
    artifact format stays uniform across all MCP proof scripts.
    """

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
    token_name = "proofs_hub_readonly"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    try:
        headers = {"Authorization": f"Bearer {bearer}"}
        async with (
            streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tx = Transcript()
            tx.heading("Hub read-only MCP — transcript")
            tx.note(
                "Captured live against the unified GUI's MCP at "
                "`127.0.0.1:8000/mcp/`. All calls are read-only; nothing on "
                "the operator's host or HF account is mutated."
            )

            tx.heading("hub_auth_status — host auth probe")
            tx.intent(
                "Ask whether the operator's host has a working HF Hub login. "
                "Used as a pre-flight before any upload/download tool."
            )
            await tx.call(session, "hub_auth_status", {})

            tx.heading("hub_repo_info — remote repo lookup")
            tx.intent(
                f"Probe a public dataset known to exist: `{KNOWN_EXISTING_REPO}`. "
                "The agent typically calls this before proposing a sync so it "
                "can warn the operator about the transfer size."
            )
            await tx.call(session, "hub_repo_info", {"repo_id": KNOWN_EXISTING_REPO})

            tx.intent(
                f"Probe a deliberately-missing repo: `{KNOWN_MISSING_REPO}`. "
                "The response collapses every failure mode (404 / 401 / "
                "network) to `exists: false` so the agent can branch on the "
                "boolean instead of parsing error strings."
            )
            await tx.call(session, "hub_repo_info", {"repo_id": KNOWN_MISSING_REPO})

            tx.heading("hub_list_jobs — Hub transfer registry")
            tx.intent(
                "List all known Hub jobs (active + recent terminals). Same "
                "source the GUI's Transfers tray polls. The `active` count "
                "is the outcome-transparent summary."
            )
            list_result = await tx.call(session, "hub_list_jobs", {})

            tx.heading("hub_job_progress — single-job snapshot")
            structured = list_result.structuredContent or {}
            jobs = structured.get("jobs", [])
            if jobs:
                first = jobs[0]["job_id"]
                tx.intent(
                    f"Snapshot the first known job (`{first}`). For active "
                    "jobs the snapshot is refreshed from the worker's "
                    "progress file before being returned."
                )
                await tx.call(session, "hub_job_progress", {"job_id": first})
            else:
                tx.intent(
                    "No live jobs in the registry today, so we probe the "
                    "error path instead. An unknown `job_id` raises so the AI "
                    "can distinguish 'job not yet started / already GC'd' "
                    "from 'job finished with status X'."
                )
                await tx.call(session, "hub_job_progress", {"job_id": "does-not-exist"})

            out = ARTIFACT_DIR / "hub_readonly_transcript.md"
            tx.write(out)
            print(f"transcript: written to {out}")
    finally:
        await _revoke_token(token_name)


if __name__ == "__main__":
    asyncio.run(main())
