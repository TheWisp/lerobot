"""Transcript proof for the edit-tier Run MCP tool (update_rlt_config).

Faking a live RLT training session over the wire (no real subprocess
running) means the agent sees the ``No active RLT session`` error path
exclusively. That is, in fact, the actual behaviour the AI agent
encounters when called against an operator's host that isn't currently
training — so the transcript captures something useful even without a
live session.

For the success path with clamping and merge semantics, see the unit
tests in ``tests/mcp/test_run_edit.py`` which patch the module global
``_active_config`` to a tmp dir.

Output: ``mcp/docs/proofs/run_edit_transcript.md``.
"""

from __future__ import annotations

import asyncio
import json
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

MCP_URL = "http://127.0.0.1:8000/mcp/"
ARTIFACT_DIR = Path(__file__).parent / "proofs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


async def _issue_token(name: str) -> str:
    store = TokenStore(default_token_store_path())
    for row in store.list_tokens():
        if row["name"] == name and row["revoked_at"] is None:
            store.revoke(name)
    try:
        return store.issue(name, [SCOPE_READ, SCOPE_EDIT])
    except ValueError:
        import secrets

        return store.issue(f"{name}-{secrets.token_hex(4)}", [SCOPE_READ, SCOPE_EDIT])


async def _revoke_token(name: str) -> None:
    TokenStore(default_token_store_path()).revoke(name)


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
    token_name = "proofs_run_edit"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    try:
        headers = {"Authorization": f"Bearer {bearer}"}
        async with (
            streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tx = Transcript()
            tx.heading("Run edit-tier MCP — transcript")
            tx.note(
                "Captured live against the unified GUI's MCP at "
                "`127.0.0.1:8000/mcp/` with NO active RLT session, which "
                "is the dominant operator state. The error responses "
                "below are the actual wire shapes the AI sees when it "
                "tries to update overrides on a host that isn't "
                "currently training — clean, actionable messages."
            )

            tx.heading("get_run_status — confirm no run is active")
            tx.intent("Pre-flight check the AI runs before proposing an override write.")
            await tx.call(session, "get_run_status", {})

            tx.heading("update_rlt_config — no active session")
            tx.intent(
                "Try to update beta without an active session. The "
                "response tells the AI WHY it failed and what to do "
                "(start an HVLA run first), not just a 409 status code."
            )
            await tx.call(session, "update_rlt_config", {"beta": 1.5})

            tx.heading("update_rlt_config — no fields provided")
            tx.intent(
                "Call with nothing to update. Same error class as a "
                "missing session but a different message — the AI can "
                "distinguish 'session missing' from 'nothing to do'."
            )
            await tx.call(session, "update_rlt_config", {})

            tx.note(
                "**For the success path** (write, clamp, partial-merge, "
                "previous_values), see the 10 unit tests in "
                "`tests/mcp/test_run_edit.py` which patch the "
                "`_active_config` module global to a tmp dir. Mocking a "
                "live training subprocess over the wire isn't worth the "
                "complexity for a transcript artifact."
            )

            out = ARTIFACT_DIR / "run_edit_transcript.md"
            tx.write(out)
            print(f"transcript: written to {out}")
    finally:
        await _revoke_token(token_name)


if __name__ == "__main__":
    asyncio.run(main())
