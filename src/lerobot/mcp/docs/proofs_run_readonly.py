"""Transcript-only proof for the read-only Run MCP tools.

These four tools (``get_run_status``, ``get_run_output``,
``get_latency_metrics``, ``get_rlt_metrics``) observe the
GUI-managed subprocess. They are pure reads with no visible GUI
side-effect, so the proof shape is a wire transcript only — same as
the Hub read-only PR.

Captures both the "no run active" path (which is what the operator's
machine looks like most of the time) and the structured-empty
responses that let the AI branch on shape rather than parse errors.

Output: ``mcp/docs/proofs/run_readonly_transcript.md``.
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


async def _issue_token(name: str) -> str:
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
    token_name = "proofs_run_readonly"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    try:
        headers = {"Authorization": f"Bearer {bearer}"}
        async with (
            streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tx = Transcript()
            tx.heading("Run read-only MCP — transcript")
            tx.note(
                "Captured live against the unified GUI's MCP at "
                "`127.0.0.1:8000/mcp/`. All calls are pure reads — no "
                "subprocess is started, stopped, or signaled."
            )

            tx.heading("get_run_status — is anything running?")
            tx.intent(
                "First call any AI agent makes when asked 'how's the robot "
                "doing'. Returns `{running: false, command: null}` when no "
                "subprocess is managed; the AI knows to suggest starting "
                "something rather than poll forever."
            )
            await tx.call(session, "get_run_status", {})

            tx.heading("get_run_output — what did the subprocess print?")
            tx.intent(
                "Snapshot of the last N captured stdout/stderr lines. With "
                "no active subprocess the buffer is empty; `truncated` "
                "remains false so the AI can branch on shape."
            )
            await tx.call(session, "get_run_output", {"last_n": 100})

            tx.heading("get_latency_metrics — performance of the active loop")
            tx.intent(
                "Latest atomic-replace snapshot for the requested source "
                "(default `teleop`). Empty-stub response when no run is "
                "active — `n_records=0` is the branch trigger."
            )
            await tx.call(session, "get_latency_metrics", {})

            tx.intent(
                "Unknown source — returns the same empty-stub shape rather "
                "than a 404, so the AI doesn't have to special-case the "
                "'wrong source name' path."
            )
            await tx.call(session, "get_latency_metrics", {"source": "made_up"})

            tx.heading("get_rlt_metrics — RLT training progress")
            tx.intent(
                "Same empty-stub pattern when no RLT session is active: "
                "`mode == 'IDLE'`. When a training run IS active, this "
                "returns the live metrics from the session's `metrics.json`."
            )
            await tx.call(session, "get_rlt_metrics", {})

            out = ARTIFACT_DIR / "run_readonly_transcript.md"
            tx.write(out)
            print(f"transcript: written to {out}")
    finally:
        await _revoke_token(token_name)


if __name__ == "__main__":
    asyncio.run(main())
