"""Transcript-only proof for the read-only Robots MCP tools.

These six tools (``list_robot_profiles``, ``get_robot_profile``,
``list_teleop_profiles``, ``get_teleop_profile``, ``list_ports``,
``get_all_port_assignments``) are strictly read-only — no motor
connections, no port-opening, no camera streams. The wire transcript
captures what the AI agent sees against the operator's actual host:
how many profiles are saved, what their shapes look like, which serial
ports the kernel exposes.

Output: ``mcp/docs/proofs/robots_readonly_transcript.md``.
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
    token_name = "proofs_robots_readonly"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    try:
        headers = {"Authorization": f"Bearer {bearer}"}
        async with (
            streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tx = Transcript()
            tx.heading("Robots read-only MCP — transcript")
            tx.note(
                "Captured live against the unified GUI's MCP at "
                "`127.0.0.1:8000/mcp/`. All calls are pure reads — no "
                "motor bus is opened, no port is connected to, no "
                "camera is started. The transcript reflects the actual "
                "operator's saved profiles and port topology."
            )

            tx.heading("list_robot_profiles — saved robot configs")
            tx.intent(
                "First call any AI agent makes when asked 'what robots "
                "does this host know about'. Returns the lightweight "
                "name+type pair; full config via get_robot_profile."
            )
            robot_list = await tx.call(session, "list_robot_profiles", {})

            structured = robot_list.structuredContent or {}
            robot_profiles = structured.get("profiles", [])
            if robot_profiles:
                first = robot_profiles[0]["name"]
                tx.intent(
                    f"Inspect the first saved robot profile (`{first}`) "
                    "to see the full on-disk config — fields, cameras, "
                    "rest_position. The AI uses this before proposing a "
                    "teleop or recording session."
                )
                await tx.call(session, "get_robot_profile", {"name": first})

            tx.intent(
                "Probe a deliberately-missing profile — the AI should "
                "see a clean tool-error message it can surface to the "
                "operator without parsing exception types."
            )
            await tx.call(session, "get_robot_profile", {"name": "does-not-exist-on-this-host"})

            tx.heading("list_teleop_profiles — saved teleop configs")
            tx.intent(
                "Saved teleop devices (leader arms, Quest 3 VR, scripted "
                "EE) for this host. Same name+type shape as robot profiles."
            )
            teleop_list = await tx.call(session, "list_teleop_profiles", {})

            structured = teleop_list.structuredContent or {}
            teleop_profiles = structured.get("profiles", [])
            if teleop_profiles:
                first = teleop_profiles[0]["name"]
                tx.intent(f"Inspect the first saved teleop profile (`{first}`).")
                await tx.call(session, "get_teleop_profile", {"name": first})

            tx.heading("list_ports — kernel-level USB serial enumeration")
            tx.intent(
                "Enumerate USB serial adapters via pyserial — kernel "
                "device-tree query, no port opened. Used to spot which "
                "/dev/ttyACM* / /dev/ttyUSB* are physically connected "
                "before assigning them to a profile."
            )
            await tx.call(session, "list_ports", {})

            tx.heading("get_all_port_assignments — saved-profile port map")
            tx.intent(
                "Cross-reference port paths against saved profile configs. "
                "Useful for spotting collisions ('two profiles claim "
                "/dev/ttyACM0 as their motor bus') and reverse-lookup "
                "('which profile owns /dev/ttyACM2?')."
            )
            await tx.call(session, "get_all_port_assignments", {})

            out = ARTIFACT_DIR / "robots_readonly_transcript.md"
            tx.write(out)
            print(f"transcript: written to {out}")
    finally:
        await _revoke_token(token_name)


if __name__ == "__main__":
    asyncio.run(main())
