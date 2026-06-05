"""Transcript proof for the edit-tier Robots MCP tools.

These nine tools mutate JSON profile files under ``~/.config/lerobot/``
— canonical operator state. The transcript walks through a full CRUD
lifecycle on a clearly-marked throwaway profile (name prefixed
``_mcp_transcript_``) so even if anything fails mid-run, the leftover
file is obvious. Each tool's call + response shape is captured for
review.

Cleanup is best-effort in a `finally` block — if the transcript dies
between create and delete, manually remove the file at:

    ~/.config/lerobot/robots/_mcp_transcript_*.json

Output: ``mcp/docs/proofs/robots_edit_transcript.md``.
"""

from __future__ import annotations

import asyncio
import json
import secrets
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

# Distinct random suffix so concurrent transcript runs (or aborted
# runs left over) don't collide.
THROWAWAY_TAG = f"_mcp_transcript_{secrets.token_hex(3)}"
THROWAWAY_ROBOT_NAME = f"{THROWAWAY_TAG}_robot"
THROWAWAY_TELEOP_NAME = f"{THROWAWAY_TAG}_teleop"


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


async def _cleanup_throwaway(session: ClientSession) -> None:
    """Belt-and-suspenders: delete the throwaway profiles even if the
    transcript exited mid-way through.
    """
    import contextlib

    for kind, name in (
        ("robot", THROWAWAY_ROBOT_NAME),
        ("teleop", THROWAWAY_TELEOP_NAME),
    ):
        with contextlib.suppress(Exception):
            await session.call_tool(f"delete_{kind}_profile", {"name": name})


async def main() -> None:
    token_name = "proofs_robots_edit"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    try:
        headers = {"Authorization": f"Bearer {bearer}"}
        async with (
            streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tx = Transcript()
            tx.heading("Robots edit-tier MCP — transcript")
            tx.note(
                "Captured live against the unified GUI's MCP at "
                "`127.0.0.1:8000/mcp/`. The transcript writes a clearly-"
                f"marked throwaway profile (name prefix `{THROWAWAY_TAG}`) "
                "to the operator's `~/.config/lerobot/robots/` and "
                "`teleops/`, walks it through the full CRUD lifecycle, "
                "then deletes it. **No motor bus is opened**, no port is "
                "connected to, no camera is started — these tools are "
                "strictly file CRUD."
            )

            try:
                tx.heading("create_robot_profile — write a new profile")
                tx.intent(
                    "Create a fresh robot profile. Response carries "
                    "`created: true` + the on-disk path so the AI can "
                    "confirm where the file landed."
                )
                await tx.call(
                    session,
                    "create_robot_profile",
                    {
                        "name": THROWAWAY_ROBOT_NAME,
                        "type": "bi_so107_follower",
                        "fields": {"port": "/dev/ttyACM0", "baudrate": 1000000},
                    },
                )

                tx.intent(
                    "Try to create it again — the name collision is "
                    "explicit, and the response hints at the right "
                    "follow-up (`use update_robot_profile`)."
                )
                await tx.call(
                    session,
                    "create_robot_profile",
                    {"name": THROWAWAY_ROBOT_NAME, "type": "bi_so107_follower"},
                )

                tx.heading("update_robot_profile — full-replace")
                tx.intent(
                    "Replace the profile. Response carries `overwrote: "
                    "true` and `previous_type` so the AI knows it "
                    "clobbered an existing one (no silent overwrite)."
                )
                await tx.call(
                    session,
                    "update_robot_profile",
                    {
                        "name": THROWAWAY_ROBOT_NAME,
                        "type": "bi_so107_follower_predictive",
                        "fields": {"port": "/dev/ttyACM2", "baudrate": 2000000},
                    },
                )

                tx.heading("assign_port_to_arm — convenience over update")
                tx.intent(
                    "Change one port field. Cheaper than reading the "
                    "whole profile + writing it back. Response carries "
                    "`previous_port` and `changed` so the AI can see "
                    "whether the assignment was a no-op."
                )
                await tx.call(
                    session,
                    "assign_port_to_arm",
                    {
                        "profile_name": THROWAWAY_ROBOT_NAME,
                        "port": "/dev/ttyACM7",
                    },
                )

                tx.intent("Same call again — `changed: false` because the value matches.")
                await tx.call(
                    session,
                    "assign_port_to_arm",
                    {
                        "profile_name": THROWAWAY_ROBOT_NAME,
                        "port": "/dev/ttyACM7",
                    },
                )

                tx.heading("rename_robot_profile")
                renamed_to = f"{THROWAWAY_ROBOT_NAME}_renamed"
                tx.intent(
                    "Rename the file AND the internal `name` field. Both "
                    "the on-disk path and the JSON contents update."
                )
                await tx.call(
                    session,
                    "rename_robot_profile",
                    {"old_name": THROWAWAY_ROBOT_NAME, "new_name": renamed_to},
                )

                tx.intent("Confirm the new name is listed.")
                list_result = await tx.call(session, "list_robot_profiles", {})
                # Show only that we can locate the renamed entry; the full
                # listing is the operator's actual host so it can be long.
                names = [p["name"] for p in (list_result.structuredContent or {}).get("profiles", [])]
                tx.note(
                    f"Renamed profile present in list: `{renamed_to in names}` ({len(names)} total profiles)"
                )

                tx.heading("Error path — rename to an existing name")
                tx.intent(
                    "Try to rename onto a slot already occupied — clean "
                    "tool-error message tells the agent what to do next."
                )
                # The freshly-renamed profile + a one-off second throwaway
                second_throwaway = f"{THROWAWAY_TAG}_collision_target"
                await tx.call(
                    session,
                    "create_robot_profile",
                    {"name": second_throwaway, "type": "bi_so107_follower"},
                )
                await tx.call(
                    session,
                    "rename_robot_profile",
                    {"old_name": renamed_to, "new_name": second_throwaway},
                )
                await tx.call(session, "delete_robot_profile", {"name": second_throwaway})

                tx.heading("delete_robot_profile — gone with diagnostics")
                tx.intent(
                    "Delete the throwaway. Response echoes back the "
                    "removed `type` + `fields_count` so the audit trail "
                    "shows WHAT was removed, not just THAT something was."
                )
                await tx.call(session, "delete_robot_profile", {"name": renamed_to})

                tx.heading("Error path — operate on missing profile")
                tx.intent(
                    "Try to delete a profile that doesn't exist — the "
                    "AI sees a clear tool-error message instead of a "
                    "silent success or HTTP status code."
                )
                await tx.call(
                    session,
                    "delete_robot_profile",
                    {"name": f"{THROWAWAY_TAG}_does_not_exist"},
                )

                tx.heading("Teleop CRUD — same shape, different directory")
                tx.intent("Teleop profiles live in `~/.config/lerobot/teleops/`.")
                await tx.call(
                    session,
                    "create_teleop_profile",
                    {
                        "name": THROWAWAY_TELEOP_NAME,
                        "type": "scripted_ee",
                    },
                )
                await tx.call(session, "delete_teleop_profile", {"name": THROWAWAY_TELEOP_NAME})

            finally:
                # Belt-and-suspenders cleanup of any leftover throwaway
                # profiles, regardless of where in the script we died.
                await _cleanup_throwaway(session)

            out = ARTIFACT_DIR / "robots_edit_transcript.md"
            out.write_text("\n".join(tx.lines))
            print(f"transcript: written to {out}")
    finally:
        await _revoke_token(token_name)


if __name__ == "__main__":
    asyncio.run(main())
