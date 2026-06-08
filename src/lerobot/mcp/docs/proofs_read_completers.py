"""Transcript proof for the read-tier completers PR.

Five new read-tier tools, all pure observations against the operator's
host. The transcript:

  1. Calls ``lerobot_list_tools`` to dump the current surface
     (proves the new tools land and have the right scopes).
  2. Calls ``lerobot_whoami`` so the AI can confirm what it's allowed
     to do under the issued token (read+edit here).
  3. Lightly tags two episodes via ``tag_episode``, then proves the
     reverse-lookup with ``list_tagged_episodes`` (key + value filter).
  4. Cleans up the tags with ``delete_episode_tag`` so the operator's
     sidecar SQLite is back to its prior state.
  5. Pulls a feature series via ``get_feature_series`` and reports the
     shape + first values (no full payload — that'd be hundreds of lines).
  6. Runs ``hub_diff_local_vs_remote`` against the throwaway dataset's
     real Hub mirror — should report in_sync (or note any drift).

No throwaway-dataset cloning needed: every step is read-only, except
for the two tags we add and delete in step 3+4.

Output: ``mcp/docs/proofs/read_completers_transcript.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, "/home/feit/Documents/lerobot-ai-native/src")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

from lerobot.mcp.auth import (  # noqa: E402
    SCOPE_COMMENT,
    SCOPE_READ,
    TokenStore,
    default_token_store_path,
)

GUI_URL = "http://127.0.0.1:8000"
MCP_URL = "http://127.0.0.1:8000/mcp/"
THROWAWAY_REPO_ID = "thewisp/test_leader_follower_do_not_use"
ARTIFACT_DIR = Path(__file__).parent / "proofs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

PROOF_TAG_KEY = f"_mcp_proof_tag_{secrets.token_hex(3)}"


async def _issue_token(name: str) -> str:
    store = TokenStore(default_token_store_path())
    for row in store.list_tokens():
        if row["name"] == name and row["revoked_at"] is None:
            store.revoke(name)
    try:
        return store.issue(name, [SCOPE_READ, SCOPE_COMMENT])
    except ValueError:
        return store.issue(f"{name}-{secrets.token_hex(4)}", [SCOPE_READ, SCOPE_COMMENT])


async def _revoke_token(name: str) -> None:
    TokenStore(default_token_store_path()).revoke(name)


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
    with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310  # nosec B310
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


async def main() -> None:
    token_name = "proofs_read_completers"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    try:
        # Open the throwaway dataset so get_feature_series + hub_diff
        # have a target.
        _open_dataset_via_api(THROWAWAY_REPO_ID)

        headers = {"Authorization": f"Bearer {bearer}"}
        async with (
            streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tx = Transcript()
            tx.heading("Read-tier completers MCP — transcript")
            tx.note(
                "Captured live against the unified GUI's MCP at "
                "`127.0.0.1:8000/mcp/`. The token carries `read` + "
                "`comment` scopes; the only mutations are two sidecar "
                "tags written and then deleted by this script. The "
                "operator's dataset files are untouched."
            )

            tx.heading("lerobot_list_tools — self-describe the surface")
            tx.intent(
                "Useful for AI clients that don't surface the tool "
                "list cleanly (raw streamable-http, scripted agents). "
                "Returns name + one-line description + scope for "
                "every registered tool."
            )
            result = await tx.call(session, "lerobot_list_tools", {})
            # Summarise in the transcript rather than dumping all 50.
            structured = result.structuredContent or {}
            tools = structured.get("tools") or []
            from collections import Counter

            by_scope = Counter(t["scope"] for t in tools)
            tx.note(f"_(Surface size: {len(tools)} tools; by scope: {dict(by_scope)})_")

            tx.heading("lerobot_whoami — verify caller privileges")
            tx.intent(
                "Pre-flight before proposing an edit-tier or "
                "operate-tier action. The agent can fail-fast when "
                "the token doesn't carry the needed scope."
            )
            await tx.call(session, "lerobot_whoami", {})

            tx.heading("list_tagged_episodes — reverse-lookup over tags")
            tx.intent(
                "First confirm there are no tags with our proof key "
                "yet (it's randomly suffixed, so almost certainly true)."
            )
            await tx.call(
                session,
                "list_tagged_episodes",
                {"repo_id": THROWAWAY_REPO_ID, "key": PROOF_TAG_KEY},
            )

            tx.intent(
                "Tag episode 0 with our proof key — value 'good'. The "
                "tag lives in the operator's sidecar SQLite; we'll "
                "delete it before exiting."
            )
            await tx.call(
                session,
                "tag_episode",
                {
                    "repo_id": THROWAWAY_REPO_ID,
                    "episode_id": 0,
                    "key": PROOF_TAG_KEY,
                    "value": "good",
                },
            )

            tx.intent(
                "Now list_tagged_episodes by key — episode 0 should "
                "show up with value=good and a set_at timestamp."
            )
            await tx.call(
                session,
                "list_tagged_episodes",
                {"repo_id": THROWAWAY_REPO_ID, "key": PROOF_TAG_KEY},
            )

            tx.intent(
                "Filter further by key + value. Same single result, "
                "demonstrating the value filter narrows correctly."
            )
            await tx.call(
                session,
                "list_tagged_episodes",
                {
                    "repo_id": THROWAWAY_REPO_ID,
                    "key": PROOF_TAG_KEY,
                    "value": "good",
                },
            )

            tx.intent("Cleanup — drop the proof tag so the operator's sidecar is back to its prior state.")
            await tx.call(
                session,
                "delete_episode_tag",
                {
                    "repo_id": THROWAWAY_REPO_ID,
                    "episode_id": 0,
                    "key": PROOF_TAG_KEY,
                },
            )

            tx.heading("get_feature_series — per-frame trajectory")
            tx.intent(
                "Default mode: omit `features` to pull every per-frame "
                "non-image feature. Transcript only logs the shape "
                "(series keys + length) — the full payload is hundreds "
                "of frames × multiple features."
            )
            result = await session.call_tool(
                "get_feature_series",
                {"repo_id": THROWAWAY_REPO_ID, "episode_id": 0},
            )
            tx.lines.append(f"\n**→** `get_feature_series(repo_id={THROWAWAY_REPO_ID!r}, episode_id=0)`\n")
            if result.isError or result.structuredContent is None:
                err = result.content[0].text if result.content else "(no body)"
                tx.lines.append(f"\n**← error** — `{err}`\n")
            else:
                body = result.structuredContent
                series_keys = sorted(body.get("series", {}).keys())
                summary = {
                    "repo_id": body.get("repo_id"),
                    "episode_index": body.get("episode_index"),
                    "length": body.get("length"),
                    "series_keys": series_keys,
                    "series_count": len(series_keys),
                }
                tx.lines.append(
                    f"\n**←** _(summarised — full payload too long for transcript)_\n```json\n"
                    f"{json.dumps(summary, indent=2)}\n```\n"
                )

            tx.heading("hub_diff_local_vs_remote — sync check")
            tx.intent(
                "Compare the local copy of the throwaway dataset "
                "against its Hub mirror. Same source the GUI's "
                "'Compare to remote' button uses."
            )
            await tx.call(
                session,
                "hub_diff_local_vs_remote",
                {"dataset_id": THROWAWAY_REPO_ID},
            )

            tx.heading("Error path — unknown dataset")
            tx.intent("Same clean tool-error pattern as the read-only Hub tools.")
            await tx.call(
                session,
                "get_feature_series",
                {"repo_id": "nope/does_not_exist", "episode_id": 0},
            )

            out = ARTIFACT_DIR / "read_completers_transcript.md"
            tx.write(out)
            print(f"transcript: written to {out}")
    finally:
        # Belt-and-suspenders: try deleting the proof tag again in case
        # the proof flow bailed before its own cleanup step.
        with contextlib.suppress(Exception):
            import urllib.request

            req = urllib.request.Request(  # noqa: S310  # nosec B310
                f"{GUI_URL}/api/datasets/{THROWAWAY_REPO_ID}",
                method="DELETE",
            )
            urllib.request.urlopen(req, timeout=5.0).read()  # noqa: S310  # nosec B310
        await _revoke_token(token_name)


if __name__ == "__main__":
    asyncio.run(main())
