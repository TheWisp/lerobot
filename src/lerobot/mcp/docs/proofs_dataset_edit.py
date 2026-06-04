"""Per-feature visible proofs for the dataset-edit MCP tools (PR 2).

Three focused before/after captures that drive the same in-memory
``PendingEdit`` queue the GUI's own UI drives, then verify via DOM probe
that the AI's proposal showed up where a human operator would see it:

  feat_propose_delete:    AI stages a delete → GUI header gains pending
                          badge, episode row shows the delete indicator.
  feat_propose_trim:      AI stages a trim → episode row's trim badge
                          appears with the new frame range.
  feat_discard_pending:   AI calls discard → pending count returns to 0.

apply_pending_edits is intentionally NOT screenshot-proved: it's the
destructive disk-mutating step, and exercising it against a real
dataset on every proof-run is bad practice. The unit tests in
``tests/mcp/test_dataset_edit.py::TestApplyToolSmoke`` and the existing
``tests/gui/test_feature_edits.py::TestApplyEdits`` cover the propose
→ apply → disk round-trip end-to-end.

Pre: a running unified GUI on http://127.0.0.1:8000 with the throwaway
dataset ``thewisp/test_leader_follower_do_not_use`` loaded. The
proofs leave the dataset untouched on disk — only the in-memory
pending-edit queue is mutated, and ``discard_pending_edits`` clears
it at the end of each proof.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "/home/feit/Documents/lerobot-ai-native/src")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from playwright.async_api import Page, async_playwright  # noqa: E402

from lerobot.mcp.auth import (  # noqa: E402
    SCOPE_EDIT,
    SCOPE_READ,
    TokenStore,
    default_token_store_path,
)

GUI_URL = "http://127.0.0.1:8000"
MCP_URL = "http://127.0.0.1:8000/mcp/"
REPO_ID = "thewisp/test_leader_follower_do_not_use"
ARTIFACT_DIR = Path(__file__).parent / "proofs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ── compose helpers (copy of proofs_e2e.py — keep this script self-contained)


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _label(img: Image.Image, text: str, color: tuple[int, int, int] = (79, 195, 247)) -> Image.Image:
    bar = 36
    out = Image.new("RGB", (img.width, img.height + bar), color=(22, 33, 62))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, img.width, bar], fill=(22, 33, 62))
    d.line([(0, bar - 1), (img.width, bar - 1)], fill=color, width=2)
    d.text((12, 8), text, fill=color, font=_font(16))
    return out


def _compose_pair(
    before: bytes,
    after: bytes,
    *,
    title: str,
    expected: str,
    observed: str,
    ok: bool,
    out: Path,
) -> None:
    from io import BytesIO

    b = _label(Image.open(BytesIO(before)).convert("RGB"), "Before")
    a = _label(Image.open(BytesIO(after)).convert("RGB"), "After")
    pair_w = b.width + a.width + 24
    header_h = 80
    canvas = Image.new("RGB", (pair_w, b.height + header_h), color=(26, 26, 46))
    d = ImageDraw.Draw(canvas)
    accent = (79, 195, 247) if ok else (255, 140, 90)
    d.text((16, 12), title, fill=accent, font=_font(20))
    verdict = ("✓  " if ok else "✗  ") + (f"Expected: {expected}    Observed: {observed}")
    d.text((16, 44), verdict, fill=(238, 238, 238), font=_font(14))
    canvas.paste(b, (0, header_h))
    canvas.paste(a, (b.width + 24, header_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)


# ── token + setup helpers ─────────────────────────────────────────────────


async def _issue_token(name: str) -> str:
    """Issue a fresh edit-scope token, suffixing on name collision."""
    store = TokenStore(default_token_store_path())
    for row in store.list_tokens():
        if row["name"] == name and row["revoked_at"] is None:
            store.revoke(name)
    try:
        return store.issue(name, [SCOPE_READ, SCOPE_EDIT])
    except ValueError:
        import secrets

        suffix = secrets.token_hex(4)
        return store.issue(f"{name}-{suffix}", [SCOPE_READ, SCOPE_EDIT])


async def _revoke_token(name: str) -> None:
    TokenStore(default_token_store_path()).revoke(name)


async def _open_dataset(page: Page, session: ClientSession) -> None:
    """Use the existing navigate_to bridge tool to open the throwaway dataset.

    This guarantees the GUI's AppState.datasets[REPO_ID] is populated
    before the edit tools fire — they require the dataset to be open.
    """
    await session.call_tool("navigate_to", {"view": "dataset", "params": {"repo_id": REPO_ID}})
    # navigate_to triggers openDataset client-side; wait for the dataset
    # to populate window.datasets.
    await page.wait_for_function(
        f"() => window.datasets && Object.keys(window.datasets).includes({REPO_ID!r})",
        timeout=15000,
    )
    await asyncio.sleep(1.0)


async def _drain_pending(session: ClientSession) -> None:
    """Reset state by discarding any leftover pending edits."""
    await session.call_tool("discard_pending_edits", {"repo_id": REPO_ID})
    await asyncio.sleep(0.4)


# ── proofs ────────────────────────────────────────────────────────────────


async def proof_propose_delete(session: ClientSession, page: Page) -> None:
    await _open_dataset(page, session)
    await _drain_pending(session)
    # Refresh the pending-edits UI badge by triggering the same poll the
    # GUI uses every few seconds. Without this, the "Before" screenshot
    # may show a stale badge from a prior run.
    await page.evaluate("() => window.refreshPendingEdits && window.refreshPendingEdits()")
    await asyncio.sleep(0.6)
    before = await page.screenshot()
    before_count = await page.evaluate("() => (window.pendingEdits || []).length")

    await session.call_tool("propose_delete_episode", {"repo_id": REPO_ID, "episode_id": 0})
    await asyncio.sleep(0.4)
    await page.evaluate("() => window.refreshPendingEdits && window.refreshPendingEdits()")
    await asyncio.sleep(0.8)
    after = await page.screenshot()

    state = await page.evaluate(
        """() => {
            const edits = window.pendingEdits || [];
            const delete_ep0 = edits.find(e => e.edit_type === 'delete' && e.episode_index === 0);
            return {count: edits.length, has_delete_ep0: !!delete_ep0};
        }"""
    )
    ok = state["count"] == before_count + 1 and state["has_delete_ep0"]
    _compose_pair(
        before,
        after,
        title="propose_delete_episode(repo_id, 0)",
        expected="pending count increases by 1, queue contains delete on episode 0",
        observed=f"count={state['count']} (was {before_count}), has_delete_ep0={state['has_delete_ep0']}",
        ok=ok,
        out=ARTIFACT_DIR / "feat_propose_delete.png",
    )
    print(f"propose_delete: {'OK' if ok else 'FAIL'} — {state}")
    await _drain_pending(session)


async def proof_propose_trim(session: ClientSession, page: Page) -> None:
    await _open_dataset(page, session)
    await _drain_pending(session)
    await page.evaluate("() => window.refreshPendingEdits && window.refreshPendingEdits()")
    await asyncio.sleep(0.6)
    before = await page.screenshot()

    # Pick a known episode + safe sub-range. Episode 0 is guaranteed to
    # exist in the test dataset; trim to a small window in the middle.
    ep_length = await page.evaluate(
        f"() => (window.episodes && window.episodes[{REPO_ID!r}] && window.episodes[{REPO_ID!r}][0]?.length) || null"
    )
    if not ep_length or ep_length < 10:
        print(f"propose_trim: SKIP — episode 0 length unknown or too short ({ep_length})")
        return
    start, end = 2, max(8, min(ep_length - 1, 10))

    await session.call_tool(
        "propose_trim_episode",
        {"repo_id": REPO_ID, "episode_id": 0, "start_frame": start, "end_frame": end},
    )
    await asyncio.sleep(0.4)
    await page.evaluate("() => window.refreshPendingEdits && window.refreshPendingEdits()")
    await asyncio.sleep(0.8)
    after = await page.screenshot()

    state = await page.evaluate(
        """() => {
            const edits = window.pendingEdits || [];
            const trim = edits.find(e => e.edit_type === 'trim' && e.episode_index === 0);
            return {count: edits.length, trim_range: trim ? [trim.params.start_frame, trim.params.end_frame] : null};
        }"""
    )
    ok = state["trim_range"] == [start, end]
    _compose_pair(
        before,
        after,
        title=f"propose_trim_episode(repo_id, 0, {start}, {end})",
        expected=f"queue contains trim on episode 0 with range [{start}, {end})",
        observed=f"count={state['count']}, trim_range={state['trim_range']}",
        ok=ok,
        out=ARTIFACT_DIR / "feat_propose_trim.png",
    )
    print(f"propose_trim: {'OK' if ok else 'FAIL'} — {state}")
    await _drain_pending(session)


async def proof_discard_pending(session: ClientSession, page: Page) -> None:
    await _open_dataset(page, session)
    await _drain_pending(session)
    # Stage one delete + one trim on episode 0 (the throwaway dataset has
    # exactly one episode, so "stage 2 edits on different episodes" isn't
    # available). Different edit_types so the queue shows variety.
    await session.call_tool("propose_delete_episode", {"repo_id": REPO_ID, "episode_id": 0})
    ep_length = await page.evaluate(
        f"() => (window.episodes && window.episodes[{REPO_ID!r}] && window.episodes[{REPO_ID!r}][0]?.length) || null"
    )
    if ep_length and ep_length >= 6:
        await session.call_tool(
            "propose_trim_episode",
            {"repo_id": REPO_ID, "episode_id": 0, "start_frame": 1, "end_frame": min(5, ep_length - 1)},
        )
    await asyncio.sleep(0.4)
    await page.evaluate("() => window.refreshPendingEdits && window.refreshPendingEdits()")
    await asyncio.sleep(0.8)
    before = await page.screenshot()
    before_count = await page.evaluate("() => (window.pendingEdits || []).length")

    await session.call_tool("discard_pending_edits", {"repo_id": REPO_ID})
    await asyncio.sleep(0.4)
    await page.evaluate("() => window.refreshPendingEdits && window.refreshPendingEdits()")
    await asyncio.sleep(0.8)
    after = await page.screenshot()
    after_count = await page.evaluate("() => (window.pendingEdits || []).length")

    ok = before_count >= 1 and after_count == 0
    _compose_pair(
        before,
        after,
        title="discard_pending_edits(repo_id)",
        expected="pending count drops to 0 after discard",
        observed=f"before={before_count}, after={after_count}",
        ok=ok,
        out=ARTIFACT_DIR / "feat_discard_pending.png",
    )
    print(f"discard_pending: {'OK' if ok else 'FAIL'} — before={before_count} after={after_count}")


# ── main ──────────────────────────────────────────────────────────────────


async def main() -> None:
    token_name = "proofs_dataset_edit"  # nosec B105 — not a credential
    bearer = await _issue_token(token_name)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            ctx = await browser.new_context(viewport={"width": 1400, "height": 900})
            page = await ctx.new_page()
            await page.goto(GUI_URL)
            await page.wait_for_function("() => typeof window.switchTab === 'function'", timeout=15000)
            await asyncio.sleep(1.0)

            headers = {"Authorization": f"Bearer {bearer}"}
            async with (
                streamablehttp_client(MCP_URL, headers=headers) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()

                await proof_propose_delete(session, page)
                await proof_propose_trim(session, page)
                await proof_discard_pending(session, page)

            await ctx.close()
            await browser.close()
    finally:
        # Always revoke; never leave a long-lived edit token after the proof.
        await _revoke_token(token_name)
        print(f"\nartifacts in: {ARTIFACT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
