"""Per-feature visible proofs for the MCP bridge.

Each user-visible bridge tool gets its own focused before/after clip
instead of being buried in the long end-to-end demo:

  feat_navigate_tab:      tab switch (view='robot')
  feat_navigate_dataset:  GUI opens a specific dataset
  feat_navigate_episode:  timeline seeks + cameras render
  feat_highlight_viewer:  target row gets cyan outline + "AI ▸" badge
  feat_notify_user:       title-flash fallback when OS notification perm is denied

Each proof:
  1. Establishes a known starting state
  2. Captures a "before" screenshot
  3. Fires one MCP tool call
  4. Probes the DOM for the expected state change
  5. Captures an "after" screenshot
  6. Composes them into a side-by-side PNG with caption + verification verdict

set_filter is intentionally absent — no GUI viewer has a filter input
today, so the tool isn't registered (the wire shape is still in
SUPPORTED_COMMAND_TYPES for when a filter UI lands).

`notify_user`'s primary path is the Web Notifications API which renders
*outside* the browser viewport and can't be captured in a Playwright
screenshot. The fallback path (title-flash on permission-denied) IS
captureable, and we explicitly deny notification permission so the
proof drives that path. Both paths run through the same `notify_user`
tool call.

Outputs land under `src/lerobot/mcp/docs/proofs/`. Re-run to regenerate.
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
    SCOPE_COMMENT,
    SCOPE_READ,
    TokenStore,
    default_token_store_path,
)

GUI_URL = "http://127.0.0.1:8000"
MCP_URL = "http://127.0.0.1:8000/mcp/"
REPO_ID = "thewisp/test_leader_follower_do_not_use"
EPISODE_ID = 0
ARTIFACT_DIR = Path(__file__).parent / "proofs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ── compose helpers ────────────────────────────────────────────────────────


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _label(img: Image.Image, text: str, color: tuple[int, int, int] = (79, 195, 247)) -> Image.Image:
    """Add a header bar to a screenshot."""
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
    """Build a single PNG: title + verdict + side-by-side before/after."""
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


# ── session helpers ────────────────────────────────────────────────────────


async def _wait_for_bridge(page: Page) -> None:
    await page.wait_for_function(
        "() => window.lerobotBridge && window.lerobotBridge.isConnected()",
        timeout=15000,
    )


async def _issue_token(name: str) -> str:
    """Issue a fresh demo token, reusing the name idempotently.

    TokenStore.issue raises IntegrityError on duplicate names (including
    revoked rows that still hold the slot). We retry once with the suffix
    "-2" rather than reach into the SQLite layer to delete rows — keeps
    this script resilient to TokenStore schema evolution.
    """
    store = TokenStore(default_token_store_path())
    for row in store.list_tokens():
        if row["name"] == name and row["revoked_at"] is None:
            store.revoke(name)
    try:
        return store.issue(name, [SCOPE_READ, SCOPE_COMMENT])
    except ValueError:
        # Name slot still held by a prior revoked token. Use a fresh name.
        import secrets

        suffix = secrets.token_hex(4)
        return store.issue(f"{name}-{suffix}", [SCOPE_READ, SCOPE_COMMENT])


async def _revoke_token(name: str) -> None:
    TokenStore(default_token_store_path()).revoke(name)


# ── proofs ─────────────────────────────────────────────────────────────────


async def proof_navigate_tab(session: ClientSession, page: Page) -> None:
    # Start on the Data tab (default) and navigate to Robot.
    await page.evaluate("() => window.switchTab && window.switchTab('data')")
    await asyncio.sleep(0.6)
    before = await page.screenshot()
    before_active = await page.evaluate("() => document.querySelector('.tab.active')?.dataset.tab || null")

    await session.call_tool("navigate_to", {"view": "robot"})
    await asyncio.sleep(1.0)
    after = await page.screenshot()
    after_active = await page.evaluate("() => document.querySelector('.tab.active')?.dataset.tab || null")

    ok = after_active == "robot" and before_active != "robot"
    _compose_pair(
        before,
        after,
        title="navigate_to(view='robot')",
        expected="active tab transitions to 'robot'",
        observed=f"active before={before_active!r} → after={after_active!r}",
        ok=ok,
        out=ARTIFACT_DIR / "feat_navigate_tab.png",
    )
    print(f"navigate_tab: {'OK' if ok else 'FAIL'} — {before_active}→{after_active}")


async def proof_navigate_dataset(session: ClientSession, page: Page) -> None:
    # Land on a known starting state: tab=robot so dataset isn't open
    await page.evaluate("() => window.switchTab && window.switchTab('robot')")
    await asyncio.sleep(0.6)
    before = await page.screenshot()

    await session.call_tool("navigate_to", {"view": "dataset", "params": {"repo_id": REPO_ID}})
    await asyncio.sleep(3.5)
    after = await page.screenshot()
    state = await page.evaluate(
        "(r) => ({opened: Object.keys(window.datasets || {}).includes(r),"
        " rows: document.querySelectorAll('[data-dataset-id=\"' + r + '\"]').length})",
        REPO_ID,
    )
    ok = state["opened"] and state["rows"] >= 1
    _compose_pair(
        before,
        after,
        title="navigate_to(view='dataset', repo_id=…)",
        expected=f"window.datasets contains {REPO_ID!r}, ≥ 1 episode row",
        observed=f"opened={state['opened']}, rows={state['rows']}",
        ok=ok,
        out=ARTIFACT_DIR / "feat_navigate_dataset.png",
    )
    print(f"navigate_dataset: {'OK' if ok else 'FAIL'} — {state}")


async def proof_navigate_episode(session: ClientSession, page: Page) -> None:
    # Pre-condition: dataset must be opened. Do it via the same MCP call,
    # so each proof is self-contained.
    await session.call_tool("navigate_to", {"view": "dataset", "params": {"repo_id": REPO_ID}})
    await asyncio.sleep(3.5)
    before = await page.screenshot()

    await session.call_tool(
        "navigate_to",
        {"view": "episode", "params": {"repo_id": REPO_ID, "episode_id": EPISODE_ID}},
    )
    await asyncio.sleep(2.5)
    after = await page.screenshot()
    state = await page.evaluate("() => ({ds: window.currentDataset, ep: window.currentEpisode})")
    ok = state["ds"] == REPO_ID and state["ep"] == EPISODE_ID
    _compose_pair(
        before,
        after,
        title="navigate_to(view='episode', episode_id=0)",
        expected=f"currentDataset={REPO_ID!r}, currentEpisode=0",
        observed=f"ds={state['ds']!r}, ep={state['ep']}",
        ok=ok,
        out=ARTIFACT_DIR / "feat_navigate_episode.png",
    )
    print(f"navigate_episode: {'OK' if ok else 'FAIL'} — {state}")


async def proof_highlight_viewer(session: ClientSession, page: Page) -> None:
    # Pre-condition: dataset opened so the row exists in the tree.
    await session.call_tool("navigate_to", {"view": "dataset", "params": {"repo_id": REPO_ID}})
    await asyncio.sleep(3.5)
    # Clear any prior highlight
    await page.evaluate(
        "() => { window.bridgeHighlights?.clear(); window.renderTree && window.renderTree(); }"
    )
    await asyncio.sleep(0.4)
    before = await page.screenshot()

    await session.call_tool("highlight_in_viewer", {"repo_id": REPO_ID, "episode_ids": [EPISODE_ID]})
    await asyncio.sleep(0.6)
    after = await page.screenshot()
    state = await page.evaluate(
        """(p) => {
            const sel = '[data-episode-row][data-dataset-id="' + p.r + '"][data-episode-id="' + p.e + '"]';
            const el = document.querySelector(sel);
            if (!el) return {found: false};
            const rect = el.getBoundingClientRect();
            return {
                found: true,
                has_class: el.classList.contains('bridge-highlight'),
                in_view: rect.top >= 0 && rect.bottom <= window.innerHeight,
                rect_top: rect.top,
            };
        }""",
        {"r": REPO_ID, "e": EPISODE_ID},
    )
    ok = state.get("has_class") and state.get("in_view")
    _compose_pair(
        before,
        after,
        title="highlight_in_viewer(repo_id, [0])",
        expected="row has class 'bridge-highlight' AND is in viewport",
        observed=f"has_class={state.get('has_class')}, in_view={state.get('in_view')}, rect.top={state.get('rect_top')}",
        ok=ok,
        out=ARTIFACT_DIR / "feat_highlight_viewer.png",
    )
    print(f"highlight_viewer: {'OK' if ok else 'FAIL'} — {state}")


async def proof_notify_user(session: ClientSession, page: Page) -> None:
    # The OS-banner path can't be captured in screenshots — Notification API
    # renders outside the viewport. Force the title-flash fallback by stubbing
    # the Notification API to report permission=denied. The same `notify_user`
    # MCP call exercises both paths; the title-flash branch is the one we can
    # observe in a screenshot.
    await page.evaluate(
        """() => {
            // Some browsers' Notification.permission is read-only. Replace the
            // whole API with a denied stub so bridge.js's ensureNotificationPermission
            // returns 'denied' immediately and falls to flashTitle().
            window.Notification = class {
                static get permission() { return 'denied'; }
                static async requestPermission() { return 'denied'; }
            };
        }"""
    )
    await page.evaluate("() => window.switchTab && window.switchTab('data')")
    await asyncio.sleep(0.4)
    title_before = await page.title()
    before = await page.screenshot()

    title_msg = "AI says: episode 3 looks like a gripper miss"
    await session.call_tool(
        "notify_user",
        {"title": title_msg, "body": "Probe me", "level": "info"},
    )

    # The flash interval is 1000 ms; sample for up to 2 s waiting for the
    # title to swap to "🔔 …".
    title_during = title_before
    deadline = asyncio.get_event_loop().time() + 2.5
    while asyncio.get_event_loop().time() < deadline:
        title_during = await page.title()
        if title_during.startswith("🔔"):
            break
        await asyncio.sleep(0.1)

    # Playwright screenshots don't include the browser chrome (where the tab
    # title actually flashes). Overlay the live document.title into the page
    # so the After screenshot shows what the user would see in their tab bar.
    await page.evaluate(
        """(t) => {
            let el = document.getElementById('__title_probe');
            if (!el) {
                el = document.createElement('div');
                el.id = '__title_probe';
                el.style.cssText =
                    'position:fixed;top:50px;left:50%;transform:translateX(-50%);'
                    + 'z-index:100000;background:#16213e;color:#4fc3f7;'
                    + 'border:2px solid #4fc3f7;border-radius:6px;'
                    + 'padding:10px 20px;font:16px monospace;'
                    + 'box-shadow:0 4px 16px rgba(0,0,0,0.6);';
                document.body.appendChild(el);
            }
            el.textContent = 'browser tab title (live):  ' + t;
        }""",
        title_during,
    )
    await asyncio.sleep(0.4)
    after = await page.screenshot()
    # Clean up the overlay so it doesn't leak into subsequent proofs.
    await page.evaluate("() => document.getElementById('__title_probe')?.remove()")
    # Also stop the title-flash interval. flashTitle() only stops on a
    # window 'focus' event; without this, a future 6th proof's "before"
    # screenshot would catch the title alternating mid-cycle.
    await page.evaluate("() => window.dispatchEvent(new Event('focus'))")
    await asyncio.sleep(0.3)

    ok = title_during.startswith("🔔") and title_msg in title_during
    _compose_pair(
        before,
        after,
        title="notify_user(title=..., body=...) — title-flash fallback",
        expected="document.title starts with '🔔 ' + title text (perm-denied path)",
        observed=f"title_before={title_before!r}, title_during={title_during!r}",
        ok=ok,
        out=ARTIFACT_DIR / "feat_notify_user.png",
    )
    print(f"notify_user: {'OK' if ok else 'FAIL'} — during={title_during!r}")


# ── runner ─────────────────────────────────────────────────────────────────


async def main() -> None:
    token = await _issue_token("mcp-proofs")
    # try/finally so an exception in any proof doesn't leak the demo
    # bearer as an active row in the production TokenStore.
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=[
                    "--disable-features=site-per-process,IsolateOrigins",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
            context = await browser.new_context(viewport={"width": 1280, "height": 720})
            page = await context.new_page()
            await page.goto(GUI_URL)
            await _wait_for_bridge(page)

            async with (
                streamablehttp_client(MCP_URL, headers={"Authorization": f"Bearer {token}"}) as (
                    read,
                    write,
                    _,
                ),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                await proof_navigate_tab(session, page)
                await proof_navigate_dataset(session, page)
                await proof_navigate_episode(session, page)
                await proof_highlight_viewer(session, page)
                await proof_notify_user(session, page)

            await context.close()
            await browser.close()
    finally:
        await _revoke_token("mcp-proofs")
    print(f"\nartifacts in: {ARTIFACT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
