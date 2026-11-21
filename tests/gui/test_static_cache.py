"""A reopened GUI must check the server before reusing cached application code."""

import httpx
import pytest

from lerobot.gui.server import app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/", "/static/index.html", "/static/flash_editor.js?v=15", "/static/rgbd_scene.js?v=12"]
)
async def test_gui_assets_revalidate_cached_content(path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["etag"]
        if path.startswith("/static/"):
            cached = await client.get(path, headers={"if-none-match": response.headers["etag"]})
            assert cached.status_code == 304
            assert cached.headers["cache-control"] == "no-cache"
            changed = await client.get(path, headers={"if-none-match": '"previous-release"'})
            assert changed.status_code == 200
            assert changed.content == response.content
