"""Saved FLASH playback must preserve the camera image's spatial detail."""

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.requires_playwright


def test_saved_clip_preserves_camera_resolution_and_fine_detail(gui_page, tmp_path, monkeypatch):
    from dataclasses import replace

    from lerobot.policies.flash.tracking_history import export_episode, save_window
    from tests.policies.flash.test_shared_image_tracking import frame
    from tests.policies.flash.test_tracking_history import new_history

    # Four-pixel-wide stripes survive the stored H.264 encode, but not playback
    # through a 320-pixel preview enlarged back to this camera's 1280 pixels.
    h, w = 720, 1280
    stripes = ((np.arange(w) // 4) % 2 * 255).astype(np.uint8)
    rgb = np.broadcast_to(stripes[None, :, None], (h, w, 3)).copy()
    recent = new_history(tmp_path / "recent")
    for i in range(3):
        recent.submit_frame(replace(frame(i), rgb=rgb, depth=np.full((h, w), 1000, np.uint16)))
    recent.close()
    saved = tmp_path / save_window(recent.directory, tmp_path)
    export_episode(saved)
    monkeypatch.setenv("LEROBOT_FLASH_HISTORY_DIR", str(tmp_path))

    gui_page.get_by_role("button", name="FLASH", exact=True).click()
    gui_page.locator("#flash-source-live").click()
    gui_page.locator('#flash-live-stream-mode button[data-mode="raw"]').click()
    gui_page.locator(".flash-live-recording-row").last.locator(".flash-live-recording-select").click()
    gui_page.wait_for_function("flashLive.episodePlayer && flashLive.frame?.frame_id === 2", timeout=20000)
    # Inspect actual pixels decoded and painted by the production player. A
    # correctly sized canvas alone would pass even with the blurry preview.
    samples = gui_page.evaluate("""() => {
        const image=flashLive.episodeImage, ctx=image.getContext('2d');
        return Array.from({length:20},(_,i)=>ctx.getImageData(402+4*i,100,1,1).data[0]);
    }""")
    assert max(samples[::2]) < 25, samples
    assert min(samples[1::2]) > 230, samples
    # Exercise the same endpoint with the explicit low-bandwidth choice too.
    # Its decoded size must remain smaller, rather than silently changing Data.
    dimensions = gui_page.evaluate("""async () => {
        const id=encodeURIComponent(flashLive.episodeDataset);
        const result={};
        for(const profile of ['native','low']) {
            const r=await fetch(`/api/datasets/${id}/episodes/0/chunk?start=0&profile=${profile}`);
            if(!r.ok)throw new Error(await r.text());
            const b=await r.arrayBuffer(), n=new DataView(b).getUint32(0,true);
            const h=JSON.parse(new TextDecoder().decode(new Uint8Array(b,4,n)));
            const p=h.parts.find(p=>p.camera==='observation.images.camera');
            result[profile]=[p.width,p.height];
        }
        return result;
    }""")
    assert dimensions == {"native": [w, h], "low": [320, 180]}
