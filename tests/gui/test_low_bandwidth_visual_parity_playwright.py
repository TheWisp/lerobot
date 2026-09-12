"""What the operator sees at Low Bandwidth is what the JPEG path shows
(design: R6): the same tile geometry, the same mask chrome, and treatments that
behave as the compositor's do.

Three things the operator saw and this pins: video tiles drawn small inside
black bars, because the canvas was the encoded size rather than the camera's;
outline and label gone when the treatment is `none`, though they are the
operator's chrome and not the treatment; and `random` redrawn every frame,
where the compositor draws one texture per episode. And one the operator did
not have to see: a muted mask row must stay out of the composite here as the
server keeps it out of the JPEG.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("av")

from playwright.sync_api import sync_playwright  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    BLOB_CENTER,
    BLOB_RADIUS,
    CAM_WIDE,
    CAMS,
    FRAMES,
    SIZES,
    GuiServer,
    blob_mask,
    build_dataset,
)

pytestmark = pytest.mark.requires_playwright

MODE_KEY = "lerobot.cameraVideoMode"
VIEWPORT = {"width": 1400, "height": 900}  # tiles far wider than the 320-pixel encode


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    yield srv
    srv.stop()


def _open(page, base, ds_id, mode):
    page.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, {json.dumps(mode)});")
    page.goto(base)
    page.wait_for_function("typeof openDataset === 'function'", timeout=15_000)
    page.evaluate("(ds) => openDataset(ds)", ds_id)
    page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
    page.evaluate(f"selectEpisode({json.dumps(ds_id)}, 0, {FRAMES})")
    if mode == "low-bandwidth":
        page.wait_for_function("window.__chunkPlayer && window.__chunkPlayer.ready()", timeout=60_000)
    else:
        page.wait_for_function(
            f"document.getElementById('frame-{CAM_WIDE.replace('.', '-')}').naturalWidth > 0", timeout=60_000
        )


def _rect(page, element_id):
    return page.evaluate(
        f"(() => {{ const r = document.getElementById('{element_id}').getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; }})()"
    )


def _mask_id(cam):
    return "mask-" + cam.replace(".", "-")


def _alpha_count(page, cam, x, y, w, h):
    """Opaque pixels in a window of the mask canvas, in the canvas's own pixels."""
    return page.evaluate(
        f"(() => {{ const c = document.getElementById('{_mask_id(cam)}'); if (!c || !c.width) return -1; const d = c.getContext('2d').getImageData({x}, {y}, {w}, {h}).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++; return n; }})()"
    )


def test_video_tiles_take_the_jpeg_tiles_geometry(server, tmp_path_factory):
    """Same viewport, same dataset: every camera's video canvas occupies the
    rectangle its JPEG <img> occupies, and its mask canvas sits on the same
    rectangle. A 320-pixel encode must not become a 320-pixel tile."""
    root = build_dataset(tmp_path_factory.mktemp("geom") / "geom")
    ds_id = server.open_dataset(root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        _open(page, server.base, ds_id, "full-quality")
        jpeg = {cam: _rect(page, "frame-" + cam.replace(".", "-")) for cam in CAMS}
        page.close()
        page = browser.new_page(viewport=VIEWPORT)
        _open(page, server.base, ds_id, "low-bandwidth")
        page.wait_for_function(
            f"document.getElementById('video-{CAM_WIDE.replace('.', '-')}').width > 0", timeout=30_000
        )
        for cam in CAMS:
            video = _rect(page, "video-" + cam.replace(".", "-"))
            assert all(abs(a - b) <= 1 for a, b in zip(video, jpeg[cam], strict=True)), (
                cam,
                "jpeg",
                jpeg[cam],
                "video",
                video,
            )
            if cam == CAM_WIDE:
                assert video[2] > 320, (
                    "the tile is wider than the encode and the picture must fill it",
                    cam,
                    video,
                )
            mask = _rect(page, _mask_id(cam)) if cam == CAM_WIDE else None
            if mask:
                assert all(abs(a - b) <= 1 for a, b in zip(mask, video, strict=True)), (
                    "mask over video",
                    mask,
                    video,
                )
        browser.close()


def test_outline_and_label_show_with_no_treatment_as_on_the_jpeg_path(server, tmp_path_factory):
    """The outline and the label name are the operator's chrome, drawn whenever
    saved masks exist; the treatment is separate. With treatment `none`, both
    modes draw the outline (edge opaque, centre transparent) and the name."""
    root = build_dataset(tmp_path_factory.mktemp("chrome") / "chrome", treatment=None)
    ds_id = server.open_dataset(root)
    h, w = SIZES[CAM_WIDE]
    cy, cx, r = BLOB_CENTER[0], BLOB_CENTER[1], BLOB_RADIUS
    with sync_playwright() as p:
        browser = p.chromium.launch()
        seen = {}
        for mode in ("full-quality", "low-bandwidth"):
            page = browser.new_page(viewport=VIEWPORT)
            _open(page, server.base, ds_id, mode)
            page.wait_for_function(
                f"(() => {{ const c = document.getElementById('{_mask_id(CAM_WIDE)}'); return c && c.width === {w} && getComputedStyle(c).display !== 'none'; }})()",
                timeout=30_000,
            )
            edge = _alpha_count(page, CAM_WIDE, cx + r - 4, cy - 4, 9, 9)
            centre = _alpha_count(page, CAM_WIDE, cx - 4, cy - 4, 9, 9)
            label = _alpha_count(
                page, CAM_WIDE, cx - r, cy - r, 90, 24
            )  # the name sits at the region's top-left
            seen[mode] = (edge, centre, label)
            page.close()
        browser.close()
    for mode, (edge, centre, label) in seen.items():
        assert edge > 0, (mode, "the outline is drawn", seen)
        assert centre == 0, (mode, "an outline, not a fill", seen)
        assert label > 10, (mode, "the label name is drawn by the region", seen)


def test_random_treatment_is_one_texture_per_episode_not_one_per_frame(server, tmp_path_factory):
    """The compositor draws the noise once per episode and recipe. The page must
    show the same pixels under the mask on every frame and after a seek back."""
    root = build_dataset(tmp_path_factory.mktemp("rnd") / "rnd", treatment={"key": "random", "params": {}})
    ds_id = server.open_dataset(root)
    cy, cx = BLOB_CENTER
    tile = "video-" + CAM_WIDE.replace(".", "-")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        _open(page, server.base, ds_id, "low-bandwidth")
        page.wait_for_function(f"document.getElementById('{tile}').width > 0", timeout=30_000)

        def patch(frame):
            page.evaluate(f"loadAllFrames({frame})")
            page.wait_for_function(
                f"window.__chunkPlayer.metrics.painted.some((q) => q.frame === {frame})", timeout=30_000
            )
            return page.evaluate(
                f"(() => {{ const c = document.getElementById('{tile}'); const s = c.width / {SIZES[CAM_WIDE][1]}; return Array.from(c.getContext('2d').getImageData(Math.round({cx} * s) - 4, Math.round({cy} * s) - 4, 8, 8).data); }})()"
            )

        a, b, c = patch(0), patch(1), patch(0)
        assert a != [100] * 0 and len(set(a)) > 4, ("the noise is there", a[:16])
        assert a == b, "the texture changed between frames"
        assert a == c, "the texture changed after a seek back to the same frame"
        browser.close()


def _treated(patch):
    """Whether an 8x8 RGBA patch reads as the solid green treatment rather than the grey strip."""
    px = [patch[i : i + 3] for i in range(0, len(patch), 4)]
    r = sum(p[0] for p in px) / len(px)
    g = sum(p[1] for p in px) / len(px)
    b = sum(p[2] for p in px) / len(px)
    return g - max(r, b) > 80


def test_a_muted_mask_stays_out_of_the_composite_as_on_the_jpeg_path(server, tmp_path_factory):
    """Muting a run of frames (the timeline's segment editing) marks the rows
    disabled; the server leaves them out of the JPEG composite. The chunk
    carries the flag and the page must apply the layer's own rule to it, or a
    mute the trainer honours would still show as treated here."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import write_episode

    root = build_dataset(
        tmp_path_factory.mktemp("muted") / "muted",
        treatment={"key": "solid", "params": {"color": [0, 200, 0]}},
    )
    h, w = SIZES[CAM_WIDE]
    muted_from = 10
    ds = LeRobotDataset("tests/chunks", root=root)
    write_episode(
        ds,
        0,
        CAM_WIDE,
        [{"ball": blob_mask(h, w)} for _ in range(FRAMES)],
        disabled_per_frame=[["ball"] if f >= muted_from else [] for f in range(FRAMES)],
    )
    ds_id = server.open_dataset(root)
    cy, cx = BLOB_CENTER
    tile = "video-" + CAM_WIDE.replace(".", "-")
    img = "frame-" + CAM_WIDE.replace(".", "-")
    frames = (0, muted_from + 2)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        _open(page, server.base, ds_id, "full-quality")
        jpeg = {}
        for f in frames:
            before = page.evaluate(f"document.getElementById('{img}').getAttribute('src')")
            page.evaluate(f"loadAllFrames({f})")
            if f != 0:
                page.wait_for_function(
                    f"document.getElementById('{img}').getAttribute('src') !== {json.dumps(before)}",
                    timeout=30_000,
                )
            src = page.evaluate(f"document.getElementById('{img}').getAttribute('src')")
            assert "masks=composited" in src, ("the JPEG tile must be asking for the composite", src)
            jpeg[f] = page.evaluate(
                f"""(src) => new Promise((res, rej) => {{ const im = new Image(); im.onload = () => {{ const c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight; const x = c.getContext('2d'); x.drawImage(im, 0, 0); res(Array.from(x.getImageData({cx} - 4, {cy} - 4, 8, 8).data)); }}; im.onerror = rej; im.src = src; }})""",
                src,
            )
        page.close()
        page = browser.new_page(viewport=VIEWPORT)
        _open(page, server.base, ds_id, "low-bandwidth")
        page.wait_for_function(f"document.getElementById('{tile}').width > 0", timeout=30_000)
        video = {}
        for f in frames:
            page.evaluate(f"loadAllFrames({f})")
            page.wait_for_function(
                f"window.__chunkPlayer.metrics.painted.some((q) => q.frame === {f})", timeout=30_000
            )
            video[f] = page.evaluate(
                f"(() => {{ const c = document.getElementById('{tile}'); const s = c.width / {w}; return Array.from(c.getContext('2d').getImageData(Math.round({cx} * s) - 4, Math.round({cy} * s) - 4, 8, 8).data); }})()"
            )
        browser.close()
    for f in frames:
        assert _treated(video[f]) == _treated(jpeg[f]), (f, "video", video[f][:12], "jpeg", jpeg[f][:12])
    assert _treated(jpeg[0]) and not _treated(jpeg[frames[1]]), (
        "the server treats the enabled frame and not the muted one",
        jpeg,
    )
