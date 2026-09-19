# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Apply-while-playing fills a camera that has no mask column yet.

A camera with no column is an absent mask TRACK: nothing is stored on any of its
frames, so the write rule fills all of it, exactly as it fills an absent frame on
a camera that has one. The column is adopted to hold the result.

It used to be refused, twice over, and both refusals were silent. The staging
endpoint rejected the camera, and the client-side drain filter dropped its frames
before the request was made -- so nothing ever reached the rejection. Driven
against a dataset with no masks, the whole episode played, no rows were staged,
Save returned 200, no column was created, and the run finished with "Apply
complete -- Save to commit". With two cameras selected and one of them adopted,
the other went the same way while the operator watched it play.

The coverage the filter uses cannot tell that case apart: it is empty both when
the dataset has no mask column and when the episode's series has not loaded, and
its own docstring says the caller must read that as "cannot filter". So the
columnless cameras are named from the schema, which knows before a frame plays,
and the column is created on the first flush that carries a row for it.

Only the segmenter is faked, at the seams the server reaches it through. The
publisher's bookkeeping, the drain, the run loop, the write-rule filter, the
staging endpoint and the edits pipeline are all real, and the masks are read
back off disk with the dataset's own accessor.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright

H, W = 48, 64
FPS, FRAMES = 10, 12
TOP = "observation.images.top"
WRIST = "observation.images.wrist"
CAMS = [TOP, WRIST]
LABEL = "ball"
PANEL = "#overlays-panel"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Pub:
    """The obs-stream writer's half: a per-camera sequence that moves on write.

    The real `publish_data_frame` files each camera's sequence against the frame
    it was published at, and the drain resolves masks back through that map, so
    the sequence has to actually move for any of it to line up.
    """

    def __init__(self, cams):
        self._seq = dict.fromkeys(cams, 0)

    def write_obs(self, obs):
        for cam in obs:
            self._seq[cam] = self._seq.get(cam, 0) + 1

    def image_seq(self, cam):
        return self._seq.get(cam, 0)


class _Worker:
    """The segmenter's half: answers every consumed frame with a mask."""

    def __init__(self, pub, cams, counts):
        self.pub = pub
        self.cameras = list(cams)
        self._counts = counts
        self._pending = []
        self._masks_seq = 0
        self._ov = dict.fromkeys(cams, 0)

    def overlay_seq(self, cam):
        return self._ov.get(cam, 0)

    def read_overlay(self, cam):
        return None

    def write_control(self, block):
        pass

    def read_latency(self):
        return {"compute_ms": 12.0}

    def masks_seq(self):
        return self._masks_seq

    def read_masks(self):
        out, self._pending = self._pending, []
        return out

    def segmented(self):
        batch = {}
        for cam in self.cameras:
            batch[cam] = {"seq": self.pub.image_seq(cam), "rle": {LABEL: self._counts}}
            self._ov[cam] += 1
        self._pending.append(batch)
        self._masks_seq += 1


def _dataset(root, repo, adopt_cams):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt

    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    for cam in CAMS:
        feats[cam] = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=repo, fps=FPS, root=root, features=feats, use_videos=True)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(FRAMES):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "apply",
                **dict.fromkeys(CAMS, img),
            }
        )
    ds.save_episode()
    ds.finalize()
    if adopt_cams:
        ds = LeRobotDataset(repo, root=root)
        adopt(ds, list(adopt_cams), [LABEL], (H, W))
    return root


@pytest.fixture
def run_apply(tmp_path, monkeypatch):
    """Play one episode with Apply armed; hand back what the operator was told
    and what reached the disk."""
    from lerobot.datasets.mask_codec import encode_mask

    blob = np.zeros((H, W), bool)
    blob[10:30, 10:40] = True
    counts = encode_mask(blob)

    def go(adopt_cams, select):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.mask_store import coverage, mask_columns
        from lerobot.gui import server as gui_server_mod
        from lerobot.gui.api import overlays as ovl

        repo = "tests/applyplay"
        root = _dataset(tmp_path / "ds", repo, adopt_cams)
        pub = _Pub(CAMS)
        worker = _Worker(pub, CAMS, counts)
        real_publish = ovl.publish_data_frame

        def publish(*a, **kw):
            real_publish(*a, **kw)
            worker.segmented()

        monkeypatch.setattr(ovl, "_data_pub", pub)
        monkeypatch.setattr(ovl, "_data_pub_cameras", list(CAMS))
        monkeypatch.setattr(ovl, "_data_pub_dataset", str(root))
        monkeypatch.setattr(ovl, "_get_live_reader", lambda: worker)
        monkeypatch.setattr(ovl, "publish_data_frame", publish)
        ovl._data_apply_pos.clear()
        monkeypatch.setattr(ovl, "_data_apply_last_seq", -1)

        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
        )
        threading.Thread(target=server.run, daemon=True).start()

        import requests

        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if requests.get(base, timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.2)
        else:
            server.should_exit = True
            pytest.fail("GUI server did not come up")

        out = {}
        with sync_playwright() as p:
            browser = p.chromium.launch()
            pg = browser.new_page(viewport={"width": 1500, "height": 950})

            # Configure would rebind the real publisher and spawn a worker; the
            # run itself still goes to the server.
            ok = '{"ok": true}'
            pg.route(
                "**/api/overlays/data/configure",
                lambda r: r.fulfill(status=200, content_type="application/json", body=ok),
            )
            pg.route(
                "**/api/overlays/data/cancel",
                lambda r: r.fulfill(status=200, content_type="application/json", body='{"parked": true}'),
            )
            pg.goto(base)
            pg.wait_for_function("typeof openDataset === 'function'", timeout=20_000)
            pg.evaluate("(d) => openDataset(d)", str(root))
            pg.wait_for_function(
                "(d) => window.datasets && window.datasets[d]", arg=str(root), timeout=60_000
            )
            pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [str(root), FRAMES])
            pg.wait_for_timeout(600)
            pg.evaluate(
                """() => { window.__toasts = []; const t = window.showToast;
                    window.showToast = (a, b, c, d) => { window.__toasts.push(a + ' :: ' + b); if (t) t(a,b,c,d); };
                    window.__status = []; const s = window.setStatus;
                    window.setStatus = (m) => { window.__status.push(m); if (s) s(m); }; }"""
            )
            pg.evaluate(
                "(sel) => { const p = document.querySelector(sel + ' .overlays-picker');"
                " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); }",
                PANEL,
            )
            pg.wait_for_function(
                "(s) => document.querySelectorAll(s + ' .overlays-cam-btn').length > 0",
                arg=PANEL,
                timeout=20_000,
            )
            pg.evaluate(
                """([s, name]) => { const row = document.querySelector(s + ' .overlays-obj-name');
                    if (row) { row.value = name; row.dispatchEvent(new Event('input', {bubbles: true})); } }""",
                [PANEL, LABEL],
            )
            pg.wait_for_timeout(900)
            pg.evaluate(
                """([s, want]) => { const set = new Set(want);
                    for (const b of document.querySelectorAll(s + ' .overlays-cam-btn'))
                      if (b.classList.contains('on') !== set.has(b.dataset.cam)) b.click(); }""",
                [PANEL, select],
            )
            pg.wait_for_timeout(400)
            out["selected"] = pg.evaluate("() => (window.Overlays.dataQuery() || {}).cameras")
            pg.evaluate(
                """(s) => { const cb = document.querySelector(s + ' .overlays-apply-cb');
                    cb.checked = true; cb.dispatchEvent(new Event('change', {bubbles: true})); }""",
                PANEL,
            )
            pg.wait_for_timeout(600)
            pg.evaluate("() => window.Overlays.applyOnTransport(true)")
            pg.wait_for_timeout(9000)
            out["toasts"] = pg.evaluate("() => window.__toasts")
            out["status"] = pg.evaluate("() => window.__status")
            requests.post(f"{base}/api/edits/apply", params={"dataset_id": str(root)}, timeout=300)
            browser.close()

        server.should_exit = True
        time.sleep(0.8)
        fresh = LeRobotDataset(repo, root=root)
        out["columns"] = sorted(mask_columns(fresh))
        out["stored"] = {c.split(".")[-1]: coverage(fresh, 0, c) for c in CAMS}
        return out

    return go


def test_a_dataset_with_no_masks_gets_its_column_and_every_frame(run_apply):
    """The reported shape of "does apply-and-play work on a new dataset": it
    played the whole episode and reported success with nothing stored. Every
    dataset is in this state until something adopts a column, so this is the
    ordinary way a dataset's masks begin."""
    out = run_apply(adopt_cams=[], select=[TOP])

    assert out["columns"], f"the run stored nothing and created no column: {out}"
    assert out["stored"]["top"] == (FRAMES, FRAMES), out["stored"]
    assert any("Save to commit" in s for s in out["status"]), out["status"]


def test_a_second_camera_is_adopted_beside_one_that_already_has_masks(run_apply):
    """Two cameras selected, one adopted. The partial form of the same defect:
    the un-adopted camera's frames went nowhere while it was on screen."""
    out = run_apply(adopt_cams=[TOP], select=CAMS)

    assert out["stored"]["top"] == (FRAMES, FRAMES), out["stored"]
    assert out["stored"]["wrist"] == (FRAMES, FRAMES), out["stored"]


def test_a_run_over_columns_that_exist_adopts_nothing_new(run_apply):
    """The complement: without it, the tests above are satisfied by a run that
    adopts indiscriminately. Nothing about the schema changes here."""
    out = run_apply(adopt_cams=[TOP], select=[TOP])

    assert out["columns"] == [TOP], f"a run over an existing column changed the schema: {out}"
    assert out["stored"]["top"] == (FRAMES, FRAMES), out["stored"]
    assert not out["toasts"], f"a run that could store everything warned anyway: {out['toasts']}"
    assert any("Save to commit" in s for s in out["status"]), out["status"]
