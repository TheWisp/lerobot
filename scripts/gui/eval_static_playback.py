"""Evaluate the static playback prototype against a dataset, read-only.

Either starts its own GUI server on a free port (with a throwaway config dir,
so the user's GUI state is untouched) or targets a running one with --base,
opens the dataset by local path, drives /static/static_playback.html in
headless Chromium through one play, a few seeks and a 2x phase, and prints
the page's metrics: request counts and bytes, first frame, seek latency,
presented and skipped frames per phase, mask paint time, decoder drops.
With a local server the server's own request log for the session is
printed too.

Usage: eval_static_playback.py <dataset_root> <episode> [--base URL]
       [--play-seconds S] [--seeks N] [--out FILE]
"""

import argparse
import json
import logging
import os
import socket
import statistics
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("LEROBOT_GUI_CONFIG_DIR", tempfile.mkdtemp(prefix="static-eval-cfg-"))

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("episode", type=int)
ap.add_argument("--play-seconds", type=float, default=6.0)
ap.add_argument("--seeks", type=int, default=4)
ap.add_argument("--out", default="eval_static_playback.json")
ap.add_argument("--base", default=None, help="use a running server at this URL instead of starting one")
args = ap.parse_args()

# The config dir must be set before the GUI server module is imported, and the
# argument parse before that so --help costs no import.
import requests  # noqa: E402
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from lerobot.gui import server as gui_server_mod  # noqa: E402

log_lines: list[str] = []


class Collect(logging.Handler):
    def emit(self, record):
        if "static-playback" in record.getMessage():
            log_lines.append(f"{time.strftime('%H:%M:%S')} {record.getMessage()}")


logging.getLogger("lerobot.gui.api.static_playback").addHandler(Collect())
logging.getLogger("lerobot.gui.api.static_playback").setLevel(logging.INFO)

srv = None
if args.base:
    base = args.base
else:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(150):
        try:
            if requests.get(base, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
t = time.perf_counter()
r = requests.post(f"{base}/api/datasets", json={"local_path": args.root}, timeout=600)
assert r.status_code == 200, r.text
dataset_id = r.json()["id"]
print(f"opened {dataset_id} in {time.perf_counter() - t:.1f} s")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    console = []
    page.on("console", lambda m: console.append(m.text))
    t_nav = time.perf_counter()
    page.goto(f"{base}/static/static_playback.html?dataset={dataset_id}&episode={args.episode}")
    page.wait_for_function("window.__playback && window.__playback.ready()", timeout=60_000)
    t_ready = time.perf_counter() - t_nav
    page.evaluate("window.__playback.mark('play1x')")
    page.evaluate("window.__playback.play()")
    page.wait_for_timeout(int(args.play_seconds * 1000))
    m0 = page.evaluate("window.__metrics")
    length = m0["manifest"]["length"]
    rng_targets = [int(length * f) for f in (0.5, 0.1, 0.9, 0.3)][: args.seeks]
    page.evaluate("window.__playback.mark('seeks')")
    for f in rng_targets:
        page.evaluate(f"window.__playback.seek({f})")
        page.wait_for_timeout(1500)
    page.evaluate("window.__playback.mark('play2x')")
    page.evaluate("window.__playback.rate(2)")
    page.wait_for_timeout(int(args.play_seconds * 1000))
    page.evaluate("window.__playback.quality()")
    m = page.evaluate("window.__metrics")
    browser.close()
if srv:
    srv.should_exit = True

frames = m["frames"]
cams = sorted({f["camera"] for f in frames})
print(f"\npage ready (manifest+masks+features+metadata) {t_ready:.2f} s; timeline {m['timeline']}")
for k, c in m["manifest"]["cameras"].items():
    print(
        f"  {k}: {c['codec']}/{c['pix_fmt']} {c['width']}x{c['height']}@{c['fps']} file {c['file_bytes'] / 1e6:.1f} MB moov_first={c['moov_first']}"
    )
print("first frame after play, ms:", m["firstFrame"])
print("seeks:", [(s["frame"], s["ms"]) for s in m["seeks"]])
for cam in cams:
    q = m["quality"].get(cam, {})
    print(f"  {cam}: decoder total {q.get('total')} dropped {q.get('dropped')}")
    for phase in ("play1x", "seeks", "play2x"):
        fs = [f for f in frames if f["camera"] == cam and f["phase"] == phase]
        if not fs:
            continue
        gaps = [f["gap"] for f in fs[1:]]
        paints = [f["paintMs"] for f in fs]
        span = fs[-1]["mediaTime"] - fs[0]["mediaTime"]
        print(
            f"      {phase}: presented {len(fs)} frames over {span:.1f} s of media, skipped {sum(g - 1 for g in gaps if g > 1)}, "
            f"paint median {statistics.median(paints):.2f} ms max {max(paints):.2f} ms, masks drawn on {sum(1 for f in fs if f['drawn'])}"
        )
reqs = m["requests"]
by_kind = {}
for r in reqs:
    key = (
        "video"
        if "/video-file/" in r["url"]
        else "masks"
        if "/masks" in r["url"]
        else "features"
        if "/features" in r["url"]
        else "manifest"
        if "/playback" in r["url"]
        else "other"
    )
    by_kind.setdefault(key, []).append(r)
for k, v in by_kind.items():
    print(
        f"  requests {k}: {len(v)}, {sum(r['bytes'] for r in v) / 1e6:.2f} MB, durations ms {sorted(r['ms'] for r in v)[:8]}{'...' if len(v) > 8 else ''}"
    )
print("errors:", m["errors"])
print(f"\nserver log: {len(log_lines)} lines")
for line in log_lines[:12]:
    print("  " + line)
if len(log_lines) > 12:
    print(f"  ... {len(log_lines) - 12} more")
Path(args.out).write_text(json.dumps({"metrics": m, "server_log": log_lines, "ready_s": t_ready}, indent=1))
print("written", args.out)
