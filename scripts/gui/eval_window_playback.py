"""Evaluate windowed playback (window_playback.html) against a dataset, read-only.

Starts its own GUI server on a free port with a throwaway config and window
cache, or targets a running server with --base; opens the dataset by local
path; drives the page in headless Chromium (localhost is a secure origin, so
WebCodecs is available) through open, play, seeks and 2x; prints the page's
metrics and, for a local server, the server's request log.

Usage: eval_window_playback.py <dataset_root> <episode> [--base URL] [--rung R]
       [--play-seconds S] [--out FILE] [--insecure]
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

os.environ.setdefault("LEROBOT_GUI_CONFIG_DIR", tempfile.mkdtemp(prefix="window-eval-cfg-"))
os.environ.setdefault("LEROBOT_WINDOW_CACHE_DIR", tempfile.mkdtemp(prefix="window-eval-cache-"))

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("episode", type=int)
ap.add_argument("--base", default=None)
ap.add_argument("--rung", default="640")
ap.add_argument("--play-seconds", type=float, default=6.0)
ap.add_argument("--out", default="eval_window_playback.json")
ap.add_argument(
    "--insecure", action="store_true", help="accept a self-signed certificate on --base (measurement only)"
)
args = ap.parse_args()

# The config and cache dirs must be set before the server module is imported.
import requests  # noqa: E402
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from lerobot.gui import server as gui_server_mod  # noqa: E402

log_lines: list[str] = []


class Collect(logging.Handler):
    def emit(self, record):
        if "window-playback" in record.getMessage():
            log_lines.append(f"{time.strftime('%H:%M:%S')} {record.getMessage()}")


logging.getLogger("lerobot.gui.api.window_playback").addHandler(Collect())
logging.getLogger("lerobot.gui.api.window_playback").setLevel(logging.INFO)

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
r = requests.post(
    f"{base}/api/datasets", json={"local_path": args.root}, timeout=600, verify=not args.insecure
)
assert r.status_code == 200, r.text
dataset_id = r.json()["id"]
print(f"opened {dataset_id} in {time.perf_counter() - t:.1f} s")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_context(
        viewport={"width": 1600, "height": 1000}, ignore_https_errors=args.insecure
    ).new_page()
    console: list[str] = []
    page.on("console", lambda m: console.append(m.text))
    page.on("pageerror", lambda e: console.append(f"PAGEERROR {e}"))
    page.goto(
        f"{base}/static/window_playback.html?dataset={dataset_id}&episode={args.episode}&rung={args.rung}&autoplay=1"
    )
    page.evaluate("window.__playback && window.__playback.mark('play1x')")
    page.wait_for_function("window.__metrics && window.__metrics.firstPicture != null", timeout=60_000)
    page.wait_for_timeout(int(args.play_seconds * 1000))
    length = page.evaluate("window.__metrics.painted.length && window.__playback.state()")
    page.evaluate("window.__playback.mark('seeks')")
    total = page.evaluate("document.getElementById('scrub').max") or "0"
    total = int(total)
    for frac in (0.5, 0.1, 0.9, 0.3):
        page.evaluate(f"window.__playback.seek({int(total * frac)})")
        page.wait_for_timeout(1500)
    page.evaluate("window.__playback.mark('play2x')")
    page.evaluate("window.__playback.rate(2)")
    page.evaluate("window.__playback.seek(0)")
    page.evaluate("window.__playback.play()")
    page.wait_for_timeout(int(args.play_seconds * 1000))
    m = page.evaluate("window.__metrics")
    browser.close()
if srv:
    srv.should_exit = True

painted = m["painted"]
marks = m["marks"]


def phase_of(t):
    ph = "start"
    for mk in marks:
        if mk["t"] <= t:
            ph = mk["phase"]
    return ph


print(f"\nbundle {m['bundle']} | first picture {m['firstPicture']} ms after open")
print("seeks (ms to first painted frame at/after target):", [(s["frame"], s["ms"]) for s in m["seeks"]])
print("stalls:", m["stalls"][:10], "..." if len(m["stalls"]) > 10 else "")
for ph in ("start", "play1x", "seeks", "play2x"):
    fs = [f for f in painted if phase_of(f["t"]) == ph]
    if len(fs) < 2:
        continue
    frames = [f["frame"] for f in fs]
    span = (fs[-1]["t"] - fs[0]["t"]) / 1000
    print(
        f"  {ph}: painted {len(fs)} frames in {span:.1f} s wall ({len(fs) / max(span, 1e-3):.1f} fps), frames {frames[0]}..{frames[-1]}"
    )
ws = m["windows"]
by = {}
for w in ws:
    by.setdefault(w["cache"], []).append(w)
for k, v in by.items():
    print(
        f"  windows {k}: {len(v)}, {sum(w['bytes'] for w in v) / 1e6:.2f} MB, ms median {statistics.median(w['ms'] for w in v):.0f} max {max(w['ms'] for w in v):.0f}, "
        f"rate Mbit/s {[w['rateMbps'] for w in v][:6]}"
    )
print("  window lengths in order:", [w["len"] for w in ws])
print("decode:", m["decode"])
print("errors:", m["errors"])
print(f"\nserver log: {len(log_lines)} lines")
for line in log_lines[:10]:
    print("  " + line)
if len(log_lines) > 10:
    print(f"  ... {len(log_lines) - 10} more")
errs = [c for c in console if "PAGEERROR" in c or "error" in c.lower()]
if errs:
    print("console errors:", errs[:5])
Path(args.out).write_text(json.dumps({"metrics": m, "server_log": log_lines, "console": console}, indent=1))
print("written", args.out)
