"""
Quest3 WebXR → PC latency probe.

Serves webxr_teleop.html over HTTPS on port 8000 and accepts a WebSocket
upgrade on /ws. The Quest's browser opens the page, enters VR, and streams
each XR frame's pose + timestamps. PC sends periodic pings; Quest echoes
them; receiver computes RTT.

Run:
    .venv/bin/python -m lerobot.robots.so107_description.experimental.teleop_quest.latency_receiver

First-run will auto-generate a self-signed cert (./cert.pem, ./key.pem).
Then on the Quest: open  https://<PC-IP>:8000/  in the headset browser,
accept the security warning (self-signed), click Connect → Enter VR.

CSV log: /tmp/quest_latency_HHMMSS.csv (one row per XR frame).
"""

from __future__ import annotations

import asyncio
import csv
import json
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aiohttp import WSMsgType, web

PORT = 8443
PING_INTERVAL = 0.1  # seconds
HERE = Path(__file__).parent
CERT = HERE / "cert.pem"
KEY = HERE / "key.pem"
HTML = HERE / "webxr_teleop.html"


def ensure_cert() -> None:
    if CERT.exists() and KEY.exists():
        return
    print("Generating self-signed cert ...")
    subprocess.check_call(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(KEY),
            "-out",
            str(CERT),
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/CN=quest-latency-probe",
        ]
    )
    print(f"  wrote {CERT}, {KEY}")


def get_lan_ip() -> str:
    """Best-effort: return our LAN IP, for printing the URL the Quest should open."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


class Stats:
    def __init__(self) -> None:
        self.rtts: list[float] = []
        self.frames = 0
        self.last_left: list | None = None
        self.last_right: list | None = None
        self.last_print = time.perf_counter()

    def add_rtt(self, ms: float) -> None:
        self.rtts.append(ms)
        if len(self.rtts) > 2000:
            self.rtts = self.rtts[-2000:]

    def maybe_print(self, last_rtt: float | None) -> None:
        now = time.perf_counter()
        if now - self.last_print < 1.0 or not self.rtts:
            return
        s = sorted(self.rtts[-200:])
        n = len(s)
        mean = sum(s) / n
        p95 = s[min(int(n * 0.95), n - 1)]
        p99 = s[min(int(n * 0.99), n - 1)]
        jitter = max(s) - min(s)
        last = f"{last_rtt:.1f}" if last_rtt is not None else "—"

        def pose_str(p):
            if p is None:
                return "—"
            return f"({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})"

        print(
            f"f={self.frames:5d}  RTT last={last:>5} mean={mean:5.1f} p95={p95:5.1f} "
            f"p99={p99:5.1f} jit={jitter:5.1f} 1way≈{mean / 2:4.1f}ms  "
            f"L={pose_str(self.last_left)}  R={pose_str(self.last_right)}"
        )
        self.last_print = now


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=64 * 1024)
    await ws.prepare(request)
    peer = request.remote
    print(f"client connected from {peer}")

    stats = Stats()
    pending: dict[int, float] = {}  # ping seq -> t_send (perf_counter ms)
    last_rtt: float | None = None

    log_path = Path(tempfile.gettempdir()) / f"quest_latency_{time.strftime('%H%M%S')}.csv"
    log_f = log_path.open("w", newline="")
    csv_w = csv.writer(log_f)
    csv_w.writerow(
        [
            "t_pc_recv_ms",
            "t_quest_xr_frame",
            "t_quest_send",
            "n_poses",
            "left_pos",
            "right_pos",
            "rtt_ms_last",
        ]
    )

    async def ping_loop() -> None:
        seq = 0
        while not ws.closed:
            t = time.perf_counter() * 1000
            pending[seq] = t
            try:
                await ws.send_json({"type": "ping", "seq": seq, "t_pc_send": t})
            except ConnectionError:
                return
            seq += 1
            await asyncio.sleep(PING_INTERVAL)

    ping_task = asyncio.create_task(ping_loop())

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            t_recv = time.perf_counter() * 1000
            data = json.loads(msg.data)
            mtype = data.get("type")
            if mtype == "pong":
                t_send = pending.pop(data["seq"], None)
                if t_send is not None:
                    rtt = t_recv - t_send
                    stats.add_rtt(rtt)
                    last_rtt = rtt
                    # report back to the page so it can show RTT in HUD
                    await ws.send_json({"type": "rtt_report", "rtt_ms": rtt})
            elif mtype == "frame":
                stats.frames += 1
                poses = data.get("poses", [])
                left = next((p["pos"] for p in poses if p["hand"] == "left"), None)
                right = next((p["pos"] for p in poses if p["hand"] == "right"), None)
                stats.last_left = left
                stats.last_right = right
                csv_w.writerow(
                    [
                        f"{t_recv:.3f}",
                        f"{data.get('t_quest_xr_frame', 0):.3f}",
                        f"{data.get('t_quest_send', 0):.3f}",
                        len(poses),
                        str(left) if left else "",
                        str(right) if right else "",
                        f"{last_rtt:.3f}" if last_rtt is not None else "",
                    ]
                )
            stats.maybe_print(last_rtt)
    finally:
        ping_task.cancel()
        log_f.close()
        print(f"client disconnected. CSV: {log_path}")

    return ws


async def index_handler(request: web.Request) -> web.Response:
    return web.Response(text=HTML.read_text(), content_type="text/html")


def main() -> int:
    ensure_cert()
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))

    lan_ip = get_lan_ip()
    print()
    print(f"  On Quest3 browser, open:  https://{lan_ip}:{PORT}/")
    print("  Accept the self-signed cert warning, then Connect → Enter VR.")
    print()
    print("Stats print once per second (rolling 20s window).")
    print()

    web.run_app(app, host="0.0.0.0", port=PORT, ssl_context=ssl_ctx, print=None)  # nosec B104 (LAN-only dev tool, intentional)
    return 0


if __name__ == "__main__":
    sys.exit(main())
