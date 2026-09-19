"""The server can be served over HTTPS, which is what makes the Data tab's Low
Bandwidth playback available away from localhost: the browser has a video
decoder only in a secure context (docs/dataset_playback.md)."""

from __future__ import annotations

import sys


def test_the_certificate_flags_reach_uvicorn(monkeypatch):
    import uvicorn

    from lerobot.gui import server as gui_server

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr(gui_server, "_mount_mcp", lambda **kw: None)
    import lerobot.gui.mdns as mdns

    monkeypatch.setattr(mdns, "advertise", lambda *a, **k: None)
    monkeypatch.setattr(mdns, "detect_lan_ip", lambda *a, **k: None)
    gui_server.run_server(host="127.0.0.1", port=0, ssl_certfile="/c.pem", ssl_keyfile="/k.pem")
    assert seen["ssl_certfile"] == "/c.pem" and seen["ssl_keyfile"] == "/k.pem"
    # And the complement: without the flags, uvicorn is asked for plain HTTP.
    seen.clear()
    gui_server.run_server(host="127.0.0.1", port=0)
    assert seen["ssl_certfile"] is None and seen["ssl_keyfile"] is None


def test_the_cli_accepts_the_flags(monkeypatch):
    from lerobot.gui import server as gui_server

    got = {}
    monkeypatch.setattr(gui_server, "run_server", lambda **kw: got.update(kw))
    monkeypatch.setattr(gui_server, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["lerobot-gui", "--ssl-certfile", "/c.pem", "--ssl-keyfile", "/k.pem"])
    gui_server.main()
    assert got["ssl_certfile"] == "/c.pem" and got["ssl_keyfile"] == "/k.pem"
