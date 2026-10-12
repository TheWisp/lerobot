"""Read-only motor telemetry through rig; guarded calls never change servo limits."""

import json
import signal
import threading
import time

import rig


class LoadLimit(RuntimeError):  # noqa: N818 - retained log/API name
    pass


class Frozen(RuntimeError):  # noqa: N818 - retained log/API name
    pass


class LoadMonitor:
    def __init__(self, folder):
        self.folder = folder
        self.phase = "idle"
        self.target = None
        self.latest = {}
        self.error = None
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        with (self.folder / "motor_loads.jsonl").open("w", buffering=1) as out:
            while not self.stop.is_set():
                began = time.monotonic()
                try:
                    s = rig.state()
                    self.latest = s
                    e = {
                        "time_s": round(began - self.started, 4),
                        "phase": self.phase,
                        "target": self.target,
                        **s,
                    }
                    self.samples.append(e)
                    out.write(json.dumps(e) + "\n")
                except Exception as exc:
                    self.error = str(exc)
                self.stop.wait(max(0, 0.05 - (time.monotonic() - began)))

    def check(self, *_):
        s = self.latest
        if getattr(self, "extra_check", None):
            self.extra_check(s)
        if s.get("halted"):
            raise Frozen(s.get("reason", "frozen"))
        if s.get("max_motor_temp_c", 0) > 50:
            raise LoadLimit("Temperature above 50 C")
        loads = s.get("load", {})
        arm = {n: abs(v) / 10 for n, v in loads.items() if n != "gripper"}
        if arm and max(arm.values()) >= 70:
            raise LoadLimit(f"Load guard: {arm}")

    def call(self, fn, target=None, recovery=False):
        self.target = target
        old = signal.signal(signal.SIGALRM, self.check)
        signal.setitimer(signal.ITIMER_REAL, 0.05, 0.05)
        try:
            if not recovery:
                self.check()
            if recovery:
                signal.setitimer(signal.ITIMER_REAL, 0)
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        phases = {}
        for s in self.samples:
            d = phases.setdefault(
                s["phase"],
                {"samples": 0, "peak_abs_load_percent": {}, "first_s": s["time_s"], "last_s": s["time_s"]},
            )
            d["samples"] += 1
            d["last_s"] = s["time_s"]
            for n, v in s.get("load", {}).items():
                d["peak_abs_load_percent"][n] = max(d["peak_abs_load_percent"].get(n, 0), abs(v) / 10)
        (self.folder / "load_summary.json").write_text(json.dumps(phases, indent=2))
        return phases
