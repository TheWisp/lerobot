---
name: gui-async-hygiene
description: What may and may not run on the GUI server's event loop, and how to offload the rest. Use when adding or editing anything under src/lerobot/gui — an endpoint, a background task, a Hub call, a subprocess — or when the GUI freezes.
---

# GUI async hygiene

The GUI is one asyncio process. A synchronous call that blocks inside an
`async def` stops **everything** — static files, websockets, SSE, other
operators' requests — for as long as it takes. This is not theoretical: a
blackholed Hub made the whole GUI unusable until kernel TCP timeouts fired,
because `whoami()` was awaited inline.

## The rules

**1. Nothing that can block runs on the loop.** Network calls, subprocesses,
large file reads, parquet rewrites, model loads. Offload them.

**2. Never the shared default executor.** `run_in_executor(None, …)` and
`asyncio.to_thread(…)` share one small pool with video decode, camera teardown
and FastAPI's own sync handlers, so a stalled Hub probe starves work with
nothing to do with it. Give each class of work its own bounded pool:

```python
_image_status_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-image-status")
```

Precedent: `_decode_executor` and `_prefetch_executor` in `api/datasets.py`.

**3. The timeout belongs in the client, not the awaiter.** Cancelling
`asyncio.wait_for(asyncio.to_thread(f))` abandons the _await_; `f` keeps running
and keeps its slot in the pool, so repeated requests leak it. Only
`HfApi(timeout=…)`, `requests(…, timeout=…)`, `subprocess.run(…, timeout=…)`
actually bound the work.

**4. A passive GET performs no network I/O.** It returns cached state plus its
freshness; refreshing is a background task or an explicit POST. `image-status`
is the worked example: the `git fetch` moved behind
`POST /training/image-status/refresh`, the probe runs in a named pool, and the
result is cached for 30s.

**5. Transfers go through the subprocess worker.** `hub_worker.py` owns
uploads and downloads end to end. The server never calls `huggingface_hub`
transfer helpers directly.

## The trap that keeps catching people

**Blocking hides behind things that do not look like I/O.** The freeze above was
`return get_auth_status()` — one level down, `whoami()`. Later,
`LeRobotDataset(repo_id)` inside an async handler resolved against the Hub with
no call in sight that looks like network. A _constructor_.

So when reviewing an async handler, do not scan for `requests` and
`subprocess`. Ask of every call: **could this reach a syscall that waits on
something outside this process?** If you cannot answer without opening the
callee, open the callee.

## What the lint does and does not cover

`scripts/lint/no_blocking_in_async.py` runs in pre-commit as a ratchet against a
baseline. Read its output correctly:

- The **shared-pool** rule is sound. `run_in_executor(None, …)` is a literal in
  the AST and cannot be disguised.
- The **blocking-call** rules are a heuristic over a denylist of names. They
  find what someone thought to list and miss everything else — they missed the
  `LeRobotDataset` case entirely.

A clean run means "none of the known shapes are present", never "this cannot
block". Real coverage needs a runtime detector measuring the symptom rather
than enumerating causes; tracked as a `[High]` item in `src/lerobot/gui/TODO.md`.

Silence a justified case with `# blocking-ok: <reason>` (startup paths that run
before the loop serves traffic, work already on a bounded pool) rather than
widening the baseline.
