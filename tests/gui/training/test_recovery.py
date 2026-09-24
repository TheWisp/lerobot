"""CPU-only recovery tests. All runs, checkpoints and logs live in tmp_path."""

import json
import struct
import sys
import time

import pytest

from lerobot.gui.training import recovery
from lerobot.gui.training.hosts import HostRegistry, TrainingHost
from lerobot.gui.training.orchestrator import Orchestrator
from lerobot.gui.training.runs import Run, RunPaths, RunRegistry, RunState
from lerobot.gui.training.transport import SubprocessTransport


def tensor_file(path):
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)


def checkpoint(root, run_id, step, *, hvla=False):
    path = root / run_id / "checkpoints" / f"{step:06d}"
    pretrained = path / "pretrained_model"
    state = path / "training_state"
    pretrained.mkdir(parents=True)
    state.mkdir()
    (pretrained / "config.json").write_text("{}")
    (pretrained / "train_config.json").write_text(
        json.dumps(
            {
                "steps": 100,
                "batch_size": 4,
                "num_workers": 0,
                "save_freq": 10,
                "scheduler": {"type": "cosine_decay_with_warmup"},
            }
        )
    )
    tensor_file(pretrained / "model.safetensors")
    (state / "training_step.json").write_text(json.dumps({"step": step}))
    if hvla:
        import zipfile

        with zipfile.ZipFile(state / "optimizer.pt", "w") as archive:
            archive.writestr("optimizer/data.pkl", b"metadata")
    else:
        tensor_file(state / "rng_state.safetensors")
        tensor_file(state / "optimizer_state.safetensors")
        (state / "optimizer_param_groups.json").write_text("{}")
        (state / "scheduler_state.json").write_text("{}")
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    host = TrainingHost(id="local", display_name="local", transport=SubprocessTransport(workdir=tmp_path))
    orch = Orchestrator(HostRegistry(hosts=[host]), RunRegistry(tmp_path / "runs"))
    now = [1000.0]
    manager = recovery.RecoveryManager(orch, clock=lambda: now[0], quiet_seconds=2, settle_timeout=20)
    monkeypatch.setattr(recovery, "crash_materials", lambda _: ([], False))
    monkeypatch.setattr(orch, "refresh", lambda _: None)
    launches = []

    def create(request):
        launches.append(request)
        child = Run(
            run_id=f"child{len(launches)}",
            host_id="local",
            recipe_name=request.recipe_name,
            dataset_id="data",
            args=request.args,
            state=RunState.RUNNING,
            created_at=now[0],
            started_at=now[0],
            idempotency_key=request.idempotency_key,
        )
        orch._runs.save(child)
        return child

    monkeypatch.setattr(orch, "start", create)
    run = Run(
        run_id="first",
        host_id="local",
        recipe_name="fake",
        dataset_id="data",
        args={"__recipe__": "__fake__", "steps": 100},
        state=RunState.RUNNING,
        created_at=900,
        started_at=900,
    )
    orch._runs.save(run)
    return manager, orch, run, now, launches


def crash(orch, run):
    run.state = RunState.STOPPED
    run.error = "exit code 139"
    orch._runs.save(run)


def settle(manager, now):
    manager.tick()
    now[0] += 3
    manager.tick()


def test_default_off_and_normal_completion_do_not_restart(env):
    manager, orch, run, now, launches = env
    crash(orch, run)
    settle(manager, now)
    assert not launches
    run.state = RunState.RUNNING
    orch._runs.save(run)
    manager.configure(run.run_id, enabled=True, delay_seconds=0)
    run.state = RunState.COMPLETED
    orch._runs.save(run)
    settle(manager, now)
    assert manager.get(run.run_id)["status"] == "completed"
    assert not launches


def test_waits_for_writes_then_uses_latest_checkpoint(env):
    manager, orch, run, now, launches = env
    checkpoint(manager.root, run.run_id, 10)
    latest = checkpoint(manager.root, run.run_id, 20)
    model = latest / "pretrained_model/model.safetensors"
    good = model.read_bytes()
    model.write_bytes(good[:8])
    manager.configure(run.run_id, enabled=True, delay_seconds=0)
    crash(orch, run)
    manager.tick()
    assert not launches
    now[0] += 3
    model.write_bytes(good)  # writer finishes before the next stability sample
    manager.tick()
    assert not launches
    now[0] += 3
    manager.tick()
    assert launches[0].args["__resumed_from_step__"] == 20
    assert launches[0].args["steps"] == 100
    assert model.read_bytes() == good


def test_corrupt_latest_falls_back_and_keeps_evidence(env):
    manager, orch, run, now, launches = env
    checkpoint(manager.root, run.run_id, 10)
    latest = checkpoint(manager.root, run.run_id, 20)
    (latest / "training_state/optimizer_state.safetensors").write_bytes(b"broken")
    manager.configure(run.run_id, enabled=True, delay_seconds=0)
    crash(orch, run)
    settle(manager, now)
    assert launches[0].args["__resumed_from_step__"] == 10
    record = manager.get(run.run_id)
    assert record["attempts"] == 1
    assert len(record["incidents"][0]["skipped_checkpoints"]) == 1
    assert record["incidents"][0]["log_path"].endswith("first/stderr.log")


def test_stop_or_disable_during_wait_prevents_resume(env):
    manager, orch, run, now, launches = env
    checkpoint(manager.root, run.run_id, 10)
    manager.configure(run.run_id, enabled=True, delay_seconds=10)
    crash(orch, run)
    manager.tick()
    manager.stop(run.run_id)
    now[0] += 30
    manager.tick()
    assert not launches
    assert not manager.get(run.run_id)["enabled"]


def test_retry_budget_persists_and_ancestor_checkpoint_is_reused(env):
    manager, orch, run, now, launches = env
    checkpoint(manager.root, run.run_id, 10)
    manager.configure(run.run_id, enabled=True, max_retries=2, delay_seconds=0)
    crash(orch, run)
    settle(manager, now)
    for _ in range(2):
        record = manager.get(run.run_id)
        child = orch._runs.load(record["active_run_id"])
        crash(orch, child)  # no checkpoint in this attempt
        manager = recovery.RecoveryManager(orch, clock=lambda: now[0], quiet_seconds=2)
        settle(manager, now)
    assert len(launches) == 2
    assert manager.get(run.run_id)["status"] == "blocked"
    assert all(request.args["__resumed_from_run__"] == "first" for request in launches)


def test_restarts_after_crash_between_resume_and_recording_child(env, monkeypatch):
    manager, orch, run, now, launches = env
    checkpoint(manager.root, run.run_id, 10)
    manager.configure(run.run_id, enabled=True, delay_seconds=0)
    crash(orch, run)
    manager.tick()
    now[0] += 3
    original = manager._adopt
    monkeypatch.setattr(manager, "_adopt", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        manager.tick()
    monkeypatch.setattr(manager, "_adopt", original)
    restarted = recovery.RecoveryManager(orch, clock=lambda: now[0])
    restarted.tick()
    assert len(launches) == 1
    assert restarted.get(run.run_id)["active_run_id"] == "child1"


def test_no_checkpoint_and_busy_diagnostics_stop_without_launch(env, monkeypatch):
    manager, orch, run, now, launches = env
    manager.configure(run.run_id, enabled=True, delay_seconds=0)
    crash(orch, run)
    settle(manager, now)
    assert "No complete checkpoint" in manager.get(run.run_id)["message"]
    assert not launches

    run.state = RunState.RUNNING
    orch._runs.save(run)
    manager.configure(run.run_id, enabled=True, delay_seconds=0)
    crash(orch, run)
    monkeypatch.setattr(recovery, "crash_materials", lambda _: (["/missing/core"], True))
    manager.tick()
    now[0] += 21
    manager.tick()
    assert "not settled" in manager.get(run.run_id)["message"]
    assert not launches


def test_hvla_requires_complete_optimizer_archive(tmp_path):
    path = checkpoint(tmp_path, "hvla", 10, hvla=True)
    assert recovery.check_checkpoint(path, 10, True) == (True, "")
    file = path / "training_state/optimizer.pt"
    file.write_bytes(file.read_bytes()[:-10])
    assert not recovery.check_checkpoint(path, 10, True)[0]


def test_manual_resume_ancestry_cannot_jump_to_later_checkpoint(env):
    manager, orch, run, now, launches = env
    parent = Run(
        run_id="parent",
        host_id="local",
        recipe_name="old",
        dataset_id="data",
        args={"__recipe__": "__fake__", "steps": 100},
        state=RunState.COMPLETED,
        created_at=800,
    )
    orch._runs.save(parent)
    checkpoint(manager.root, "parent", 10)
    checkpoint(manager.root, "parent", 80)
    run.args.update(__resumed_from_run__="parent", __resumed_from_step__=10, steps=200, batch_size=8)
    orch._runs.save(run)
    manager.configure(run.run_id, enabled=True, delay_seconds=0)
    crash(orch, run)
    settle(manager, now)
    assert launches[0].args["__resumed_from_step__"] == 10
    assert launches[0].args["steps"] == 200
    assert launches[0].args["batch_size"] == 8


def test_background_loop_recovers_without_browser_requests(env):
    manager, orch, run, now, launches = env
    checkpoint(manager.root, run.run_id, 10)
    manager.interval = 0.01
    manager.quiet_seconds = 0
    manager.configure(run.run_id, enabled=True, delay_seconds=0)
    crash(orch, run)
    manager.start()
    try:
        deadline = time.monotonic() + 3
        while not launches and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(launches) == 1
    finally:
        manager.close()


def test_existing_training_state_loader_preserves_next_update(tmp_path):
    import torch

    from lerobot.common.train_utils import load_training_state, save_training_state
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=2,
        num_decay_steps=20,
        peak_lr=0.01,
        decay_lr=0.001,
    )
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    scheduler = config.build(optimizer, 20)
    for _ in range(5):
        weight.square().sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    save_training_state(tmp_path, 5, optimizer, scheduler)
    resumed_weight = torch.nn.Parameter(weight.detach().clone())
    resumed_opt = torch.optim.AdamW([resumed_weight], lr=0.01)
    resumed_scheduler = config.build(resumed_opt, 20)
    step, resumed_opt, resumed_scheduler = load_training_state(tmp_path, resumed_opt, resumed_scheduler)
    assert step == 5
    assert resumed_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for param, opt, schedule in [
        (weight, optimizer, scheduler),
        (resumed_weight, resumed_opt, resumed_scheduler),
    ]:
        param.square().sum().backward()
        opt.step()
        schedule.step()
    torch.testing.assert_close(weight, resumed_weight, rtol=0, atol=0)
    assert resumed_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_real_local_exit_code_survives_liveness_probe(tmp_path):
    host = TrainingHost(id="local", display_name="local", transport=SubprocessTransport(workdir=tmp_path))
    orch = Orchestrator(HostRegistry(hosts=[host]), RunRegistry(tmp_path))
    paths = RunPaths.for_run("exit", tmp_path)
    first = orch._client_for_host(host, paths)
    pid = first.launch([sys.executable, "-c", "raise SystemExit(7)"], {}, paths.root, paths.stderr_log)
    second = orch._client_for_host(host, paths)
    assert first is second
    run = Run(
        run_id="exit",
        host_id="local",
        recipe_name="local exit",
        dataset_id="synthetic",
        args={"steps": 100},
        state=RunState.RUNNING,
        created_at=time.time(),
        started_at=time.time(),
        session_id=pid,
    )
    orch._runs.save(run)
    deadline = time.monotonic() + 10
    while orch.list_runs()[0].state == RunState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.02)
    assert second.exit_code(pid) == 7
    events = [json.loads(line) for line in paths.events_jsonl.read_text().splitlines()]
    assert next(event for event in events if event["type"] == "process_exit")["exit_code"] == 7


def test_real_cpu_process_crashes_resumes_and_completes_without_browser(tmp_path, monkeypatch):
    from lerobot.gui.training.orchestrator import StartRequest

    worker = tmp_path / "cpu_worker.py"
    worker.write_text("""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from lerobot.common.train_utils import save_training_state, load_training_state
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
out = Path(sys.argv[1])
resume = Path(sys.argv[2]) if len(sys.argv) > 2 else None
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scheduler = CosineDecayWithWarmupSchedulerConfig(1, 4, 0.01, 0.001).build(optimizer, 4)
step = 0
if resume:
    model.load_state_dict(load_file(str(resume / 'pretrained_model/model.safetensors')))
    step, optimizer, scheduler = load_training_state(resume, optimizer, scheduler)
    (out / 'loaded.json').write_text(json.dumps({'step':step,'lr':optimizer.param_groups[0]['lr']}))
for step in range(step + 1, 5):
    model(torch.ones(1, 1)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    if step % 2 == 0:
        ckpt = out / 'checkpoints' / f'{step:06d}'
        pm = ckpt / 'pretrained_model'
        pm.mkdir(parents=True)
        save_file(model.state_dict(), str(pm / 'model.safetensors'))
        (pm / 'config.json').write_text('{}')
        (pm / 'train_config.json').write_text(json.dumps({'steps':4,'scheduler':{'type':'cosine_decay_with_warmup'}}))
        save_training_state(ckpt, step, optimizer, scheduler)
    if not resume and step == 2:
        raise SystemExit(7)
""")
    host = TrainingHost(id="local", display_name="local", transport=SubprocessTransport(workdir=tmp_path))
    orch = Orchestrator(HostRegistry(hosts=[host]), RunRegistry(tmp_path / "runs"))

    def launch(host, run, paths):
        command = [sys.executable, str(worker), str(paths.root)]
        if run.args.get("__resume_checkpoint__"):
            command.append(run.args["__resume_checkpoint__"])
        return orch._client_for_host(host, paths, run).launch(
            command, {"CUDA_VISIBLE_DEVICES": ""}, paths.root, paths.stderr_log
        )

    monkeypatch.setattr(orch, "_launch_worker", launch)
    monkeypatch.setattr(recovery, "crash_materials", lambda _: ([], False))
    run = orch.start(
        StartRequest(
            host_id="local",
            recipe_name="CPU recovery test",
            dataset_id="synthetic",
            args={"__recipe__": "__fake__", "steps": 4},
        )
    )
    manager = recovery.RecoveryManager(orch, interval=0.03, quiet_seconds=0.03)
    manager.configure(run.run_id, enabled=True, delay_seconds=0, _initial=True)
    manager.start()
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            record = manager.get(run.run_id)
            if record["status"] in recovery.FINISHED:
                break
            time.sleep(0.05)
        assert record["status"] == "completed", record
        assert record["attempts"] == 1
        assert record["incidents"][0]["exit_code"] == 7
        child_root = manager.root / record["active_run_id"]
        loaded = json.loads((child_root / "loaded.json").read_text())
        assert loaded["step"] == 2
        groups = json.loads(
            (
                manager.root / run.run_id / "checkpoints/000002/training_state/optimizer_param_groups.json"
            ).read_text()
        )
        assert loaded["lr"] == groups[0]["lr"]
        assert (child_root / "checkpoints/000004/training_state/training_step.json").exists()
    finally:
        manager.close()
        for candidate in orch.list_runs():
            if candidate.state not in {RunState.COMPLETED, RunState.STOPPED, RunState.FAILED}:
                orch.stop(candidate.run_id)
