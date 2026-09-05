# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recording which training image a run actually used.

`lerobot-training:dev-local` is rebuilt whenever the trainer changes, so runs
recording only that tag could not be attributed to the code they ran. The
rendering half is tested under node; this covers the backend, the wiring, and
that runs predating it keep working.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from lerobot.gui.training.orchestrator import _apply_image_identity
from lerobot.gui.training.transport import SubprocessClient, _parse_image_identity

_STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"

CREATED = "2026-08-10T14:22:31.123456789Z"
REVISION = "9469cdf39abcdef0123456789abcdef012345678"


# ── The docker output parser ─────────────────────────────────────────────────


def test_parses_a_normal_inspect_line():
    assert _parse_image_identity(f"{CREATED}\t{REVISION}\n") == (CREATED, REVISION)


def test_a_missing_label_is_not_a_revision():
    """Docker prints '<no value>' for a missing label. Mistaking that for a
    revision would show it in the UI on every image built before the label."""
    created, revision = _parse_image_identity(f"{CREATED}\t<no value>\n")

    assert created == CREATED
    assert revision is None


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (f"{CREATED}\t\n", (CREATED, None)),  # label present but empty
        (f"{CREATED}\n", (CREATED, None)),  # no tab at all
        ("", (None, None)),  # docker produced nothing
        ("\n  \n", (None, None)),  # whitespace only
        ("\t" + REVISION, (None, REVISION)),  # date missing, label present
    ],
)
def test_partial_and_empty_output(stdout, expected):
    assert _parse_image_identity(stdout) == expected


def test_a_login_shell_banner_is_not_read_as_the_build_date():
    """Regression: the parser took the first non-blank line. Over SSH the remote
    login shell can print before the command runs, and that banner would have
    been reported as the image's build date."""
    banner = "Welcome to Ubuntu 24.04 LTS\nLast login: Mon Aug 10\n"

    assert _parse_image_identity(banner + f"{CREATED}\t{REVISION}\n") == (CREATED, REVISION)


def test_a_crossed_field_is_caught_rather_than_displayed():
    """A sha in the date slot renders as a plausible answer, so the parser
    asserts instead. Callers run it inside a try that logs and degrades."""
    with pytest.raises(AssertionError, match="fields crossed"):
        _parse_image_identity(f"{REVISION}\t{REVISION}\n")


def test_a_revision_is_never_split_on_whitespace():
    """Tab-separated, not whitespace: the older helper in api/training.py splits
    on whitespace and would misread any date format containing a space."""
    assert _parse_image_identity("2026-08-10 14:22:31 +0000\t" + REVISION) == (
        "2026-08-10 14:22:31 +0000",
        REVISION,
    )


# ── The transport ────────────────────────────────────────────────────────────


def test_missing_docker_yields_no_identity(monkeypatch, tmp_path):
    """On the launch path — an exception here would fail a run over a label."""

    def _no_docker(*a, **k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", _no_docker)
    from lerobot.gui.training.transport import SubprocessTransport

    client = SubprocessClient(SubprocessTransport(workdir=tmp_path))

    assert client.image_identity("whatever:tag") == (None, None)


def test_nonzero_exit_yields_no_identity(monkeypatch, tmp_path):
    """An absent image exits non-zero; that is not an error worth propagating."""

    def _fail(*a, **k):
        return subprocess.CompletedProcess(a[0] if a else [], 1, stdout="", stderr="No such image")

    monkeypatch.setattr(subprocess, "run", _fail)
    from lerobot.gui.training.transport import SubprocessTransport

    client = SubprocessClient(SubprocessTransport(workdir=tmp_path))

    assert client.image_identity("absent:tag") == (None, None)


def _ssh_client():
    from lerobot.gui.training.ssh_transport import SshClient
    from lerobot.gui.training.transport import SshTransport

    return SshClient(SshTransport(user="u", host="h", port=22))


def test_the_ssh_command_is_a_template_docker_accepts():
    """Regression: the format string was written with quadrupled braces in a
    plain literal, so nothing collapsed them and every remote call died with
    'template parsing error'. Assert the text actually sent."""
    sent = {}

    def _fake_exec(cmd, **kw):
        sent["cmd"] = cmd
        return subprocess.CompletedProcess([], 0, stdout=b"2026-01-01T00:00:00Z\tabc123\n")

    client = _ssh_client()
    client._exec = _fake_exec
    client.image_identity("some:tag")

    assert "{{.Created}}" in sent["cmd"], sent["cmd"]
    assert "{{{{" not in sent["cmd"], "quadrupled braces reach docker verbatim"
    assert "{{index .Config.Labels" in sent["cmd"]


def test_both_transports_agree_on_the_same_docker_output():
    """The identity must come from the host that will run the image; for an SSH
    host that is not the GUI's machine. Both must read it identically."""
    payload = f"{CREATED}\t{REVISION}\n"

    def _local(*a, **k):
        return subprocess.CompletedProcess([], 0, stdout=payload, stderr="")

    from lerobot.gui.training.transport import SubprocessTransport

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "run", _local)
        local = SubprocessClient(SubprocessTransport(workdir=Path("/tmp"))).image_identity("t")

    ssh_client = _ssh_client()
    ssh_client._exec = lambda cmd, **kw: subprocess.CompletedProcess([], 0, stdout=payload.encode())

    assert local == ssh_client.image_identity("t") == (CREATED, REVISION)


def test_ssh_timeout_returns_a_tuple_rather_than_raising():
    """The Protocol promises a tuple; SubprocessClient already swallows this."""
    client = _ssh_client()

    def _timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 10)

    client._exec = _timeout

    assert client.image_identity("t") == (None, None)


# ── Recording onto the run ───────────────────────────────────────────────────


@pytest.fixture
def orch_and_run(tmp_path):
    """An orchestrator with one saved run, ready to have identity stamped on it."""
    from lerobot.gui.training.hosts import HostRegistry, TrainingHost
    from lerobot.gui.training.orchestrator import Orchestrator
    from lerobot.gui.training.runs import Run, RunRegistry, RunState
    from lerobot.gui.training.transport import SubprocessTransport

    hosts = HostRegistry(
        hosts=[
            TrainingHost(
                id="h",
                display_name="H",
                transport=SubprocessTransport(workdir=tmp_path / "wd"),
                capabilities={},
            )
        ]
    )
    runs = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hosts, run_registry=runs)
    run = Run(
        run_id="deadbeef1234",
        host_id="h",
        recipe_name="hvla_flow_s1",
        dataset_id="ds/one",
        args={"__image__": "lerobot-training:dev-local", "batch_size": 128},
        state=RunState.PENDING,
        created_at=0.0,
    )
    runs.save(run)
    return orch, runs, run


class _Client:
    """Stand-in transport returning whatever the test wants."""

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises

    def image_identity(self, tag):
        if self._raises is not None:
            raise self._raises
        return self._result


def test_identity_is_recorded_and_persisted(orch_and_run):
    """Persisted, not just in memory: the launch path re-loads the run for its
    post-prep race check and would discard an unsaved mutation."""
    orch, runs, run = orch_and_run

    _apply_image_identity(
        run, orch._resolve_image_identity(_Client((CREATED, REVISION)), "lerobot-training:dev-local")
    )
    runs.save(run)
    reloaded = runs.load(run.run_id)

    assert reloaded.args["__image_created__"] == CREATED
    assert reloaded.args["__image_revision__"] == REVISION
    assert reloaded.args["batch_size"] == 128, "unrelated args must be untouched"
    assert reloaded.args["__image__"] == "lerobot-training:dev-local"


def test_a_transport_failure_never_breaks_a_launch(orch_and_run):
    """Failing a real run because a label could not be read would be worse than
    the bug being fixed."""
    orch, runs, run = orch_and_run

    identity = orch._resolve_image_identity(_Client(raises=RuntimeError("ssh died")), "img")

    assert identity["__image_created__"] is None
    assert identity["__image_resolved__"] == "img", "the tag is known without docker"


def test_no_date_or_revision_is_written_when_nothing_is_known(orch_and_run):
    """Absence is the signal: the frontend tells 'not recorded' from 'recorded
    as nothing' only that way. The resolved tag still lands — it comes from the
    launch argv, not from docker."""
    orch, runs, run = orch_and_run

    _apply_image_identity(run, orch._resolve_image_identity(_Client((None, None)), "img"))
    args = run.args

    assert "__image_created__" not in args
    assert "__image_revision__" not in args
    assert args["__image_resolved__"] == "img"


def test_the_resolved_tag_names_a_default_image_run(orch_and_run):
    """'(default)' names nothing, and means something different once the pin
    moves."""
    orch, runs, run = orch_and_run
    run.args.pop("__image__")
    runs.save(run)
    ci_tag = "ghcr.io/thewisp/lerobot-training:feat-gui-training-deploy-proto-e6bf147"

    _apply_image_identity(run, orch._resolve_image_identity(_Client((CREATED, REVISION)), ci_tag))

    assert run.args["__image_resolved__"] == ci_tag


def test_a_date_without_a_revision_is_still_recorded(orch_and_run):
    """The common case for any image built before the label existed."""
    orch, runs, run = orch_and_run

    _apply_image_identity(run, orch._resolve_image_identity(_Client((CREATED, None)), "img"))
    args = run.args

    assert args["__image_created__"] == CREATED
    assert "__image_revision__" not in args


def test_resolving_the_identity_writes_nothing(orch_and_run):
    """The unit-level half of the launch-path race: resolution must be pure
    with respect to storage, so no caller can accidentally persist a stale run.
    The end-to-end consequence is covered by
    ``test_stopping_during_the_image_pull_is_not_undone_by_the_launch_path``,
    which is the test that actually fails if the write comes back."""
    from lerobot.gui.training.runs import RunState

    orch, runs, stale = orch_and_run  # `stale` was loaded before the pull

    stopped = runs.load(stale.run_id)
    stopped.advance(RunState.STOPPED)
    runs.save(stopped)

    def _explode(_run):
        raise AssertionError("resolving the image identity must not persist anything")

    monkeypatch_save = runs.save
    runs.save = _explode
    try:
        identity = orch._resolve_image_identity(_Client((CREATED, REVISION)), "img")
        _apply_image_identity(stale, identity)
    finally:
        runs.save = monkeypatch_save

    assert runs.load(stale.run_id).state is RunState.STOPPED
    assert stale.args["__image_resolved__"] == "img", "the identity is still computed"


def test_stopping_during_the_image_pull_is_not_undone_by_the_launch_path(tmp_path):
    """The severe one, driven through `_prepare_and_launch` rather than its parts.

    `RunRegistry.save` rewrites the whole record. The run object the prep thread
    holds was loaded before the pull, which on a cold cache is minutes long. If
    anything on that path writes it back after the user stops the run, PENDING
    is restored, the race check below sees PENDING and launches — and the state
    later reconciles to STOPPED while a container trains, releasing the
    per-host lock so a second run can start on the same GPU.

    Asserting on the two helpers in isolation does NOT cover this: they pass
    whether or not the caller persists the stale object. A mutation that moves
    the apply back onto `run` is caught here and nowhere else.
    """
    from lerobot.gui.training.hosts import HostRegistry, TrainingHost
    from lerobot.gui.training.orchestrator import Orchestrator
    from lerobot.gui.training.runs import Run, RunPaths, RunRegistry, RunState
    from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

    transport = SubprocessTransport(workdir=tmp_path / "wd")
    host = TrainingHost(id="h", display_name="H", transport=transport, capabilities={})
    runs = RunRegistry(runs_dir=tmp_path / "runs")

    launched: list[str] = []

    class _StopsMidPull(SubprocessClient):
        """Cache hit, then the user presses Stop while we resolve the image."""

        def ensure_prereqs(self, *, sudo_password: str | None = None) -> None:
            return None

        def image_inspect(self, tag: str) -> bool:
            return True

        def image_identity(self, tag: str):
            orch.stop(run.run_id)  # the user, during the pull
            return CREATED, REVISION

        def launch(self, *a, **k):
            launched.append("launched")
            return "session-1"

    client = _StopsMidPull(transport)
    orch = Orchestrator(
        host_registry=HostRegistry(hosts=[host]),
        run_registry=runs,
        make_client_fn=lambda _t: client,
    )
    run = Run(
        run_id="raceable123",
        host_id="h",
        recipe_name="hvla_flow_s1",
        dataset_id="ds/one",
        args={"__image__": "img:tag"},
        state=RunState.PENDING,
        created_at=0.0,
    )
    runs.save(run)
    paths = RunPaths.for_run(run.run_id, runs.runs_dir)
    paths.ensure_exists()
    orch._build_command = lambda _run, _paths: ["docker", "run", "--rm", "img:tag"]

    orch._prepare_and_launch(host, run.run_id, paths)

    assert runs.load(run.run_id).state is RunState.STOPPED, (
        "a run stopped during the pull was resurrected and launched"
    )
    assert launched == [], "no worker may be spawned after the user stopped the run"


def test_resume_does_not_inherit_the_source_runs_identity(orch_and_run):
    """resume() copies the whole args dict. Leaving an unresolved field alone
    would show the previous run's build date and sha as this run's."""
    orch, runs, run = orch_and_run
    run.args.update(
        {
            "__image_resolved__": "old:tag",
            "__image_created__": "2026-01-01T00:00:00Z",
            "__image_revision__": "0" * 12,
        }
    )

    _apply_image_identity(run, orch._resolve_image_identity(_Client((None, None)), "new:tag"))

    assert run.args["__image_resolved__"] == "new:tag"
    assert "__image_created__" not in run.args, "stale date must be cleared, not kept"
    assert "__image_revision__" not in run.args


# ── Wiring ───────────────────────────────────────────────────────────────────


def test_the_orchestrator_records_identity_after_ensuring_the_image():
    """Resolving before the pull returns nothing for any image not already
    cached, so this pins the order, not just the call."""
    src = (Path(__file__).resolve().parents[2] / "src/lerobot/gui/training/orchestrator.py").read_text(
        encoding="utf-8"
    )
    ensure = src.index("self._ensure_image(client, image, remote)")
    resolve = src.index("self._resolve_image_identity(client, image)")
    apply_to_reloaded = src.index("_apply_image_identity(run_after, image_identity)")

    assert ensure < resolve, "identity must be resolved after the image is guaranteed present"
    assert resolve < apply_to_reloaded, "and applied only to the post-race-check run"


def test_the_frontend_uses_the_shared_identity_module():
    """The formatter could be correct and unwired — which is what the bug was."""
    training_js = (_STATIC / "training.js").read_text(encoding="utf-8")
    index_html = (_STATIC / "index.html").read_text(encoding="utf-8")

    assert "TrainingImageIdentity.text(args)" in training_js
    assert "training_image_identity.js" in index_html, "module must be loaded before training.js"
    assert index_html.index("training_image_identity.js") < index_html.index("training.js?v=")


@pytest.mark.asyncio
async def test_the_async_git_helper_reads_head():
    """The build path is a coroutine; the blocking _git would stall the loop."""
    from lerobot.gui.api.training import _git_async

    repo = Path(__file__).resolve().parents[2]
    head = await _git_async(["rev-parse", "HEAD"], repo)

    assert head is not None and len(head) == 40, f"expected a full sha, got {head!r}"
    assert head.isalnum()


@pytest.mark.asyncio
async def test_the_async_git_helper_is_quiet_outside_a_checkout(tmp_path):
    """A pip-installed GUI has no .git; the label is then simply omitted."""
    from lerobot.gui.api.training import _git_async

    assert await _git_async(["rev-parse", "HEAD"], tmp_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("head", "expect_label"),
    [("a" * 40, True), (None, False)],
)
async def test_the_local_image_build_stamps_its_revision(monkeypatch, head, expect_label):
    """Only CI stamped this label, so the reader in _docker_image_inspect found
    nothing for dev-local. Asserts the argv, not the source text: computing
    label_args and forgetting to spread it is the way this actually breaks.
    """
    from lerobot.gui.api import training as api

    captured: list[str] = []

    class _Proc:
        returncode = 0
        stdout = None

        async def wait(self):
            return 0

    async def _fake_exec(*argv, **kwargs):
        captured.extend(argv)
        p = _Proc()
        p.stdout = _AsyncLines([])
        return p

    monkeypatch.setattr(api, "_git_async", lambda *a, **k: _async_value(head))
    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", _fake_exec)
    await api._run_image_build(Path("/repo"))

    assert captured[:2] == ["docker", "build"]
    if expect_label:
        assert "--label" in captured
        assert f"org.opencontainers.image.revision={head}" in captured
    else:
        assert "--label" not in captured, "no HEAD means no label rather than an empty one"


async def _async_value(v):
    return v


class _AsyncLines:
    """Minimal async-iterable stand-in for proc.stdout."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


# ── Rendering (JS) ───────────────────────────────────────────────────────────


def test_no_test_name_is_exactly_forty_characters():
    """TruffleHog's Lob detector matches ``test_`` + 35 chars, so a 40-character
    test name is reported as a verified secret and fails the branch's CI. Two
    names in this file hit it; this stops the next one silently doing so."""
    import re as _re

    names = _re.findall(r"^def (test_\w+)", Path(__file__).read_text(encoding="utf-8"), _re.M)

    assert names, "no tests found — the pattern is wrong"
    assert [n for n in names if len(n) == 40] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_rendering_js():
    test_js = Path(__file__).parent / "training_image_identity.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stdout + result.stderr
