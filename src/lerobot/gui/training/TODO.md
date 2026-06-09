# Training — TODO

Tracked follow-ups for the training subsystem. Colocated with [`DESIGN.md`](DESIGN.md) so the rationale and the open work live next to each other.

## Recently fixed

- **Form-pane scroll** — picking SmolVLA (38 scalar fields) made the form taller than the viewport with no scrollbar; the bottom of the form (Start / Cancel buttons) was clipped. Root cause: `#training-detail` inherited the parent `.tab-content { overflow: hidden }` and didn't add its own `overflow-y: auto`. Sibling `.model-detail` did this correctly; `#training-detail` was the new container I added and missed the pattern. Fixed by mirroring `.model-detail`'s rule.
- **Duplicate Cancel button on the form** — `Start a training run` had a Cancel in the header AND a Cancel in the footer next to Start. The header one wasn't aligned to the form (it was in the title row) and was redundant. Removed the header copy; kept the footer one next to Start (standard form-action position).
- **DataLoader shared-memory OOM in container** — every camera-using policy crashed in worker process 0 with `RuntimeError: unable to allocate shared memory(shm) ... Resource temporarily unavailable (11)` partway through the first batch. Root cause: Docker's default `/dev/shm` is 64 MiB; PyTorch DataLoader workers use `/dev/shm` to ship tensors to the main process. SmolVLA at batch=16 with the user's 4-camera dataset (512² uint8) = ~200 MB per batch, instantly over the limit. ACT/Diffusion smokes presumably stayed under 64 MiB by accident (smaller batches or fewer cameras). Fixed by adding `--shm-size=8g` to the docker run argv in `recipes.py` (both `_build_docker_command` and `_build_hvla_flow_s1_command`). 8 GiB is conservative for typical ML batches; an alternative is `--ipc=host` (uses host's IPC namespace) but that breaks isolation on shared hosts.

## Footguns

- **Docker group not picked up by existing shell sessions** — adding the user to the `docker` group with `usermod -aG docker $USER` updates `/etc/group` but does NOT backfill into running sessions (groups are cached at login). Every existing shell needs either `sg docker -c "..."` wrapping for `docker` commands or `newgrp docker` to refresh in-place. The orchestrator's image-prep path (`docker image inspect`, `docker pull`, `docker run`) needs daemon socket access and hits this immediately. Workaround: start the GUI as `sg docker -c "nohup python -m lerobot.gui ... &"` (the `nohup` is required because `sg`'s subshell exits and would otherwise SIGHUP the GUI). Real fix: user logs out + back in once, then `groups` shows `docker` and the wrapper is no longer needed. Documenting only — not worth automating around.

- **`api/run.py` resolves `lerobot-record` via PATH, not absolute path** — when the GUI server is started without activating the conda env (e.g. `sg docker -c "python -m lerobot.gui ..."` with explicit Python path), the subprocess inherits the parent shell's PATH. If that PATH has `/home/feit/miniforge3/bin` (base env) before `/home/feit/miniforge3/envs/lerobot/bin`, then `lerobot-record` resolves to the BASE env's binary — which on this workstation points at a different lerobot checkout and is missing the `feetech` extra. Result: ImportError on `scservo_sdk` after the policy loads cleanly. Workaround: prepend `PATH=/home/feit/miniforge3/envs/lerobot/bin:$PATH` on GUI startup. Real fix: change every `args = ["lerobot-record", ...]` (and the analogous `lerobot-eval`, `lerobot-replay` spawns) in `gui/api/run.py` to `args = [sys.executable, "-m", "lerobot.scripts.lerobot_record", ...]`. Then the subprocess uses the SAME Python as the GUI server — no PATH dependency. ~5 line change per spawn site; should cover all GUI-launched lerobot CLI invocations.

- **`wall_x` codebase dir vs `wallx` extra name (cosmetic; deferred rename)** — historical naming mismatch. `tests/policies/test_registry_symmetry.py` papers over it via `DIR_TO_EXTRA_OVERRIDES`. The clean fix is renaming the extra `wallx` → `wall_x` with a back-compat alias (`wallx = ["lerobot[wall_x]"]`). Touches pyproject.toml, `policies-all`, and any user install docs that say `pip install lerobot[wallx]`. Defer until the next pyproject sweep.

- **Surface policy health warnings in the GUI itself, not just the log** — `gui/server.py` startup now runs `log_policy_registry_health` in a background thread and writes warnings to the log. The next layer of UX: `gui/api/training.py:list_policies` (a) records `_policy_load_failures` from `_ensure_policy_configs_loaded` instead of silently dropping them, (b) merges that with `check_policy_registry_health()` results, (c) returns a `warnings: list[{policy, missing_module, install_hint}]` field on the policies endpoint, and (d) frontend renders an unobtrusive banner above the policy picker: "N policies hidden — missing deps. [Show details]". Bigger UX change than the current commit; structural pieces (probe + install hint) are already in place.

- **Dynamic CI test under `--extra all`** — extend `tests/policies/` with a parametrized test that asserts `get_policy_class(name)` succeeds for every `PreTrainedConfig.get_known_choices()` entry. Gate to `full_tests.yml` / `latest_deps_tests.yml` (which already do `uv sync --extra all`). The current static `test_registry_symmetry.py` catches "policy exists without an extra"; the dynamic version catches "extra exists but doesn't actually list the right deps". Both run, defense in depth.

- **Update the user's `uv sync` curated extras list** — the conda-env-setup memory at `/home/feit/.claude/projects/-home-feit-Documents-lerobot/memory/MEMORY.md` lists the curated `--extra` set. As of this commit the recommended invocation is:

  ```bash
  uv sync --locked \
    --extra policies-all \
    --extra dataset --extra training --extra viz \
    --extra hardware --extra feetech --extra dynamixel --extra gamepad \
    --extra kinematics --extra intelrealsense \
    --extra gui --extra dev --extra test \
    --python /home/feit/miniforge3/envs/lerobot/bin/python --active
  ```

  Key change: `--extra policies-all` replaces the ad-hoc enumeration of `hvla` + `hilserl` (and the implicitly-missing `smolvla`/`pi`/`xvla`/etc.). The host env now provably has every policy importable, with the same one-liner that the Docker training image uses.

- **Dockerfile edits invisible until CI rebuilds + DEFAULT_IMAGE bumps** — the orchestrator's `_ensure_image` is pull-only; it does `docker image inspect` then `docker pull`, never `docker build`. So a `docker/Dockerfile.training` edit (e.g. adding `--extra smolvla` to fix the num2words crash) has zero effect on runs until: (1) the branch is pushed, (2) CI builds the new tag (~9-17 min), and (3) `DEFAULT_IMAGE` in `recipes.py` is manually bumped to the new short-sha. THREE manual steps on a Dockerfile one-liner. Hot-fix retag is the get-unstuck workaround: `printf 'FROM <current-image>\nRUN cd /lerobot && uv pip install <missing-dep>==<version>\n' | sg docker -c 'docker build -t <current-image> -'` — builds a thin layer on top of the cached image and reuses the same tag. ~10 seconds. Throwaway state (only on the workstation; overwritten next CI rebuild). Permanent fix is the env-vs-source split described in [`DESIGN.md`](DESIGN.md) § Image tag derivation + live source mount — once landed, dep changes still need a rebuild but the orchestrator detects the mismatch and either builds locally (workstation) or says "push branch" (ephemeral).

## UX

- **Stats card poll latency on big checkpoints** — `_sync_checkpoints_manifest` hashes new checkpoint files synchronously inside the poll() request thread. A 1 GB Diffusion checkpoint takes ~3.5 s at ~300 MB/s, freezing the run detail pane until the manifest is appended. Fix: defer sha256 to a background worker; the manifest entry appears one poll cycle later but the request returns instantly. Surfaced during the C5 prototyping audit.
- **Form re-open race after cancel** — `trainingShowStartForm` works reliably on first open but Playwright tests showed it flakily fails to render the policy-fields container on the second open after a cancel. Workaround in the screenshot driver was to dispatch via `evaluate()`; the real fix is whatever's making the second render time out. Surfaced while regenerating screenshots.
- **`gui_common` field metadata** — the dynamic policy catalog surfaces every scalar field (ACT 23, Diffusion 33, SmolVLA 38). Render time is fine; reader load is not. Support `dataclasses.field(metadata={"gui_common": True})` on config classes so config authors mark which fields surface by default vs. behind an "Advanced" collapse. Lives in the dataclass (where the author already is), not in the GUI codebase.
- **Surface run dir path in the detail view** — users looking for "where did my model go?" check `~/Documents/lerobot/outputs/` (the upstream `lerobot-train` default) and find nothing. The GUI's runs root is `$LEROBOT_RUNS_DIR` (defaults to `~/.cache/lerobot/runs/`). Add a one-line path string to the run detail view + a paragraph in the README.
- **Live-source-mount dev mode** — when iterating on `lerobot/` code, the full edit → CI rebuild → GHCR pull loop is ~17–27 min. A `__mount_src__=true` knob in the recipe builder would bind-mount the local `src/lerobot/` into the container, skipping the rebuild for pure-Python changes. Dev-only; never the default.

- **Per-checkpoint picker on the Run tab** — today the policy-checkpoint dropdown (`#run-policy-checkpoint`) and debug-model dropdown (`#run-teleop-debug-model`) expose **only one option per run** — the latest (`is_last`) checkpoint, via `default_policy_path`. To support workflows like "did training regress at step 45k? test step 40k instead" or "eval each checkpoint, pick the best", surface every checkpoint in the picker.
  - Server side: already shipped. `GET /api/models/sources/{path}/models` returns each run with a full `checkpoints[]` list; each entry has `policy_path`, `step`, `is_last`, `has_training_state`, `model_size_bytes`, `num_parameters`. Same for `GET /api/models/run/{path}/checkpoints`. No new endpoint needed.
  - UI shape (pick one):
    - Option A — nested `<optgroup>` per run inside the existing `<select>`: run name = optgroup label, one option per checkpoint with step + size + is_last marker.
    - Option B — two-level picker: run dropdown + checkpoint dropdown that appears after pick. Cleaner for many runs / many checkpoints, costs one extra click for the common case.
  - Default selection: the `is_last` checkpoint (behavior identical to today).
  - Each option's `value` must remain the server-emitted `c.policy_path` (never client-concatenated — that's the bug this whole layer of work fixed).
  - Each option must carry `data-policy-type="${m.policy_type}"` AND `data-run-path="${m.path}"` so `_onPolicyCheckpointChange` / `_prefillPolicyFields` continue to work without regex-stripping anything.
  - HVLA policies: confirm `_prefillPolicyFields` still resolves the right run when a non-last checkpoint is picked (it keys off run path, not checkpoint path — should be unaffected, but cover with a manual test).
  - Flat checkpoints (converted S2 VLM): `num_checkpoints == 1`, no sub-picker needed — keep collapsed as a single option in the optgroup.
  - RLT entries (`default_policy_path == null`): already filtered out of this picker (`_modelCheckpointOptions` skips entries with no `default_policy_path`) and surfaced separately in the HVLA RLT field. The new picker must continue to skip them.
  - Tests to add: extend `tests/hvla/test_checkpoint_format.py::TestGUIScanner` with a multi-checkpoint run fixture asserting every entry has a usable `policy_path` and exactly one has `is_last=True`.
  - Out of scope until landed: per-checkpoint "delete" / "compare two checkpoints" / step-range filter. Track separately when surfaced.

- **`_dir_has_step_subdirs` rejects `checkpoint-<N>` naming (pre-existing)** — HVLA's `train.py` writes `checkpoints/checkpoint-<N>/...` but the GUI scanner's `_dir_has_step_subdirs` only accepts names where `lstrip("0").isdigit()` is True, which excludes the `checkpoint-` prefix. Result: HVLA runs that use the upstream naming are invisible in the Model tab. `tests/hvla/test_checkpoint_format.py::test_scanner_finds_hvla` has been failing on `main` for at least the current branch's lifetime. Fix is one extra regex match in `_dir_has_step_subdirs`. Tracked but not blocking the prototype.

## Build / deployment

- **BuildKit cache mount for `uv sync`** — the training Dockerfile uses `RUN uv sync --no-cache` which re-downloads PyPI dependencies on every dep-dirty CI build (~9 min). Adding `RUN --mount=type=cache,target=/root/.cache/uv uv sync --extras...` would persist the uv cache between CI runs and drop dep-dirty builds from 9–13 min to 4–6 min. Pull time unchanged. Already supported by `docker/build-push-action@v6` cache infra.

## Observability

- **Real progress parsing** — `progress.json` is only written by the legacy fake-runner. Real `lerobot-train` and HVLA `flow_matching` don't write it, so the detail-view Loss field shows `—` for real runs (Step is derived from the latest checkpoint as a fallback). Parse `step=N loss=X` from the stderr tail and write to `progress.json` from the orchestrator side — same poll cycle that picks up checkpoints.

## Next phases

- **SSH transport** (`SshClient` implementation) — `TransportClient` Protocol expansion already landed (`f969e7602`). Implement each method as `subprocess.run(['ssh', user@host, ...])` / `scp` / `tmux`. Test loopback first (`ssh $USER@127.0.0.1`); then a leased server. See [`DESIGN.md`](DESIGN.md) § Host setup UX for the "Add SSH host" dialog spec.
- **Ephemeral provider (Nebius)** — vendor SDK as a LeRobot optional extra, paste-token auth in v1, auto-destroy + scheduled-delete backstop, cost-ceiling enforcement. Lands on top of the SSH transport (Ephemeral pods use SshClient once spawned). See [`DESIGN.md`](DESIGN.md) § Cost discipline + § Authentication.
- **`_drop_run_metadata` via transport** — `delete_run` / `clear_terminal_runs` currently use `pathlib` + `shutil.rmtree` directly. Works for workstation (bind-mount = same disk). For SSH hosts the rmtree needs to route via `client.rmtree(path)` — add to the Protocol and update the helper. Deferred from the abstraction PR (`f969e7602`) because it doesn't gate SSH transport work.
- **Training MCP tools** — read-only first (`list_hosts`, `list_runs`, `get_run_snapshot`) so AI agents can drive testing without needing browser automation. Write tools (`start_run`, `stop_run`, `delete_run`) come after the read surface stabilises.
