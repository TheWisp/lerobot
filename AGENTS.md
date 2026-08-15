This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Commit Messages

Commit messages are durable review and handoff documentation, not just labels for
the diff. For every non-trivial change, use an imperative summary followed by a
body that lets a future reviewer understand the decision without reconstructing
the conversation or reverse-engineering the patch. The body must explain:

- the concrete problem or failure mode that motivated the change;
- the behavior or invariant chosen, and why;
- important compatibility implications, tradeoffs, and risks; and
- the verification performed, when it adds useful evidence.

Do not merely restate which files or symbols changed. A subject-only message is
acceptable only for genuinely mechanical, self-explanatory changes.

## Follow-ups and known problems

**GitHub Issues is the single backlog** — bugs and improvements alike,
distinguished by label. Anything with a completion state goes there and is
closed by the PR that fixes it. Use `gh issue list` / `gh issue create`; the
issues are as much a part of this repository as the code.

Do **not** add work to the `TODO.md` files — a markdown checklist cannot close,
assign, or link to the PR that resolves it, which is how one accumulated 209
entries with 22 already shipped. They are being migrated and are meant to
drain: when you pick an entry up, open the issue, **delete the entry**, and
reference the issue number in the commit. Leaving it behind creates two records
of the same work. Don't convert the backlog in bulk; check an entry is still
true first.

### Write the problem, not the plan

An issue's job is to make a problem findable and understood later. State what
goes wrong, what it costs, and the evidence — the file, the measurement, the
observed behaviour. Point at a direction or two if there is a real fork worth
recording, and stop there.

Do **not** ship a task list. A checklist of steps is a design review nobody
attended, written before the work was scheduled, against a codebase that will
have moved by then. Whoever picks it up will know more than you do now; a stale
plan either misleads them or gets silently ignored, and both outcomes make the
issue less trustworthy than if it had only described the problem.

The test: if the issue would still be correct after someone refactors the module
it is about, it is describing a problem. If it would be wrong, it is describing
a solution.

### Label impact and effort at creation

Every issue gets one `impact:` (high/med/low) and one `effort:` (S/M/L) label
when it is opened — not later, in a triage pass that never happens. Priority is
not a third label; it is what falls out of the pair, which is why the two are
kept orthogonal:

```bash
gh issue list --label "impact:high" --label "effort:S"    # what to do next
```

Effort is a guess and should be re-checked when the issue is picked up, not
trusted months on. Add `discussion` when the issue is blocked on a decision
rather than on effort, so it stays out of that query.

What does belong in the repository is documentation rather than work — design
decisions, invariants, and why an approach was rejected. These have no
completion state and must sit next to the code they describe.

When a document describes behaviour that does not exist, annotate it **at the
claim** (`**NOT IMPLEMENTED.** …`) and link the issue. Correcting it silently
removes the evidence of how it survived; a design doc that reads as shipped is
how several defects went unnoticed for months.

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
