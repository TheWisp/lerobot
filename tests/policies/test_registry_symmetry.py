"""Structural symmetry between the policy codebase and pyproject.toml extras.

Catches the class of bug that has bitten this branch three times already:
a new policy lands under ``src/lerobot/policies/<name>/``, gets registered
with ``@PreTrainedConfig.register_subclass("<name>")``, and (silently) has
no matching ``<name>`` extra in pyproject.toml — so no single
``uv sync --extra ...`` invocation can install its deps. Some environment
(usually the training Docker image or the host conda env) ends up missing
the transitive package, and the user discovers it at runtime when the
policy crashes mid-init.

Two asserts, both static (no torch/transformers/etc imports — runs in CI
tier-1 in milliseconds):

- :func:`test_every_policy_dir_has_an_extra` — every codebase subdir must
  map to a pyproject extra (empty extra is fine; serves as the canonical
  install handle).
- :func:`test_policies_all_covers_every_policy_extra` — the
  ``policies-all`` meta-extra must reference every policy's extra (so
  ``--extra policies-all`` installs every policy's runtime deps in one
  command).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES_DIR = REPO_ROOT / "src" / "lerobot" / "policies"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Codebase dir → pyproject extra name. Most are identical; the exceptions
# are documented here. New mismatches MUST be added here AND ideally fixed
# (rename the extra to match the dir) in a follow-up.
DIR_TO_EXTRA_OVERRIDES: dict[str, str] = {
    # codebase: wall_x/ → extra: wallx (historical name mismatch; renaming
    # the extra would break downstream `pip install lerobot[wallx]` users)
    "wall_x": "wallx",
    # Pi-family shares one extra — pi0, pi05, pi0_fast (and pi_gemma.py at
    # the file level) all use the same transformers + scipy deps. The
    # single `pi` extra covers them; no separate `pi0`/`pi05`/`pi0_fast`
    # extras are warranted.
    "pi0": "pi",
    "pi05": "pi",
    "pi0_fast": "pi",
}

# Codebase subdirs that are explicitly NOT standalone policies (helpers,
# shared backbones, etc.) and therefore don't need their own extra.
NON_POLICY_DIRS: set[str] = {
    "__pycache__",
    # add here if a future subdir is a shared module rather than a policy
}

# Policy extras intentionally excluded from ``policies-all`` — usually
# because they need special install instructions (flash-attn, vendor SDKs).
# Listed here so the second test doesn't fail on them. New exclusions must
# be justified in a comment AND mentioned in docker/Dockerfile.training.
POLICIES_ALL_EXCLUSIONS: set[str] = {
    "groot",  # needs flash-attn with platform-specific install
}


def _policy_dirs() -> set[str]:
    """Codebase policy subdirs — anything with an __init__.py that we
    treat as a policy module. Excludes helpers in :data:`NON_POLICY_DIRS`
    and any non-policy single-file modules at the top of policies/."""
    return {
        p.name
        for p in POLICIES_DIR.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and (p / "__init__.py").exists()
        and p.name not in NON_POLICY_DIRS
    }


def _load_extras() -> dict[str, list[str]]:
    return tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]


def test_every_policy_dir_has_an_extra() -> None:
    """Every ``src/lerobot/policies/<name>/`` subdir must map to a declared
    extra in pyproject.toml. An empty extra (``foo = []``) is valid and
    encouraged — it serves as the canonical install handle so callers can
    write ``pip install 'lerobot[foo]'`` consistently whether or not foo
    has deps today. If foo gains a dep tomorrow, every install site
    already references the right name."""
    extras = _load_extras()
    declared = set(extras.keys())
    expected = {DIR_TO_EXTRA_OVERRIDES.get(d, d) for d in _policy_dirs()}
    missing = expected - declared
    assert not missing, (
        f"Policies without a matching extra in pyproject.toml: {sorted(missing)}.\n"
        f"Fix: add `<name> = [...]` to pyproject.toml's "
        f"[project.optional-dependencies] (empty list is fine if base deps suffice) "
        f"AND include `lerobot[<name>]` in `policies-all`."
    )


def test_policies_all_covers_every_policy_extra() -> None:
    """The ``policies-all`` meta-extra must reference every policy's extra
    so a single ``uv sync --extra policies-all`` installs every policy's
    runtime deps. Without this, environments (training Docker image, host
    conda env, ephemeral cloud pods) inevitably drift on which policies
    they can actually load."""
    extras = _load_extras()
    assert "policies-all" in extras, (
        "pyproject.toml must declare a `policies-all` meta-extra that lists "
        "every policy's individual extra. See the comment above its definition."
    )

    policies_all_refs = {
        ref.removeprefix("lerobot[").removesuffix("]")
        for ref in extras["policies-all"]
        if ref.startswith("lerobot[") and ref.endswith("]")
    }
    expected = {DIR_TO_EXTRA_OVERRIDES.get(d, d) for d in _policy_dirs()} - POLICIES_ALL_EXCLUSIONS
    missing = expected - policies_all_refs
    assert not missing, (
        f"`policies-all` is missing these policy extras: {sorted(missing)}.\n"
        f"Fix: add `lerobot[<name>]` to policies-all in pyproject.toml. "
        f"If the omission is deliberate (e.g. needs special install), add "
        f"to POLICIES_ALL_EXCLUSIONS in this test with a justifying comment "
        f"AND a corresponding note in docker/Dockerfile.training."
    )
