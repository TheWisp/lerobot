# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One unimportable policy package must not empty the policy catalog.

`_ensure_policy_configs_loaded` wraps each module import in
`contextlib.suppress`, which reads as covering this. It does not:
`pkgutil.walk_packages` imports each *package* itself, to read its `__path__`,
and that import happens inside the walk rather than in the loop body. A package
that raises on import therefore escapes the suppress entirely.

Observed on a host whose transformers rejects `wall_x`'s vendored Qwen config:
`/api/training/policies` returned 500 and the training form's policy selector
rendered blank, with every other policy importable. `onerror` is what contains
it, so these tests pin the mechanism rather than the one package that exposed it
-- the next incompatibility will be a different package.
"""

import contextlib
import importlib
import pkgutil
import sys
import types

import pytest


@pytest.fixture
def broken_package_tree(tmp_path):
    """A package with one importable child and one that raises on import."""
    root = tmp_path / "fakepolicies"
    (root / "good").mkdir(parents=True)
    (root / "bad").mkdir()
    (root / "__init__.py").write_text("")
    (root / "good" / "__init__.py").write_text("")
    (root / "bad" / "__init__.py").write_text("raise RuntimeError('unimportable policy')\n")
    pkg = types.ModuleType("fakepolicies")
    pkg.__path__ = [str(root)]
    sys.modules["fakepolicies"] = pkg
    yield pkg
    for name in [n for n in sys.modules if n.startswith("fakepolicies")]:
        del sys.modules[name]


def _walk(pkg, **kwargs):
    seen = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(pkg.__path__, prefix="fakepolicies.", **kwargs):
        with contextlib.suppress(Exception):
            importlib.import_module(modname)
        seen.append(modname)
    return seen


def test_suppress_alone_does_not_contain_a_broken_package(broken_package_tree):
    """The failure mode, so the fix below is not asserting into a vacuum."""
    with pytest.raises(RuntimeError, match="unimportable policy"):
        _walk(broken_package_tree)


def test_onerror_keeps_the_remaining_packages_discoverable(broken_package_tree):
    seen = _walk(broken_package_tree, onerror=lambda _name: None)
    assert "fakepolicies.good" in seen, "a healthy package must still be enumerated"


def test_the_gui_walkers_all_pass_onerror():
    """Whichever catalog loses its walk, the symptom is an empty picker."""
    import inspect

    from lerobot.gui.api import robot, run, training

    for module in (training, robot, run):
        source = inspect.getsource(module)
        for call in source.split("pkgutil.walk_packages(")[1:]:
            head = call[: call.index(")")]
            assert "onerror" in head, f"{module.__name__} walks without onerror"
