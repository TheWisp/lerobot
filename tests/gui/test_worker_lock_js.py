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
"""The worker-count box is frozen when the GPU image pipeline is chosen.

See worker_lock.test.js for the assertions. This runs them, and pins that the
form actually calls the binder -- a correct predicate that nothing invokes would
leave the box live and the run quietly ignoring it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_worker_lock_decision_js():
    test_js = Path(__file__).parent / "worker_lock.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_policy_form_binds_the_worker_lock():
    """The binder must be wired where the fields are rendered, not merely defined."""
    source = (Path(__file__).parents[2] / "src" / "lerobot" / "gui" / "static" / "training.js").read_text()
    assert "function trainingBindWorkerLock(" in source, "the binder must exist"
    rendered = source[source.index("container.innerHTML = primaryHtml") :]
    rendered = rendered[: rendered.index("\n}")]
    assert "trainingBindWorkerLock(container)" in rendered, (
        "the policy form must call the binder after rendering its fields"
    )


def test_no_policy_declares_the_shared_training_fields_itself():
    """One definition, not one per policy.

    These are TrainPipelineConfig fields. While they lived in the HVLA schema,
    selecting any other policy offered neither -- the option was supported by the
    trainer and invisible in the form. A second definition would bring that back
    for whichever policy carried it.
    """
    import asyncio
    import inspect

    from lerobot.gui.api.training import list_policies

    result = list_policies()
    if inspect.iscoroutine(result):
        result = asyncio.new_event_loop().run_until_complete(result)
    policies = result if isinstance(result, list) else result.get("policies", result)
    assert policies, "the catalog must not be empty"
    for policy in policies:
        names = {f["name"] for f in policy.get("fields", [])}
        offending = names & {"data_path", "num_workers"}
        assert not offending, (
            f"{policy.get('type_name')} declares {offending} in its own schema; "
            "they belong in the shared TRAINING_FIELDS"
        )
