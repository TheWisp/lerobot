# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Where the GUI keeps state the user would notice losing.

Configured dataset sources, which datasets to reopen on next launch, saved
robot and teleop profiles: decisions the user made, not a cache. Losing any of
them is work lost, which is why they live under ``~/.config`` rather than
``~/.cache``.

One overridable base, because the GUI also runs as a subprocess — in tests and
in the e2e flows that launch it for real. A subprocess re-imports these modules
and cannot see a monkeypatched constant, so an environment variable is the only
channel that reaches it. Without that, those runs write the developer's actual
config: it has left the GUI opening with a "Failed to open dataset" toast
pointing at a deleted pytest directory, and creating profile directories on a
machine that had none.

Resolved at import so a subprocess picks up the variable it was launched with.
In-process callers that need to redirect should patch the module constants
built from this, which is what the test fixtures do.
"""

from __future__ import annotations

import os
from pathlib import Path

GUI_CONFIG_DIR_ENV = "LEROBOT_GUI_CONFIG_DIR"


def gui_config_dir() -> Path:
    """Base directory for the GUI's own config. Env var wins."""
    return Path(os.environ.get(GUI_CONFIG_DIR_ENV) or Path.home() / ".config" / "lerobot")
