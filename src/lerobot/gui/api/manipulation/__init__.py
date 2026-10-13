#!/usr/bin/env python

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

"""Key-frame manipulation API.

The historical HTTP prefix remains compatible with workers and saved tooling.
Python code imports the owning module explicitly; there is no legacy runner or
forwarding facade. Import all endpoint modules before assembling the router.
"""

from fastapi import APIRouter

from . import (
    acquisition,
    camera,
    demos,
    execution,
    groups,
    holds,
    planning,
    recording,
    rendering,
    replay,
    runtime,
    session,
    tracking,
    worker,
)

router = APIRouter(prefix="/api/pregrasp", tags=["manipulation"])
for module in (
    session,
    camera,
    acquisition,
    tracking,
    execution,
    recording,
    replay,
    rendering,
    demos,
    groups,
    worker,
):
    router.include_router(module.router)

__all__ = [
    "router",
    "runtime",
    "session",
    "camera",
    "acquisition",
    "worker",
    "tracking",
    "holds",
    "planning",
    "execution",
    "demos",
    "recording",
    "replay",
    "rendering",
    "groups",
]
