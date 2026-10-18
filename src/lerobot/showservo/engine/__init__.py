# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The act engine: a demo replayed as one continuous motion, gated by Milestones checked from the camera.

Designed in ``src/lerobot/showservo/docs/act_execution.md``. It holds no GUI state: it reads an arm (kinematics and
limits), a playback (the motor step's queue) and a scene log that perception and the joint readings write into, so the
GUI's adapter and the tests drive it the same way.
"""
