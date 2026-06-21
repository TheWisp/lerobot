# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Do-as-I-Do MVP: reconstruct a tabletop hand demo from the top RealSense
(metric depth + calibrated extrinsic), then retarget it to the SO-107.

Experimental prototype on branch ``proto/gui-debug-vision``. The reconstruction
half is pure perception (no motor motion): SAM3 object mask + metric depth +
RANSAC table plane + a hand/wrist estimate, all lifted into the robot base frame
via the existing ``click_target_extrinsics.json`` transform.
"""
