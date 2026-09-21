# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Few-shot-after-examine manipulation: placement transfer without object pose.

PROTOTYPE. One teleop demo of a manipulation transfers to new object placements
using only SAM3 masks, dense ViT features, proprioception, and a self-calibrated
table homography — no mesh, no pose estimation, no simulator. See README.md in
this package for the system design, what is proven, and what still needs the rig.
"""

from lerobot.fewshot.events import detect_interaction
from lerobot.fewshot.planar import PlanarDemo, apply_homography, fit_homography
from lerobot.fewshot.registration import RegistrationResult, Sim2, fit_similarity, ransac_register
from lerobot.fewshot.templates import BankMatch, Template, TemplateBank

__all__ = [
    "BankMatch",
    "PlanarDemo",
    "RegistrationResult",
    "Sim2",
    "Template",
    "TemplateBank",
    "apply_homography",
    "detect_interaction",
    "fit_homography",
    "fit_similarity",
    "ransac_register",
]
