# S1 action policy for Hierarchical VLA.
#
# Currently uses ACTWithVLM from policies/act_vlm/.
# Swappable — can be replaced with diffusion/flow-matching policy later.
# The only contract is: S1 accepts "observation.s2_latent" [B, 2048]
# and "observation.s2_latent_age" [B, 1] in the input batch.

from lerobot.policies.act_vlm.modeling_act_vlm import ACTWithVLMPolicy, S2_LATENT_KEY, S2_AGE_KEY
from lerobot.policies.act_vlm.configuration_act_vlm import ACTWithVLMConfig

__all__ = ["ACTWithVLMPolicy", "ACTWithVLMConfig", "S2_LATENT_KEY", "S2_AGE_KEY"]
