# ruff: noqa
"""Inspect aloha_sim_insertion_scripted (matches our insertion sim) for the reach-to-object
build: episodes, frames/ep, keys, image/state/action shapes."""

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/aloha_sim_insertion_scripted")
print("num_episodes:", ds.num_episodes, "| total frames:", len(ds), flush=True)
it = ds[0]
print("sample keys:", [k for k in it.keys()], flush=True)
for k, v in it.items():
    try:
        print(f"  {k:32s} {tuple(v.shape)} {v.dtype}")
    except Exception:
        print(f"  {k:32s} {type(v)} {v}")
ei = np.array(ds.hf_dataset["episode_index"])
lens = [int((ei == e).sum()) for e in range(min(ds.num_episodes, 5))]
print("first 5 episode lengths:", lens, flush=True)
