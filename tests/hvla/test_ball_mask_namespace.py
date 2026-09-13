"""The HVLA wrapper must consume the same mask namespace as the dataset reader."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset


class Columns(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return {name: values[key] for name, values in self.items()}
        return super().__getitem__(key)

    @property
    def column_names(self):
        return list(self)


class TinyDataset:
    def __init__(self):
        self.hf_dataset = Columns(
            {
                "action": [[0.0], [1.0], [2.0]],
                "observation.state": [[0.0], [1.0], [2.0]],
                "episode_index": [0, 0, 0],
                "masks.top_l": ["[]", "[]", "[]"],
            }
        )
        self.meta = SimpleNamespace(
            features={
                "masks.top_l": {"mask_labels": ["yellow ball"], "mask_size": [4, 4]},
            }
        )

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "action": torch.tensor(self.hf_dataset["action"][index]),
            "observation.state": torch.tensor(self.hf_dataset["observation.state"][index]),
            "observation.images.top_l": torch.ones(3, 4, 4),
            "masks.top_l": "[]",
            "index": torch.tensor(index),
        }


@pytest.mark.parametrize("mode", ["ball_view", "ball_token", "ball_aux"])
def test_hvla_accepts_current_namespace_and_reads_the_saved_mask(mode):
    images = ["observation.images.top_l"]
    if mode == "ball_view":
        images.append("observation.images.ball_view")
    dataset = FlowMatchingDataset(
        TinyDataset(),
        s2_latents=None,
        chunk_size=2,
        image_keys=images,
        resize_to=(4, 4),
        state_position_std_floor=0.0,
        ball_source="observation.images.top_l",
        **{mode: True},
    )
    sample = dataset[0]
    assert dataset._ball_mask_key == "masks.top_l"
    if mode == "ball_view":
        assert torch.count_nonzero(sample["observation.images.ball_view"]) == 0
    else:
        assert sample["observation.ball"].tolist() == [-1.0, -1.0, 0.0]
