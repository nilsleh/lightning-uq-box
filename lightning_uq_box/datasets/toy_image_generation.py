# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Toy Image Generation Dataset."""

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ToyImageGenerationDataset(Dataset):
    """Toy Image Generation Dataset."""

    def __init__(
        self, num_samples: int = 10, in_channels: int = 5, cond_channels: int = 1
    ) -> None:
        """Initialize a new instance of Toy Image Generation Dataset."""
        super().__init__()

        self.num_samples = num_samples
        self.num_input_channels = in_channels
        self.num_cond_channels = cond_channels
        self.images = [
            torch.randn(in_channels, 64, 64) for val in range(self.num_samples)
        ]
        self.cond_images = [
            torch.randn(cond_channels, 64, 64) for val in range(self.num_samples)
        ]

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> Tensor:
        """Retrieve single sample from the dataset.

        Args:
            index: index value to index dataset
        """
        return {
            "input": self.images[index],
            "condition": self.cond_images[index],
            "index": index,
            "aux": "random_aux_data",
        }
