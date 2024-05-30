# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Toy Image Generation Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToyImageGenerationDataset


class ToyImageGenerationDatamodule(LightningDataModule):
    """Toy Image Generation Datamodule for Testing."""

    def __init__(self, batch_size: int = 10, **kwargs) -> None:
        """Initialize a new instance of Toy Image Generation Datamodule.

        Args:
            batch_size: batch size
            kwargs: additional arguments for dataset class
        """
        super().__init__()
        self.batch_size = batch_size
        self.kwargs = kwargs

    def train_dataloader(self):
        """Return Train Dataloader."""
        return DataLoader(
            ToyImageGenerationDataset(**self.kwargs), batch_size=self.batch_size
        )

    def val_dataloader(self):
        """Return Val Dataloader."""
        return DataLoader(
            ToyImageGenerationDataset(**self.kwargs), batch_size=self.batch_size
        )

    def calib_dataloader(self):
        """Return Calib Dataloader."""
        return DataLoader(
            ToyImageGenerationDataset(**self.kwargs), batch_size=self.batch_size
        )

    def test_dataloader(self):
        """Return Test Dataloader."""
        return DataLoader(
            ToyImageGenerationDataset(**self.kwargs), batch_size=self.batch_size
        )
