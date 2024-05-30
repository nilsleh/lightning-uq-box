# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Lightning Modules to train Diffusers Models."""

from typing import Any

import torch
from diffusers.models import ModelMixin
from diffusers.schedulers import SchedulerMixin
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import BaseModule


class UnconditionalImageDiffusionModel(BaseModule):
    """Unconditional Image Generation Model."""

    def __init__(
        self,
        net: ModelMixin,
        noise_scheduler: SchedulerMixin,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        """Initialize a new instance of the Unconditional Image Diffusion Model.

        Args:
            net: The neural network to use for training, Unet or Transformer
                architecture from diffusers, for an example of adapting
                a model, see the `diffuser docs <https://huggingface.co/docs/diffusers/training/adapt_a_model>`_.
            noise_scheduler: The noise scheduler to use for training.
            optimizer: The optimizer to use for training.
            lr_scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.net = net
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_fn = torch.nn.MSELoss()

        # TODO add ema

        # https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline

    def training_step(self, batch: dict[str, Tensor]) -> torch.Tensor:
        """Training Step of Diffusion Model.

        Args:
            batch: The batch of data to use for training

        Returns:
            loss of training step
        """
        # https://huggingface.co/docs/diffusers/tutorials/basic_training
        input = batch[self.input_key]
        noise = torch.randn_like(input)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (input.size(0),),
            device=input.device,
        ).long()

        noisy_x = self.noise_scheduler.add_noise(input, noise, timesteps)
        noise_pred = self.net(noisy_x, timesteps, return_dict=False)[0]

        loss = self.loss_fn(noise_pred, noise)
        self.log("train_loss", loss, batch_size=input.size(0))

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class ConditionalLatentImageDiffusionModel(BaseModule):
    """Conditional Image Generation Model."""

    def __init__(
        self,
        vae: ModelMixin,
        diffusion_net: ModelMixin,
        noise_scheduler: SchedulerMixin,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        """Initialize a new instance of the Conditional Image Diffusion Model.

        Args:
            vae: The variational autoencoder to use for encoding input data
                to latent space and decoding latent space to generate images
            diffusion_net: The neural network to use for diffusion, Unet or Transformer
                architecture from diffusers, for an example of adapting
                a model, see the `diffuser docs <https://huggingface.co/docs/diffusers/training/adapt_a_model>`_.
            noise_scheduler: The noise scheduler to use for training.
            optimizer: The optimizer to use for training.
            lr_scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.vae = vae
        self.diffusion_net = diffusion_net
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_fn = torch.nn.MSELoss()

        # TODO add ema

        # https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline
        # TODO: for VAE idea

    def training_step(self, batch: dict[str, Tensor]) -> torch.Tensor:
        """Training Step of Diffusion Model.

        Args:
            batch: The batch of data to use for training

        Returns:
            loss of training step
        """
        # https://huggingface.co/docs/diffusers/tutorials/basic_training
        import pdb

        pdb.set_trace()
        input = batch[self.input_key]
        noise = torch.randn_like(input)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (input.size(0),),
            device=input.device,
        ).long()

        noisy_x = self.noise_scheduler.add_noise(input, noise, timesteps)
        noise_pred = self.net(noisy_x, timesteps, return_dict=False)[0]

        loss = self.loss_fn(noise_pred, noise)
        self.log({"train_loss": loss}, batch_size=input.size(0))

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}
