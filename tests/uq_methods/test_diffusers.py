# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Diffuser Tests."""

from pathlib import Path

import pytest
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

model_config_paths = [
    "tests/configs/diffusers/unconditional_unet.yaml"
    # "tests/configs/diffusers/conditional_image_unet.yaml",
    # "tests/configs/diffusion/conditional_image_transformer.yaml",
]

data_config_paths = ["tests/configs/diffusers/toy_dataset.yaml"]


class TestConditionalDiffusionTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)
        model = instantiate(model_conf.uq_method)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=2,
            log_every_n_steps=1,
            default_root_dir=str(tmp_path),
            logger=CSVLogger(str(tmp_path)),
        )
        # laplace only uses test
        trainer.fit(model, datamodule)
