"""Experiment generator to setup an experiment based on a config file."""

from typing import Any, Dict, List, Optional

import torch.nn as nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from uq_method_box.train_utils import NLL, QuantileLoss
from uq_method_box.uq_methods import (
    BaseModel,
    DeepEnsembleModel,
    MCDropoutModel,
    QuantileRegressionModel,
)


def retrieve_loss_fn(
    loss_fn_name: str, quantiles: Optional[List[float]] = None
) -> nn.Module:
    """Retrieve the desired loss function.

    Args:
        loss_fn_name: name of the loss function, one of ['mse', 'nll', 'quantile']

    Returns
        desired loss function module
    """
    if loss_fn_name == "mse":
        return nn.MSELoss()
    elif loss_fn_name == "nll":
        return NLL()
    elif loss_fn_name == "quantile":
        return QuantileLoss(quantiles)
    else:
        raise ValueError("Your loss function choice is not supported.")


def generate_base_model(
    config: Dict[str, Any],
    model: Optional[nn.Module] = None,
    criterion: Optional[nn.Module] = None,
) -> LightningModule:
    """Generate a configured base model.

    Args:
        config: config dictionary

    Returns:
        configured pytorch lightning module
    """
    if criterion is None:
        criterion = retrieve_loss_fn(config["model"]["loss_fn"])

    if config["model"]["base_model"] == "base_model":
        return BaseModel(config, model, criterion)

    elif config["model"]["base_model"] == "mc_dropout":
        return MCDropoutModel(config, model, criterion)

    elif config["model"]["base_model"] == "quantile_regression":
        return QuantileRegressionModel(config, model)

    else:
        raise ValueError("Your base_model choice is currently not supported.")


def generate_ensemble_model(
    config: Dict[str, Any], ensemble_members: List[nn.Module]
) -> LightningModule:
    """Generate an ensemble model.

    Args:
        config: config dictionary
        ensemble_members: ensemble models

    Returns:
        configureed ensemble lightning module
    """
    if config["model"]["ensemble"] == "deep_ensemble":
        return DeepEnsembleModel(config, ensemble_members)

    else:
        raise ValueError("Your ensemble choice is currently not supported.")


def generate_datamodule(config: Dict[str, Any]) -> LightningDataModule:
    """Generate LightningDataModule from config file.

    Args:
        config: config dictionary

    Returns:
        configured datamodule for experiment
    """
    pass


def generate_trainer(config: Dict[str, Any]) -> Trainer:
    """Generate a pytorch lightning trainer."""
    loggers = [
        CSVLogger(config["experiment"]["save_dir"], name="csv_logs"),
        WandbLogger(
            save_dir=config["experiment"]["save_dir"],
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            resume="allow",
            config=config,
            mode=config["wandb"].get("mode", "online"),
        ),
    ]

    track_metric = "train_loss"
    mode = "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["experiment"]["save_dir"],
        save_top_k=1,
        monitor=track_metric,
        mode=mode,
        every_n_epochs=1,
    )

    return Trainer(
        **config["pl"],
        default_root_dir=config["experiment"]["save_dir"],
        callbacks=[checkpoint_callback],
        logger=loggers
    )
