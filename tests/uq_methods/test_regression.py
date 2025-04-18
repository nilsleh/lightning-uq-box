# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Regression Tasks."""

import glob
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.main import get_uq_box_cli
from lightning_uq_box.uq_methods import DeepEnsembleRegression

model_config_paths = [
    "tests/configs/regression/zigzag.yaml",
    "tests/configs/regression/masked_ensemble_mse.yaml",
    "tests/configs/regression/masked_ensemble_nll.yaml",
    "tests/configs/regression/mc_dropout_mse.yaml",
    "tests/configs/regression/mc_dropout_nll.yaml",
    "tests/configs/regression/mean_variance_estimation.yaml",
    "tests/configs/regression/qr_model.yaml",
    "tests/configs/regression/der.yaml",
    "tests/configs/regression/bnn_vi_elbo.yaml",
    "tests/configs/regression/bnn_vi.yaml",
    "tests/configs/regression/bnn_vi_batched.yaml",
    "tests/configs/regression/bnn_vi_lv_first_batched.yaml",
    "tests/configs/regression/bnn_vi_lv_first.yaml",
    "tests/configs/regression/bnn_vi_lv_last.yaml",
    "tests/configs/regression/swag.yaml",
    "tests/configs/regression/sgld_mse.yaml",
    "tests/configs/regression/sgld_nll.yaml",
    "tests/configs/regression/dkl.yaml",
    "tests/configs/regression/due.yaml",
    "tests/configs/regression/cards.yaml",
    "tests/configs/regression/sngp.yaml",
    "tests/configs/regression/vbll.yaml",
    "tests/configs/regression/mixture_density.yaml",
    "tests/configs/regression/density_layer.yaml",
]

data_config_paths = ["tests/configs/regression/toy_regression.yaml"]


class TestRegressionTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        args = [
            "--config",
            model_config_path,
            "--config",
            data_config_path,
            "--trainer.accelerator",
            "cpu",
            "--trainer.max_epochs",
            "2",
            "--trainer.log_every_n_steps",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--trainer.logger",
            "CSVLogger",
            "--trainer.logger.save_dir",
            str(tmp_path),
        ]

        cli = get_uq_box_cli(args)
        if "laplace" not in model_config_path:
            cli.trainer.fit(cli.model, cli.datamodule)
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

        # assert predictions are saved
        assert os.path.exists(
            os.path.join(cli.trainer.default_root_dir, cli.model.pred_file_name)
        )


posthoc_config_paths = [
    "tests/configs/regression/conformal_qr.yaml",
    "tests/configs/regression/conformal_qr_with_module.yaml",
]


class TestPosthoc:
    @pytest.mark.parametrize("model_config_path", posthoc_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        args = [
            "--config",
            model_config_path,
            "--config",
            data_config_path,
            "--trainer.accelerator",
            "cpu",
            "--trainer.max_epochs",
            "1",
            "--trainer.log_every_n_steps",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--trainer.logger",
            "CSVLogger",
            "--trainer.logger.save_dir",
            str(tmp_path),
        ]

        cli = get_uq_box_cli(args)
        cli.trainer.fit(cli.model, train_dataloaders=cli.datamodule.calib_dataloader())
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


ensemble_model_config_paths = [
    "tests/configs/regression/mc_dropout_mse.yaml",
    "tests/configs/regression/mc_dropout_nll.yaml",
    "tests/configs/regression/mean_variance_estimation.yaml",
]


class TestDeepEnsemble:
    @pytest.fixture(
        params=[
            (model_config_path, data_config_path)
            for model_config_path in ensemble_model_config_paths
            for data_config_path in data_config_paths
        ]
    )
    def ensemble_members_dict(
        self, request, tmp_path_factory: TempPathFactory
    ) -> list[dict[str, Any]]:
        model_config_path, data_config_path = request.param
        # train networks for deep ensembles
        ckpt_paths = []
        for i in range(5):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            args = [
                "--config",
                model_config_path,
                "--config",
                data_config_path,
                "--trainer.accelerator",
                "cpu",
                "--trainer.max_epochs",
                "1",
                "--trainer.log_every_n_steps",
                "1",
                "--trainer.default_root_dir",
                str(tmp_path),
            ]

            cli = get_uq_box_cli(args)
            cli.trainer.fit(cli.model, cli.datamodule)

            # Find the .ckpt file in the lightning_logs directory
            ckpt_file = glob.glob(
                os.path.join(
                    str(tmp_path),
                    "lightning_logs",
                    "version_*",
                    "checkpoints",
                    "*.ckpt",
                )
            )[0]
            ckpt_paths.append({"base_model": cli.model, "ckpt_path": ckpt_file})

        return ckpt_paths

    def test_deep_ensemble(
        self, ensemble_members_dict: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsembleRegression(ensemble_members_dict)

        datamodule = ToyHeteroscedasticDatamodule()

        trainer = Trainer(accelerator="cpu", default_root_dir=str(tmp_path))

        trainer.test(ensemble_model, datamodule=datamodule)

        assert os.path.exists(
            os.path.join(trainer.default_root_dir, ensemble_model.pred_file_name)
        )

    def test_mve_gmm_single_model(self, tmp_path: Path) -> None:
        """Test whether DeepEnsembleRegression reduces to a single MVE model for
        one ensemble member.
        """
        seed_everything(123)
        mve_path = "tests/configs/regression/mean_variance_estimation.yaml"
        data_config_path = data_config_paths[0]

        args = [
            "--config",
            mve_path,
            "--config",
            data_config_path,
            "--trainer.accelerator",
            "cpu",
            "--trainer.max_epochs",
            "1",
            "--trainer.log_every_n_steps",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
        ]

        cli = get_uq_box_cli(args)
        cli.trainer.fit(cli.model, cli.datamodule)

        # Find the .ckpt file in the lightning_logs directory
        ckpt_file = glob.glob(
            os.path.join(
                str(tmp_path), "lightning_logs", "version_*", "checkpoints", "*.ckpt"
            )
        )[0]
        mve_model = cli.model
        trained_model = [{"base_model": mve_model, "ckpt_path": ckpt_file}]

        pred_model_mve_gmm = DeepEnsembleRegression(ensemble_members=trained_model)

        pred_mve = mve_model.predict_step(cli.datamodule.X_test)
        pred_mve_gmm = pred_model_mve_gmm.predict_step(cli.datamodule.X_test)

        # pred_uct, aleatoric_uct, pred
        for key in set(pred_mve.keys()) & set(pred_mve_gmm.keys()):
            aa = pred_mve[key].squeeze()
            bb = pred_mve_gmm[key].squeeze()
            # NOTE: max(abs(aa-bb)) ~ 1.2e-07, that's quite high, even for float32.
            # Either one of the hard-cocded eps values, or too much of
            # sqrt(exp(log(...))**2).
            assert torch.allclose(aa, bb, rtol=0, atol=1e-6), (
                f"{torch.max(torch.abs(aa-bb))=}"
            )


frozen_config_paths = [
    "tests/configs/regression/mc_dropout_nll.yaml",
    "tests/configs/regression/bnn_vi_elbo.yaml",
    "tests/configs/regression/dkl.yaml",
    "tests/configs/regression/due.yaml",
    "tests/configs/regression/sngp.yaml",
    "tests/configs/regression/qr_model.yaml",
]


class TestFrozenBackbone:
    @pytest.mark.parametrize("model_config_path", frozen_config_paths)
    def test_freeze_backbone(self, model_config_path: str) -> None:
        cli = get_uq_box_cli(
            ["--config", model_config_path, "--model.freeze_backbone", "True"]
        )
        model = cli.model
        try:
            assert not all(
                [param.requires_grad for param in model.model.model[0].parameters()]
            )
            assert all(
                [param.requires_grad for param in model.model.model[-1].parameters()]
            )
        except AttributeError:
            # check that entire feature extractor is frozen
            assert not all(
                [param.requires_grad for param in model.feature_extractor.parameters()]
            )
