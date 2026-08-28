# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""DKL regression tests.

The config-driven smoke tests in test_classification.py/test_regression.py
train a DKL model and call ``trainer.test(...)`` on the *same in-memory
object*, so they never reload a checkpoint, and they use dict-style toy
datasets, so they never see a torchvision-style ``(Tensor, int)`` sample.
Both gaps hid real bugs.
"""

import torch
from conftest import minimal_trainer_kwargs
from lightning import LightningDataModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lightning_uq_box.uq_methods import DKLClassification, DKLRegression

NUM_CLASSES = 4
NUM_FEATURES = 16


class _TinyBackbone(nn.Module):
    """Small feature extractor so the GP layer has something to sit on."""

    def __init__(self, num_outputs: int = NUM_FEATURES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, num_outputs)
        )
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TupleDataset(Dataset):
    """Dataset returning ``(Tensor, int)``, as torchvision's do."""

    def __init__(self, n: int = 64, regression: bool = False) -> None:
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 8, 8, generator=generator)
        if regression:
            self.y = torch.randn(n, 1, generator=generator)
        else:
            # Plain Python ints, not tensors -- what torchvision returns.
            self.y = torch.randint(0, NUM_CLASSES, (n,), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class _ToyDataModule(LightningDataModule):
    """Datamodule over :class:`_TupleDataset` in the dict format DKL expects."""

    def __init__(self, n: int = 64, regression: bool = False) -> None:
        super().__init__()
        self.regression = regression
        # compute_initial_values samples up to 1000 points and splits them with
        # .chunk(10), so the dataset must yield 10 non-empty chunks.
        self.dataset = _TupleDataset(n, regression)
        self.x = self.dataset.x

    def _loader(self) -> DataLoader:
        def collate(batch: list) -> dict[str, torch.Tensor]:
            images, targets = zip(*batch)
            stacked = (
                torch.stack(targets)
                if self.regression
                else torch.tensor(targets, dtype=torch.long)
            )
            return {"input": torch.stack(images), "target": stacked}

        return DataLoader(self.dataset, batch_size=16, collate_fn=collate)

    def train_dataloader(self) -> DataLoader:
        return self._loader()

    def val_dataloader(self) -> DataLoader:
        return self._loader()

    def test_dataloader(self) -> DataLoader:
        return self._loader()

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int = 0) -> dict:
        return batch


def _classifier(**kwargs) -> DKLClassification:
    """Build a DKL classifier over the tiny backbone.

    Args:
        **kwargs: overrides passed to ``DKLClassification``

    Returns:
        the model
    """
    return DKLClassification(
        feature_extractor=_TinyBackbone(),
        n_inducing_points=8,
        num_classes=NUM_CLASSES,
        gp_kernel="RBF",
        **kwargs,
    )


class TestDKL:
    def test_fit_with_non_tensor_targets(self, accelerator_config, tmp_path) -> None:
        """Training must work on a dataset whose labels are plain ints.

        ``compute_initial_values`` used to ``torch.stack`` the raw targets,
        which raises "expected Tensor as element 0 in argument 0, but got int"
        from ``configure_optimizers`` -- before the first training step -- for
        any torchvision classification dataset.
        """
        model = _classifier()
        Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=1)
        ).fit(model, _ToyDataModule())

        assert model.dkl_model_built

    def test_forward_does_not_rescale_features(
        self, accelerator_config, tmp_path
    ) -> None:
        """forward() must be ``gp(feature_extractor(x))``, as in DUE.

        A ``ScaleToBounds`` used to sit between the two, which invalidates the
        lengthscale that ``compute_initial_values`` fits to the unscaled
        features.
        """
        datamodule = _ToyDataModule()
        model = _classifier()
        Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=1)
        ).fit(model, datamodule)
        model.eval()

        inputs = datamodule.x[:8].to(model.device)
        with torch.no_grad():
            actual = model.forward(inputs)
            expected = model.gp_layer(model.feature_extractor(inputs))

        assert torch.allclose(actual.mean, expected.mean)
        assert torch.allclose(actual.stddev, expected.stddev)

    def test_load_from_checkpoint(self, accelerator_config, tmp_path) -> None:
        """A checkpoint must reload with its GP weights and posterior intact.

        The GP layer is built lazily in ``configure_optimizers``, so loading a
        checkpoint used to hit a model with no ``gp_layer`` and raise
        ``RuntimeError`` listing every GP tensor as an unexpected key.
        """
        datamodule = _ToyDataModule()
        model = _classifier()
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=2)
        )
        trainer.fit(model, datamodule)

        ckpt_path = tmp_path / "dkl_classification.ckpt"
        trainer.save_checkpoint(ckpt_path)
        reloaded = DKLClassification.load_from_checkpoint(
            ckpt_path,
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )

        # num_data scales the ELBO's KL term, so it has to round-trip too.
        assert reloaded.elbo_fn.num_data == model.elbo_fn.num_data

        model.eval()
        reloaded.eval()
        inputs = datamodule.x[:8].to(model.device)
        with torch.no_grad():
            expected = model.forward(inputs)
            actual = reloaded.forward(inputs.to(reloaded.device))

        # predict_step samples the likelihood, so compare the GP posterior.
        assert torch.allclose(expected.mean, actual.mean.to(expected.mean.device))
        assert torch.allclose(expected.stddev, actual.stddev.to(expected.stddev.device))

    def test_test_with_ckpt_path(self, accelerator_config, tmp_path) -> None:
        """``Trainer.test(ckpt_path=...)`` must run -- it used to raise."""
        datamodule = _ToyDataModule()
        model = _classifier()
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=1)
        )
        trainer.fit(model, datamodule)
        ckpt_path = tmp_path / "dkl_test_path.ckpt"
        trainer.save_checkpoint(ckpt_path)

        results = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path)).test(
            _classifier(), datamodule, ckpt_path=str(ckpt_path), verbose=False
        )

        assert "testAcc" in results[0]

    def test_regression_load_from_checkpoint(
        self, accelerator_config, tmp_path
    ) -> None:
        """The checkpoint round-trip must hold for DKLRegression too."""
        datamodule = _ToyDataModule(regression=True)
        model = DKLRegression(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_targets=1,
            gp_kernel="RBF",
        )
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=1)
        )
        trainer.fit(model, datamodule)

        ckpt_path = tmp_path / "dkl_regression.ckpt"
        trainer.save_checkpoint(ckpt_path)
        reloaded = DKLRegression.load_from_checkpoint(
            ckpt_path,
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_targets=1,
            gp_kernel="RBF",
        )

        model.eval()
        reloaded.eval()
        inputs = datamodule.x[:8].to(model.device)
        with torch.no_grad():
            expected = model.forward(inputs)
            actual = reloaded.forward(inputs.to(reloaded.device))

        assert torch.allclose(expected.mean, actual.mean.to(expected.mean.device))
