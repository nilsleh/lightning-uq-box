# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Utility Callbacks applicable to multiple methods."""

import matplotlib.pyplot as plt
from lightning import Callback, LightningModule, Trainer
from torch import Tensor


class LogImageSamples(Callback):
    """Callback for logging sampled images during training."""

    def __init__(self, num_samples=8, every_n_steps=100, **kwargs):
        """Initialize the callback.

        Args:
            num_samples: The number of samples to generate.
            every_n_steps: The number of steps between logging.
            kwargs: Additional arguments to pass to the model
                sampling function
        """
        super().__init__()
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps
        self.kwargs = kwargs

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ) -> None:
        """Check step and log samples.

        Args:
            trainer: The trainer object.
            pl_module: The UQ Method Lightning Module
            outputs: The output of the training step (e.g. loss)
            batch: The current batch of data
            batch_idx: The index of the current batch
        """
        if self.trainer.global_step % self.log_samples_every_n_steps == 0:
            sampled_imgs = self.generate_imgs(pl_module)

            if not isinstance(trainer.logger, list):
                loggers = [trainer.logger]
            for logger in loggers:
                curr_logger = logger.experiment
                if hasattr(curr_logger, "add_image"):
                    curr_logger.add_image(
                        f"sample_{self.trainer.global_step}",
                        sampled_imgs,
                        global_step=trainer.current_epoch,
                    )
                elif hasattr(logger, "log_image"):
                    curr_logger.log_image(
                        f"sample_{self.trainer.global_step}",
                        sampled_imgs,
                        global_step=trainer.current_epoch,
                    )
                else:
                    continue

    def generate_imgs(self, pl_module: LightningModule) -> Tensor:
        """Generate Image samples from the UQ-Method.

        By default it calls the sampling method of the UQ-Method.

        Can be overwritten by subclasses to generate images in a different way
        and still use the logging functionality.

        Args:
            pl_module: The module to generate images from.
        """
        return pl_module.sample(**self.kwargs)


class LogSegmentationPreds(Callback):
    """Callback for logging segmentation masks during training."""

    def __init__(self, num_samples=8, every_n_steps=100):
        """Initialize the callback.

        Args:
            num_samples: The number of samples to log, will be sampled from the batch
                at the current trainer step
            every_n_steps: The number of steps between logging.
        """
        super().__init__()
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ) -> None:
        """Check step and log segmentation predictions.

        Args:
            trainer: The trainer object.
            pl_module: The UQ Method Lightning Module
            outputs: The output of the training step (e.g. loss)
            batch: The current batch of data
            batch_idx: The index of the current batch
        """
        if trainer.global_step % self.every_n_steps == 0:
            # do forward pass of input batch and log the segmentation masks
            # or actually a predict step and log the segmentation masks with UQ etc.
            pred_dict = pl_module.predict_step(batch[pl_module.input_key])

            pred_dict = {
                key: val[: self.num_samples, ...].detach()
                for key, val in pred_dict.items()
            }
            pred_dict[pl_module.target_key] = batch[pl_module.target_key][
                : self.num_samples, ...
            ]
            pred_dict[pl_module.input_key] = batch[pl_module.input_key][
                : self.num_samples, ...
            ]

            for i in range(pred_dict["pred"].shape[0]):
                # add figure of pred, uct, and target
                pred = pred_dict["pred"][i, ...].permute(1, 2, 0)

                pred_uct = pred_dict["pred_uct"][i, ...]

                input = pred_dict[pl_module.input_key][i, ...]
                input = input if input.dim() == 2 else input.permute(1, 2, 0)

                target = pred_dict[pl_module.target_key][i, ...]
                target = target if target.dim() == 2 else target.permute(1, 2, 0)

                fig, axs = plt.subplots(1, 4, figsize=(16, 4))

                axs[0].imshow(input)
                axs[0].set_title(f"{pl_module.input_key}")
                axs[0].axis("off")

                axs[1].imshow(pred)
                axs[1].set_title("Prediction")
                axs[1].axis("off")

                axs[2].imshow(pred_uct)
                axs[2].set_title("Uncertainty")
                axs[2].axis("off")

                axs[3].imshow(target)
                axs[3].set_title(f"{pl_module.target_key}")
                axs[3].axis("off")

                for logger in trainer.loggers:
                    curr_logger = logger.experiment
                    import pdb

                    pdb.set_trace()
                    if "tensorboard" in str(curr_logger.__class__):
                        curr_logger.add_figure(
                            f"pred_{trainer.global_step}_{i}",
                            fig,
                            global_step=trainer.global_step,
                        )
                    elif "wandb" in str(curr_logger.__class__):
                        curr_logger.log({"prediction": fig}, step=trainer.global_step)
                    else:
                        continue

                plt.close(fig)
