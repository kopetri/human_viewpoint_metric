import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import lightning as L
import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers
import torch


def parse_ckpt(path, return_first=True, pattern=None):
    """Resolve a checkpoint path or directory to a .ckpt file path."""
    if Path(path).is_file():
        print("Loading checkpoint: ", path)
        if return_first:
            return path
        return [path]
    ckpts = [p.as_posix() for p in Path(path).glob("**/*") if p.suffix == ".ckpt"]
    if pattern:
        ckpts = [ckpt for ckpt in ckpts if pattern in Path(ckpt).stem]
    if return_first:
        ckpt = ckpts[0]
        print("Loading checkpoint: ", ckpt)
        return ckpt
    print("Found {} checkpoints.".format(len(ckpts)))
    return ckpts


class Trainer(L.Trainer):
    def __init__(self, project_name):
        self.project_name = project_name
        self.parser = ArgumentParser("Training of {}".format(project_name))
        self.parser.add_argument("--seed", default=None, type=int, help="Random Seed")
        self.parser.add_argument("--accelerator", default="gpu", type=str, help="Accelerator to use: cpu or gpu")
        self.parser.add_argument(
            "--devices", default=-1, nargs="+", type=int, help="Number of GPUs to use. -1 uses all."
        )
        self.parser.add_argument("--precision", default="16-mixed", type=str, help="Floating point precision")
        self.parser.add_argument("--dev", action="store_true", help="Activate Lightning fast_dev_run for debugging")
        self.parser.add_argument("--overfit", default=0, type=int, help="Overfit on this many batches (0 = disabled)")
        self.parser.add_argument("--profiler", default=None, type=str, help='"simple" or "advanced"')
        self.parser.add_argument("--min_epochs", default=10, type=int, help="Minimum number of epochs")
        self.parser.add_argument("--max_epochs", default=50, type=int, help="Maximum number of epochs")
        self.parser.add_argument("--worker", default=8, type=int, help="DataLoader worker count")
        self.parser.add_argument("--detect_anomaly", action="store_true", help="Enable PyTorch anomaly detection")
        self.parser.add_argument("--learning_rate_decay", default=0.99999, type=float, help="Multiplicative LR decay")
        self.parser.add_argument(
            "--reduce_lr_on_plateau", default=0, type=int, help="ReduceLROnPlateau patience (0 = disabled)"
        )
        self.parser.add_argument(
            "--early_stop_patience", default=0, type=int, help="EarlyStopping patience (0 = disabled)"
        )
        self.parser.add_argument("--name", default=None, help="Name of the training run")
        self.parser.add_argument("--log_every_n_steps", default=50, type=int, help="Logging interval in steps")
        self.parser.add_argument("--ckpt_every_n_epochs", default=None, type=int, help="Save checkpoint every N epochs")
        self.parser.add_argument("--ckpt_every_n_steps", default=None, type=int, help="Save checkpoint every N steps")
        self.parser.add_argument("--save_code_base", default=1, type=int, help="Save code base to W&B")
        self.parser.add_argument(
            "--checkpoint_metric", default=["valid_loss"], nargs="+", type=str, help="Metric for best-checkpoint saving"
        )
        self.parser.add_argument(
            "--mode", default=["min"], nargs="+", type=str, help="min or max for checkpoint_metric"
        )
        self.__initialized__ = False
        self.__args__ = None

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def setup(self, train=True, **kwargs):
        self.__args__ = self.parser.parse_args()
        args = vars(self.__args__)
        args.update(**kwargs)
        self.__args__ = Namespace(**args)

        if self.__args__.detect_anomaly:
            print("Enabling anomaly detection")
            torch.autograd.set_detect_anomaly(True)

        if sys.platform == "win32":
            self.__args__.worker = 6

        if self.__args__.seed is None:
            self.__args__.seed = random.randrange(4294967295)
        L.seed_everything(self.__args__.seed)

        if self.__args__.name is None or self.__args__.dev:
            logger = None
        else:
            logger = pl_loggers.WandbLogger(
                project=self.project_name,
                name=self.__args__.name,
                log_model=bool(self.__args__.save_code_base),
            )

        callbacks = []

        if self.__args__.learning_rate_decay and logger:
            callbacks += [pl_callbacks.LearningRateMonitor()]

        callbacks += [
            pl_callbacks.ModelCheckpoint(
                verbose=True,
                save_top_k=1,
                filename="{epoch}-{" + metric + "}",
                monitor=metric,
                mode=mode,
            )
            for metric, mode in zip(self.__args__.checkpoint_metric, self.__args__.mode, strict=False)
        ]

        if self.__args__.ckpt_every_n_epochs:
            callbacks += [
                pl_callbacks.ModelCheckpoint(
                    verbose=True,
                    save_top_k=-1,
                    every_n_epochs=self.__args__.ckpt_every_n_epochs,
                    filename="{epoch}",
                )
            ]

        if self.__args__.ckpt_every_n_steps:
            callbacks += [
                pl_callbacks.ModelCheckpoint(
                    verbose=True,
                    save_top_k=-1,
                    every_n_train_steps=self.__args__.ckpt_every_n_steps,
                    filename="{step}",
                )
            ]

        if self.__args__.early_stop_patience > 0:
            callbacks += [
                pl_callbacks.EarlyStopping(
                    monitor=metric,
                    min_delta=0.0,
                    patience=self.__args__.early_stop_patience,
                    verbose=True,
                    mode=mode,
                )
                for metric, mode in zip(self.__args__.checkpoint_metric, self.__args__.mode, strict=False)
            ]

        # Unwrap single-element device list so Lightning accepts it for non-GPU accelerators
        devices = self.__args__.devices
        if isinstance(devices, list) and len(devices) == 1:
            devices = devices[0]

        if train:
            super().__init__(
                fast_dev_run=self.__args__.dev,
                accelerator=self.__args__.accelerator,
                devices=devices,
                log_every_n_steps=self.__args__.log_every_n_steps,
                overfit_batches=self.__args__.overfit,
                precision=self.__args__.precision,
                min_epochs=self.__args__.min_epochs,
                max_epochs=self.__args__.max_epochs,
                logger=logger,
                callbacks=callbacks,
                deterministic=True,
                profiler=self.__args__.profiler,
                **kwargs,
            )
        else:
            super().__init__(
                accelerator=self.__args__.accelerator,
                devices=devices,
                precision=self.__args__.precision,
                deterministic=True,
                **kwargs,
            )

        return self.__args__
