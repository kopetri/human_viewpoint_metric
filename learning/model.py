from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

import lightning as L
import lightning.pytorch.loggers as pl_loggers
import torch

from learning.metrics import BinaryClassMetrics, accuracy
from learning.network import ViewModel


class LightningModule(L.LightningModule):
    """Inlined from pytorch_utils.module — no external dependency needed."""

    def __init__(self, opt=None, **kwargs):
        super().__init__()
        if opt is None:
            self.opt = Namespace(**kwargs)
            if "hparams" in vars(self.opt):
                self.opt = self.opt.hparams
        else:
            self.opt = opt
        self.save_hyperparameters(vars(self.opt))
        self.learning_rate = self.opt.learning_rate
        self.validation_outputs = []

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        outputs = self(batch, batch_idx, "valid")
        self.validation_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        return self(batch, batch_idx, "test")

    def log_value(self, key, value, split, batch_size):
        self.log(
            "{split}_{key}".format(split=split, key=key),
            value,
            prog_bar=True,
            on_epoch=split != "train",
            on_step=split == "train",
            batch_size=batch_size,
        )

    def log_image(self, key, images, **kwargs):
        if self.logger and isinstance(self.logger, pl_loggers.WandbLogger):
            self.logger.log_image(key=key, images=images, **kwargs)

    def on_save_checkpoint(self, _checkpoint) -> None:
        if not (isinstance(self.logger, pl_loggers.WandbLogger) and getattr(self.opt, "save_code_base", False)):
            return
        path = Path(".", self.logger.experiment.project, self.logger.experiment.id, "code")
        zipfile = path / "code.zip"
        if not zipfile.exists():
            path.mkdir(parents=True, exist_ok=True)
            code_base = [
                f
                for f in Path(".").glob("**/*")
                if f.suffix == ".py" and not any(s in f.as_posix() for s in ["venv", "wandb", "lightning_log"])
            ]
            with ZipFile(zipfile.as_posix(), "w") as codezip:
                for code in code_base:
                    codezip.write(code)

    def forward(self, batch, batch_idx, split):
        raise NotImplementedError


class ViewQualityModel(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        mlp_layers: list | None = None,
        dropout: float = 0.2,
        decoder: str = "binary",
        learning_rate_decay: float = 0.0,
        reduce_lr_on_plateau: int = 0,
    ):
        super().__init__(
            learning_rate=learning_rate,
            mlp_layers=mlp_layers or [256, 64],
            dropout=dropout,
            decoder=decoder,
            learning_rate_decay=learning_rate_decay,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
        )
        decoder = self.opt.decoder if "decoder" in self.opt else "binary"
        self.model = ViewModel(
            mlp_layers=self.opt.mlp_layers,
            dropout=self.opt.dropout,
            sigmoid=False,
            decoder=decoder,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.metrics = BinaryClassMetrics()
        self.test_output = None
        self.pred = []
        self.gt = []

    def get_prediction(self, x0, x1):
        if "decoder" not in self.opt or self.opt.decoder == "binary":
            y_hat = self.model(x0, x1)
        elif self.opt.decoder == "goodness":
            score0 = self.model(x0)
            score1 = self.model(x1)
            y_hat = score0 - score1
        else:
            raise ValueError(self.opt.decoder)
        return y_hat

    def forward(self, batch, batch_idx, split):
        x0, x1, y, y_inv = batch
        batch_size = x0.shape[0]

        if split in ["train", "valid"]:
            y_hat = self.get_prediction(x0, x1)
            loss = self.criterion(y_hat, y)
            acc = accuracy(y_hat.sigmoid(), y)
            self.log_value("loss", loss, split, batch_size)
            self.log_value("acc", acc, split, batch_size)
            return loss
        y_hat = self.get_prediction(x0, x1)
        y_hat_inv = self.get_prediction(x1, x0)
        y_hat = torch.cat([y_hat, y_hat_inv], dim=0)
        y = torch.cat([y, y_inv], dim=0)

        loss = self.criterion(y_hat, y)
        self.metrics.set_data(y_hat.sigmoid(), y)
        acc = self.metrics.compute_accuracy()
        auc = self.metrics.compute_AUC()
        aupr = self.metrics.compute_AUPR()
        self.log_value("acc", acc, split, batch_size)
        self.log_value("auc", auc, split, batch_size)
        self.log_value("aupr", aupr, split, batch_size)
        self.log_value("loss", loss, split, batch_size)
        return y, y_inv, y_hat, y_hat_inv

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        if self.opt.learning_rate_decay > 0:
            print("Adding learning rate decay: {}".format(self.opt.learning_rate_decay))
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiplicativeLR(
                    optimizer,
                    lr_lambda=[lambda _: self.opt.learning_rate_decay],
                ),
                "interval": "step",
                "frequency": 1,
                "strict": True,
            }
            return [optimizer], [scheduler]
        if self.opt.reduce_lr_on_plateau > 0:
            print("Adding learning rate scheduler on plateau")
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    "min",
                    patience=self.opt.reduce_lr_on_plateau,
                ),
                "monitor": "val_loss",
            }
            return [optimizer], [scheduler]
        return optimizer


class LabelingModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = ViewModel(mlp_layers=hparams.mlp_layers, dropout=hparams.dropout, sigmoid=False)
        self.labels = {}

    def forward(self, x0, x1):
        self.model.eval()
        return self.model(x0, x1).sigmoid()

    def test_step(self, batch, batch_idx):
        x0, x1 = batch["image0"], batch["image1"]
        y_hat = self.model(x0, x1).sigmoid()
        for img0, img1, pred in zip(batch["path0"], batch["path1"], y_hat, strict=False):
            model_id_0, img_idx_0, _ = img0.split("/")[-1].split(".")
            model_id_1, img_idx_1, _ = img1.split("/")[-1].split(".")
            assert model_id_0 == model_id_1, "wrong model ids: {}, {}".format(model_id_0, model_id_1)
            if model_id_0 not in self.labels:
                self.labels[model_id_0] = torch.zeros(1000)
            if pred > 0.5:  # noqa: PLR2004
                self.labels[model_id_0][int(img_idx_0.replace("off_", ""))] += 1
            else:
                self.labels[model_id_0][int(img_idx_1.replace("off_", ""))] += 1
        return 0
