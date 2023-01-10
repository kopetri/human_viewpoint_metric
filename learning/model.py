import pytorch_lightning as pl
from pytorch_utils.module import LightningModule
import torch
from learning.network import ViewModel
from learning.metrics import BinaryClassMetrics, accuracy

class ViewQualityModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ViewModel(mlp_layers=self.opt.mlp_layers, dropout=self.opt.dropout, sigmoid=False, decoder=self.opt.decoder if 'decoder' in self.opt else 'binary')
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.metrics = BinaryClassMetrics()
        self.test_output = None
        self.pred = []
        self.gt = []
        
    def get_prediction(self, x0, x1):
        if not 'decoder' in self.opt or self.opt.decoder == 'binary':
            y_hat = self.model(x0, x1)
        elif self.opt.decoder == 'goodness':
            score0 = self.model(x0)
            score1 = self.model(x1)
            y_hat = (score0 - score1)
        else:
            raise ValueError(self.opt.decoder)
        return y_hat

    def forward(self, batch, batch_idx, split):
        x0, x1, y, y_inv = batch
        B = x0.shape[0]
        
        if split in ["train", "valid"]:
            y_hat = self.get_prediction(x0, x1)
            loss = self.criterion(y_hat, y)
            acc = accuracy(y_hat.sigmoid(), y)
            # Logging
            self.log_value("loss", loss, split, B)
            self.log_value("acc", acc, split, B)
            return loss
        else:
            y_hat     = self.get_prediction(x0, x1)
            y_hat_inv = self.get_prediction(x1, x0)

            # combine prediction
            y_hat = torch.cat([y_hat, y_hat_inv], dim=0)
            y     = torch.cat([y,     y_inv],     dim=0)

            loss = self.criterion(y_hat, y)
            self.metrics.set_data(y_hat.sigmoid(), y)
            acc = self.metrics.compute_accuracy()
            auc = self.metrics.compute_AUC()
            aupr = self.metrics.compute_AUPR()
            self.log_value('acc', acc, split, B)
            self.log_value('auc', auc, split, B)
            self.log_value('aupr', aupr, split, B)
            self.log_value('loss', loss, split, B)
            return y, y_inv, y_hat, y_hat_inv

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        if self.opt.learning_rate_decay > 0:
            print("Adding learning rate decay: {}".format(self.opt.learning_rate_decay))
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=[lambda step: self.opt.learning_rate_decay]),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
            return [optimizer], [scheduler]
        elif self.opt.reduce_lr_on_plateau > 0:
            print("Adding learning rate scheduler on plateau")
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.opt.reduce_lr_on_plateau),
                'monitor': 'val_loss'
            }
            return [optimizer], [scheduler]
        else:
            return optimizer

class LabelingModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = ViewModel(mlp_layers=hparams.mlp_layers, dropout=hparams.dropout, sigmoid=False)
        self.labels = {}

    def forward(self, x0, x1):
        self.model.eval()
        y_hat = self.model(x0, x1)
        y_hat = y_hat.sigmoid()
        return y_hat

    def test_step(self, batch, batch_idx):
        x0, x1 = batch['image0'], batch['image1']
        y_hat = self.model(x0, x1)
        y_hat = y_hat.sigmoid()
        for img0, img1, pred in zip(batch['path0'], batch['path1'], y_hat):
            model_id_0, img_idx_0, _ = img0.split('/')[-1].split('.')
            model_id_1, img_idx_1, _ = img1.split('/')[-1].split('.')
            assert model_id_0 == model_id_1, "wrong model ids: {}, {}".format(model_id_0, model_id_1)
            if not model_id_0 in self.labels:
                self.labels[model_id_0] = torch.zeros(1000)
            if pred > 0.5:
                self.labels[model_id_0][int(img_idx_0.replace("off_", ""))] += 1
            else:
                self.labels[model_id_0][int(img_idx_1.replace("off_", ""))] += 1
        return 0
