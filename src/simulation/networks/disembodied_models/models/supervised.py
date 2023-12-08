import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks.disembodied_models.models.common import create_decoder, create_encoder


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = create_encoder([3, 32, 32, 64, 64])
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.encoder(x)
        return self.out(x)

    def shared_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
