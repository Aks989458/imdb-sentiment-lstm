import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy

class SentimentLSTM(LightningModule):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(1)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(torch.sigmoid(logits), y.int())
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
