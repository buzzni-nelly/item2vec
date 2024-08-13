import pytorch_lightning as pl
import torch
from torch import nn, optim


class Item2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super(Item2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, items, samples):
        item_embeddings = self.embeddings(items)
        item_embeddings = item_embeddings.unsqueeze(1)
        sample_embeddings = self.embeddings(samples)
        sample_embeddings = sample_embeddings.transpose(1, 2)
        scores = torch.bmm(item_embeddings, sample_embeddings)
        scores = scores.squeeze(1)
        return scores


class Item2VecModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        dropout: float = 0.5,
    ):
        super(Item2VecModule, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.item2vec = Item2Vec(vocab_size, embed_dim)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, focus_items, context_items):
        return self.item2vec(focus_items, context_items)

    def training_step(self, batch, batch_idx):
        focus_items, context_items, labels = batch
        scores = self.forward(focus_items, context_items)
        loss = self.criterion(scores, labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        focus_items, context_items, labels = batch
        scores = self(focus_items, context_items)
        loss = self.criterion(scores, labels)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
