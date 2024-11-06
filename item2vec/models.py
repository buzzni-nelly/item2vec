import pytorch_lightning as pl
import torch
from torch import nn, optim


class Item2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super(Item2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Xavier 초기화 적용
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, items, samples):
        item_embeddings = self.embeddings(items)
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
    ):
        super(Item2VecModule, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.item2vec = Item2Vec(vocab_size, embed_dim)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, focus_items, context_items):
        return self.item2vec(focus_items, context_items)

    def training_step(self, batch, batch_idx):
        focus_items, context_items, labels = batch
        scores = self.forward(focus_items, context_items)
        loss = self.criterion(scores, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class Item2VecBPRModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super(Item2VecBPRModule, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.item2vec = Item2Vec(vocab_size, embed_dim)

    def forward(self, focus_items, positive_items, negative_items):
        pos_scores = self.item2vec(focus_items, positive_items)
        neg_scores = self.item2vec(focus_items, negative_items)
        return pos_scores, neg_scores

    def bpr_loss(self, pos_scores, neg_scores):
        if torch.isnan(pos_scores).any() or torch.isnan(neg_scores).any():
            print("NaN detected in pos_scores or neg_scores")
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    def training_step(self, batch, batch_idx):
        focus_items, positive_items, negative_items = batch
        pos_scores, neg_scores = self.forward(focus_items, positive_items, negative_items)
        loss = self.bpr_loss(pos_scores, neg_scores)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
