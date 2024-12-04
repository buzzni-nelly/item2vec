import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        max_len: int = 50,
        scale: int = 100,
        dropout: float = 0.05,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.Tensor([scale])) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BERT4Rec(nn.Module):
    def __init__(self, num_items: int, embed_dim: int, num_heads: int, num_layers: int, max_len: int, dropout=0.1):
        super(BERT4Rec, self).__init__()
        self.num_items = num_items
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.item_embeddings = nn.Embedding(num_items + 2, embed_dim, padding_idx=self.pad_token_idx)
        self.position_embeddings = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_seqs: torch.Tensor, src_key_padding_mask: torch.Tensor):
        embeddings = self.item_embeddings(input_seqs)
        embeddings = self.position_embeddings(embeddings)
        encoder_output = self.transformer_encoder(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )
        return encoder_output


class Bert4RecModule(pl.LightningModule):

    def __init__(
        self,
        num_items: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_len: int = 50,
        dropout: float = 0.1,
    ):
        super(Bert4RecModule, self).__init__()
        self.num_items = num_items
        self.mask_token_idx = num_items + 0
        self.pad_token_idx = num_items + 1
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout

        self.bert4rec = BERT4Rec(
            num_items=num_items,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )

    def training_step(self, batch: list[torch.Tensor], idx: int):
        # input_seqs shape: (4, 20)
        # src_key_padding_mask shape: (4, 20)
        # last_idxs shape: (4)
        # positive_idxs shape: (4)
        # negative_idxs shape: (4)
        input_seqs, src_key_padding_mask, last_idxs, positive_idxs, negative_idxs = batch
        # logits shape: (4, 20, 16)
        logits = self.bert4rec(input_seqs, src_key_padding_mask=src_key_padding_mask)
        # output shape: (4, 16)
        output = logits[torch.arange(logits.size(0)), last_idxs, :]

        # positive_embeddings shape: (4, 16)
        # negative_embeddings shape: (4, 16)
        positive_embeddings = self.bert4rec.item_embeddings.weight[positive_idxs]
        negative_embeddings = self.bert4rec.item_embeddings.weight[negative_idxs]

        # positive_scores shape: (4, 20)
        # negative_scores shape: (4, 20)
        positive_scores = torch.sum(output * positive_embeddings, dim=-1)
        negative_scores = torch.sum(output * negative_embeddings, dim=-1)

        # BPR Loss 계산
        train_loss = self.bpr_loss(positive_scores, negative_scores)
        self.log("train_loss", train_loss)
        return train_loss

    def forward(self):
        pass

    def bpr_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor):
        loss = -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))
        return loss

    def configure_optimizers(self) -> Optimizer:
        return optim.AdamW(self.parameters(), lr=0.1, weight_decay=0.001)


class Bert4RecTrainDataset(Dataset):
    def __init__(self, negative_k: int = 10):
        self.negative_k = negative_k
        self.num_items = 10
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1
        self.max_len = 20

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx):
        num_pads = random.randint(1, 3)
        input_seqs = torch.randint(0, self.num_items, (self.max_len,))
        input_seqs[-num_pads:] = self.pad_token_idx
        padding_mask = input_seqs == self.pad_token_idx

        last_idx = (~padding_mask).sum(dim=0) - 1

        # Sampling positive_idx & masking process
        positive_idx = input_seqs[last_idx].item()
        input_seqs[last_idx] = self.mask_token_idx

        # Sampling a negative_idx
        negative_idx = positive_idx
        while negative_idx == positive_idx:
            negative_idx = torch.randint(0, self.num_items, ()).item()

        return input_seqs, padding_mask, last_idx, positive_idx, negative_idx


class Bert4RecDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 4
        self.num_workers = 2
        self.train_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Bert4RecTrainDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=True,
        )


def main():

    data_module = Bert4RecDataModule()

    bert4rec = Bert4RecModule(
        num_items=10,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_len=20,
        dropout=0.1,
    )

    trainer = Trainer(
        limit_train_batches=10,
        max_epochs=5,
    )
    trainer.fit(model=bert4rec, datamodule=data_module)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
