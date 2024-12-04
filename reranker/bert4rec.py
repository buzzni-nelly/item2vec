import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from item2vec.volume import Volume


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
    def __init__(
        self,
        num_items: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_len: int,
        dropout=0.1,
    ):
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
        encoder_output = self.transformer_encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
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
        lr: float = 0.001,
        weight_decay: float = 0.001,
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
        self.lr = lr
        self.weight_decay = weight_decay

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
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def forward(self):
        pass

    def bpr_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor):
        loss = -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))
        return loss

    def configure_optimizers(self) -> Optimizer:
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class Bert4RecTrainDataset(Dataset):
    def __init__(self, volume: Volume, max_len: int = 50):
        self.histories = volume.migrate_user_histories()
        self.num_items = volume.vocab_size()
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, idx: int):
        history_pidxs = self.histories[idx]
        pad_len = self.max_len - len(history_pidxs)

        input_seqs = history_pidxs + ([self.pad_token_idx] * pad_len)
        input_seqs = torch.tensor(input_seqs, dtype=torch.long)
        padding_mask = input_seqs == self.pad_token_idx

        last_idx = (~padding_mask).sum(dim=0) - 1

        # Sampling positive_idx & masking process
        positive_pidx = input_seqs[last_idx].item()
        input_seqs[last_idx] = self.mask_token_idx

        # Sampling a negative_idx
        negative_pidx = positive_pidx
        while negative_pidx == positive_pidx:
            negative_pidx = torch.randint(0, self.num_items, ()).item()

        return input_seqs, padding_mask, last_idx, positive_pidx, negative_pidx


class Bert4RecDataModule(pl.LightningDataModule):
    def __init__(self, volume: Volume, batch_size: int = 32, num_workers: int = 2, max_len: int = 50):
        super().__init__()
        self.volume = volume
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Bert4RecTrainDataset(volume=self.volume, max_len=self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=True,
        )
