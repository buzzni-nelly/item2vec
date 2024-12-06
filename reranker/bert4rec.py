import random

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
        self.position_embeddings = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_len,
            dropout=dropout,
        )

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
        input_seqs, src_key_padding_mask, masked_idx, positive_idxs, negative_idxs = batch
        # logits shape: (4, 20, 16)
        logits = self.bert4rec(input_seqs, src_key_padding_mask=src_key_padding_mask)
        # output shape: (4, 16)
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1).to(self.device)
        output = logits[batch_indices, masked_idx]

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

    def validation_step(self, batch: list[torch.Tensor], idx: int):
        input_seqs, src_key_padding_mask, last_idxs, ground_truth_items = batch
        logits = self.bert4rec(input_seqs, src_key_padding_mask=src_key_padding_mask)
        output = logits[torch.arange(logits.size(0)), last_idxs, :]
        scores = torch.matmul(output, self.bert4rec.item_embeddings.weight[:-2].T)

        mrr = self.calc_mrr(scores, ground_truth_items)
        self.log("val_mrr", mrr, prog_bar=True)

        recall_5 = self.calc_recall_at_k(scores, ground_truth_items, 5)
        ndcg_5 = self.calc_ndcg_at_k(scores, ground_truth_items, 5)
        self.log(f"val_recall@5", recall_5, prog_bar=True)
        self.log(f"val_ndcg@5", ndcg_5, prog_bar=True)

        recall_20 = self.calc_recall_at_k(scores, ground_truth_items, 20)
        ndcg_20 = self.calc_ndcg_at_k(scores, ground_truth_items, 20)
        self.log(f"val_recall@20", recall_20, prog_bar=True)
        self.log(f"val_ndcg@20", ndcg_20, prog_bar=True)

    def calc_mrr(self, scores: torch.Tensor, ground_truth_items: torch.Tensor):
        gt_scores = scores.gather(1, ground_truth_items.unsqueeze(1)).squeeze(1)
        ranks = (scores > gt_scores.unsqueeze(1)).sum(dim=1) + 1
        reciprocal_ranks = 1.0 / ranks.float()
        mrr = torch.mean(reciprocal_ranks).item()
        return mrr

    def calc_recall_at_k(self, scores: torch.Tensor, ground_truth_items: torch.Tensor, k: int):
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        hits = (topk_indices == ground_truth_items.unsqueeze(1)).any(dim=1).float()
        recall = torch.mean(hits).item()
        return recall

    def calc_ndcg_at_k(self, scores: torch.Tensor, ground_truth_items: torch.Tensor, k: int):
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        match_positions = (topk_indices == ground_truth_items.unsqueeze(1)).nonzero(as_tuple=False)
        dcg = torch.zeros(scores.size(0), device=scores.device)
        dcg[match_positions[:, 0]] = 1.0 / torch.log2(match_positions[:, 1].float() + 2)
        idcg = 1.0
        ndcg = torch.mean(dcg / idcg).item()
        return ndcg

    def forward(self):
        pass

    def bpr_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor):
        loss = -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))
        return loss

    def configure_optimizers(self) -> Optimizer:
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def import_item_embeddings(self, item_embeddings: torch.Tensor):
        mask_embeddings = torch.zeros((1, item_embeddings.size(1))).to(item_embeddings.device)
        padding_embeddings = torch.zeros((1, item_embeddings.size(1))).to(item_embeddings.device)
        extended_embeddings = torch.cat([item_embeddings, mask_embeddings, padding_embeddings], dim=0)
        self.bert4rec.item_embeddings.weight.data.copy_(extended_embeddings)


class Bert4RecTrainDataset(Dataset):
    def __init__(self, volume: Volume, max_len: int = 50):
        self.histories = volume.migrate_user_histories()
        self.num_items = volume.vocab_size()
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1
        self.max_len = max_len
        self.num_masked = 2

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, idx: int):
        history_pidxs = self.histories[idx]
        seq_len = len(history_pidxs)
        pad_len = self.max_len - seq_len

        input_seqs = history_pidxs + ([self.pad_token_idx] * pad_len)
        input_seqs = torch.tensor(input_seqs, dtype=torch.long)
        padding_mask = input_seqs == self.pad_token_idx

        valid_positions = [i for i in range(seq_len)]
        masked_positions = random.sample(valid_positions, min(self.num_masked, seq_len))
        masked_positions = torch.tensor(masked_positions, dtype=torch.long)

        positive_pidxs = input_seqs[masked_positions].clone()
        input_seqs[masked_positions] = self.mask_token_idx

        negative_pidxs = torch.randint(0, self.num_items, (len(masked_positions),), dtype=torch.long)
        for i in range(len(negative_pidxs)):
            while negative_pidxs[i] == positive_pidxs[i]:
                negative_pidxs[i] = torch.randint(0, self.num_items, ()).item()

        return input_seqs, padding_mask, masked_positions, positive_pidxs, negative_pidxs


class Bert4RecValidDataset(Dataset):
    def __init__(self, volume: Volume, max_len: int = 50):
        self.histories = volume.migrate_user_histories()
        self.num_items = volume.vocab_size()
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.histories) // 500

    def __getitem__(self, idx: int):
        history_pidxs = self.histories[idx]
        pad_len = self.max_len - len(history_pidxs)

        input_seqs = history_pidxs + ([self.pad_token_idx] * pad_len)
        input_seqs = torch.tensor(input_seqs, dtype=torch.long)
        padding_mask = input_seqs == self.pad_token_idx

        last_idx = (~padding_mask).sum(dim=0) - 1

        ground_truth_item = input_seqs[last_idx].item()
        input_seqs[last_idx] = self.mask_token_idx

        return input_seqs, padding_mask, last_idx, ground_truth_item


class Bert4RecDataModule(pl.LightningDataModule):
    def __init__(self, volume: Volume, batch_size: int = 32, num_workers: int = 2, max_len: int = 50):
        super().__init__()
        self.volume = volume
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.train_dataset = None
        self.valid_dataset = None

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
            collate_fn=self.bert4rec_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            Bert4RecValidDataset(volume=self.volume, max_len=self.max_len),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=False,
        )

    def bert4rec_collate_fn(self, batch):
        """
        Bert4Rec 데이터셋을 위한 collate_fn.

        Args:
            batch (list of tuples): 각 샘플은 (input_seqs, padding_mask, masked_positions, positive_pidxs, negative_pidxs) 형태.

        Returns:
            Tensor 형태의 배치 데이터:
            - input_seqs: (batch_size, max_len)
            - padding_masks: (batch_size, max_len)
            - masked_positions: (batch_size, max_masked)
            - positive_pidxs: (batch_size, max_masked)
            - negative_pidxs: (batch_size, max_masked)
        """
        # 모든 샘플에서 텐서 추출
        input_seqs = torch.stack([item[0] for item in batch])  # shape: (batch_size, max_len)
        padding_masks = torch.stack([item[1] for item in batch])  # shape: (batch_size, max_len)

        # 각 샘플의 마스크된 위치와 positive, negative 인덱스를 패딩 처리
        max_masked = max(len(item[2]) for item in batch)  # 배치에서 최대 마스크 개수
        batch_size = len(batch)

        pad_token_idx = self.train_dataset.pad_token_idx

        # 배치를 위한 텐서 초기화 (패딩은 0으로)
        masked_positions = torch.full((batch_size, max_masked), pad_token_idx, dtype=torch.long)
        positive_pidxs = torch.full((batch_size, max_masked), pad_token_idx, dtype=torch.long)
        negative_pidxs = torch.full((batch_size, max_masked), pad_token_idx, dtype=torch.long)

        # 각 샘플의 데이터를 텐서에 채워 넣기
        for i, (seq, mask, masked_pos, pos_idx, neg_idx) in enumerate(batch):
            length = len(masked_pos)  # 현재 샘플의 마스크 된 위치 개수
            masked_positions[i, :length] = masked_pos
            positive_pidxs[i, :length] = pos_idx
            negative_pidxs[i, :length] = neg_idx

        return input_seqs, padding_masks, masked_positions, positive_pidxs, negative_pidxs
