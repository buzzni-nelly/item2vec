import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from item2vec.volume import Volume
from reranker.attention import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder
from reranker.encoding import PositionalEncoding


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout=0.1,
    ):
        super(CrossAttention, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )
        decoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> tuple[Tensor, Tensor]:
        encoder_output, encoder_weights = self.transformer_encoder(
            x,
            x,
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        decoder_output, decoder_weights = self.transformer_decoder(
            x,
            encoder_output,
            encoder_output,
            src_key_padding_mask=src_key_padding_mask,
        )
        return decoder_output, decoder_weights


class CARCA(pl.LightningModule):

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
        super(CARCA, self).__init__()
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

        self.item_embeddings = nn.Embedding(num_items + 2, embed_dim, padding_idx=self.pad_token_idx)
        self.dropout = nn.Dropout(dropout)
        self.position_embeddings = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        self.cross_attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        input_seqs: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        last_idxs: torch.Tensor,
        candidate_idxs: torch.Tensor = None,
    ):
        embeddings = self.item_embeddings(input_seqs)
        embeddings = self.dropout(embeddings)
        embeddings = self.position_embeddings(embeddings)

        logits, _ = self.cross_attention(embeddings, src_key_padding_mask=src_key_padding_mask)
        output = logits[torch.arange(logits.size(0)), last_idxs]

        if candidate_idxs is not None:
            candidate_embeddings = self.cross_attention.item_embeddings.weight[candidate_idxs]
            scores = torch.matmul(output, candidate_embeddings.T)
        else:
            candidate_embeddings = self.cross_attention.item_embeddings.weight[:-2]
            scores = torch.matmul(output, candidate_embeddings.T)

        return scores

    def training_step(self, batch: list[torch.Tensor], idx: int):
        # input_seqs shape: (256, 50)
        # src_key_padding_mask shape: (256, 50)
        # masked_idxs shape: (256, 2)
        # positive_idxs shape: (256, 2)
        # negative_idxs shape: (256, 2)
        input_seqs, src_key_padding_mask, masked_idxs, positive_idxs, negative_idxs = batch

        embeddings = self.item_embeddings(input_seqs)
        embeddings = self.dropout(embeddings)
        embeddings = self.position_embeddings(embeddings)

        # logits shape: (256, 50, 128)
        logits, _ = self.cross_attention(embeddings, src_key_padding_mask=src_key_padding_mask)
        # batch_indices shape: (256, 1)
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1).to(self.device)
        # output shape: (256, 2, 128)
        output = logits[batch_indices, masked_idxs]

        # positive_embeddings shape: (256, 2, 128)
        # negative_embeddings shape: (256, 2, 128)
        positive_embeddings = self.item_embeddings.weight[positive_idxs]
        negative_embeddings = self.item_embeddings.weight[negative_idxs]

        # positive_scores shape: (256, 2)
        # negative_scores shape: (256, 2)
        positive_scores = torch.sum(output * positive_embeddings, dim=-1)
        negative_scores = torch.sum(output * negative_embeddings, dim=-1)

        # BPR Loss 계산
        train_loss = self.bpr_loss(positive_scores, negative_scores)
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch: list[torch.Tensor], idx: int):
        input_seqs, src_key_padding_mask, last_idxs, ground_truth_items = batch

        embeddings = self.item_embeddings(input_seqs)
        embeddings = self.dropout(embeddings)
        embeddings = self.position_embeddings(embeddings)

        logits, _ = self.cross_attention(embeddings, src_key_padding_mask=src_key_padding_mask)
        output = logits[torch.arange(logits.size(0)), last_idxs, :]
        scores = torch.matmul(output, self.item_embeddings.weight[:-2].T)

        mrr = self.calc_mrr(scores, ground_truth_items)
        self.log("val_mrr", mrr, prog_bar=True)

        hr_5 = self.calc_hr_at_k(scores, ground_truth_items, k=5)
        ndcg_5 = self.calc_ndcg_at_k(scores, ground_truth_items, k=5)
        self.log(f"val_hr@5", hr_5, prog_bar=True)
        self.log(f"val_ndcg@5", ndcg_5, prog_bar=True)

        hr_10 = self.calc_hr_at_k(scores, ground_truth_items, k=10)
        ndcg_10 = self.calc_ndcg_at_k(scores, ground_truth_items, k=10)
        self.log(f"val_hr@10", hr_10, prog_bar=True)
        self.log(f"val_ndcg@10", ndcg_10, prog_bar=True)

        hr_20 = self.calc_hr_at_k(scores, ground_truth_items, k=20)
        ndcg_20 = self.calc_ndcg_at_k(scores, ground_truth_items, k=20)
        self.log(f"val_hr@20", hr_20, prog_bar=True)
        self.log(f"val_ndcg@20", ndcg_20, prog_bar=True)

    def test_step(self, batch: list[torch.Tensor], idx: int):
        self.validation_step(batch, idx)

    def configure_optimizers(self) -> Optimizer:
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def calc_mrr(self, scores: torch.Tensor, ground_truth_items: torch.Tensor):
        gt_scores = scores.gather(1, ground_truth_items.unsqueeze(1)).squeeze(1)
        ranks = (scores > gt_scores.unsqueeze(1)).sum(dim=1) + 1
        reciprocal_ranks = 1.0 / ranks.float()
        mrr = torch.mean(reciprocal_ranks).item()
        return mrr

    def calc_hr_at_k(self, scores: torch.Tensor, ground_truth_items: torch.Tensor, k: int):
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        hits = (topk_indices == ground_truth_items.unsqueeze(1)).any(dim=1).float()
        hit_rate = torch.mean(hits).item()
        return hit_rate

    def calc_ndcg_at_k(self, scores: torch.Tensor, ground_truth_items: torch.Tensor, k: int):
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        match_positions = (topk_indices == ground_truth_items.unsqueeze(1)).nonzero(as_tuple=False)
        dcg = torch.zeros(scores.size(0), device=scores.device)
        dcg[match_positions[:, 0]] = 1.0 / torch.log2(match_positions[:, 1].float() + 2)
        idcg = 1.0
        ndcg = torch.mean(dcg / idcg).item()
        return ndcg

    def bpr_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor):
        loss = -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))
        return loss

    def import_item_embeddings(self, item_embeddings: torch.Tensor):
        mask_embeddings = torch.zeros((1, item_embeddings.size(1))).to(item_embeddings.device)
        padding_embeddings = torch.zeros((1, item_embeddings.size(1))).to(item_embeddings.device)
        extended_embeddings = torch.cat([item_embeddings, mask_embeddings, padding_embeddings], dim=0)
        self.item_embeddings.weight.data.copy_(extended_embeddings)


class CarcaTrainDataset(Dataset):
    def __init__(self, volume: Volume, max_len: int = 50):
        self.histories = volume.migrate_user_histories()
        self.num_items = volume.vocab_size()
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1
        self.max_len = max_len
        self.num_masked = 10

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, idx: int):
        history_pidxs = self.histories[idx]
        seq_len = len(history_pidxs)
        pad_len = self.max_len - seq_len

        input_seqs = history_pidxs + ([self.pad_token_idx] * pad_len)
        input_seqs = torch.tensor(input_seqs, dtype=torch.long)
        padding_mask = input_seqs == self.pad_token_idx

        valid_positions = list(range(seq_len))
        if seq_len >= self.num_masked:
            masked_positions = random.sample(valid_positions, self.num_masked)
        else:
            masked_positions = random.choices(valid_positions, k=self.num_masked)

        masked_positions = torch.tensor(masked_positions, dtype=torch.long)

        positive_pidxs = input_seqs[masked_positions].clone()
        input_seqs[masked_positions] = self.mask_token_idx

        negative_pidxs = torch.randint(0, self.num_items, (len(masked_positions),), dtype=torch.long)
        for i in range(len(negative_pidxs)):
            while negative_pidxs[i] == positive_pidxs[i]:
                negative_pidxs[i] = torch.randint(0, self.num_items, ()).item()

        return input_seqs, padding_mask, masked_positions, positive_pidxs, negative_pidxs


class CarcaValidDataset(Dataset):
    def __init__(self, volume: Volume, max_len: int = 50):
        self.histories = volume.migrate_user_histories(condition="greater")
        self.num_items = volume.vocab_size()
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, idx: int) -> tuple:
        history_pidxs = self.histories[idx]
        pad_len = self.max_len - len(history_pidxs)

        input_seqs = history_pidxs + ([self.pad_token_idx] * pad_len)
        input_seqs = torch.tensor(input_seqs, dtype=torch.long)
        padding_mask = input_seqs == self.pad_token_idx

        last_idx = (~padding_mask).sum(dim=0) - 1

        ground_truth_item = input_seqs[last_idx].item()
        input_seqs[last_idx] = self.mask_token_idx

        return input_seqs, padding_mask, last_idx, ground_truth_item


class CarcaDataModule(pl.LightningDataModule):
    def __init__(self, volume: Volume, batch_size: int = 32, num_workers: int = 2, max_len: int = 50):
        super().__init__()
        self.volume = volume
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CarcaTrainDataset(volume=self.volume, max_len=self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=True,
            collate_fn=self.carca_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            CarcaValidDataset(volume=self.volume, max_len=self.max_len),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            CarcaValidDataset(volume=self.volume, max_len=self.max_len),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=False,
        )

    def carca_collate_fn(self, batch):
        """
        CARCA 데이터셋을 위한 collate_fn.

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
