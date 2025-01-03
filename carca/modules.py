import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from carca.attention import CrossAttention
from carca.encoding import PositionalEncoding
from item2vec.volume import Volume


class CARCA(pl.LightningModule):

    def __init__(
        self,
        num_items: int,
        num_category1: int,
        num_category2: int,
        num_category3: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_len: int = 50,
        dropout: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        encoder_residual_strategy_1: str = "none",
        encoder_residual_strategy_2: str = "none",
        decoder_residual_strategy_1: str = "none",
        decoder_residual_strategy_2: str = "none",
    ):
        super(CARCA, self).__init__()
        self.num_items = num_items
        self.mask_token_idx = num_items + 0
        self.pad_token_idx = num_items + 1

        self.num_category1 = num_category1
        self.num_category2 = num_category2
        self.num_category3 = num_category3

        self.cat1_mask_token_idx = num_category1 + 0
        self.cat2_mask_token_idx = num_category2 + 0
        self.cat3_mask_token_idx = num_category3 + 0

        self.cat1_pad_token_idx = num_category1 + 1
        self.cat2_pad_token_idx = num_category2 + 1
        self.cat3_pad_token_idx = num_category3 + 1

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.item_embeddings = nn.Embedding(num_items + 2, embed_dim, padding_idx=self.mask_token_idx)

        # Category Embeddings
        self.category1_embeddings = nn.Embedding(num_category1 + 2, 4, padding_idx=self.cat1_mask_token_idx)
        self.category2_embeddings = nn.Embedding(num_category2 + 2, 4, padding_idx=self.cat1_mask_token_idx)
        self.category3_embeddings = nn.Embedding(num_category3 + 2, 4, padding_idx=self.cat1_mask_token_idx)

        self.dropout = nn.Dropout(dropout)
        self.position_embeddings = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        self.concat_linear = nn.Linear(embed_dim + (4 * 3), embed_dim)

        self.cross_attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
            encoder_residual_strategy_1=encoder_residual_strategy_1,
            encoder_residual_strategy_2=encoder_residual_strategy_2,
            decoder_residual_strategy_1=decoder_residual_strategy_1,
            decoder_residual_strategy_2=decoder_residual_strategy_2,
        )

    def forward(
        self,
        seq_pidxs: torch.Tensor,
        seq_cat1_cidxs: torch.Tensor,
        seq_cat2_cidxs: torch.Tensor,
        seq_cat3_cidxs: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        src_mask: torch.Tensor,
        candidate_pidxs: torch.Tensor,
    ):
        batch_indices = torch.arange(seq_pidxs.size(0)).unsqueeze(1).to(self.device)
        seq_pidxs[batch_indices, src_mask] = self.mask_token_idx
        seq_cat1_cidxs[batch_indices, src_mask] = self.cat1_mask_token_idx
        seq_cat2_cidxs[batch_indices, src_mask] = self.cat2_mask_token_idx
        seq_cat3_cidxs[batch_indices, src_mask] = self.cat3_mask_token_idx

        seq_item_emb = self.item_embeddings(seq_pidxs)  # (batch_size, seq_len, embed_dim)

        # category embeddings
        seq_cat1_emb = self.category1_embeddings(seq_cat1_cidxs)  # (batch_size, seq_len, embed_dim)
        seq_cat2_emb = self.category2_embeddings(seq_cat2_cidxs)  # (batch_size, seq_len, embed_dim)
        seq_cat3_emb = self.category3_embeddings(seq_cat3_cidxs)  # (batch_size, seq_len, embed_dim)

        # concat
        embeddings = torch.cat(tensors=[seq_item_emb, seq_cat1_emb, seq_cat2_emb, seq_cat3_emb], dim=-1)
        embeddings = self.concat_linear(embeddings)  # (batch_size, seq_len, embed_dim)
        embeddings = self.position_embeddings(embeddings)

        logits, _ = self.cross_attention(embeddings, src_key_padding_mask=src_key_padding_mask)
        output = logits[torch.arange(logits.size(0)), src_mask.squeeze(1), :]

        # output shape: torch.Size([batch_size, embed_dim])
        # candidate_item_embeddings shape: torch.Size([batch_size, num_candidate, embed_dim])
        candidate_item_embeddings = self.item_embeddings(candidate_pidxs)
        scores = torch.matmul(output.unsqueeze(1), candidate_item_embeddings.transpose(1, 2))
        scores = scores.squeeze(1)

        # Sort scores and candidates in descending order
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        sorted_candidates = torch.gather(candidate_pidxs, dim=-1, index=sorted_indices)

        return sorted_scores, sorted_candidates

    def training_step(self, batch: list[torch.Tensor], idx: int):
        # seq_pidxs shape: (256, 50)
        # category_cidxs shape: (256, 50)
        # src_key_padding_mask shape: (256, 50)
        # src_mask shape: (256, 2)
        # positive_idxs shape: (256, 2)
        # negative_idxs shape: (256, 2)

        (
            seq_pidxs,
            seq_cat1_cidxs,
            seq_cat2_cidxs,
            seq_cat3_cidxs,
            src_key_padding_mask,
            src_mask,
            positive_idxs,
            negative_idxs,
        ) = batch

        batch_indices = torch.arange(seq_pidxs.size(0)).unsqueeze(1).to(self.device)
        seq_pidxs[batch_indices, src_mask] = self.mask_token_idx
        seq_cat1_cidxs[batch_indices, src_mask] = self.cat1_mask_token_idx
        seq_cat2_cidxs[batch_indices, src_mask] = self.cat2_mask_token_idx
        seq_cat3_cidxs[batch_indices, src_mask] = self.cat3_mask_token_idx

        # item embeddings
        seq_item_emb = self.item_embeddings(seq_pidxs)  # (batch_size, seq_len, embed_dim)

        # category embeddings
        seq_cat1_emb = self.category1_embeddings(seq_cat1_cidxs)  # (batch_size, seq_len, embed_dim)
        seq_cat2_emb = self.category2_embeddings(seq_cat2_cidxs)  # (batch_size, seq_len, embed_dim)
        seq_cat3_emb = self.category3_embeddings(seq_cat3_cidxs)  # (batch_size, seq_len, embed_dim)

        # concat
        embeddings = torch.cat([seq_item_emb, seq_cat1_emb, seq_cat2_emb, seq_cat3_emb], dim=-1)
        embeddings = self.concat_linear(embeddings)  # (batch_size, seq_len, embed_dim)
        embeddings = self.dropout(embeddings)
        embeddings = self.position_embeddings(embeddings, combine=True)

        # logits shape: (256, 50, 128)
        logits, _ = self.cross_attention(embeddings, src_key_padding_mask=src_key_padding_mask)
        # batch_indices shape: (256, 1)
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1).to(self.device)
        # output shape: (256, 2, 128)
        output = logits[batch_indices, src_mask]

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
        self.log("train_loss", train_loss, prog_bar=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch: list[torch.Tensor], idx: int):
        seq_pidxs, seq_cat1_cidxs, seq_cat2_cidxs, seq_cat3_cidxs, src_key_padding_mask, src_mask, labeled_pidxs = batch

        batch_indices = torch.arange(seq_pidxs.size(0)).unsqueeze(1).to(self.device)
        seq_pidxs[batch_indices, src_mask] = self.mask_token_idx
        seq_cat1_cidxs[batch_indices, src_mask] = self.cat1_mask_token_idx
        seq_cat2_cidxs[batch_indices, src_mask] = self.cat2_mask_token_idx
        seq_cat3_cidxs[batch_indices, src_mask] = self.cat3_mask_token_idx

        # item embeddings
        seq_item_emb = self.item_embeddings(seq_pidxs)  # (batch_size, seq_len, embed_dim)

        # category embeddings
        seq_cat1_emb = self.category1_embeddings(seq_cat1_cidxs)  # (batch_size, seq_len, embed_dim)
        seq_cat2_emb = self.category2_embeddings(seq_cat2_cidxs)  # (batch_size, seq_len, embed_dim)
        seq_cat3_emb = self.category3_embeddings(seq_cat3_cidxs)  # (batch_size, seq_len, embed_dim)

        # concat
        embeddings = torch.cat(tensors=[seq_item_emb, seq_cat1_emb, seq_cat2_emb, seq_cat3_emb], dim=-1)
        embeddings = self.concat_linear(embeddings)  # (batch_size, seq_len, embed_dim)
        # embeddings = self.dropout(embeddings)
        embeddings = self.position_embeddings(embeddings)

        logits, _ = self.cross_attention(embeddings, src_key_padding_mask=src_key_padding_mask)
        output = logits[torch.arange(logits.size(0)), src_mask.squeeze(-1), :]
        scores = torch.matmul(output, self.item_embeddings.weight[:-2].T)

        mrr = self.calc_mrr(scores, labeled_pidxs)
        self.log("val_mrr", mrr, prog_bar=True, sync_dist=True)

        hr_5 = self.calc_hr_at_k(scores, labeled_pidxs, k=5)
        ndcg_5 = self.calc_ndcg_at_k(scores, labeled_pidxs, k=5)
        self.log(f"val_hr@5", hr_5, prog_bar=True, sync_dist=True)
        self.log(f"val_ndcg@5", ndcg_5, prog_bar=True, sync_dist=True)

        hr_10 = self.calc_hr_at_k(scores, labeled_pidxs, k=10)
        ndcg_10 = self.calc_ndcg_at_k(scores, labeled_pidxs, k=10)
        self.log(f"val_hr@10", hr_10, prog_bar=True, sync_dist=True)
        self.log(f"val_ndcg@10", ndcg_10, prog_bar=True, sync_dist=True)

        hr_20 = self.calc_hr_at_k(scores, labeled_pidxs, k=20)
        ndcg_20 = self.calc_ndcg_at_k(scores, labeled_pidxs, k=20)
        self.log(f"val_hr@20", hr_20, prog_bar=True, sync_dist=True)
        self.log(f"val_ndcg@20", ndcg_20, prog_bar=True, sync_dist=True)

    def test_step(self, batch: list[torch.Tensor], idx: int):
        self.validation_step(batch, idx)

    def configure_optimizers(self) -> Optimizer:
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def calc_mrr(self, scores: torch.Tensor, labeled_pidxs: torch.Tensor):
        gt_scores = scores.gather(1, labeled_pidxs.unsqueeze(1)).squeeze(1)
        ranks = (scores > gt_scores.unsqueeze(1)).sum(dim=1) + 1
        reciprocal_ranks = 1.0 / ranks.float()
        mrr = torch.mean(reciprocal_ranks).item()
        return mrr

    def calc_hr_at_k(self, scores: torch.Tensor, labeled_pidxs: torch.Tensor, k: int):
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        hits = (topk_indices == labeled_pidxs.unsqueeze(1)).any(dim=1).float()
        hit_rate = torch.mean(hits).item()
        return hit_rate

    def calc_ndcg_at_k(self, scores: torch.Tensor, labeled_pidxs: torch.Tensor, k: int):
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        match_positions = (topk_indices == labeled_pidxs.unsqueeze(1)).nonzero(as_tuple=False)
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
        self.volume = volume
        self.num_items = volume.vocab_size()
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1

        self.num_category1, self.num_category2, self.num_category3 = volume.count_categories()

        self.cat1_mask_token_idx = self.num_category1 + 0
        self.cat2_mask_token_idx = self.num_category2 + 0
        self.cat3_mask_token_idx = self.num_category3 + 0

        self.cat1_pad_token_idx = self.num_category1 + 1
        self.cat2_pad_token_idx = self.num_category2 + 1
        self.cat3_pad_token_idx = self.num_category3 + 1

        self.max_len = max_len
        self.num_masked = 10

    def __len__(self) -> int:
        return self.volume.get_user_history_count(condition="training")

    def __getitem__(self, idx: int):
        seq_pidxs, cat1_cidxs, cat2_cidxs, cat3_cidxs = self.volume.get_user_history(idx + 1, condition="training")
        seq_len = len(seq_pidxs)
        pad_len = self.max_len - seq_len

        seq_pidxs = seq_pidxs + ([self.pad_token_idx] * pad_len)
        seq_pidxs = torch.tensor(seq_pidxs, dtype=torch.long)

        cat1_cidxs = cat1_cidxs + ([self.cat1_pad_token_idx] * pad_len)
        cat1_cidxs = torch.tensor(cat1_cidxs, dtype=torch.long)

        cat2_cidxs = cat2_cidxs + ([self.cat2_pad_token_idx] * pad_len)
        cat2_cidxs = torch.tensor(cat2_cidxs, dtype=torch.long)

        cat3_cidxs = cat3_cidxs + ([self.cat3_pad_token_idx] * pad_len)
        cat3_cidxs = torch.tensor(cat3_cidxs, dtype=torch.long)

        src_key_padding_mask = seq_pidxs == self.pad_token_idx

        valid_positions = torch.arange(seq_len).tolist()
        src_mask = random.choices(valid_positions, k=min(seq_len // 2, self.num_masked))
        src_mask = torch.tensor(src_mask, dtype=torch.long)

        missing_count = self.num_masked - len(src_mask)
        if missing_count > 0:
            additional_values = random.choices(src_mask.tolist(), k=missing_count)
            src_mask = torch.cat([src_mask, torch.tensor(additional_values, dtype=torch.long)])

        positive_pidxs = seq_pidxs[src_mask].clone()
        negative_pidxs = torch.randint(0, self.num_items, (len(src_mask),), dtype=torch.long)
        for i in range(len(negative_pidxs)):
            while negative_pidxs[i] == positive_pidxs[i]:
                negative_pidxs[i] = torch.randint(0, self.num_items, ()).item()

        return (
            seq_pidxs,
            cat1_cidxs,
            cat2_cidxs,
            cat3_cidxs,
            src_key_padding_mask,
            src_mask,
            positive_pidxs,
            negative_pidxs,
        )


class CarcaValidDataset(Dataset):
    def __init__(self, volume: Volume, max_len: int = 50):
        self.volume = volume

        self.num_items = volume.vocab_size()
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1

        self.num_category1, self.num_category2, self.num_category3 = volume.count_categories()

        self.cat1_mask_token_idx = self.num_category1 + 0
        self.cat2_mask_token_idx = self.num_category2 + 0
        self.cat3_mask_token_idx = self.num_category3 + 0

        self.cat1_pad_token_idx = self.num_category1 + 1
        self.cat2_pad_token_idx = self.num_category2 + 1
        self.cat3_pad_token_idx = self.num_category3 + 1

        self.max_len = max_len

    def __len__(self) -> int:
        return self.volume.get_user_history_count(condition="test")

    def __getitem__(self, idx: int) -> tuple:
        seq_pidxs, cat1_cidxs, cat2_cidxs, cat3_cidxs = self.volume.get_user_history(idx + 1, condition="test")
        seq_len = len(seq_pidxs)
        pad_len = self.max_len - len(seq_pidxs)

        seq_pidxs = seq_pidxs + ([self.pad_token_idx] * pad_len)
        seq_pidxs = torch.tensor(seq_pidxs, dtype=torch.long)

        cat1_cidxs = cat1_cidxs + ([self.cat1_pad_token_idx] * pad_len)
        cat1_cidxs = torch.tensor(cat1_cidxs, dtype=torch.long)

        cat2_cidxs = cat2_cidxs + ([self.cat2_pad_token_idx] * pad_len)
        cat2_cidxs = torch.tensor(cat2_cidxs, dtype=torch.long)

        cat3_cidxs = cat3_cidxs + ([self.cat3_pad_token_idx] * pad_len)
        cat3_cidxs = torch.tensor(cat3_cidxs, dtype=torch.long)

        src_key_padding_mask = seq_pidxs == self.pad_token_idx
        src_mask = torch.tensor([seq_len - 1], dtype=torch.long)

        labeled_pidxs = seq_pidxs[src_mask].item()

        return (
            seq_pidxs,
            cat1_cidxs,
            cat2_cidxs,
            cat3_cidxs,
            src_key_padding_mask,
            src_mask,
            labeled_pidxs,
        )


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
