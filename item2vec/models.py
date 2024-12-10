import pathlib

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import Optimizer
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class Item2Vec(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super(Item2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self):
        return self.embeddings.weight


class LightGCNConv(MessagePassing):
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr="sum")

    def forward(self, x, edge_index):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        col_deg = degree(col, x.size(0))
        col_deg_inv = col_deg.pow(-1)
        col_deg_inv[col_deg_inv == float("inf")] = 0.0
        col_norm = col_deg_inv[col]

        # 인기기반 row 처리
        # row_deg = degree(row, x.size(0))
        # row_min, row_max = row_deg.min(), row_deg.max()
        # row_deg_norm = 1 + (row_deg - row_min) * (2 - 1) / (row_max - row_min + 1e-9)
        # row_norm = row_deg_norm[row]

        # idf
        # row_deg = degree(row, x.size(0))
        # row_deg_inv = row_deg.pow(-1)
        # row_deg_inv[row_deg_inv == float('inf')] = 0.0
        # row_min, row_max = row_deg_inv.min(), row_deg_inv.max()
        # row_deg_norm = 1 + (row_deg_inv - row_min) * (2 - 1) / (row_max - row_min + 1e-9)
        # row_norm = row_deg_norm[row]

        # norm = col_norm * row_norm
        return self.propagate(edge_index, x=x, norm=col_norm)

    def message(self, x_j, norm):
        msg = norm.view(-1, 1) * x_j
        return msg


class GraphBPRItem2Vec(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        purchase_edge_index_path: pathlib.Path,
        embed_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        dropout: float = 0.0,
    ):
        super(GraphBPRItem2Vec, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        edge_df = pd.read_csv(purchase_edge_index_path.as_posix())
        sources, targets = edge_df["source"].values, edge_df["target"].values
        purchase_edge_index = torch.tensor([sources, targets], dtype=torch.long)
        self.register_buffer("purchase_edge_index", purchase_edge_index)

        self.item2vec = Item2Vec(self.vocab_size, embed_dim=self.embed_dim)
        self.conv = LightGCNConv()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def setup(self, stage=None):
        self.purchase_edge_index = self.purchase_edge_index.to(self.device)

    def forward(self):
        pass

    def dot_product(self, x, y):
        y = y.transpose(1, 2)
        scores = torch.bmm(x, y)
        scores = scores.squeeze(1)
        return scores

    def bpr_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        margins: torch.Tensor,
        boost_factor: float = 2.0,
    ) -> torch.Tensor:
        """
        Modified BPR Loss with dynamic margin adjustment for purchased samples.

        Args:
            pos_scores: Positive sample scores (S_pos)
            neg_scores: Negative sample scores (S_neg)
            margins: Tensor of margins (1 for purchase, 0 otherwise)
            boost_factor: Additional factor to amplify purchased samples

        Returns:
            Computed BPR loss as a tensor
        """
        base_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores - margins))
        boosted_loss = base_loss + (margins * boost_factor * (1 - torch.sigmoid(pos_scores - neg_scores)))
        return boosted_loss.mean()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        seq_focus_items, seq_pos_items, seq_margins, seq_neg_items = batch
        # seq_margins = 5 * seq_margins # val_graph_dot_ndcg@20=0.360
        seq_margins = 1 * seq_margins

        embeddings = self.get_graph_embeddings(num_layers=3)
        seq_focus_embeddings = embeddings[seq_focus_items]
        seq_pos_embeddings = embeddings[seq_pos_items]
        seq_neg_embeddings = embeddings[seq_neg_items]

        seq_pos_scores = self.dot_product(seq_focus_embeddings, seq_pos_embeddings)
        seq_neg_scores = self.dot_product(seq_focus_embeddings, seq_neg_embeddings)

        train_loss = self.bpr_loss(seq_pos_scores, seq_neg_scores, seq_margins)
        self.log("train_loss", train_loss, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        sources, labels = batch
        vanilla_embeddings = self.item2vec()
        graph_embeddings = self.get_graph_embeddings(num_layers=3)

        cos_ndcg = self.calc_cosine_ndcg(vanilla_embeddings, sources, labels, k=20)
        dot_ndcg = self.calc_dot_product_ndcg(vanilla_embeddings, sources, labels, k=20)
        graph_dot_ndcg = self.calc_dot_product_ndcg(graph_embeddings, sources, labels, k=20)
        graph_cos_ndcg = self.calc_cosine_ndcg(graph_embeddings, sources, labels, k=20)
        graph_dot_recall = self.calc_dot_product_ndcg(graph_embeddings, sources, labels, k=20)

        self.log("val_cos_ndcg@20", cos_ndcg.mean(), prog_bar=True, logger=True, sync_dist=True)
        self.log("val_dot_ndcg@20", dot_ndcg.mean(), prog_bar=True, logger=True, sync_dist=True)
        self.log("val_graph_dot_ndcg@20", graph_dot_ndcg.mean(), prog_bar=True, logger=True, sync_dist=True)
        self.log("val_graph_cos_ndcg@20", graph_cos_ndcg.mean(), prog_bar=True, logger=True, sync_dist=True)
        self.log("val_graph_dot_recall@20", graph_dot_recall.mean(), prog_bar=True, logger=True, sync_dist=True)

    def calc_cosine_ndcg(self, embeddings: torch.Tensor, sources: torch.Tensor, labels: torch.Tensor, k: int = 20):
        source_embeddings = embeddings[sources]
        all_scores = F.cosine_similarity(source_embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        _, top_indices = torch.topk(all_scores, k, dim=-1)
        top_indices = top_indices.squeeze(1)
        relevance = (top_indices == labels).float()
        gains = 1 / torch.log2(torch.arange(2, k + 2).float()).to(relevance.device)
        ndcg = (relevance * gains).sum(dim=-1)
        return ndcg

    def calc_dot_product_ndcg(self, embeddings: torch.Tensor, sources: torch.Tensor, labels: torch.Tensor, k: int = 20):
        source_embeddings = embeddings[sources]
        all_scores = torch.matmul(source_embeddings, embeddings.T)
        _, top_indices = torch.topk(all_scores, k, dim=-1)
        top_indices = top_indices.squeeze(1)
        relevance = (top_indices == labels).float()
        gains = 1 / torch.log2(torch.arange(2, k + 2).float()).to(relevance.device)
        ndcg = (relevance * gains).sum(dim=-1)
        return ndcg

    def configure_optimizers(self) -> Optimizer:
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_graph_embeddings(self, num_layers: int = 2) -> torch.Tensor:
        initial_embeddings = self.item2vec()
        x = initial_embeddings
        layers = [initial_embeddings]  # layers = []
        for _ in range(num_layers):
            weights = self.conv(x, self.purchase_edge_index)
            layers.append(weights)
            x = x + weights
        mean_weights = torch.mean(torch.stack(layers, dim=-1), dim=-1)  # mean_weights = self.layer_norm(mean_weights)
        return initial_embeddings + self.dropout(mean_weights)
