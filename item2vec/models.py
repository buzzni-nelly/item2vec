import pathlib

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


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
        embedding_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super(Item2VecModule, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.item2vec = Item2Vec(vocab_size, embedding_dim)
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


class BPRItem2VecModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super(BPRItem2VecModule, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.item2vec = Item2Vec(vocab_size, embedding_dim)

    def forward(self, focus_items, positive_items, negative_items):
        pos_scores = self.item2vec(focus_items, positive_items)
        neg_scores = self.item2vec(focus_items, negative_items)
        return pos_scores, neg_scores

    def bpr_loss(self, pos_scores, neg_scores):
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    def training_step(self, batch, batch_idx):
        focus_items, positive_items, negative_items = batch
        pos_scores, neg_scores = self.forward(
            focus_items, positive_items, negative_items
        )
        loss = self.bpr_loss(pos_scores, neg_scores)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class LightGCNConv(MessagePassing):
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr="mean")  # 평균 집계 방식

    def forward(self, x, edge_index) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class GraphItem2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        edge_index: torch.Tensor,
        embedding_dim: int = 128,
        num_layers: int = 2,
    ):
        super(GraphItem2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.edge_index = edge_index
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

    def forward(self, items, samples) -> torch.Tensor:
        embeddings = self.get_graph_embeddings()
        item_embeddings = embeddings[items]
        sample_embeddings = embeddings[samples]

        sample_embeddings = sample_embeddings.transpose(1, 2)
        scores = torch.bmm(item_embeddings, sample_embeddings)
        scores = scores.squeeze(1)
        return scores

    def get_graph_embeddings(self) -> torch.Tensor:
        x = self.embeddings.weight

        all_embeddings = [x]
        for conv in self.convs:
            x = conv(x, self.edge_index)
            all_embeddings.append(x)

        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        return final_embeddings


class GraphBPRItem2VecModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        edge_index_path: pathlib.Path,
        embed_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super(GraphBPRItem2VecModule, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.vocab_size = vocab_size
        self.embedding_dim = embed_dim
        self._edge_index_path = edge_index_path

        self.item2vec = None

    def setup(self, stage: str = None):
        edge_df = pd.read_csv(self._edge_index_path.as_posix())
        edge_index = torch.tensor(
            [edge_df["source"].values, edge_df["target"].values], dtype=torch.long
        )
        edge_index = edge_index.to(self.device)
        self.item2vec = GraphItem2Vec(
            self.vocab_size, edge_index, embedding_dim=self.embedding_dim
        )

    def forward(self, focus_items, positive_items, negative_items) -> tuple[torch.Tensor, torch.Tensor]:
        pos_scores = self.item2vec(focus_items, positive_items)
        neg_scores = self.item2vec(focus_items, negative_items)
        return pos_scores, neg_scores

    def bpr_loss(self, pos_scores, neg_scores) -> torch.Tensor:
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        focus_items, positive_items, negative_items = batch
        pos_scores, neg_scores = self.forward(
            focus_items, positive_items, negative_items
        )
        loss = self.bpr_loss(pos_scores, neg_scores)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def get_graph_embeddings(self) -> torch.Tensor:
        return self.item2vec.get_graph_embeddings()


