import pathlib

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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
        sequential_edge_index: torch.Tensor,
        embed_dim: int = 128,
        num_layers: int = 2,
    ):
        super(GraphItem2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.sequential_edge_index = sequential_edge_index
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
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
            x = conv(x, self.sequential_edge_index)
            all_embeddings.append(x)

        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        return final_embeddings

    def get_similar_pids(self, pids: list[int], k: int = 10, largest: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.get_graph_embeddings()  # [16070, 128]
        pids = torch.tensor(pids, device=embeddings.device)
        pid_embeddings = embeddings[pids]  # [len(pids), 128]
        scores = F.cosine_similarity(embeddings.unsqueeze(0), pid_embeddings.unsqueeze(1), dim=-1)
        similarities, indices = torch.topk(scores, k, dim=-1, largest=largest)
        return similarities, indices


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
        self.embed_dim = embed_dim

        edge_df = pd.read_csv(edge_index_path.as_posix())
        sources = edge_df["source"].values
        targets = edge_df["target"].values
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        self.register_buffer("edge_index", edge_index)
        self.item2vec = GraphItem2Vec(
            self.vocab_size, self.edge_index, embed_dim=self.embed_dim
        )

    def setup(self, stage=None):
        self.item2vec.sequential_edge_index = self.item2vec.sequential_edge_index.to(self.device)

    def forward(
        self, focus_items, positive_items, negative_items
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        sources, labels = batch
        cos_ndcg = self.calc_cosine_ndcg(sources, labels, k=20)
        dot_ndcg = self.calc_dotproduct_ndcg(sources, labels, k=20)
        self.log("val_cos_ndcg@20", cos_ndcg.mean(), prog_bar=True, logger=True)
        self.log("val_dot_ndcg@20", dot_ndcg.mean(), prog_bar=True, logger=True)

    def calc_cosine_ndcg(self, sources: torch.Tensor, labels: torch.Tensor, k: int = 20):
        all_embeddings = self.get_graph_embeddings()
        source_embeddings = all_embeddings[sources]

        all_scores = F.cosine_similarity(source_embeddings.unsqueeze(1), all_embeddings.unsqueeze(0), dim=-1)

        _, top_indices = torch.topk(all_scores, k, dim=-1)
        top_indices = top_indices.squeeze(1)
        relevance = (top_indices == labels).float()
        gains = 1 / torch.log2(torch.arange(2, k + 2).float()).to(relevance.device)
        ndcg = (relevance * gains).sum(dim=-1)
        return ndcg

    def calc_dotproduct_ndcg(self, sources: torch.Tensor, labels: torch.Tensor, k: int = 20):
        all_embeddings = self.get_graph_embeddings()
        source_embeddings = all_embeddings[sources]

        all_scores = torch.matmul(source_embeddings, all_embeddings.T)

        _, top_indices = torch.topk(all_scores, k, dim=-1)
        top_indices = top_indices.squeeze(1)
        relevance = (top_indices == labels).float()
        gains = 1 / torch.log2(torch.arange(2, k + 2).float()).to(relevance.device)
        ndcg = (relevance * gains).sum(dim=-1)
        return ndcg

    def configure_optimizers(self) -> Optimizer:
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def get_graph_embeddings(self) -> torch.Tensor:
        return self.item2vec.get_graph_embeddings()
