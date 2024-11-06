import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch_geometric.nn import GCNConv


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


class GraphItem2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super(GraphItem2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # 아이템 임베딩
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

        # GNN 레이어 추가
        self.gnn = GCNConv(embedding_dim, embedding_dim)

    def forward(self, items, samples, edge_index):
        node_embeddings = self.embeddings.weight  # shape: (vocab_size, embedding_dim)

        updated_node_embeddings = self.gnn(node_embeddings, edge_index)

        item_embeddings = updated_node_embeddings[items]       # shape: (batch_size, embedding_dim)
        sample_embeddings = updated_node_embeddings[samples]   # shape: (batch_size, num_samples, embedding_dim)

        sample_embeddings = sample_embeddings.transpose(1, 2)  # shape: (batch_size, embedding_dim, num_samples)
        scores = torch.bmm(item_embeddings.unsqueeze(1), sample_embeddings)  # shape: (batch_size, 1, num_samples)
        scores = scores.squeeze(1)  # shape: (batch_size, num_samples)
        return scores


class GraphItem2VecBPRModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        edge_index,
        embed_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super(GraphItem2VecBPRModule, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.edge_index = edge_index  # edge_index를 저장합니다.
        self.item2vec = GraphItem2Vec(vocab_size, embed_dim)

    def forward(self, focus_items, positive_items, negative_items):
        pos_scores = self.item2vec(focus_items, positive_items, self.edge_index)
        neg_scores = self.item2vec(focus_items, negative_items, self.edge_index)
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
