import os

from pytorch_lightning.callbacks import ModelCheckpoint

from item2vec.configs import settings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import pytorch_lightning as pl
from torch_geometric.data import Dataset, DataLoader
import pandas as pd

import random

from lightgcn.volume import Volume


# LightGCNConv 클래스 정의
class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")  # 합산 집계

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# LightGCN 모델 정의
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers: int = 1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(self.num_users + self.num_items, embedding_dim)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, edge_index):
        x = self.embeddings.weight
        all_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)
        embeddings = torch.stack(all_embeddings, dim=0).mean(0)
        return embeddings[: self.num_users], embeddings[self.num_users :]


class LightGCNModel(pl.LightningModule):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, edge_index):
        super().__init__()
        self.model = LightGCN(num_users, num_items, embedding_dim, num_layers)
        self.edge_index = edge_index
        self.num_users = num_users
        self.num_items = num_items

    def setup(self, stage: str) -> None:
        self.edge_index = self.edge_index.to(self.device)

    def forward(self):
        user_embeddings, item_embeddings = self.model(self.edge_index)
        return user_embeddings, item_embeddings

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        user_embeddings, item_embeddings = self.model(self.edge_index)

        user_embeddings = user_embeddings[users]  # [1024, 10, 64]
        pos_item_embeddings = item_embeddings[pos_items]  # [1024, 10, 64]
        neg_item_embeddings = item_embeddings[neg_items]  # [1024, 10, 64]

        pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=-1)  # [1024, 10]
        neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=-1)  # [1024, 10]

        loss = torch.nn.functional.softplus(neg_scores - pos_scores).mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class VolumeDataset(Dataset):
    def __init__(self, volume: Volume):
        super(VolumeDataset, self).__init__()
        self.num_users = volume.count_users()
        self.num_items = volume.count_items()
        self.traces = volume.list_traces()

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        user_id, positive_item_id = trace["user_id"], trace["item_id"]
        user_ids = [user_id] * 10
        positive_item_ids = [positive_item_id] * 10
        negative_item_ids = random.choices(range(0, self.num_items - 1), k=10)
        return (
            torch.LongTensor(user_ids),
            torch.LongTensor(positive_item_ids),
            torch.LongTensor(negative_item_ids),
        )


# Models
EMBED_DIM = settings.embed_dim

# Optimizers and training envs
LR = settings.lr
WEIGHT_DECAY = settings.weight_decay

# Trainers
TRAINER_STRATEGY = settings.trainer_strategy
TRAINER_PRECISION = settings.trainer_precision
TRAINER_LIMIT_TRAIN_BATCHES = settings.trainer_limit_train_batches
TRAINER_LIMIT_VAL_BATCHES = settings.trainer_limit_val_batches
TRAINER_LIMIT_TEST_BATCHES = settings.trainer_limit_test_batches
TRAINER_MAX_EPOCHS = settings.trainer_max_epochs
TRAINER_PROFILER = settings.trainer_profiler

# DataModules
DATAMODULE_BATCH_SIZE = settings.datamodule_batch_size
DATAMODULE_NUM_WORKERS = settings.datamodule_num_workers
DATAMODULE_NEGATIVE_K = settings.datamodule_negative_k

# Checkpoints
CHECKPOINT_DIRPATH = settings.checkpoint_dirpath
CHECKPOINT_MONITOR = settings.checkpoint_monitor
CHECKPOINT_MODE = settings.checkpoint_mode
CHECKPOINT_EVERY_N_TRAIN_STEPS = settings.checkpoint_every_n_train_steps
CHECKPOINT_FILENAME = settings.checkpoint_filename
CKPT_PATH = settings.checkpoint_path

# Wandb
WANDB_CONFIG = settings.dict()

# 메인 코드
if __name__ == "__main__":
    # Volume 클래스 초기화 및 데이터 생성
    volume = Volume(site="aboutpet", model="lightgcn", version="v1")
    # volume.migrate_traces(start_date=datetime(2024, 8, 1))
    # volume.migrate_items()
    # volume.migrate_users()
    # volume.generate_edge_indices_csv()

    num_users = volume.count_users()
    num_items = volume.count_items()

    click_edge_path = os.path.join(volume.workspace_path, "edge.click.indices.csv")
    edge_df = pd.read_csv(click_edge_path)
    sources = edge_df["source"].values
    targets = edge_df["target"].values
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    dataset = VolumeDataset(volume)
    data_loader = DataLoader(dataset, batch_size=4000, shuffle=True)

    # 모델 초기화
    embedding_dim = 64
    num_layers = 3
    model = LightGCNModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        edge_index=edge_index,
    )

    # 트레이너 설정 및 학습 시작
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIRPATH,
                monitor="train_loss",
                mode="min",
                every_n_train_steps=CHECKPOINT_EVERY_N_TRAIN_STEPS,
                filename=CHECKPOINT_FILENAME,
                save_last=True,
            ),
            # EarlyStopping(
            #     monitor=CHECKPOINT_MONITOR,
            #     mode=CHECKPOINT_MODE,
            #     patience=5,
            #     verbose=True
            # )
        ],
    )
    trainer.fit(model, data_loader)
