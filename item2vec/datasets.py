import random
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from item2vec.volume import Volume


class SkipGramBPRDataset(Dataset):
    def __init__(self, pair_path: Path, volume: Volume, negative_k: int = 10):
        self.negative_k = negative_k
        self.pair_path = pair_path

        csv_path = self.pair_path.as_posix()
        pairs_df = pd.read_csv(csv_path)
        self.pairs = pairs_df.to_numpy()

        self.item_ids = volume.pids()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        target, positive = self.pairs[idx]
        negatives = random.sample(self.item_ids, self.negative_k)
        target_tensor = torch.LongTensor([target])
        positive_tensor = torch.LongTensor([positive] * self.negative_k)
        negative_tensor = torch.LongTensor(negatives)
        return target_tensor, positive_tensor, negative_tensor


class SkipGramBPRDataModule(LightningDataModule):
    def __init__(
        self,
        volume: Volume,
        batch_size: int = 512,
        num_workers: int = 8,
        negative_k: int = 9,
    ):
        super().__init__()
        self.train_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negative_k = negative_k
        self.volume = volume
        self.vocab_size = volume.vocab_size()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_datasets()

    def load_datasets(self):
        pair_path = self.volume.workspace_path.joinpath("item.pairs.csv")
        return SkipGramBPRDataset(pair_path, volume=self.volume, negative_k=self.negative_k)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=True,
        )
