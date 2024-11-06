import random
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

from item2vec import vocab


class SkipGramDataset(Dataset):
    def __init__(self, pair_path: Path, negative_k: int = 9):
        self.negative_k = negative_k
        self.pair_path = pair_path

        csv_path = self.pair_path.as_posix()
        pairs_df = pd.read_csv(csv_path)
        self.pairs = pairs_df.to_numpy()

        self.item_ids = vocab.pids()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        target, positive = self.pairs[idx]
        negatives = random.sample(self.item_ids, self.negative_k)
        target_tensor = torch.LongTensor([target])
        samples_tensor = torch.LongTensor([positive, *negatives])
        labels_tensor = torch.FloatTensor([1] + [0] * self.negative_k)
        return target_tensor, samples_tensor, labels_tensor


class SkipGramDataModule(LightningDataModule):
    def __init__(
        self,
        pair_paths: list[Path],
        item_path: Path,
        vocab_size: int,
        batch_size: int = 512,
        num_workers: int = 8,
        negative_k: int = 9,
    ):
        super().__init__()
        self.train_dataset = None
        self.item_path = item_path
        self.pair_paths = pair_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negative_k = negative_k
        self.vocab_size = vocab_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            datasets = self.load_datasets()
            self.train_dataset = ConcatDataset(datasets)

    def load_datasets(self):
        datasets = []
        total_count = 0
        iteration = tqdm(self.pair_paths, desc="Loading datasets...")

        for path in iteration:
            dataset = SkipGramDataset(path, negative_k=self.negative_k)
            total_count += len(dataset)
            datasets.append(dataset)
            iteration.set_postfix(total_count=total_count)

        return datasets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=True,
        )


class SkipGramBPRDataset(Dataset):
    def __init__(self, pair_path: Path, negative_k: int = 10):
        self.negative_k = negative_k
        self.pair_path = pair_path

        csv_path = self.pair_path.as_posix()
        pairs_df = pd.read_csv(csv_path)
        self.pairs = pairs_df.to_numpy()

        self.item_ids = vocab.pids()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        target, positive = self.pairs[idx]
        negatives = random.sample(self.item_ids, self.negative_k)

        # BPR Loss를 위해 필요한 텐서들
        target_tensor = torch.LongTensor([target])
        positive_tensor = torch.LongTensor([positive] * self.negative_k)
        negative_tensor = torch.LongTensor(negatives)

        # BPR에서는 positive와 negative를 따로 반환합니다.
        return target_tensor, positive_tensor, negative_tensor


class SkipGramBPRDataModule(LightningDataModule):
    def __init__(
            self,
            pair_paths: list[Path],
            item_path: Path,
            vocab_size: int,
            batch_size: int = 512,
            num_workers: int = 8,
            negative_k: int = 9,
    ):
        super().__init__()
        self.train_dataset = None
        self.item_path = item_path
        self.pair_paths = pair_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negative_k = negative_k
        self.vocab_size = vocab_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            datasets = self.load_datasets()
            self.train_dataset = ConcatDataset(datasets)

    def load_datasets(self):
        datasets = []
        total_count = 0
        iteration = tqdm(self.pair_paths, desc="Loading datasets...")

        for path in iteration:
            dataset = SkipGramBPRDataset(path, negative_k=self.negative_k)
            total_count += len(dataset)
            datasets.append(dataset)
            iteration.set_postfix(total_count=total_count)

        return datasets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=True,
        )
