import json
import random
from abc import ABC
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from item2vec import vocab


class SkipGramDataset(IterableDataset, ABC):

    size: int | None = None

    def __init__(
        self,
        pairs_paths: list[Path],
        negative_k: int = 9,
    ):
        self.negative_k = negative_k
        self.pairs_paths = pairs_paths
        self.size = None

        self.item_ids = vocab.pids()

    def __iter__(self):
        random.shuffle(self.pairs_paths)
        for pair_path in self.pairs_paths:
            with open(pair_path, "r") as pairs:
                for pair in pairs:
                    target, positive = json.loads(pair)
                    negatives = random.sample(self.item_ids, self.negative_k)
                    target = torch.LongTensor([target])
                    samples = torch.LongTensor([positive, *negatives])
                    labels = [1] + [0] * self.negative_k
                    labels = torch.FloatTensor(labels)
                    yield target, samples, labels

    def __len__(self) -> int:
        if self.size:
            print(f"Dataset size is {self.size:,}")
            return self.size
        size = 0
        for pair_path in tqdm(self.pairs_paths, desc="Counting dataset size.."):
            with open(pair_path, "r") as pairs:
                size += sum(1 for _ in pairs)
        self.size = size
        print(f"Dataset size is {self.size:,}")
        return size


class SkipGramDataModule(LightningDataModule):
    def __init__(
        self,
        pair_paths: list[Path],
        item_path: Path,
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
        self.vocab_size = vocab.size()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_datasets()

    def load_datasets(self):
        return SkipGramDataset(
            pairs_paths=self.pair_paths[:],
            negative_k=self.negative_k,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # persistent_workers=True,
        )
