import json
import os
import random
from abc import ABC
from multiprocessing import Pool
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset, Dataset, ConcatDataset
from tqdm import tqdm

from item2vec import vocab


class SkipGramIterableDataset(IterableDataset, ABC):

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
            pairs = []
            with open(pair_path, "r") as lines:
                for pair in lines:
                    pairs.append(json.loads(pair))
            for target, positive in pairs:
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


class SkipGramDataset(Dataset):
    def __init__(
        self,
        pair_path: Path,
        negative_k: int = 9,
    ):
        self.negative_k = negative_k
        self.pairs_path = pair_path

        self.item_ids = vocab.pids()

        with open(self.pairs_path, "r") as p:
            pairs = json.load(p)
            self.pairs = list(map(tuple, pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        target, positive = self.pairs[idx]
        negatives = random.sample(self.item_ids, self.negative_k)
        target = torch.LongTensor([target])
        samples = torch.LongTensor([positive, *negatives])
        labels = [1] + [0] * self.negative_k
        labels = torch.FloatTensor(labels)
        return target, samples, labels


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
            datasets = self.load_datasets()
            self.train_dataset = ConcatDataset(datasets)

    def _load(self, path):
        return SkipGramDataset(path, negative_k=self.negative_k)

    def load_datasets(self):
        worker_number = os.cpu_count() // 2
        with Pool(worker_number) as pool:
            iteration = tqdm(
                pool.imap(self._load, self.pair_paths),
                total=len(self.pair_paths),
                desc="Loading datasets...",
            )
            datasets = list(iteration)

        total_count = sum(len(x) for x in datasets)
        print(f"Total items loaded: {total_count}")
        return datasets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
        )
