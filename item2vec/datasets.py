import random

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from item2vec.volume import Volume


class SkipGramBPRTrainDataset(Dataset):
    def __init__(self, volume: Volume, negative_k: int = 10):
        self.volume = volume
        self.idxs = volume.pidxs()
        self.negative_k = negative_k

    def __len__(self) -> int:
        return self.volume.count_skip_grams()

    def __getitem__(self, idx):
        x = self.volume.get_skip_gram(idx + 1)
        sources, targets, margins = x.source_pidx, x.target_pidx, x.is_purchased
        negatives = random.sample(self.idxs, self.negative_k)

        target_tensor = torch.LongTensor([sources])
        positive_tensor = torch.LongTensor([targets] * self.negative_k)
        negative_tensor = torch.LongTensor(negatives)
        margin_tensor = torch.LongTensor([margins] * self.negative_k)
        return (
            target_tensor,
            positive_tensor,
            margin_tensor,
            negative_tensor,
        )


class SkipGramBPRValidDataset(Dataset):
    def __init__(self, volume: Volume):
        sequences = volume.list_click2purchase_sequences()
        self.pairs = [(x.source_pidx, x.target_pidx) for x in sequences]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        source_tensor = torch.LongTensor([source])
        target_tensor = torch.LongTensor([target])
        return source_tensor, target_tensor


class SkipGramBPRDataModule(LightningDataModule):
    def __init__(
        self,
        volume: Volume,
        batch_size: int = 128,
        num_workers: int = 8,
        negative_k: int = 9,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negative_k = negative_k
        self.volume = volume
        self.vocab_size = volume.vocab_size()

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SkipGramBPRTrainDataset(volume=self.volume, negative_k=self.negative_k)
        elif stage == "valid":
            self.valid_dataset = SkipGramBPRValidDataset(volume=self.volume)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            SkipGramBPRValidDataset(volume=self.volume),
            batch_size=self.batch_size // 2,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            pin_memory=True,
            shuffle=False,
        )
