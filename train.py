import glob
import os
from pathlib import Path

import clients
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import directories
from configs import settings
from item2vec.dataset import SkipGramDataModule
from item2vec.model import Item2VecModule


os.environ["WANDB_API_KEY"] = settings.wandb_api_key

# Models
EMBED_DIM = settings.embed_dim

# Optimizers and training envs
LR = settings.lr
WEIGHT_DECAY = settings.weight_decay
DROPOUT = settings.dropout

# Trainers
TRAINER_STRATEGY = settings.trainer_strategy
TRAINER_PRECISION = settings.trainer_precision
TRAINER_LIMIT_TRAIN_BATCHES = settings.trainer_limit_train_batches
TRAINER_LIMIT_VAL_BATCHES = settings.trainer_limit_val_batches
TRAINER_LIMIT_TEST_BATCHES = settings.trainer_limit_test_batches
TRAINER_MAX_EPOCHS = settings.trainer_max_epochs

# DataModules
DATAMODULE_BATCH_SIZE = settings.datamodule_batch_size
DATAMODULE_K = settings.datamodule_k
DATAMODULE_NUM_WORKERS = settings.datamodule_num_workers

# Checkpoints
CHECKPOINT_DIRPATH = settings.checkpoint_dirpath
CHECKPOINT_MONITOR = settings.checkpoint_monitor
CHECKPOINT_MODE = settings.checkpoint_mode
CHECKPOINT_EVERY_N_TRAIN_STEPS = settings.checkpoint_every_n_train_steps
CKPT_PATH = settings.checkpoint_path

# Wandb
WANDB_CONFIG = settings.dict()


def download_files(
    source_path: str,
    always: bool = False,
):

    filepath = directories.data.joinpath("user_items_*.pairs.jsonl")
    filepath = filepath.as_posix()
    pair_paths = list(map(Path, glob.glob(filepath)))

    if always or len(pair_paths) == 0:
        ceph = clients.ceph.CephClient()
        destination_path = directories.data.as_posix()
        ceph.download_dir(source_path, destination_path)


def main():
    download_files(
        source_path=settings.download_source_path,
        always=False,
    )

    filepath = directories.data.joinpath("user_items_*.pairs.jsonl")
    filepath = filepath.as_posix()
    pair_paths = list(map(Path, glob.glob(filepath)))
    item_path = directories.data.joinpath("items.data")

    data_module = SkipGramDataModule(
        pair_paths=pair_paths,
        item_path=item_path,
        batch_size=DATAMODULE_BATCH_SIZE,
        num_workers=DATAMODULE_NUM_WORKERS,
    )

    item2vec = Item2VecModule(
        vocab_size=data_module.vocab_size,
        embed_dim=EMBED_DIM,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
    )

    wandb.init(project="item2vec", config=WANDB_CONFIG)
    wandb.watch(item2vec, log="all", log_freq=1)

    trainer = Trainer(
        limit_train_batches=TRAINER_LIMIT_TRAIN_BATCHES,
        limit_val_batches=TRAINER_LIMIT_VAL_BATCHES,
        limit_test_batches=TRAINER_LIMIT_TEST_BATCHES,
        max_epochs=TRAINER_MAX_EPOCHS,
        strategy=TRAINER_STRATEGY,
        precision=TRAINER_PRECISION,
        logger=WandbLogger(),
        callbacks=[
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIRPATH,
                monitor=CHECKPOINT_MONITOR,
                mode=CHECKPOINT_MODE,
                every_n_train_steps=CHECKPOINT_EVERY_N_TRAIN_STEPS,
            ),
        ],
    )
    trainer.fit(model=item2vec, datamodule=data_module)
    wandb.finish()


if __name__ == "__main__":
    main()
