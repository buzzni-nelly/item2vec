import glob
import os
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import directories
import wandb
from configs import settings
from item2vec import vocab
from item2vec.datasets import SkipGramDataModule
from item2vec.models import Item2VecModule

os.environ["WANDB_API_KEY"] = settings.wandb_api_key

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
CKPT_PATH = settings.checkpoint_path

# dataset path
PAIRS_PATH = directories.pairs.as_posix()
ITEM_PATH = directories.item.as_posix()

# Wandb
WANDB_CONFIG = settings.dict()


def main():
    try:
        pair_paths = glob.glob(PAIRS_PATH)
        pair_paths = list(map(Path, pair_paths))
        item_path = Path(ITEM_PATH)

        data_module = SkipGramDataModule(
            pair_paths=pair_paths,
            item_path=item_path,
            vocab_size=vocab.size(),
            batch_size=DATAMODULE_BATCH_SIZE,
            num_workers=DATAMODULE_NUM_WORKERS,
            negative_k=DATAMODULE_NEGATIVE_K,
        )

        item2vec = Item2VecModule(
            vocab_size=data_module.vocab_size,
            embed_dim=EMBED_DIM,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        wandb.init(project="item2vec", config=WANDB_CONFIG)
        wandb.watch(item2vec, log="all", log_freq=1)

        trainer = Trainer(
            limit_train_batches=TRAINER_LIMIT_TRAIN_BATCHES,
            max_epochs=TRAINER_MAX_EPOCHS,
            logger=WandbLogger(),
            profiler=TRAINER_PROFILER,
            callbacks=[
                ModelCheckpoint(
                    dirpath=CHECKPOINT_DIRPATH,
                    monitor=CHECKPOINT_MONITOR,
                    mode=CHECKPOINT_MODE,
                    every_n_train_steps=CHECKPOINT_EVERY_N_TRAIN_STEPS,
                    save_last=True,
                ),
            ],
        )
        trainer.fit(model=item2vec, datamodule=data_module, ckpt_path=CKPT_PATH)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
