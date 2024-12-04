import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from reranker.configs import settings
from item2vec.volume import Volume
from reranker.bert4rec import Bert4RecDataModule, Bert4RecModule

os.environ["WANDB_API_KEY"] = settings.wandb_api_key

# Models
EMBED_DIM = settings.embed_dim
NUM_HEADS = settings.num_heads
NUM_LAYERS = settings.num_layers
MAX_LEN = settings.max_len
DROPOUT = settings.dropout

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
CHECKPOINT_MONITOR = settings.checkpoint_monitor
CHECKPOINT_MODE = settings.checkpoint_mode
CHECKPOINT_EVERY_N_TRAIN_STEPS = settings.checkpoint_every_n_train_steps
CHECKPOINT_FILENAME = settings.checkpoint_filename
CKPT_PATH = settings.checkpoint_path

# Wandb
WANDB_CONFIG = settings.dict()


def main():
    settings.print()

    volume = Volume(site="aboutpet", model="item2vec", version="v1")
    CHECKPOINT_PATH = volume.workspace_path

    data_module = Bert4RecDataModule(
        volume=volume,
        batch_size=DATAMODULE_BATCH_SIZE,
        num_workers=DATAMODULE_NUM_WORKERS,
    )

    bert4rec = Bert4RecModule(
        num_items=volume.vocab_size(),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    trainer = Trainer(
        limit_train_batches=TRAINER_LIMIT_TRAIN_BATCHES,
        max_epochs=TRAINER_MAX_EPOCHS,
        logger=WandbLogger(),
        profiler=TRAINER_PROFILER,
        precision=TRAINER_PRECISION,
        callbacks=[
            ModelCheckpoint(
                dirpath=CHECKPOINT_PATH,
                monitor=CHECKPOINT_MONITOR,
                mode=CHECKPOINT_MODE,
                every_n_train_steps=CHECKPOINT_EVERY_N_TRAIN_STEPS,
                filename=CHECKPOINT_FILENAME,
                save_last=True,
            ),
            EarlyStopping(
                monitor=CHECKPOINT_MONITOR,
                mode=CHECKPOINT_MODE,
                patience=5,
                verbose=True,
            ),
        ],
    )
    trainer.fit(model=bert4rec, datamodule=data_module, ckpt_path=CKPT_PATH)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
