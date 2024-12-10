import os

import directories

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from item2vec.configs import Settings
from item2vec.datasets import SkipGramBPRDataModule
from item2vec.models import GraphBPRItem2VecModule
from item2vec.volume import Volume

settings = Settings.load(directories.config("gsshop", "item2vec", "kt"))

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
CHECKPOINT_FILENAME = settings.checkpoint_filename
CKPT_PATH = settings.checkpoint_path

# Wandb
WANDB_CONFIG = settings.dict()


def main():
    settings.print()

    volume = Volume(company_id="gsshop", model="item2vec", version="v1")

    data_module = SkipGramBPRDataModule(
        volume=volume,
        batch_size=DATAMODULE_BATCH_SIZE,
        num_workers=DATAMODULE_NUM_WORKERS,
        negative_k=DATAMODULE_NEGATIVE_K,
    )

    item2vec = GraphBPRItem2VecModule(
        vocab_size=data_module.vocab_size,
        purchase_edge_index_path=volume.workspace_path.joinpath("edge.purchase.indices.csv"),
        embed_dim=EMBED_DIM,
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
                dirpath=CHECKPOINT_DIRPATH,
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
    trainer.fit(model=item2vec, datamodule=data_module, ckpt_path=CKPT_PATH)


if __name__ == "__main__":
    main()
