import os
import pathlib
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from configs import settings
from item2vec.datasets import SkipGramBPRDataModule
from item2vec.models import GraphBPRItem2VecModule
from item2vec.volume import Volume
from scripts import script_5

os.environ["WANDB_API_KEY"] = settings.wandb_api_key
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


def delete_checkpoints():
    directory = pathlib.Path(settings.checkpoint_dirpath)
    for file in directory.glob("*"):
        if file.is_file():
            file.unlink()


def main():
    try:
        delete_checkpoints()

        settings.print()

        volume = Volume(site="aboutpet", model="item2vec", version="v1")
        volume.migrate_traces(start_date=datetime(2024, 8, 1))
        volume.migrate_items()
        volume.generate_pairs_csv()
        volume.generate_edge_indices_csv()

        data_module = SkipGramBPRDataModule(
            volume=volume,
            batch_size=DATAMODULE_BATCH_SIZE,
            num_workers=DATAMODULE_NUM_WORKERS,
            negative_k=DATAMODULE_NEGATIVE_K,
        )

        item2vec = GraphBPRItem2VecModule(
            vocab_size=data_module.vocab_size,
            edge_index_path=volume.workspace_path.joinpath("edge.indices.csv"),
            embed_dim=EMBED_DIM,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        # wandb.init(project="item2vec", config=WANDB_CONFIG)
        # wandb.watch(item2vec, log="all", log_freq=1)

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
                    filename=CHECKPOINT_FILENAME,
                    save_last=True,
                ),
                EarlyStopping(
                    monitor=CHECKPOINT_MONITOR,
                    mode=CHECKPOINT_MODE,
                    patience=2,
                    verbose=True
                )
            ],
        )
        trainer.fit(model=item2vec, datamodule=data_module, ckpt_path=CKPT_PATH)
        script_5.main()
    except Exception as e:
        print(e)
    finally:
        pass
        # wandb.finish()


if __name__ == "__main__":
    while True:
        main()
