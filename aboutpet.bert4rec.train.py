import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from item2vec.configs import settings as item2vec_settings
from reranker.configs import settings as bert4rec_settings
from item2vec.volume import Volume
from item2vec.models import GraphBPRItem2VecModule
from reranker.bert4rec import Bert4RecDataModule, Bert4RecModule

os.environ["WANDB_API_KEY"] = bert4rec_settings.wandb_api_key

# Models
EMBED_DIM = bert4rec_settings.embed_dim
NUM_HEADS = bert4rec_settings.num_heads
NUM_LAYERS = bert4rec_settings.num_layers
MAX_LEN = bert4rec_settings.max_len
DROPOUT = bert4rec_settings.dropout

# Optimizers and training envs
LR = bert4rec_settings.lr
WEIGHT_DECAY = bert4rec_settings.weight_decay

# Trainers
TRAINER_STRATEGY = bert4rec_settings.trainer_strategy
TRAINER_PRECISION = bert4rec_settings.trainer_precision
TRAINER_LIMIT_TRAIN_BATCHES = bert4rec_settings.trainer_limit_train_batches
TRAINER_LIMIT_VAL_BATCHES = bert4rec_settings.trainer_limit_val_batches
TRAINER_LIMIT_TEST_BATCHES = bert4rec_settings.trainer_limit_test_batches
TRAINER_MAX_EPOCHS = bert4rec_settings.trainer_max_epochs
TRAINER_PROFILER = bert4rec_settings.trainer_profiler

# DataModules
DATAMODULE_BATCH_SIZE = bert4rec_settings.datamodule_batch_size
DATAMODULE_NUM_WORKERS = bert4rec_settings.datamodule_num_workers
DATAMODULE_NEGATIVE_K = bert4rec_settings.datamodule_negative_k

# Checkpoints
CHECKPOINT_MONITOR = bert4rec_settings.checkpoint_monitor
CHECKPOINT_MODE = bert4rec_settings.checkpoint_mode
CHECKPOINT_EVERY_N_TRAIN_STEPS = bert4rec_settings.checkpoint_every_n_train_steps
CHECKPOINT_FILENAME = bert4rec_settings.checkpoint_filename
CHECKPOINT_DIRPATH = bert4rec_settings.checkpoint_dirpath
CKPT_PATH = bert4rec_settings.checkpoint_path

# Wandb
WANDB_CONFIG = bert4rec_settings.dict()


def main():
    bert4rec_settings.print()

    volume = Volume(site="aboutpet", model="item2vec", version="v1")

    item2vec_module = GraphBPRItem2VecModule.load_from_checkpoint(
        f"{item2vec_settings.checkpoint_dirpath}/last.ckpt",
        vocab_size=volume.vocab_size(),
        purchase_edge_index_path=volume.workspace_path.joinpath("edge.purchase.indices.csv"),
        embed_dim=128,
    )

    bert4rec_module = Bert4RecModule(
        num_items=volume.vocab_size(),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    item_embeddings = item2vec_module.get_graph_embeddings()
    bert4rec_module.import_item_embeddings(item_embeddings)

    datamodule = Bert4RecDataModule(
        volume=volume,
        batch_size=DATAMODULE_BATCH_SIZE,
        num_workers=DATAMODULE_NUM_WORKERS,
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
    trainer.fit(model=bert4rec_module, datamodule=datamodule, ckpt_path=CKPT_PATH)
    trainer.test(model=bert4rec_module, datamodule=datamodule, ckpt_path=CKPT_PATH)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
