import os

import torch

import directories
import item2vec
import reranker

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from item2vec.configs import Settings
from item2vec.volume import Volume
from item2vec.models import GraphBPRItem2Vec
from reranker.carca import CarcaDataModule, CARCA

item2vec_config_path = directories.config("aboutpet", "item2vec", "v1")
item2vec_settings = item2vec.configs.Settings.load(item2vec_config_path)

carca_config_path = directories.config("aboutpet", "carca", "v1")
carca_settings = reranker.configs.Settings.load(carca_config_path)

os.environ["WANDB_API_KEY"] = carca_settings.wandb_api_key

# Models
EMBED_DIM = carca_settings.embed_dim
NUM_HEADS = carca_settings.num_heads
NUM_LAYERS = carca_settings.num_layers
MAX_LEN = carca_settings.max_len
DROPOUT = carca_settings.dropout

# Optimizers and training envs
LR = carca_settings.lr
WEIGHT_DECAY = carca_settings.weight_decay

# Trainers
TRAINER_STRATEGY = carca_settings.trainer_strategy
TRAINER_PRECISION = carca_settings.trainer_precision
TRAINER_LIMIT_TRAIN_BATCHES = carca_settings.trainer_limit_train_batches
TRAINER_LIMIT_VAL_BATCHES = carca_settings.trainer_limit_val_batches
TRAINER_LIMIT_TEST_BATCHES = carca_settings.trainer_limit_test_batches
TRAINER_MAX_EPOCHS = carca_settings.trainer_max_epochs
TRAINER_PROFILER = carca_settings.trainer_profiler

# DataModules
DATAMODULE_BATCH_SIZE = carca_settings.datamodule_batch_size
DATAMODULE_NUM_WORKERS = carca_settings.datamodule_num_workers
DATAMODULE_NEGATIVE_K = carca_settings.datamodule_negative_k

# Checkpoints
CHECKPOINT_MONITOR = carca_settings.checkpoint_monitor
CHECKPOINT_MODE = carca_settings.checkpoint_mode
CHECKPOINT_EVERY_N_TRAIN_STEPS = carca_settings.checkpoint_every_n_train_steps
CHECKPOINT_FILENAME = carca_settings.checkpoint_filename
CHECKPOINT_DIRPATH = carca_settings.checkpoint_dirpath
CKPT_PATH = carca_settings.checkpoint_path

# Wandb
WANDB_CONFIG = carca_settings.dict()


def main():
    carca_settings.print()

    volume = Volume(company_id="aboutpet", model="item2vec", version="v1")

    item2vec_module = GraphBPRItem2Vec.load_from_checkpoint(
        f"{item2vec_settings.checkpoint_dirpath}/last.ckpt",
        vocab_size=volume.vocab_size(),
        purchase_edge_index_path=volume.workspace_path.joinpath("edge.purchase.indices.csv"),
        embed_dim=128,
    )

    carca = CARCA(
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
    carca.import_item_embeddings(item_embeddings)

    datamodule = CarcaDataModule(
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
    trainer.fit(model=carca, datamodule=datamodule, ckpt_path=CKPT_PATH)
    trainer.test(model=carca, datamodule=datamodule, ckpt_path=CKPT_PATH)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
