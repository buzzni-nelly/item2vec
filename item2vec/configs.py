from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    company_id: str
    model: str
    version: str

    # dataset
    skipgram_window_size: int = 5
    blacklist: list[str] = []

    # train
    embed_dim: int = 128  # 32
    num_layers: int = 2

    lr: float = 1e-3
    weight_decay: float = 1e-2

    datamodule_batch_size: int = 2**8
    datamodule_num_workers: int = 10
    datamodule_negative_k: int = 10

    trainer_max_epochs: int = 2
    trainer_limit_val_batches: int | float = 1.0
    trainer_limit_train_batches: int | float = 1.0
    trainer_limit_test_batches: int | float = 1.0
    trainer_strategy: str = "auto"  # deepspeed_stage_1
    trainer_precision: str = "16"
    trainer_profiler: str = "simple"

    checkpoint_dirpath: str = "/tmp/aboutpet/item2vec/checkpoints"
    checkpoint_monitor: str = "val_graph_dot_ndcg@20"
    checkpoint_filename: str = "{epoch}-{step}-{train_loss:.2f}"
    checkpoint_mode: str = "max"
    checkpoint_every_n_train_steps: int = 10_000
    ckpt_path: str | None = "last"

    def print(self):
        for k, v in self.model_dump().items():
            print(k, v)

    @staticmethod
    def load(filepath: Path) -> "Settings":
        with filepath.open("r") as file:
            config = yaml.safe_load(file)
            return Settings(**config)

    def to_dict(self):
        return self.model_dump()

