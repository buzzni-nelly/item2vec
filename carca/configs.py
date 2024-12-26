from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    company_id: str
    model: str
    version: str

    embed_dim: int = 128
    num_heads: int = 1
    num_layers: int = 1
    max_len: int = 50
    encoder_residual_strategy_1: str = "none"
    encoder_residual_strategy_2: str = "none"
    decoder_residual_strategy_1: str = "none"
    decoder_residual_strategy_2: str = "none"

    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-2

    datamodule_batch_size: int = 2**8
    datamodule_num_workers: int = 10
    datamodule_negative_k: int = 10

    trainer_max_epochs: int = 100
    trainer_limit_val_batches: int | float = 1.0
    trainer_limit_train_batches: int | float = 1.0
    trainer_limit_test_batches: int | float = 1.0
    trainer_strategy: str = "auto"  # deepspeed_stage_1
    trainer_precision: str = "16"
    trainer_profiler: str = "simple"

    checkpoint_dirpath: str = "/tmp/aboutpet/bert4rec/checkpoints"
    checkpoint_monitor: str = "val_ndcg@10"
    checkpoint_filename: str = "{epoch}-{step}-{train_loss:.2f}"
    checkpoint_mode: str = "max"
    checkpoint_every_n_train_steps: int = 10_000
    ckpt_path: str | None = "last"

    wandb_api_key: str = "7290cd5cb94c29300893438a08b4b6aa844149f3"

    def print(self):
        for k, v in self.model_dump().items():
            print(k, v)

    @staticmethod
    def load(filepath: Path) -> "Settings":
        import yaml

        with filepath.open("r") as file:
            config = yaml.safe_load(file)
            return Settings(**config)

    def to_dict(self):
        return self.model_dump()
