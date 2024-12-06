from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    max_len: int = 50
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

    checkpoint_dirpath: str = "/tmp/bert4rec/checkpoints"
    checkpoint_monitor: str = "val_ndcg@20"
    checkpoint_filename: str = "{epoch}-{step}-{train_loss:.2f}"
    checkpoint_mode: str = "max"
    checkpoint_every_n_train_steps: int = 10_000
    checkpoint_path: str | None = "last"

    wandb_api_key: str = "7290cd5cb94c29300893438a08b4b6aa844149f3"

    def print(self):
        for k, v in self.model_dump().items():
            print(k, v)


settings = Settings()
