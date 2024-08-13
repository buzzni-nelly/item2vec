from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    embed_dim: int = 128
    encoder_dim: int = 128
    K: int = 20

    lr: float = 2e-4
    weight_decay: float = 1e-5
    dropout: float = 0.2

    datamodule_batch_size: int = 2**13
    datamodule_k: int = 19
    datamodule_num_workers: int = 6

    trainer_max_epochs: int = 1000
    trainer_limit_val_batches: int | float = 1.0
    trainer_limit_train_batches: int | float = 1.0
    trainer_limit_test_batches: int | float = 1.0
    trainer_strategy: str = "auto"  # deepspeed_stage_1
    trainer_precision: str = "16-mixed"

    checkpoint_dirpath: str = "checkpoints"
    checkpoint_monitor: str = "train_loss"
    checkpoint_mode: str = "min"
    checkpoint_every_n_train_steps: int = 0
    checkpoint_path: str | None = "last"

    download_source_path: str = "/item2vec/20240701-20240812"

    wandb_api_key: str = ""


settings = Settings()
