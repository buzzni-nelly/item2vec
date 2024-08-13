from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    embed_dim: int = 128
    encoder_dim: int = 128

    lr: float = 1e-3
    weight_decay: float = 1e-2

    datamodule_batch_size: int = 2 ** 14
    datamodule_num_workers: int = 22

    trainer_max_epochs: int = 1000
    trainer_limit_val_batches: int | float = 1.0
    trainer_limit_train_batches: int | float = 1.0
    trainer_limit_test_batches: int | float = 1.0
    trainer_strategy: str = "auto"  # deepspeed_stage_1
    trainer_precision: str = "16-mixed"

    checkpoint_dirpath: str = "checkpoints"
    checkpoint_monitor: str = "train_loss"
    checkpoint_mode: str = "min"
    checkpoint_every_n_train_steps: int = 1
    checkpoint_path: str | None = "last"

    wandb_api_key: str = "7290cd5cb94c29300893438a08b4b6aa844149f3"


settings = Settings()
