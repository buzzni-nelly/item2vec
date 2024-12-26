import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import directories
from item2vec.configs import Settings as Item2vecSettings
from item2vec.datasets import SkipGramBPRDataModule
from item2vec.modules import GraphBPRItem2Vec
from item2vec.volume import Volume


def main(company_id: str, version: str):
    model = "item2vec"
    config_path = directories.config(company_id, model, version)
    item2vec_settings = Item2vecSettings.load(config_path)
    item2vec_settings.print()

    logger = WandbLogger(project=f"{company_id}.{model}")
    logger.log_hyperparams(item2vec_settings.to_dict())

    volume = Volume(company_id=company_id, model=model, version=version)

    data_module = SkipGramBPRDataModule(
        volume=volume,
        batch_size=item2vec_settings.datamodule_batch_size,
        num_workers=item2vec_settings.datamodule_num_workers,
        negative_k=item2vec_settings.datamodule_negative_k,
    )

    purchase_edge_index_path = volume.workspace_path.joinpath("edge.purchase.indices.csv")
    item2vec = GraphBPRItem2Vec(
        vocab_size=data_module.vocab_size,
        purchase_edge_index_path=purchase_edge_index_path,
        embed_dim=item2vec_settings.embed_dim,
        lr=item2vec_settings.lr,
        weight_decay=item2vec_settings.weight_decay,
        num_layers=item2vec_settings.num_layers,
    )

    trainer = Trainer(
        limit_train_batches=item2vec_settings.trainer_limit_train_batches,
        max_epochs=item2vec_settings.trainer_max_epochs,
        logger=logger,
        profiler=item2vec_settings.trainer_profiler,
        precision=item2vec_settings.trainer_precision,
        callbacks=[
            ModelCheckpoint(
                dirpath=volume.checkpoints_dirpath,
                monitor=item2vec_settings.checkpoint_monitor,
                mode=item2vec_settings.checkpoint_mode,
                every_n_train_steps=item2vec_settings.checkpoint_every_n_train_steps,
                filename=item2vec_settings.checkpoint_filename,
                save_last=True,
            ),
            EarlyStopping(
                monitor=item2vec_settings.checkpoint_monitor,
                mode=item2vec_settings.checkpoint_mode,
                patience=5,
                verbose=True,
            ),
        ],
    )
    trainer.fit(model=item2vec, datamodule=data_module, ckpt_path=item2vec_settings.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning Trainer 실행")
    parser.add_argument(
        "--company-id",
        type=str,
        required=True,
        help="처리할 회사 ID를 입력하세요."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="모델 버전을 입력하세요."
    )
    args = parser.parse_args()

    main(args.company_id, args.version)
