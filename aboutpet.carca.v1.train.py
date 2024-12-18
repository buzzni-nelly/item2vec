import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import directories
from carca.configs import Settings as CarcaSettings
from carca.modules import CarcaDataModule, CARCA
from item2vec.configs import Settings as Item2vecSettings
from item2vec.modules import GraphBPRItem2Vec
from item2vec.volume import Volume


def main():
    item2vec_config_path = directories.config("aboutpet", "item2vec", "v1")
    carca_config_path = directories.config("aboutpet", "carca", "v1")
    item2vec_settings = Item2vecSettings.load(item2vec_config_path)
    carca_settings = CarcaSettings.load(carca_config_path)
    carca_settings.print()

    logger = WandbLogger(project="aboutpet.carca")
    logger.log_hyperparams(item2vec_settings.to_dict())

    volume_i = Volume(company_id="aboutpet", model="item2vec", version="v1")
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")

    item2vec_checkpoint_path = volume_i.workspace_path.joinpath("checkpoints", "last.ckpt")
    carca_checkpoint_dir_path = volume_c.workspace_path.joinpath("checkpoints")

    purchase_edge_index_path = volume_i.workspace_path.joinpath("edge.purchase.indices.csv")
    item2vec_module = GraphBPRItem2Vec.load_from_checkpoint(
        checkpoint_path=item2vec_checkpoint_path,
        vocab_size=volume_i.vocab_size(),
        purchase_edge_index_path=purchase_edge_index_path,
        embed_dim=item2vec_settings.embed_dim,
        num_layers=item2vec_settings.num_layers,
    )

    num_category1, num_category2, num_category3 = volume_i.count_categories()
    carca = CARCA(
        num_items=volume_i.vocab_size(),
        num_category1=num_category1,
        num_category2=num_category2,
        num_category3=num_category3,
        embed_dim=carca_settings.embed_dim,
        num_heads=carca_settings.num_heads,
        num_layers=carca_settings.num_layers,
        max_len=carca_settings.max_len,
        dropout=carca_settings.dropout,
        lr=carca_settings.lr,
        weight_decay=carca_settings.weight_decay,
    )

    item_embeddings = item2vec_module.get_graph_embeddings(num_layers=item2vec_settings.num_layers)
    carca.import_item_embeddings(item_embeddings)

    datamodule = CarcaDataModule(
        volume=volume_i,
        batch_size=carca_settings.datamodule_batch_size,
        num_workers=carca_settings.datamodule_num_workers,
    )

    trainer = Trainer(
        limit_train_batches=carca_settings.trainer_limit_train_batches,
        max_epochs=carca_settings.trainer_max_epochs,
        logger=logger,
        profiler=carca_settings.trainer_profiler,
        precision=carca_settings.trainer_precision,
        callbacks=[
            ModelCheckpoint(
                dirpath=carca_checkpoint_dir_path,
                monitor=carca_settings.checkpoint_monitor,
                mode=carca_settings.checkpoint_mode,
                every_n_train_steps=carca_settings.checkpoint_every_n_train_steps,
                filename=carca_settings.checkpoint_filename,
                save_last=True,
            ),
            EarlyStopping(
                monitor=carca_settings.checkpoint_monitor,
                mode=carca_settings.checkpoint_mode,
                patience=10,
                verbose=True,
            ),
        ],
    )
    trainer.fit(model=carca, datamodule=datamodule, ckpt_path=carca_settings.ckpt_path)
    trainer.test(model=carca, datamodule=datamodule, ckpt_path=carca_settings.ckpt_path)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
