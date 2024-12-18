from datetime import datetime

import torch

import directories
import carca
import tools
from item2vec.volume import Volume
from carca.modules import CARCA


def export_onnx():
    volume_i = Volume(company_id="aboutpet", model="item2vec", version="v1")
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")

    carca_config_path = directories.config("aboutpet", "carca", "v1")
    carca_settings = carca.configs.Settings.load(carca_config_path)

    checkpoint_path = volume_c.workspace_path.joinpath("checkpoints", "last.ckpt")
    model = CARCA.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_items=volume_i.vocab_size(),
        embed_dim=carca_settings.embed_dim,
        num_heads=carca_settings.num_heads,
        num_layers=carca_settings.num_layers,
        max_len=carca_settings.max_len,
        dropout=carca_settings.dropout,
        lr=carca_settings.lr,
        weight_decay=carca_settings.weight_decay,
    )
    model.eval()
    model.cpu()

    batch_size = 32
    max_len = 50
    num_candidates = 10
    num_items = volume_i.vocab_size()

    input_seqs = torch.randint(0, num_items, (batch_size, max_len), dtype=torch.int64)
    src_key_padding_mask = torch.randint(0, 2, (batch_size, max_len), dtype=torch.bool)
    last_idxs = torch.randint(0, max_len, (batch_size, 1), dtype=torch.int64)
    candidate_idxs = torch.randint(0, num_items, (batch_size, num_candidates), dtype=torch.int64)

    torch.onnx.export(
        model=model,
        args=(input_seqs, src_key_padding_mask, last_idxs, candidate_idxs),
        f=volume_c.workspace_path.joinpath("CARCA.onnx"),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_seqs", "src_key_padding_mask", "masked_idxs", "candidate_idxs"],
        output_names=["output"],
        dynamic_axes={
            "input_seqs": {0: "batch_size", 1: "sequence_length"},
            "src_key_padding_mask": {0: "batch_size", 1: "sequence_length"},
            "masked_idxs": {0: "batch_size", 1: "masked_idx_dim"},
            "candidate_idxs": {0: "batch_size", 1: "num_candidates"},
            "output": {0: "batch_size", 1: "num_scores"},
        },
    )


def export_sqlite3():
    volume_i = Volume(company_id="aboutpet", model="item2vec", version="v1")
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")

    tools.migrate_tables(
        old_sqlite3_path=volume_i.sqlite3_path,
        new_sqlite3_path=volume_c.sqlite3_path,
        tables=["item"],
    )


def compress():
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")
    tools.compress(
        file_paths=[volume_c.workspace_path.joinpath("CARCA.onnx"), volume_c.sqlite3_path],
        tar_gz_path=volume_c.workspace_path.joinpath(f"{datetime.now().strftime('%Y%m%d%H%M%S')}.tar.gz"),
    )


def remove():
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")
    volume_c.workspace_path.joinpath("CARCA.onnx").unlink()
    volume_c.sqlite3_path.unlink()


def main():
    export_onnx()
    export_sqlite3()
    compress()
    remove()


if __name__ == "__main__":
    main()
