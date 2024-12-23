import pathlib
import re
from datetime import datetime

import torch
import z3

import carca
import directories
import tools
from carca.modules import CARCA
from item2vec.volume import Volume


def export_onnx():
    volume_i = Volume(company_id="aboutpet", model="item2vec", version="v1")
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")

    carca_config_path = directories.config("aboutpet", "carca", "v1")
    carca_settings = carca.configs.Settings.load(carca_config_path)

    checkpoint_path = volume_c.checkpoints_dirpath.joinpath("last.ckpt")
    num_items = volume_i.vocab_size()
    num_category1, num_category2, num_category3 = volume_i.count_categories()
    model = CARCA.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_items=num_items,
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
    model.eval()
    model.cpu()

    batch_size = 32
    max_len = 50
    num_candidates = 10
    num_items = volume_i.vocab_size()
    num_category1, num_category2, num_category3 = volume_i.count_categories()

    seq_pidxs = torch.randint(0, num_items, (batch_size, max_len), dtype=torch.int64)
    seq_cat1_cidxs = torch.randint(0, num_category1, (batch_size, max_len), dtype=torch.int64)
    seq_cat2_cidxs = torch.randint(0, num_category2, (batch_size, max_len), dtype=torch.int64)
    seq_cat3_cidxs = torch.randint(0, num_category3, (batch_size, max_len), dtype=torch.int64)
    src_key_padding_mask = torch.randint(0, 2, (batch_size, max_len), dtype=torch.bool)
    src_mask = torch.randint(0, max_len, (batch_size, 1), dtype=torch.int64)
    candidate_pidxs = torch.randint(0, num_items, (batch_size, num_candidates), dtype=torch.int64)

    torch.onnx.export(
        model=model,
        args=(
            seq_pidxs,
            seq_cat1_cidxs,
            seq_cat2_cidxs,
            seq_cat3_cidxs,
            src_key_padding_mask,
            src_mask,
            candidate_pidxs,
        ),
        f=volume_c.onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=[
            "seq_pidxs",
            "seq_cat1_cidxs",
            "seq_cat2_cidxs",
            "seq_cat3_cidxs",
            "src_key_padding_mask",
            "src_mask",
            "candidate_pidxs",
        ],
        output_names=["output"],
        dynamic_axes={
            "seq_pidxs": {0: "batch_size", 1: "sequence_length"},
            "seq_cat1_cidxs": {0: "batch_size", 1: "sequence_length"},
            "seq_cat2_cidxs": {0: "batch_size", 1: "sequence_length"},
            "seq_cat3_cidxs": {0: "batch_size", 1: "sequence_length"},
            "src_key_padding_mask": {0: "batch_size", 1: "sequence_length"},
            "src_mask": {0: "batch_size", 1: "dim_src_mask"},
            "candidate_pidxs": {0: "batch_size", 1: "num_candidates"},
            "output": {0: "batch_size", 1: "num_scores"},
        },
    )
    return volume_c.onnx_path


def export_sqlite3():
    volume_i = Volume(company_id="aboutpet", model="item2vec", version="v1")
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")

    tools.migrate_tables(
        old_sqlite3_path=volume_i.sqlite3_path,
        new_sqlite3_path=volume_c.sqlite3_path,
        tables=["item", "category1", "category2", "category3"],
    )
    return volume_c.sqlite3_path


def clear_workspace():

    def clear_directory(directory_path: pathlib.Path):
        """재귀적으로 디렉토리 내 모든 파일 및 하위 디렉토리를 삭제합니다."""
        for x in directory_path.iterdir():
            if x.is_file():
                x.unlink()
            elif x.is_dir():
                clear_directory(x)
        directory_path.rmdir()

    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")

    workspace_path = volume_c.workspace_path

    if workspace_path.exists() and workspace_path.is_dir():
        for item in workspace_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                clear_directory(item)


def compress():
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")
    filename = datetime.now().strftime("%Y%m%d%H%M%S")
    tools.compress(
        file_paths=[volume_c.onnx_path, volume_c.sqlite3_path],
        tar_gz_path=volume_c.workspace_path.joinpath(f"{filename}.tar.gz"),
    )
    return volume_c.workspace_path.joinpath(f"{filename}.tar.gz")


def upload_s3(filepath: pathlib.Path = None):
    z3.put_object(
        bucket_name="ailab-recommenders",
        object_key=f"aboutpet/carca/v1/{filepath.name}",
        local_path=filepath,
    )


def clear_s3_retaining_k(k: int = 20):
    keys = z3.list_object_keys(
        bucket_name="ailab-recommenders",
        prefix="aboutpet/carca/v1/",
    )
    print("Original keys:", keys)

    pattern = r"^aboutpet/carca/v1/\d{14}\.tar\.gz$"
    filtered_keys = [key for key in keys if re.match(pattern, key)]
    filtered_keys.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]), reverse=True)
    retained_keys = filtered_keys[:k]
    files_to_delete = set(filtered_keys) - set(retained_keys)
    print("Files to delete:", files_to_delete)

    for file_key in files_to_delete:
        z3.delete_object(bucket_name="ailab-recommenders", object_key=file_key)
        print(f"Deleted: {file_key}")

    print("Retained keys:", retained_keys)
    return retained_keys


def redeploy():
    tools.redeploy(
        deployment="aiaas-aboutpet-carca-v1-inference-deployment",
        namespace="aiaas-inference-dev",
        context="service-eks",
    )
    tools.redeploy(
        deployment="aiaas-aboutpet-carca-v1-inference-deployment",
        namespace="aiaas-inference-prod",
        context="service-eks",
    )


def main():
    # export_onnx()
    # export_sqlite3()
    # tarfile_path = compress()
    # upload_s3(tarfile_path)
    # clear_workspace()
    # clear_s3_retaining_k(k=20)
    redeploy()


if __name__ == "__main__":
    main()
