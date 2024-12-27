import argparse
import pathlib
import re
from datetime import datetime

import pytz
import torch
import z3

import carca
import directories
import tools
from carca.modules import CARCA
from item2vec.volume import Volume


def export_onnx(company_id: str, version: str):
    volume_i = Volume(company_id=company_id, model="item2vec", version=version)
    volume_c = Volume(company_id=company_id, model="carca", version=version)

    carca_config_path = directories.config(company_id, "carca", version)
    carca_settings = carca.configs.Settings.load(carca_config_path)

    checkpoint_path = volume_c.checkpoints_dirpath / "ndcg@10.max.ckpt"
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


def export_sqlite3(company_id: str, version: str):
    volume_i = Volume(company_id=company_id, model="item2vec", version=version)
    volume_c = Volume(company_id=company_id, model="carca", version=version)

    tools.migrate_tables(
        old_sqlite3_path=volume_i.sqlite3_path,
        new_sqlite3_path=volume_c.sqlite3_path,
        tables=["item", "category1", "category2", "category3"],
    )
    return volume_c.sqlite3_path


def compress(company_id: str, version: str):
    volume_c = Volume(company_id=company_id, model="carca", version=version)
    filename = datetime.now(tz=pytz.timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S")
    tools.compress(
        file_paths=[volume_c.onnx_path, volume_c.sqlite3_path],
        tar_gz_path=volume_c.workspace_path.joinpath(f"{filename}.tar.gz"),
    )
    return volume_c.workspace_path.joinpath(f"{filename}.tar.gz")


def upload_s3(filepath: pathlib.Path, company_id: str, version: str):
    z3.put_object(
        bucket_name="ailab-recommenders",
        object_key=f"{company_id}/carca/{version}/{filepath.name}",
        local_path=filepath,
    )


def clear_s3_retaining_k(company_id: str, version: str, k: int = 20):
    keys = z3.list_object_keys(
        bucket_name="ailab-recommenders",
        prefix=f"{company_id}/carca/{version}/",
    )
    print("Original keys:", keys)

    pattern = fr"^{company_id}/carca/{version}/\d+\.tar\.gz$"
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
    parser = argparse.ArgumentParser(description="Volume 작업을 실행합니다.")
    parser.add_argument(
        "--company-id",
        type=str,
        required=True,
        help="회사 ID를 입력하세요."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="버전 정보를 입력하세요."
    )
    args = parser.parse_args()

    company_id = args.company_id
    version = args.version

    _ = export_onnx(company_id, version)
    __ = export_sqlite3(company_id, version)
    tarfile_path = compress(company_id, version)
    upload_s3(tarfile_path, company_id, version)
    clear_s3_retaining_k(company_id, version, k=20)
    redeploy()


if __name__ == "__main__":
    main()
