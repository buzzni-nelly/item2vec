import argparse

import numpy as np
import onnx
import onnxruntime as ort
import requests
import torch
import math
from tqdm import tqdm

from carca.modules import CarcaDataModule
from item2vec.volume import Volume


def check_onnx(onnx_path: str):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("Completed loading onnx model & validation.")
    return model


def run_inference(session: ort.InferenceSession, input_data: dict):
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    input_feed = {name: data for name, data in zip(input_names, input_data.values())}
    outputs = session.run(output_names, input_feed)
    return outputs


def ndcg_at_k(sorted_pidxs, label_pidx, k=10):
    sorted_pidxs = list(sorted_pidxs)
    label_pidx = label_pidx.item()
    if label_pidx not in sorted_pidxs:
        return 0.0

    rank = sorted_pidxs.index(label_pidx)
    if rank < k:
        return 1 / math.log2(rank + 2)
    return 0.0


def send_performance_metric(
    company_id: str,
    model_type: str,
    model_version: str,
    metric_function: str,
    metric_value: float
):
    """
    Sends a POST request to the performance metric endpoint.

    Args:
        company_id (str): Company ID.
        model_type (str): Model type.
        model_version (str): Model version.
        metric_function (str): Metric function name.
        metric_value (float): Metric value.

    Returns:
        Response: The HTTP response from the server.
    """
    url = "http://aiaas.isolated.buzzni.com/metric-collector/performance_metric"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json"
    }
    payload = {
        "company_id": company_id,
        "model_type": model_type,
        "model_version": model_version,
        "metric_function": metric_function,
        "metric_value": metric_value
    }

    response = requests.post(url, headers=headers, json=payload)
    return response


def main(company_id: str, version: str):
    volume_i = Volume(company_id=company_id, model="item2vec", version=version)
    volume_c = Volume(company_id=company_id, model="carca", version=version)

    datamodule = CarcaDataModule(volume=volume_i, batch_size=64, num_workers=1)

    check_onnx(volume_c.onnx_path)
    session = ort.InferenceSession(volume_c.onnx_path)

    ndcg_scores = []
    val_loader = datamodule.val_dataloader()
    tqdm_iterator = tqdm(val_loader, desc="Evaluating", unit="batch")
    for batch in tqdm_iterator:
        seq_pidxs, cat1_cidxs, cat2_cidxs, cat3_cidxs, src_key_padding_mask, src_mask, label_pidxs = batch

        label_pidxs = label_pidxs.unsqueeze(-1)
        candidate_pidxs = torch.cat([torch.randint(0, volume_i.vocab_size(), (seq_pidxs.size(0), 1000)), label_pidxs], dim=-1)

        input_example = {
            "seq_pidxs": seq_pidxs.numpy(),
            "seq_cat1_cidxs": cat1_cidxs.numpy(),
            "seq_cat2_cidxs": cat2_cidxs.numpy(),
            "seq_cat3_cidxs": cat3_cidxs.numpy(),
            "src_key_padding_mask": src_key_padding_mask.numpy(),
            "src_mask": src_mask.numpy(),
            "candidate_pidxs": candidate_pidxs.numpy(),
        }

        topk_scores, topk_indices = run_inference(session, input_example)

        batch_ndcg = []
        for idxs, label in zip(topk_indices, label_pidxs):
            ndcg = ndcg_at_k(idxs, label)
            batch_ndcg.append(ndcg)

        ndcg_scores.extend(batch_ndcg)
        current_mean_ndcg = np.mean(ndcg_scores)
        tqdm_iterator.set_postfix({"ndcg@10": f"{current_mean_ndcg:.4f}"})

    mean_ndcg = np.mean(ndcg_scores)
    print("Final Mean NDCG@10_{1000}:", mean_ndcg)

    send_performance_metric(
        company_id=company_id,
        model_type="carca",
        model_version=version,
        metric_function="ndcg@10_{1000}",
        metric_value=float(mean_ndcg),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ONNX model with NDCG@10")
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
