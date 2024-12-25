import numpy as np
import onnx
import onnxruntime as ort
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


if __name__ == "__main__":

    volume_i = Volume(company_id="aboutpet", model="item2vec", version="v1")
    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")

    datamodule = CarcaDataModule(volume=volume_i, batch_size=64, num_workers=1)

    check_onnx(volume_c.onnx_path)
    session = ort.InferenceSession(volume_c.onnx_path)

    ndcg_scores = []
    val_loader = datamodule.val_dataloader()
    tqdm_iterator = tqdm(val_loader, desc="Evaluating", unit="batch")
    for batch in tqdm_iterator:
        seq_pidxs, cat1_cidxs, cat2_cidxs, cat3_cidxs, src_key_padding_mask, src_mask, label_pidxs = batch

        label_pidxs = label_pidxs.unsqueeze(-1)
        candidate_pidxs = torch.cat([torch.randint(0, 15000, (seq_pidxs.size(0), 100)), label_pidxs], dim=-1)

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
    print("Final Mean NDCG@10:", mean_ndcg)
