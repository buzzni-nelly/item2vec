import json

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import clients
from item2vec.models import GraphBPRItem2VecModule
from item2vec.volume import Volume


def debug(
    target_item,
    top_k_items,
    top_k_scores,
    show: bool = False,
):
    if not show:
        return
    print("")
    print(
        target_item["name"],
        target_item["category1"],
        target_item["category2"],
        target_item["purchase_count"],
    )

    print("=" * 20)
    for item, score in zip(top_k_items[:10], top_k_scores[:10]):
        if target_item["category1"] != item["category1"]:
            continue
        print(
            item["name"],
            item["category1"],
            item["category2"],
            item["purchase_count"],
            round(float(score), 4),
        )
    print("=" * 20)


def load_items():
    items_df = pd.read_csv("/Users/nelly/PycharmProjects/item2vec/assets/items.csv")
    items_dict = items_df.to_dict(orient="records")
    items_dict = {item["pdid"]: item for item in items_dict}
    return items_dict


def load_embeddings(volume: Volume, embedding_dim: int = 128):
    model_path = "/tmp/checkpoints/last.ckpt"
    vocab_size = volume.vocab_size()
    item2vec_module = GraphBPRItem2VecModule.load_from_checkpoint(
        model_path, vocab_size=vocab_size, embedding_dim=embedding_dim
    )
    item2vec_module.eval()
    item2vec_module.freeze()

    embeddings = item2vec_module.get_graph_embeddings()
    return embeddings


def cosine_topk(
    embeddings: torch.Tensor, target: int, volume: Volume, k=100
):
    items = volume.items()
    similarities = F.cosine_similarity(embeddings[target].unsqueeze(0), embeddings, dim=1)
    top_k_values, top_k_pids = torch.topk(similarities, k)

    top_k_pdids = [volume.pid2pdid(int(x)) for x in top_k_pids]
    top_k_items = [items[pdid] for pdid in top_k_pdids]
    top_k_scores = top_k_values
    return top_k_items, top_k_pids, top_k_scores


def dot_product_topk(
    embeddings: torch.Tensor, target: int, volume: Volume, items: dict, k=100
):
    similarities = torch.mm(embeddings[target].unsqueeze(0), embeddings.T).squeeze(0)
    top_k_values, top_k_pids = torch.topk(similarities, k)

    top_k_pdids = [volume.pid2pdid(int(x)) for x in top_k_pids]
    top_k_items = [items[pdid] for pdid in top_k_pdids]
    top_k_scores = top_k_values
    return top_k_items, top_k_pids, top_k_scores


def rerank(
    top_k_items: list[dict],
    top_k_pids: torch.Tensor,
    top_k_scores: torch.Tensor,
    pid2pdid: dict,
    items: dict,
):
    k, device = len(top_k_items), top_k_pids.device

    top_k_p_counts = [x["purchase_count"] for x in top_k_items]
    top_k_c_counts = [x["click_count"] for x in top_k_items]

    top_k_soft_values = F.softmax(top_k_scores, dim=0)
    top_k_soft_p_values = F.softmax(
        torch.tensor(top_k_p_counts, device=device) / k, dim=0
    )
    top_k_soft_c_values = F.softmax(
        torch.tensor(top_k_c_counts, device=device) / (k**2), dim=0
    )

    combined_values = top_k_soft_values + top_k_soft_p_values + top_k_soft_c_values
    combined_values, merged_soft_indices = torch.sort(combined_values, descending=True)

    reranked_indices = top_k_pids[merged_soft_indices]
    reranked_k_pdids = [pid2pdid[int(x)] for x in reranked_indices]
    reranked_items = [items[pdid] for pdid in reranked_k_pdids]
    reranked_scores = combined_values
    return reranked_items, reranked_scores


def main(embed_dim=128, candidate_k: int = 100):
    # Load vocabulary and model

    volume = Volume(site="aboutpet", model="item2vec", version="v1")

    embeddings = load_embeddings(volume, embedding_dim=embed_dim)
    items = volume.items()

    unknown_pids = [x["pid"] for x in items.values() if x["name"] == "UNKNOWN"]
    embeddings[unknown_pids] = torch.zeros(embed_dim, device=embeddings.device)

    cos_original_predictions, cos_reranked_predictions = {}, {}
    for pid in tqdm(
        range(embeddings.shape[0]), desc="추천 점수를 계산 및 Redis 할당 중입니다.."
    ):
        if pid in unknown_pids:
            continue

        pdid = volume.pid2pdid(pid)
        query_item = items[pdid]
        category1, category2 = query_item["category1"], query_item["category2"]
        k = candidate_k

        cos_top_k_items, cos_top_k_pids, cos_top_k_scores = cosine_topk(
            embeddings=embeddings,
            target=pid,
            volume=volume,
            k=k,
        )

        debug(
            query_item,
            cos_top_k_items,
            cos_top_k_scores,
            show=False,
        )

        cos_original_predictions[pdid] = [
            {"pdid": x["pdid"], "score": round(float(s), 4)}
            for x, s in zip(cos_top_k_items, cos_top_k_scores)
            if category1 == x["category1"]
        ]

    pipeline = clients.redis.aiaas_6.pipeline()
    for k, v in cos_original_predictions.items():
        key = f"i2i:aboutpet:i2v:v2:{k}"
        pipeline.set(key, json.dumps(v))
        pipeline.expire(key, 30 * 24 * 60 * 60)

    pipeline.execute()

    print(len(cos_reranked_predictions))


if __name__ == "__main__":
    main()
