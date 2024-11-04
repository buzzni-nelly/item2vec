import json
import math

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import clients
from item2vec import vocab
from item2vec.models import Item2VecModule


def debug(
    target_item,
    top_k_items,
    top_k_scores,
    reranked_items,
    reranked_scores,
    show: bool = False,
):
    if not show:
        return

    print(
        target_item["name"],
        target_item["category1"],
        target_item["category2"],
        target_item["click_count"],
    )
    print("=" * 10)
    count = 0
    for item, score in zip(top_k_items, top_k_scores):
        if target_item["category1"] != item["category1"]:
            continue
        print(
            item["name"],
            item["category1"],
            item["category2"],
            item["purchase_count"],
            score,
        )
        count += 1
        if count > 10:
            break
    print("**" * 10)
    count = 0
    for item, score in zip(reranked_items, reranked_scores):
        if target_item["category1"] != item["category1"]:
            continue
        print(
            item["name"],
            item["category1"],
            item["category2"],
            item["purchase_count"],
            score,
        )
        count += 1
        if count > 10:
            break
    print("=" * 10)


def load_items():
    items_df = pd.read_csv("/Users/nelly/PycharmProjects/item2vec/assets/items.csv")
    items_dict = items_df.to_dict(orient="records")
    items_dict = {item["pdid"]: item for item in items_dict}
    return items_dict


def load_embeddings(embed_dim=64):
    model_path = "/tmp/checkpoints/last.ckpt"
    vocab_size = vocab.size()
    item2vec_module = Item2VecModule.load_from_checkpoint(
        model_path, vocab_size=vocab_size, embed_dim=embed_dim
    )
    item2vec_module.eval()
    item2vec_module.freeze()

    embeddings = item2vec_module.item2vec.embeddings.weight.data
    return embeddings


def main(embed_dim=64):
    # Load vocabulary and model
    pdid2pid = vocab.load()
    pid2pdid = {v: k for k, v in pdid2pid.items()}

    # embeddings.shape = torch.Size([34728, 64])
    embeddings = load_embeddings(embed_dim=embed_dim)
    items_dict = load_items()
    device = embeddings.device

    unknown_pids = [x["pid"] for x in items_dict.values() if x["name"] == "UNKNOWN"]
    embeddings[unknown_pids] = torch.zeros(embed_dim, device=device)

    pipeline = clients.redis.aiaas_6.pipeline()
    for i in tqdm(
        range(embeddings.shape[0]), desc="추천 점수를 계산 및 Redis 할당 중입니다.."
    ):
        if i in unknown_pids:
            continue

        pdid = pid2pdid[i]
        target_item = items_dict[pdid]
        category1, category2 = target_item["category1"], target_item["category2"]
        device = embeddings.device

        similarities = F.cosine_similarity(
            embeddings[i].unsqueeze(0), embeddings, dim=1
        )
        # similarities = torch.matmul(embeddings[i], embeddings.T)

        top_k_values, top_k_indices = torch.topk(similarities, 100)

        top_k_pdids = [pid2pdid[int(x)] for x in top_k_indices]
        top_k_items = [items_dict[pdid] for pdid in top_k_pdids]
        top_k_p_counts = [x["purchase_count"] for x in top_k_items]
        top_k_scores = top_k_values

        category_scores = [
            1 if x["category2"] == category2 else 0.9 for x in top_k_items
        ]
        top_k_soft_values = F.softmax(top_k_values * math.e, dim=0)
        top_k_soft_p_counts = F.softmax(
            torch.tensor(top_k_p_counts, device=device) / math.e, dim=0
        )

        merged_soft_values = top_k_soft_values + top_k_soft_p_counts
        merged_soft_values = merged_soft_values * torch.tensor(
            category_scores, device=device
        )

        merged_soft_values, merged_soft_indices = torch.sort(
            merged_soft_values, descending=True
        )
        reranked_indices = top_k_indices[merged_soft_indices]

        reranked_k_pdids = [pid2pdid[int(x)] for x in reranked_indices]
        reranked_items = [items_dict[pdid] for pdid in reranked_k_pdids]
        reranked_scores = merged_soft_values

        click_count = target_item["click_count"]
        if click_count < 50:
            continue

        debug(
            target_item,
            top_k_items,
            top_k_scores,
            reranked_items,
            reranked_scores,
            show=False,
        )

        scores = [
            {"pdid": x["pdid"], "score": round(float(s), 4)}
            for x, s in zip(reranked_items, reranked_scores)
            if category1 == x["category1"]
        ]

        key = f"i2i:aboutpet:i2v:v1:{pdid}"
        pipeline.set(key, json.dumps(scores))
        pipeline.expire(key, 30 * 24 * 60 * 60)

    pipeline.execute()


if __name__ == "__main__":
    main()
